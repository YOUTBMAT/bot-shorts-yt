"""
Bot Automático de Publicação de YouTube Shorts.

Pipeline: gera roteiro (Gemini) -> gera narração (TTS: Google Cloud -> OpenAI
-> ElevenLabs, com fallback automático) -> transcreve
a narração com timestamps por palavra (faster-whisper) -> monta vídeo com
legendas sincronizadas palavra-a-palavra e legendas dinâmicas (MoviePy) ->
publica no YouTube.

Este arquivo mantém a mesma arquitetura do projeto original (tudo em um
único arquivo). A única dependência nova é `faster-whisper`, necessária
para a sincronização de legendas por palavra — ver justificativa técnica
na seção de transcrição, mais abaixo.

Fase 6: a biblioteca de mídia (seção 0) ganhou uma camada `MediaProvider`
(`LocalMediaProvider` / `PexelsMediaProvider`) por trás do `MediaManager`,
permitindo buscar vídeos de fundo automaticamente no Pexels quando
`PEXELS_API_KEY` está configurada, com fallback automático e transparente
para a biblioteca local se o Pexels falhar. Nenhuma dependência nova foi
adicionada (usa `urllib` da stdlib).
"""

import os
import re
import json
import math
import time
import random
import bisect
import unicodedata
import urllib.request
import urllib.error
import urllib.parse
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable

import numpy as np
from PIL import Image

# Compat: moviepy 1.0.3 usa `Image.ANTIALIAS` internamente (resizer PIL, em
# moviepy/video/fx/resize.py — usado quando o OpenCV/cv2 não está
# disponível, e este projeto NÃO depende de opencv). `Image.ANTIALIAS` foi
# removido no Pillow >= 10.0 (era só um alias de `Image.LANCZOS` desde o
# Pillow 2.7, mantido por compatibilidade). Sem este shim, TODO
# `clip.resize(...)` (usado no Ken Burns e no redimensionamento do vídeo de
# fundo) lança `AttributeError: module 'PIL.Image' has no attribute
# 'ANTIALIAS'`. Não altera nada além de re-registrar o nome antigo — o
# `Image.BILINEAR` usado diretamente mais abaixo neste arquivo não foi
# afetado pela remoção (só o ANTIALIAS foi removido no Pillow 10; os demais
# filtros nomeados como constante, como BILINEAR, foram restaurados na 9.4).
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech
from openai import OpenAI
from faster_whisper import WhisperModel
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    VideoClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
import moviepy.audio.fx.all as afx
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
from models.media import FundoInvalidoError, MediaItem

try:
    import resource
except ImportError:
    resource = None


# ===========================================================================
# CONFIGURAÇÕES
# ===========================================================================
# Todas as constantes do projeto ficam centralizadas aqui. Nenhum "número
# mágico" deve aparecer solto no restante do código.

# --- Roteiro (Gemini) ---
MODEL_PRIORITY: List[str] = ["gemini-3.6-flash"]
MAX_RETRIES: int = 2  # tentativas por modelo antes de passar para o próximo
ROTEIRO_MIN_PALAVRAS: int = 65
ROTEIRO_MAX_PALAVRAS: int = 85
ROTEIRO_TOLERANCIA_PALAVRAS: int = 15  # margem aceita além do min/max antes de rejeitar o roteiro
TITULO_MAX_CARACTERES: int = 90
RETRY_BACKOFF_BASE: float = 1.0  # segundos; dobra a cada nova tentativa (1, 2, 4...)
RETRY_BACKOFF_JITTER: float = 0.5  # variação aleatória (0 a X segundos) somada à espera

# --- Narração (TTS com fallback entre provedores) ---
# Prioridade automática, decidida em runtime pelas variáveis de ambiente
# disponíveis (ver _provedores_tts_configurados()): OpenAI TTS -> Google
# Cloud TTS -> ElevenLabs (mantido como fallback opcional/legado).
GOOGLE_TTS_VOICE_NAME: str = "pt-BR-Chirp3-HD-Charon"  # voz HD (Chirp 3) em pt-BR
GOOGLE_TTS_LANGUAGE_CODE: str = "pt-BR"
GOOGLE_TTS_TIMEOUT_SECONDS: float = 120.0

OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE: str = "onyx"  # voz grave, natural para narração de Shorts
OPENAI_TTS_TIMEOUT_SECONDS: float = 120.0

VOICE_ID: str = "pNInz6obpgDQGcFmaJgB"  # ElevenLabs, voz "Adam" (fallback legado)
ELEVEN_MODEL_ID: str = "eleven_multilingual_v2"
ELEVENLABS_TIMEOUT_SECONDS: float = 120.0  # evita que uma chamada travada prenda o job do GitHub Actions

TTS_TENTATIVAS_POR_PROVEDOR: int = 2  # antes de cair para o próximo provedor; mesmo padrão do MODEL_PRIORITY do Gemini

NARRACAO_PATH: str = "narracao.mp3"

# --- Vídeo (MoviePy) ---
VIDEO_FUNDO_PATH: str = "fundo.mp4"  # fallback legado: usado só se a biblioteca de mídia estiver vazia
VIDEO_SAIDA_PATH: str = "video_final.mp4"
VIDEO_WIDTH: int = 1080
ASPECT_RATIO: float = 9 / 16  # formato vertical (Shorts)
# Altura EXATA correspondente a VIDEO_WIDTH em 9:16 — usada como alvo fixo
# (não apenas como resultado "esperado" de um resize proporcional) para
# forçar TODO segmento de fundo a ter sempre a mesma altura em pixels. Ver
# `_redimensionar_para_vertical()` e `normalize_frame_size()`: sem isso,
# `clip.resize(width=VIDEO_WIDTH)` deixa a altura final a cargo do
# arredondamento interno do MoviePy a partir da altura ORIGINAL de cada
# vídeo de fundo — que varia de arquivo para arquivo — e diferentes
# trechos de B-roll (Fase 4) podiam terminar com alturas ligeiramente
# diferentes (ex.: 1918 vs 1921px) mesmo depois do crop 9:16. Isso não dava
# erro no crop/resize em si (cada segmento é internamente consistente) —
# só estourava depois, na composição final, quando um segmento posterior
# do vídeo concatenado (com sua própria altura) era desenhado sob máscaras/
# overlays construídos contra o tamanho do PRIMEIRO segmento
# (`largura_final, altura_final = fundo_com_vinheta.size`), causando
# exatamente `operands could not be broadcast together with shapes
# (1921,1080,3) (1918,1080,1)`.
VIDEO_HEIGHT: int = round(VIDEO_WIDTH / ASPECT_RATIO)
CROP_OFFSET_MAX_FRACTION: float = 0.35  # quanto do espaço sobrando o corte pode usar p/ não centralizar
DURACAO_MINIMA_SEGURANCA: float = 0.05  # segundos; piso usado contra timestamps/durações degeneradas

# --- Múltiplos B-rolls por vídeo (Fase 4) ---
# Em vez de um único fundo do início ao fim, o Short é dividido em trechos
# e cada trecho usa um vídeo de fundo diferente (com seu próprio Ken
# Burns) — ver `_definir_segmentos_broll()` / `_montar_fundo_multiplo()`.
BROLL_SEGMENTO_DURACAO_ALVO: float = 6.0  # segundos; duração alvo de cada trecho de B-roll
BROLL_SEGMENTO_DURACAO_MINIMA: float = 3.0  # segundos; nunca fecha um trecho final menor que isso sozinho
BROLL_MAX_SEGMENTOS: int = 6  # teto de trechos por vídeo (também o nº de candidatos pedidos ao MediaManager)
BROLL_MAX_TENTATIVAS_POR_SEGMENTO: int = 2  # candidatos alternativos testados por trecho antes de reaproveitar o último que funcionou
# Folga mínima (fim do vídeo de origem menos a duração que o trecho
# precisa) para variar o ponto inicial do corte em vez de sempre começar
# em 0s (Fase 2, item 12: evita sempre usar os mesmos primeiros segundos
# de um vídeo de fundo comprido — normalmente intro/logo/frame estático).
BROLL_FOLGA_MINIMA_PARA_VARIAR_INICIO: float = 1.0  # segundos

# --- B-roll contextual por trecho (Fase 2) ---
# Cada trecho de B-roll (ver BROLL_SEGMENTO_DURACAO_ALVO acima) busca
# candidatos com base no texto REALMENTE narrado NAQUELE trecho (não o
# roteiro inteiro) — ver `_texto_do_segmento()` / `_selecionar_broll_por_segmento()`.
BROLL_CANDIDATOS_POR_TRECHO_ALVO: int = 3  # candidatos pontuados o bastante para parar de buscar mais queries
BROLL_MAX_QUERIES_POR_TRECHO: int = 4  # cascata de queries tentadas por trecho antes de desistir (item 19: limite de requests)
# Pesos do score de cada candidato (soma 1.0) — ver `_pontuar_candidato_broll()`.
BROLL_PESO_RELEVANCIA: float = 0.50   # posição da query na cascata (a 1ª query é a mais específica ao trecho)
BROLL_PESO_QUALIDADE: float = 0.20    # resolução do arquivo de vídeo
BROLL_PESO_ORIENTACAO: float = 0.15   # proximidade do aspect ratio de origem com 9:16
BROLL_PESO_DURACAO: float = 0.10      # folga confortável entre a duração do vídeo e a do trecho
BROLL_PESO_DIVERSIDADE: float = 0.05  # penalidade leve (não eliminatória) se o vídeo já foi usado neste Short

# --- Biblioteca de mídia (MediaManager) ---
MEDIA_VIDEOS_DIR: str = "assets/videos"  # pode ter subpastas por tema (minecraft/, parkour/, ...)
MEDIA_IMAGENS_DIR: str = "assets/imagens"
MEDIA_EXTENSOES_VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
MEDIA_EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_VIDEO_DURACAO_MINIMA: float = 5.0  # segundos; vídeos mais curtos são ignorados
MEDIA_RESOLUCAO_MINIMA_MENOR_LADO: int = 480  # px; ignora vídeos com o menor lado (largura OU altura) abaixo disso
MEDIA_HISTORICO_PATH: str = "media_history.json"  # últimas mídias usadas, para não repetir
MEDIA_HISTORICO_TAMANHO: int = 10  # quantos últimos usados guardar/evitar repetir
MEDIA_MAX_TENTATIVAS: int = 3  # quantos arquivos diferentes tentar antes de desistir

# --- Provedor remoto: Pexels (opcional — só ativa se PEXELS_API_KEY existir) ---
# A chave NUNCA fica no código: só é lida da variável de ambiente abaixo.
PEXELS_API_KEY_ENV_VAR: str = "PEXELS_API_KEY"
PEXELS_API_URL: str = "https://api.pexels.com/videos/search"
PEXELS_MAX_RESULTS: int = 15  # quantos resultados pedir na busca (antes de filtrar/baixar)
PEXELS_MAX_DOWNLOADS: int = 6  # nunca baixa mais que isso por execução (pedido: entre 4 e 6)
PEXELS_MIN_DOWNLOADS_DESEJADOS: int = 4
PEXELS_TIMEOUT_SECONDS: float = 15.0  # por requisição (busca OU download de 1 arquivo)
PEXELS_ORIENTACAO: str = "landscape"  # priorizamos 16:9, cortado depois para 9:16 no pipeline
PEXELS_FPS_PREFERIDO: int = 30
PEXELS_RESOLUCAO_PREFERIDA: Tuple[int, int] = (1920, 1080)
MIN_WIDTH: int = 1280  # ignora automaticamente vídeos com resolução menor que essa
# User-Agent "de navegador" enviado em toda requisição ao Pexels (busca E
# download). O Pexels roda atrás de Cloudflare, e o Cloudflare bloqueia por
# assinatura de bot (HTTP 403, corpo "error code: 1010") requisições sem um
# User-Agent — ou com o default do urllib ("Python-urllib/3.x"), que é um
# dos assinantes mais reconhecidos como automação. A API Key sendo válida
# não evita esse bloqueio: ele acontece ANTES da chave ser sequer avaliada
# pela API do Pexels, na camada de Cloudflare. Nenhuma outra informação de
# identificação é necessária — só um UA que não bata com um cliente HTTP de
# biblioteca "cru".
PEXELS_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CACHE_DIR: str = "cache/pexels"
CACHE_METADATA_PATH: str = "cache/pexels/metadata.json"
CACHE_TTL_HOURS: float = 24.0  # depois disso, o cache expira e pode baixar de novo
MAX_PALAVRAS_CHAVE_PESQUISA: int = 4  # quantas palavras-chave extrair do roteiro
PEXELS_MIN_PALAVRA_LEN: int = 4  # ignora palavras menores que isso na extração de palavras-chave

# --- Identidade visual (cor de destaque compartilhada) ---
# Uma ÚNICA cor de destaque usada tanto nas legendas quanto na barra de
# progresso, para que os dois elementos pareçam parte do mesmo projeto (um
# amarelo puro nas legendas e um dourado na barra, como era antes, lia como
# duas identidades visuais diferentes coexistindo por acaso).
ACCENT_COLOR_HEX: str = "#FFD100"   # dourado quente, usado no texto ativo
ACCENT_COLOR_RGB: Tuple[int, int, int] = (255, 209, 0)  # mesmo tom, para a barra

# --- Legendas sincronizadas por palavra (faster-whisper) ---
TRANSCRIPTION_ENGINE: str = "faster-whisper"
WHISPER_MODEL_SIZE: str = "small"  # bom equilíbrio precisão/velocidade em CPU
WHISPER_COMPUTE_TYPE: str = "int8"  # quantização int8: mais rápido e mais leve em CPU
WHISPER_LANGUAGE: str = "pt"
LEGEND_FONT: str = "DejaVu-Sans-Bold"
LEGEND_FONT_SIZE: int = 58
NORMAL_COLOR: str = "white"
ACTIVE_COLOR: str = ACCENT_COLOR_HEX
STROKE_COLOR: str = "black"
STROKE_WIDTH: int = 3
LEGEND_SHADOW_OFFSET: Tuple[int, int] = (3, 4)  # deslocamento (x, y) da sombra em px
LEGEND_SHADOW_OPACITY: float = 0.45
LEGEND_WIDTH_FRACTION: float = 0.82  # largura máxima da área de legenda (< 1 - 2*SAFE_AREA_SIDE, ver seção "Safe area" abaixo)
LEGEND_WORD_SPACING: int = 18  # espaço horizontal (px) entre palavras
LEGEND_LINE_SPACING: int = 12  # espaço vertical (px) entre linhas, se o grupo quebrar em 2
POP_SCALE: float = 1.08  # escala máxima do "pop" da palavra ativa (100% -> 108% -> 100%)
POP_DURATION: float = 0.22  # duração do pop, em segundos, ao ativar uma palavra
MIN_GROUP_WORDS: int = 3
MAX_GROUP_WORDS: int = 5
PAUSA_MINIMA_QUEBRA: float = 0.35  # segundos; pausa que sugere quebra natural de grupo

# --- Safe area (Fase 3) ---
# Todas as margens de segurança do Short centralizadas aqui, como frações
# da tela (0 a 1). Nenhum outro ponto do código deve definir uma margem
# de segurança "solta" — quem precisar de uma referência de posição usa
# estas constantes. Valores pensados para a UI real do YouTube Shorts:
# o topo costuma ter o nome do canal/ícones, a base tem título, botão de
# inscrição e a barra de descrição, e a lateral direita tem a coluna de
# curtir/comentar/compartilhar.
SAFE_AREA_TOP: float = 0.10     # topo: evita colidir com barra de progresso/notificações do app
SAFE_AREA_BOTTOM: float = 0.20  # base: evita título, inscrever-se e a barra de descrição do Shorts
SAFE_AREA_SIDE: float = 0.06    # laterais: evita a coluna de curtir/comentar/compartilhar à direita

# --- Animação de entrada/saída das legendas (Fase 3) ---
# Cada grupo de legenda entra com um fade + leve "scale-in" (nunca um
# bounce) e sai com um fade curto — sutil o bastante para não competir
# com o "pop" da palavra ativa, que continua acontecendo por cima disso.
LEGEND_ANIM_IN_DURATION: float = 0.16   # segundos; duração da entrada (fade + scale)
LEGEND_ANIM_OUT_DURATION: float = 0.12  # segundos; duração da saída (só fade, sem encolher)
LEGEND_ANIM_SCALE_START: float = 0.92   # escala inicial do "scale-in" padrão (92% -> 100%)

# --- Hook (Fase 3) ---
# Tratamento visual especial só para o PRIMEIRO grupo de legenda/trecho de
# B-roll do vídeo (os primeiros segundos, decisivos para retenção em
# Shorts): entrada de legenda um pouco mais pronunciada e um "punch" de
# zoom rápido no B-roll, que se resolve em `HOOK_ZOOM_PUNCH_DURATION`
# segundos — depois disso, o hook segue exatamente o mesmo Ken Burns
# contínuo do restante do vídeo.
HOOK_ANIM_SCALE_START: float = 0.86    # entrada do hook (mais forte que o padrão, ainda sem bounce)
HOOK_ZOOM_PUNCH: float = 1.06          # zoom extra somado só no início do primeiro B-roll
HOOK_ZOOM_PUNCH_DURATION: float = 0.6  # segundos; janela em que o "punch" decai até zero

# --- Edição dinâmica de vídeo (Ken Burns / zoom / pan) ---
KEN_BURNS_ZOOM_MAX: float = 1.03  # zoom lento e contínuo ao longo do vídeo inteiro
BLOCK_ZOOM_LEVELS: List[float] = [1.00, 1.05, 1.08, 1.03]  # varia a cada troca de legenda
PAN_MAX_FRACTION: float = 0.035  # deslocamento horizontal/vertical máx. (fração da tela)
ZOOM_TRANSICAO_DURACAO: float = 0.4  # segundos; suaviza a troca de nível de zoom entre grupos

# --- Fade de abertura/fechamento ---
FADE_TIME: float = 0.4  # segundos (entre 0.3 e 0.5, conforme especificado)

# --- Barra de progresso ---
BAR_HEIGHT: int = 5  # px (entre 5 e 8, conforme especificado; reduzido de 6 para menos presença)
BAR_COLOR: Tuple[int, int, int] = ACCENT_COLOR_RGB  # mesma cor de destaque das legendas
BAR_BACKGROUND_COLOR: Tuple[int, int, int] = (35, 35, 35)
BAR_OPACITY: float = 0.65  # reduzida de 0.9: a barra era o único elemento "sempre ligado" do vídeo

# --- Vinheta ---
VIGNETTE_INTENSITY: float = 0.14  # 0 (nenhuma) a 1 (forte); reduzida para ficar mais discreta

# --- Modo de diagnóstico (instrumentação da pipeline de vídeo) ---
# Quando DEBUG_RENDER=False (padrão), todo o custo é um único `if` por
# chamada — não abre arquivo, não salva PNG, não imprime nada.
DEBUG_RENDER: bool = False
DEBUG_RENDER_DIR: str = "debug_render"  # pasta onde os frames/logs de diagnóstico são salvos
DEBUG_RENDER_INSTANTE_FRACAO: float = 0.5  # em que ponto da duração salvar o frame (50% por padrão)

# --- Renderização ---
RENDER_PRESET: str = "fast"  # ~25% mais rápido que "medium" no mesmo bitrate, sem perda perceptível
RENDER_BITRATE: str = "8000k"
RENDER_THREADS: int = 4

# --- Áudio (mixagem: narração + música de fundo) ---
MUSICA_PATH: str = "musica.mp3"
MUSIC_VOLUME: float = 0.18  # ganho da música (baixo, para ficar discreta)
VOICE_VOLUME: float = 1.0  # ganho da narração (deve ficar sempre mais alta)
MUSIC_FADE_IN: float = 0.8  # segundos (entre 0.5 e 1.0)
MUSIC_FADE_OUT: float = 0.8  # segundos (entre 0.5 e 1.0)
MUSIC_LOOP: bool = True  # repete a música automaticamente se ela for mais curta
ENABLE_SOFT_LIMITER: bool = False  # compressão/saturação (tanh) é opcional; ver nota abaixo
VOICE_COMPRESSION_DRIVE: float = 1.35  # intensidade da saturação, só usada se ENABLE_SOFT_LIMITER=True
MUSIC_VOLUME_FALLBACK_FRACTION: float = 0.3  # fração de VOICE_VOLUME usada se MUSIC_VOLUME estiver mal configurado

# --- Logs ---
LARGURA_LOG_TEMPOS: int = 22  # largura usada para alinhar os "...." no resumo de tempos

# --- YouTube ---
YOUTUBE_CATEGORY_ID: str = "27"  # Educação
YOUTUBE_TAGS: List[str] = ["shorts", "curiosidades", "fatos interessantes", "ciencia", "viral"]
YOUTUBE_DESCRICAO_TEMPLATE: str = (
    "{roteiro}\n\n"
    "💡 Gostou dessa curiosidade? Inscreva-se no canal para acompanhar mais vídeos diários!\n\n"
    "#shorts #curiosidades #fatos #ciencia #conhecimento"
)

# Palavras que nunca devem ficar sozinhas no fim de uma linha de legenda
# (artigos, preposições e contrações comuns em português) — usadas para
# decidir onde é seguro quebrar um grupo em duas linhas, quando necessário.
# As comparações são sempre feitas com o texto já normalizado (sem acento,
# minúsculo), então não é preciso listar as formas acentuadas aqui.
PALAVRAS_INSEPARAVEIS = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "ao", "aos", "pelo", "pela", "pelos", "pelas",
    "num", "numa", "nesse", "nessa", "neste", "nesta",
    "desse", "dessa", "deste", "desta", "de", "em", "com", "sem",
}

# Palavras (e categorias) consideradas "importantes" pelo detector de
# palavras-chave (etapa 8). O detector APENAS marca `is_keyword=True` — ele
# não aplica nenhum efeito visual; isso fica reservado para uma fase futura.
PALAVRAS_CHAVE = {
    "nasa", "lua", "marte", "sol", "terra", "universo",
    "buraco", "negro", "dna", "oceano",
    "bilhoes", "bilhao", "milhoes", "milhao",
    "gigante", "minusculo", "maior", "menor",
    "primeiro", "primeira", "ultimo", "ultima",
    "recorde", "ano", "anos",
}

# Palavras ignoradas na extração de palavras-chave de busca do
# PexelsMediaProvider (seção 0): além dos artigos/preposições de
# PALAVRAS_INSEPARAVEIS, inclui verbos auxiliares, pronomes e conectivos
# comuns que não ajudam a descrever visualmente o roteiro. Comparação
# sempre feita com a palavra já normalizada (sem acento, minúscula).
STOPWORDS_PESQUISA = PALAVRAS_INSEPARAVEIS | {
    "que", "para", "mas", "por", "sao", "foi", "ser", "estar", "esta",
    "este", "essa", "esse", "isso", "aquilo", "aquele", "aquela",
    "ele", "ela", "eles", "elas", "voce", "nao", "sim", "quando", "onde",
    "como", "porque", "porem", "tambem", "ja", "ainda", "so", "apenas",
    "cada", "todo", "toda", "todos", "todas", "outro", "outra", "outros",
    "outras", "seu", "sua", "seus", "suas", "meu", "minha", "meus",
    "minhas", "tem", "tinha", "vai", "vou", "pode", "podem", "havia",
    "houve", "muito", "muita", "muitos", "muitas", "mais", "menos",
    "entre", "sobre", "apos", "antes", "depois", "assim", "entao",
    "qual", "quais", "quem", "isto", "estes", "estas", "aqueles",
    "aquelas", "haver", "fazer", "fez", "fazem",
}

# Palavras adicionais, além de STOPWORDS_PESQUISA, ignoradas SÓ na geração
# de queries visuais do Pexels (Fase 3): quantidades, comparativos e
# unidades. Gramaticalmente são substantivos/adjetivos válidos (por isso
# não entram em STOPWORDS_PESQUISA, usada também para outros fins), mas
# não têm nenhum valor de busca visual — "anos", "milhoes", "maior" não
# correspondem a nenhuma imagem/vídeo de estoque específico. Comparação
# sempre feita com a palavra já normalizada (sem acento, minúscula).
PEXELS_PALAVRAS_ABSTRATAS = {
    "ano", "anos", "vez", "vezes", "bilhao", "bilhoes", "milhao", "milhoes",
    "mil", "recorde", "maior", "menor", "melhor", "pior", "primeiro",
    "primeira", "ultimo", "ultima", "gigante", "minusculo", "enorme",
    "imenso", "pequeno", "grande", "metro", "metros", "quilometro",
    "quilometros", "centimetro", "centimetros", "quilo", "quilos", "grama",
    "gramas", "tonelada", "toneladas", "grau", "graus", "porcento",
    "percentual", "segundo", "segundos", "minuto", "minutos", "hora",
    "horas", "dia", "dias", "mes", "meses", "seculo", "seculos", "vezes",
}

# Dicionário PT -> lista de queries visuais em inglês (Fase 3): usado por
# `_gerar_queries_visuais()` (seção 0, PexelsMediaProvider) para converter
# palavras do roteiro em buscas que realmente retornam imagens relevantes
# no Pexels — em vez da palavra isolada em português (que, sozinha, é
# ambígua ou pouco pesquisada no banco: "aneis", "saturno" sozinhos
# retornam muito menos/pior que "saturn rings"). Cada entrada tem de 2 a 4
# variações para dar diversidade de material entre B-rolls do mesmo tema.
# Deliberadamente uma heurística por dicionário (sem IA/embeddings) — fora
# de escopo desta fase, e desnecessário para o resultado pedido.
TERMOS_VISUAIS_PT_EN: Dict[str, List[str]] = {
    # Espaço / astronomia
    "saturno": ["saturn planet", "saturn rings", "planet rings", "solar system"],
    "jupiter": ["jupiter planet", "gas giant planet", "solar system"],
    "marte": ["mars planet", "red planet", "mars surface"],
    "lua": ["moon surface", "full moon", "moon night sky"],
    "sol": ["sun solar flare", "sunrise sky", "sun close up"],
    "terra": ["planet earth space", "earth from space"],
    "estrela": ["stars night sky", "starry sky"],
    "estrelas": ["stars night sky", "starry sky", "galaxy stars"],
    "galaxia": ["galaxy space", "milky way night sky"],
    "universo": ["universe space", "deep space stars", "galaxy space"],
    "planeta": ["planet space", "solar system planet"],
    "asteroide": ["asteroid space", "asteroid field"],
    "cometa": ["comet space", "comet night sky"],
    "buraco": ["black hole space", "space vortex"],
    "foguete": ["rocket launch", "space rocket"],
    "astronauta": ["astronaut space", "astronaut spacewalk"],
    "satelite": ["satellite space", "satellite orbit earth"],
    "nasa": ["nasa space", "space exploration"],
    "gravidade": ["zero gravity space", "gravity space"],
    # Oceano / vida marinha
    "oceano": ["ocean underwater", "deep ocean", "ocean waves aerial"],
    "mar": ["sea waves", "ocean sea aerial"],
    "baleia": ["blue whale", "whale underwater", "ocean whale", "marine life"],
    "cachalote": ["sperm whale underwater", "sperm whale", "whale echolocation", "sperm whale ocean"],
    "orca": ["orca whale underwater", "killer whale ocean"],
    "tubarao": ["shark underwater", "shark ocean"],
    "golfinho": ["dolphin underwater", "dolphin ocean"],
    "polvo": ["octopus underwater", "octopus ocean"],
    "peixe": ["fish underwater", "fish swimming ocean"],
    "recife": ["coral reef underwater", "coral reef ocean"],
    # Natureza / geografia
    "floresta": ["forest aerial", "forest trees", "rainforest"],
    "deserto": ["desert dunes", "desert landscape"],
    "montanha": ["mountain landscape", "mountain range aerial"],
    "vulcao": ["volcano eruption", "volcano lava"],
    "gelo": ["ice glacier", "ice landscape"],
    "glaciar": ["glacier ice", "glacier landscape aerial"],
    "terremoto": ["earthquake damage", "seismic disaster"],
    "tempestade": ["storm clouds", "thunderstorm lightning"],
    "relampago": ["lightning storm", "lightning strike sky"],
    "furacao": ["hurricane satellite", "storm clouds aerial"],
    "cachoeira": ["waterfall nature", "waterfall aerial"],
    # Ciência / corpo humano
    "dna": ["dna strand", "dna helix science"],
    "celula": ["human cell microscope", "cell biology animation"],
    "cerebro": ["human brain", "brain science"],
    "coracao": ["human heart", "heartbeat medical"],
    "atomo": ["atom science", "atom particle animation"],
    "molecula": ["molecule science", "chemistry molecule"],
    "virus": ["virus microscope", "virus cell"],
    "bacteria": ["bacteria microscope", "bacteria cell"],
    # Animais terrestres / dinossauros
    "dinossauro": ["dinosaur fossil", "dinosaur animation"],
    "leao": ["lion wildlife", "lion savanna"],
    "elefante": ["elephant wildlife", "elephant savanna"],
    "tigre": ["tiger wildlife", "tiger jungle"],
    "urso": ["bear wildlife", "bear forest"],
    "aranha": ["spider macro", "spider web"],
    "formiga": ["ant macro", "ant colony"],
    "abelha": ["bee macro", "bee flower"],
    # Tecnologia
    "robo": ["robot technology", "robotics"],
    "inteligencia": ["artificial intelligence technology", "ai technology"],
    "computador": ["computer technology", "computer chip"],
    "internet": ["internet network technology", "data network"],
}

# Conceitos CONTEXTUAIS (Fase 2.1): diferente de TERMOS_VISUAIS_PT_EN (uma
# entidade nomeada específica), aqui a chave é uma palavra-gatilho —
# substantivo genérico, adjetivo/qualificador ou verbo de ação — que por
# si só já indica um CONCEITO visual, mesmo sem nenhuma entidade nomeada
# por perto no trecho. É o que resolve o caso relatado na Fase 2.1: o
# trecho "uma criatura real, que navega pelas profundezas" não menciona
# nenhum animal específico (nada bate em TERMOS_VISUAIS_PT_EN), mas
# "criatura"/"navega"/"profundezas" já bastam para produzir queries
# compostas e visuais como "deep sea creature" — em vez de cair nas
# palavras soltas em português que sobrariam do roteiro ("estamos",
# "falando", "monstro"). Usada como 2ª camada em `_gerar_queries_visuais`,
# só depois das entidades diretas — ver comentário lá para a ordem de
# prioridade completa. Mesma lógica de "português -> conceito em inglês
# composto", só que no nível do CONCEITO em vez da ENTIDADE.
CONCEITOS_CONTEXTUAIS_PT_EN: Dict[str, List[str]] = {
    # Criatura/monstro sem nome específico (contexto: mar/profundezas)
    "criatura": ["deep sea creature", "ocean creature underwater", "sea creature swimming"],
    "monstro": ["deep sea creature", "sea monster silhouette underwater"],
    "predador": ["ocean predator", "marine predator underwater"],
    # Profundidade / ambiente
    "profundezas": ["deep ocean", "deep sea underwater"],
    "profundo": ["deep ocean", "deep underwater"],
    "profunda": ["deep ocean", "deep underwater"],
    "abismo": ["deep sea abyss", "ocean trench underwater"],
    # Perigo / letalidade
    "perigoso": ["dangerous marine animal", "dangerous ocean animal"],
    "perigosa": ["dangerous marine animal", "dangerous ocean animal"],
    "perigosos": ["dangerous marine animal", "dangerous ocean animal"],
    "mortal": ["deadly ocean creature", "dangerous marine animal"],
    "mortais": ["deadly ocean creature", "dangerous marine animal"],
    "veneno": ["venomous animal", "poisonous creature"],
    "venenoso": ["venomous animal underwater", "poisonous creature"],
    "venenosa": ["venomous animal underwater", "poisonous creature"],
    # Ações visuais (item 7 da Fase 2.1)
    "nada": ["marine animal swimming", "ocean creature swimming underwater"],
    "nadar": ["marine animal swimming", "ocean creature swimming underwater"],
    "nadando": ["marine animal swimming", "ocean creature swimming underwater"],
    "navega": ["ocean creature swimming", "underwater creature moving"],
    "navegando": ["ocean creature swimming", "underwater creature moving"],
    "atacando": ["predator attack ocean", "shark attack underwater"],
    "ataque": ["predator attack ocean", "shark attack underwater"],
    "atacar": ["predator attack ocean", "shark attack underwater"],
    "correndo": ["person running", "running motion"],
    "correr": ["person running", "running motion"],
    "voando": ["bird flying sky", "flying animal wings"],
    "voar": ["bird flying sky", "flying animal wings"],
    "decolando": ["rocket launch", "rocket liftoff"],
    "decolagem": ["rocket launch", "rocket liftoff"],
    "explodindo": ["explosion slow motion", "explosion close up"],
    "explosao": ["explosion slow motion", "explosion close up"],
    # Qualificadores gerais (item 4/13)
    "gigante": ["giant animal", "massive creature"],
    "gigantesco": ["giant animal", "massive creature"],
    "minusculo": ["microscopic close up", "tiny creature macro"],
    "misterioso": ["mysterious underwater scene", "mysterious dark ocean"],
    "misteriosa": ["mysterious underwater scene", "mysterious dark ocean"],
    "escuro": ["dark ocean underwater", "deep dark sea"],
    "escura": ["dark ocean underwater", "deep dark sea"],
    "antigo": ["ancient ruins", "ancient artifact"],
    "antiga": ["ancient ruins", "ancient artifact"],
    "esconde": ["hidden underwater mystery", "deep ocean mystery"],
    "escondido": ["hidden underwater mystery", "deep ocean mystery"],
    "arma": ["dangerous marine animal", "ocean predator"],
    "armas": ["dangerous marine animal", "ocean predator"],
}

# Fallback genérico (Fase 2.1): usado APENAS quando nenhuma palavra do
# trecho bate em `TERMOS_VISUAIS_PT_EN` nem em
# `CONCEITOS_CONTEXTUAIS_PT_EN` — ou seja, quando não há absolutamente
# nada de visual reconhecível no texto (ex.: trecho só com conectivos).
# Sempre queries compostas e em inglês — NUNCA mais uma palavra solta em
# português como acontecia antes desta fase (a causa raiz do bug
# relatado: "estamos", "corpo", "morte" chegando ao Pexels como query).
PEXELS_QUERIES_FALLBACK_GENERICAS: List[str] = [
    "nature landscape",
    "abstract background loop",
    "cinematic slow motion",
    "city time lapse",
]




# ===========================================================================
# 0. BIBLIOTECA DE MÍDIA (MediaProvider + MediaManager)
# ===========================================================================
# Arquitetura (Fase 6):
#
#   MediaManager
#         │  (nunca toca disco nem API diretamente)
#         ▼
#   MediaProvider  (interface)
#         │
#         ├──── LocalMediaProvider   (Fase 5 — migrado sem mudança de comportamento)
#         └──── PexelsMediaProvider  (Fase 6 — novo)
#
# O MediaManager decide COMO escolher (histórico, evitar repetição,
# fallback entre providers). Cada MediaProvider decide APENAS de onde a
# mídia vem e como validá-la/descartá-la. Nenhum dos dois sabe nada sobre
# Ken Burns, legendas, roteiro etc.
class MediaProvider(ABC):
    """Contrato único que qualquer fonte de mídia deve implementar (Local,
    Pexels, Pixabay, ...). Responsabilidade única: fornecer mídia — listar
    candidatos e validar/descartar o candidato escolhido. Nada mais: não
    decide QUAL escolher, não guarda histórico de uso entre execuções, não
    sabe nada do resto do pipeline. Isso é trabalho do `MediaManager`.
    """

    @abstractmethod
    def listar_videos(self, contexto: Optional[str] = None) -> List[MediaItem]:
        """Retorna candidatos a vídeo de fundo, já prontos para uso (ex.:
        já baixados e em disco, no caso de um provider remoto).

        `contexto` é opcional: providers que precisam de mais informação
        para buscar mídia relevante (ex.: o roteiro do vídeo, usado pelo
        PexelsMediaProvider para gerar palavras-chave de busca) o recebem
        aqui. Providers que não precisam (ex.: Local) simplesmente o
        ignoram — a assinatura é a mesma para todos.
        """

    @abstractmethod
    def validar_video(self, item: MediaItem) -> bool:
        """Validação completa (cara: abre o arquivo de verdade) do
        candidato JÁ escolhido pelo MediaManager — nunca é chamada em toda
        a lista de candidatos, só no vencedor do sorteio."""

    def descartar(self, item: MediaItem) -> None:
        """Hook opcional: o provider pode reagir a um descarte (ex.: um
        candidato passou na validação leve mas falhou de verdade durante a
        renderização). Default: nada a fazer — sobrescrito só por quem
        precisa (ex.: PexelsMediaProvider invalida o cache do item)."""
        return None

    def listar_videos_contextual(
        self, texto: str, duracao_alvo: float, ids_evitar: Optional[set] = None
    ) -> List[MediaItem]:
        """Hook opcional (Fase 2 — B-roll contextual por trecho): providers
        capazes de gerar candidatos relevantes para um TRECHO específico da
        narração (não o roteiro inteiro) sobrescrevem isto, tipicamente
        gerando uma cascata de queries a partir de `texto` e pontuando os
        resultados (relevância/qualidade/orientação/duração/diversidade).

        `ids_evitar` é o conjunto de identificadores (aqui, caminhos de
        arquivo) já usados em OUTROS trechos deste mesmo Short — usado
        para preferir diversidade visual entre cenas, nunca para excluir
        de forma absoluta (ver `_pontuar_candidato_broll`).

        Default: delega para `listar_videos(texto)` — o mesmo
        comportamento de antes da Fase 2, sem quebrar nenhum provider que
        não sabe (ou não precisa saber) trabalhar por trecho — caso do
        `LocalMediaProvider`, que ignora `contexto`/`texto` de qualquer
        forma."""
        return self.listar_videos(texto)


def _validar_arquivo_video(caminho: str) -> bool:
    """Validação completa de um arquivo de vídeo, compartilhada por todos
    os MediaProviders (Fase 5): existe, abre de verdade, tem duração
    mínima, tem resolução adequada e consegue decodificar pelo menos um
    frame de verdade (não só ler os metadados do header). Extraída como
    função solta (em vez de método duplicado em cada provider) porque a
    lógica é idêntica independente de o arquivo ter vindo do disco local
    ou de um download do Pexels — só muda de onde o arquivo chegou até
    aqui.

    Qualquer falha aqui (arquivo ausente, corrompido, curto demais,
    pequeno demais ou sem frames decodificáveis) faz o candidato ser
    descartado pelo `MediaManager` SEM entrar no histórico — ver
    `MediaManager._escolher_de()`."""
    if not os.path.exists(caminho) or os.path.getsize(caminho) <= 0:
        return False

    clip = None
    try:
        clip = VideoFileClip(caminho)

        if not clip.duration or clip.duration < MEDIA_VIDEO_DURACAO_MINIMA:
            return False

        largura, altura = clip.size
        if not largura or not altura or min(largura, altura) < MEDIA_RESOLUCAO_MINIMA_MENOR_LADO:
            return False

        # "possui frames válidos": força a decodificação de verdade de um
        # frame (não só a leitura dos metadados do header, que podem estar
        # ok mesmo num arquivo com payload corrompido/truncado).
        clip.get_frame(min(0.1, clip.duration / 2))
        return True
    except Exception:
        return False
    finally:
        if clip is not None:
            clip.close()


class LocalMediaProvider(MediaProvider):
    """Fonte de mídia local: escaneia `assets/videos/` (com subpastas por
    tema) e `assets/imagens/`, e serve os arquivos encontrados. Migrado
    sem mudança de comportamento a partir do `MediaManager` da Fase 5 —
    toda a lógica de disco do projeto (antes espalhada dentro dele) agora
    mora aqui. É o único provider que sabe o que é um `os.walk`.

    Ignora `contexto` por completo: a busca é sempre "a biblioteca local
    inteira", não depende do roteiro.

    Escaneia cada pasta apenas uma vez por execução (cache em memória em
    `self._videos`/`self._imagens`); chamadas repetidas não tocam o disco
    de novo.
    """

    def __init__(
        self,
        pasta_videos: str = MEDIA_VIDEOS_DIR,
        pasta_imagens: str = MEDIA_IMAGENS_DIR,
    ) -> None:
        self.pasta_videos = pasta_videos
        self.pasta_imagens = pasta_imagens
        self._videos: Optional[List[MediaItem]] = None
        self._imagens: Optional[List[MediaItem]] = None

    # --- listagem (com cache em memória) --------------------------------
    def listar_videos(self, contexto: Optional[str] = None) -> List[MediaItem]:
        """Lista os vídeos válidos da biblioteca local. Se ela estiver
        vazia, cai de volta para o único arquivo legado `VIDEO_FUNDO_PATH`,
        se ele existir — um projeto que ainda não montou `assets/videos/`
        continua funcionando exatamente como antes, sem configuração
        extra."""
        if self._videos is not None:
            return self._videos

        videos = self._escanear(self.pasta_videos, MEDIA_EXTENSOES_VIDEO, "video")

        if not videos and os.path.exists(VIDEO_FUNDO_PATH) and self._validacao_leve(VIDEO_FUNDO_PATH):
            print(
                f"ℹ️ Biblioteca de mídia local vazia em '{self.pasta_videos}'; "
                f"usando o arquivo legado '{VIDEO_FUNDO_PATH}'."
            )
            videos = [MediaItem(caminho=VIDEO_FUNDO_PATH, categoria="legado", tipo="video")]

        self._videos = videos
        return self._videos

    def listar_imagens(self) -> List[MediaItem]:
        """Lista as imagens válidas da biblioteca local (escaneia só uma
        vez). Preparado para uso futuro — nada no pipeline atual consome
        imagens como fundo ainda. Não faz parte da interface
        `MediaProvider` (nenhum outro provider serve imagens ainda), então
        fica como método extra só do Local."""
        if self._imagens is None:
            self._imagens = self._escanear(self.pasta_imagens, MEDIA_EXTENSOES_IMAGEM, "imagem")
        return self._imagens

    def _escanear(self, pasta: str, extensoes: set, tipo: str) -> List[MediaItem]:
        itens: List[MediaItem] = []
        if not os.path.isdir(pasta):
            return itens

        for raiz, _subpastas, arquivos in os.walk(pasta):
            categoria = os.path.relpath(raiz, pasta)
            categoria = "" if categoria == "." else categoria
            for nome in arquivos:
                if os.path.splitext(nome)[1].lower() not in extensoes:
                    continue
                caminho = os.path.join(raiz, nome)
                if not self._validacao_leve(caminho):
                    continue
                itens.append(MediaItem(caminho=caminho, categoria=categoria, tipo=tipo))

        return itens

    # --- validação --------------------------------------------------------
    def _validacao_leve(self, caminho: str) -> bool:
        """Validação barata, aplicada a TODA a biblioteca durante o escaneio:
        só confirma que o arquivo existe e não está vazio. Não abre o vídeo
        (isso seria caro demais para repetir em toda a biblioteca)."""
        try:
            return os.path.getsize(caminho) > 0
        except OSError:
            return False

    def validar_video(self, item: MediaItem) -> bool:
        return _validar_arquivo_video(item.caminho)

    def validar_imagem(self, item: MediaItem) -> bool:
        """Validação completa de imagem — mesmo princípio: só é chamada no
        candidato sorteado, usando PIL (já é dependência do projeto)."""
        try:
            with Image.open(item.caminho) as img:
                img.verify()
            return True
        except Exception:
            return False


def _componentes_score_candidato_broll(
    video: dict,
    arquivo: dict,
    indice_query: int,
    total_queries: int,
    duracao_alvo: float,
    ja_usados: Optional[set] = None,
) -> Dict[str, float]:
    """Calcula os 5 fatores (0.0-1.0 cada) usados por
    `_pontuar_candidato_broll()`, sem combiná-los em score final. Extraído
    como função própria (Fase 2.1.1, item 3 — diagnóstico) só para que o
    log de diagnóstico possa mostrar a composição do score de cada
    candidato (relevância/qualidade/orientação/duração/diversidade)
    sem duplicar a lógica de cálculo — `_pontuar_candidato_broll()` chama
    esta função e só faz a soma ponderada. Nenhuma mudança de
    comportamento/matemática: mesmos fatores, mesmas fórmulas."""
    ja_usados = ja_usados or set()

    # --- relevância: 1.0 na 1ª query, decaindo linearmente até perto de 0 ---
    relevancia = 1.0 - (indice_query / max(1, total_queries))

    # --- qualidade: resolução do arquivo vs a preferida ---
    largura_arquivo = arquivo.get("width") or 0
    largura_preferida = PEXELS_RESOLUCAO_PREFERIDA[0]
    qualidade = min(1.0, largura_arquivo / largura_preferida) if largura_preferida else 0.0

    # --- orientação: aspect ratio de origem vs 9:16 (ASPECT_RATIO) ---
    largura_video = video.get("width") or 0
    altura_video = video.get("height") or 0
    if largura_video and altura_video:
        aspect_origem = largura_video / altura_video
        # distância normalizada: 0 = exatamente 9:16, 1 = o mais distante
        # que consideramos plausível (bem quadrado ou bem panorâmico).
        distancia = abs(aspect_origem - ASPECT_RATIO) / ASPECT_RATIO
        orientacao = max(0.0, 1.0 - min(distancia, 1.5) / 1.5)
    else:
        orientacao = 0.5  # sem informação de aspect ratio — nem penaliza nem beneficia

    # --- duração: folga confortável sobre o trecho, sem exagero ---
    duracao_video = video.get("duration") or 0
    if duracao_alvo > 0 and duracao_video > 0:
        folga = duracao_video / duracao_alvo
        if folga < 1.0:
            duracao_score = 0.0  # não deveria acontecer (já filtrado antes), mas nunca pontua acima do alvo
        elif folga <= 3.0:
            duracao_score = (folga - 1.0) / 2.0  # cresce suavemente até 3x a duração do trecho
        else:
            duracao_score = 1.0 - min((folga - 3.0) / 10.0, 0.5)  # vídeos MUITO mais longos perdem um pouco de pontos
    else:
        duracao_score = 0.5

    # --- diversidade: penaliza (não zera) reuso dentro do mesmo Short ---
    video_id = str(video.get("id"))
    diversidade = 0.0 if video_id in ja_usados else 1.0

    return {
        "relevancia": relevancia,
        "qualidade": qualidade,
        "orientacao": orientacao,
        "duracao": duracao_score,
        "diversidade": diversidade,
    }


def _pontuar_candidato_broll(
    video: dict,
    arquivo: dict,
    indice_query: int,
    total_queries: int,
    duracao_alvo: float,
    ja_usados: Optional[set] = None,
) -> float:
    """Pontua (0-100) um candidato de B-roll do Pexels para um trecho
    específico (Fase 2). Combina 5 fatores com pesos fixos (soma 1.0,
    ver `BROLL_PESO_*`):

    - Relevância (50%): posição da query que encontrou o vídeo na cascata
      gerada por `_gerar_queries_visuais()` — a 1ª query é sempre a mais
      específica ao trecho (dicionário visual PT->EN), então um resultado
      da 1ª query pontua mais que um da 4ª.
    - Qualidade (20%): resolução do arquivo de vídeo escolhido
      (`_escolher_arquivo()`), normalizada contra `PEXELS_RESOLUCAO_PREFERIDA`.
    - Orientação (15%): proximidade do aspect ratio de ORIGEM do vídeo
      com 9:16 — vídeos já verticais pontuam mais, mas um horizontal bom
      ainda pontua razoavelmente (nunca é descartado só por isso; o
      pipeline já sabe cortar/zoom para vertical).
    - Duração (10%): favorece vídeos com folga confortável sobre a
      duração do trecho (nem curtos demais — já filtrados antes por
      `MEDIA_VIDEO_DURACAO_MINIMA` — nem desproporcionalmente longos).
    - Diversidade (5%): penalidade leve (não eliminatória) se este vídeo
      já foi usado em outro trecho do MESMO Short — dá preferência a
      variedade sem nunca escolher um candidato irrelevante só para
      evitar repetição (item 10 da Fase 2: "é melhor reutilizar um vídeo
      bom do que usar um vídeo completamente irrelevante").

    Pura e sem I/O — testável isoladamente com dicts feitos à mão.
    """
    c = _componentes_score_candidato_broll(
        video, arquivo, indice_query, total_queries, duracao_alvo, ja_usados
    )
    score = (
        BROLL_PESO_RELEVANCIA * c["relevancia"]
        + BROLL_PESO_QUALIDADE * c["qualidade"]
        + BROLL_PESO_ORIENTACAO * c["orientacao"]
        + BROLL_PESO_DURACAO * c["duracao"]
        + BROLL_PESO_DIVERSIDADE * c["diversidade"]
    )
    return round(score * 100, 2)


class PexelsMediaProvider(MediaProvider):
    """Provider remoto: busca, baixa e cacheia vídeos do banco Pexels a
    partir de palavras-chave extraídas do roteiro. Usa SOMENTE a API
    oficial (`PEXELS_API_URL`) via `urllib` (stdlib) — sem scraping, sem
    Selenium, sem Playwright, sem navegador, sem dependência nova.

    Responsabilidade única: entregar `MediaItem`s de vídeos já baixados e
    prontos para uso. Cache, expiração (`CACHE_TTL_HOURS`) e limite de
    downloads por execução (`PEXELS_MAX_DOWNLOADS`) ficam inteiramente
    aqui dentro — o resto do projeto não sabe (nem precisa saber) que os
    arquivos vieram da internet.

    A API Key NUNCA fica no código: só é aceita via parâmetro no
    construtor, e quem monta o provider (`_criar_media_manager()`, mais
    abaixo) só a lê de `os.environ[PEXELS_API_KEY_ENV_VAR]`.
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: str = CACHE_DIR,
        metadata_path: str = CACHE_METADATA_PATH,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.metadata_path = metadata_path
        os.makedirs(self.cache_dir, exist_ok=True)
        self._metadata: Dict[str, dict] = self._carregar_metadata()
        # Diagnóstico seguro (Fase 2): confirma que a Secret chegou até
        # aqui e informa só o TAMANHO dela — nunca o valor. Ajuda a
        # distinguir rapidamente "a key nem chegou no ambiente" de "a key
        # chegou mas foi rejeitada pela API" só olhando o log.
        print(
            f"🔑 Pexels: API key presente (comprimento: {len(self.api_key) if self.api_key else 0} "
            f"caracteres)."
        )
        # Blacklist em memória, válida só para esta execução (não é
        # persistida): guarda os IDs de vídeos descartados por
        # `descartar()` (ex.: passaram na validação mas falharam de
        # verdade na renderização), para que um retry na mesma execução
        # não busque/baixe o mesmo vídeo de novo — sem isso, como
        # `descartar()` remove a entrada do metadata, o vídeo deixaria de
        # ser reconhecido como "já visto" e poderia ser rebaixado e
        # escolhido de novo no mesmo `escolher_fundo_e_criar_video()`.
        self._rejeitados_nesta_execucao: set = set()
        # Fase 2 (B-roll contextual): cache de resultados de busca EM
        # MEMÓRIA, válido só durante esta execução — se duas cenas do
        # mesmo Short gerarem a mesma query (comum: "saturn" pode
        # aparecer na cascata de mais de um trecho), a segunda NUNCA
        # dispara uma nova requisição HTTP, só reaproveita a lista já
        # obtida. Chave = string da query; valor = lista de vídeos crus
        # retornados pela API (ou lista vazia, se a busca falhou — também
        # cacheado, para não tentar de novo a mesma query com erro
        # repetidamente dentro da mesma execução).
        self._cache_busca_execucao: Dict[str, List[dict]] = {}
        # Contador de downloads efetivos nesta execução, compartilhado
        # entre o fluxo antigo (`listar_videos`) e o novo por-trecho
        # (`listar_videos_contextual`) — item 19 da Fase 2: nunca deixa o
        # total de downloads da execução (não só de uma chamada isolada)
        # ultrapassar `PEXELS_MAX_DOWNLOADS`.
        self._downloads_nesta_execucao: int = 0

    # --- cache / metadata ---------------------------------------------------
    def _carregar_metadata(self) -> Dict[str, dict]:
        if not os.path.exists(self.metadata_path):
            return {}
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as arquivo:
                dados = json.load(arquivo)
            return dados if isinstance(dados, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _salvar_metadata(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.metadata_path) or ".", exist_ok=True)
            with open(self.metadata_path, "w", encoding="utf-8") as arquivo:
                json.dump(self._metadata, arquivo, ensure_ascii=False, indent=2)
        except OSError as err:
            # Nunca derruba a execução por causa do cache — é só uma
            # otimização de banda/quota, não uma etapa crítica.
            print(f"⚠️ Pexels: não foi possível salvar cache ({err}); seguindo sem persistir.")

    def _cache_valido(self, entrada: dict) -> bool:
        caminho = entrada.get("caminho")
        baixado_em = entrada.get("baixado_em", 0)
        if not caminho or not os.path.exists(caminho):
            return False
        idade_horas = (time.time() - baixado_em) / 3600
        return idade_horas < CACHE_TTL_HOURS

    def _videos_em_cache_validos(self) -> List[MediaItem]:
        """Retorna os itens de cache ainda válidos e, de quebra, expurga do
        metadata (em memória) as entradas expiradas ou com arquivo
        ausente — a remoção efetiva do arquivo em disco fica a cargo de
        `limpar_cache_expirado()`, chamado ao final da execução."""
        validos: List[MediaItem] = []
        for video_id, entrada in list(self._metadata.items()):
            if self._cache_valido(entrada):
                validos.append(
                    MediaItem(caminho=entrada["caminho"], categoria="pexels/cache", tipo="video")
                )
        return validos

    # --- queries visuais (Fase 2, refinado na Fase 2.1) --------------------
    def _gerar_queries_visuais(self, texto: str) -> List[str]:
        """Gera até `MAX_PALAVRAS_CHAVE_PESQUISA` queries de busca VISUAIS,
        SEMÂNTICAS e compostas a partir de `texto` (o roteiro inteiro, na
        chamada de `listar_videos()`, ou o texto de UM trecho, na chamada
        de `listar_videos_contextual()` — funciona igual para os dois).

        Fase 2.1 (auditoria de qualidade das queries): a versão anterior
        caía, na ausência de uma entidade conhecida, para as próprias
        palavras em português que sobrassem dos filtros — o que produzia
        queries inúteis para trechos sem substantivo concreto reconhecido
        (ex.: "estamos", "falando", "corpo", "morte" viravam queries
        literais, e o Pexels não achava nada). Esta versão nunca mais
        expõe uma palavra solta em português como query: passa a usar
        DUAS camadas de dicionário — a antiga (`TERMOS_VISUAIS_PT_EN`,
        entidades nomeadas: "cachalote", "saturno"...) e uma nova
        (`CONCEITOS_CONTEXTUAIS_PT_EN`, palavras-gatilho de CONTEXTO —
        substantivo genérico, qualificador ou ação — que já bastam para um
        conceito visual mesmo sem entidade nomeada por perto: "criatura",
        "profundezas", "nadando", "perigoso"...). Se nada do texto bater em
        nenhum dos dois, cai para um pequeno conjunto de queries genéricas
        JÁ COMPOSTAS em inglês (`PEXELS_QUERIES_FALLBACK_GENERICAS`) — nunca
        mais para uma palavra crua.

        Heurística determinística por dicionário (sem IA/embeddings/NLP
        pesado, de propósito — ver item 18 da Fase 2.1): olha o TEXTO
        INTEIRO do trecho (não só as primeiras palavras), então um conceito
        que só aparece no meio ou no fim ("...que navega pelas
        profundezas") ainda é encontrado.

        Prioridade (mais específico primeiro, cascata preservada): 1)
        entidades nomeadas; 2) conceitos/contexto; 3) fallback genérico —
        só quando as duas primeiras camadas não renderam NADA. Diferença
        importante em relação à versão anterior: se as camadas 1+2 juntas
        renderem menos que `MAX_PALAVRAS_CHAVE_PESQUISA` queries, a função
        devolve só o que encontrou (queries melhores, não mais queries —
        ver item 10 da Fase 2.1), em vez de completar o restante com
        palavras soltas de qualidade duvidosa.
        """
        contagem: Dict[str, int] = {}
        ordem_primeira_ocorrencia: List[str] = []
        for palavra in re.findall(r"[A-Za-zÀ-ÿ]+", texto):
            normalizada = _normalizar_palavra(palavra)
            if (
                len(normalizada) < PEXELS_MIN_PALAVRA_LEN
                or normalizada in STOPWORDS_PESQUISA
                or normalizada in PEXELS_PALAVRAS_ABSTRATAS
            ):
                continue
            if normalizada not in contagem:
                ordem_primeira_ocorrencia.append(normalizada)
            contagem[normalizada] = contagem.get(normalizada, 0) + 1

        # Ordena por frequência (desc) e, em empate, pela ordem de
        # primeira aparição no texto — só usado para desempatar QUAL
        # entidade/conceito entra primeiro quando o trecho tem mais
        # candidatos do que cabe em MAX_PALAVRAS_CHAVE_PESQUISA.
        palavras_por_prioridade = sorted(
            contagem.keys(),
            key=lambda p: (-contagem[p], ordem_primeira_ocorrencia.index(p)),
        )

        queries: List[str] = []
        vistas: set = set()

        def _adicionar_todas(candidatas: List[str]) -> None:
            for query in candidatas:
                if query not in vistas and len(queries) < MAX_PALAVRAS_CHAVE_PESQUISA:
                    vistas.add(query)
                    queries.append(query)

        # 1ª camada — entidades nomeadas conhecidas (a mais específica: um
        # animal/objeto/planeta identificado explicitamente no texto).
        for palavra in palavras_por_prioridade:
            if len(queries) >= MAX_PALAVRAS_CHAVE_PESQUISA:
                break
            if palavra in TERMOS_VISUAIS_PT_EN:
                _adicionar_todas(TERMOS_VISUAIS_PT_EN[palavra])

        # 2ª camada (Fase 2.1) — conceito/contexto: sem entidade nomeada
        # específica, mas com uma palavra-gatilho (substantivo genérico,
        # qualificador ou ação) que já descreve um conceito visual.
        for palavra in palavras_por_prioridade:
            if len(queries) >= MAX_PALAVRAS_CHAVE_PESQUISA:
                break
            if palavra in CONCEITOS_CONTEXTUAIS_PT_EN:
                _adicionar_todas(CONCEITOS_CONTEXTUAIS_PT_EN[palavra])

        # 3ª camada — nada no texto bateu em nenhum dos dois dicionários:
        # fallback genérico já composto em inglês, nunca uma palavra crua.
        if not queries:
            _adicionar_todas(PEXELS_QUERIES_FALLBACK_GENERICAS)

        return queries

    # --- busca / download ---------------------------------------------------
    def _ler_corpo_erro(self, err: "urllib.error.HTTPError") -> str:
        """Lê o corpo de uma resposta de erro HTTP com segurança (nunca
        levanta exceção — usado só para diagnóstico)."""
        try:
            bruto = err.read()
            return bruto.decode("utf-8", errors="replace") if bruto else ""
        except Exception:
            return ""

    def _sanitizar_corpo_erro(self, corpo: str, tamanho_max: int = 300) -> str:
        """Trunca e remove QUALQUER ocorrência da API key do corpo de erro
        antes de logar (defesa extra: mesmo que um proxy/CDN ecoasse a URL
        completa de volta no corpo, a key nunca vai parar no log)."""
        if not corpo:
            return "(corpo vazio)"
        sanitizado = corpo.replace(self.api_key, "***") if self.api_key else corpo
        sanitizado = " ".join(sanitizado.split())  # colapsa quebras de linha/espaços em excesso
        if len(sanitizado) > tamanho_max:
            sanitizado = sanitizado[:tamanho_max] + "…"
        return sanitizado

    def _log_diagnostico_erro(self, query: str, status: int, corpo_erro: str) -> None:
        """Log de diagnóstico seguro para falhas do Pexels (Fase 2): NUNCA
        imprime a API key nem qualquer header — só metadados úteis para
        depuração (presença/tamanho da key, endpoint, método, status HTTP
        e um corpo de erro sanitizado)."""
        print(
            f"🔎 Pexels diagnóstico: query='{query}' metodo=GET endpoint={PEXELS_API_URL} "
            f"status={status} chave_presente={bool(self.api_key)} "
            f"chave_tamanho={len(self.api_key) if self.api_key else 0} "
            f"corpo={self._sanitizar_corpo_erro(corpo_erro)}"
        )

    def _buscar(self, query: str) -> List[dict]:
        """Consulta a API oficial do Pexels (GET /videos/search com header
        `Authorization: <key>`, sem prefixo "Bearer" — formato exigido pela
        API do Pexels). Erros de rede/HTTP/JSON viram RuntimeError com
        prefixo "❌ Pexels: ..." — capturados por `listar_videos()`, que
        segue para a próxima query em vez de interromper a busca inteira
        por causa de uma falha pontual (Fase 8: o pipeline nunca quebra só
        por causa do Pexels).

        Trata separadamente 401 (key inválida), 403 (acesso negado — key
        revogada/conta suspensa/plano insuficiente, não é rede), 429
        (limite de requisições excedido), 5xx (erro do lado do Pexels,
        provavelmente temporário), timeout/indisponibilidade de rede, e
        resposta com JSON inválido — cada caso com uma mensagem específica
        que já aponta a causa mais provável, em vez de um "falhou (HTTP
        xxx)" genérico para tudo.
        """
        parametros = urllib.parse.urlencode(
            {"query": query, "per_page": PEXELS_MAX_RESULTS, "orientation": PEXELS_ORIENTACAO}
        )
        url = f"{PEXELS_API_URL}?{parametros}"
        requisicao = urllib.request.Request(
            url,
            headers={
                "Authorization": self.api_key,
                # Sem isso, o Cloudflare na frente do Pexels bloqueia a
                # requisição com HTTP 403 / "error code: 1010" ANTES mesmo
                # de a API Key ser avaliada — ver comentário de
                # `PEXELS_USER_AGENT` acima.
                "User-Agent": PEXELS_USER_AGENT,
            },
        )

        try:
            with urllib.request.urlopen(requisicao, timeout=PEXELS_TIMEOUT_SECONDS) as resposta:
                # Defensivo: em produção `resposta` sempre tem `.status`
                # (http.client.HTTPResponse), mas dublês de teste minimalistas
                # (ver `test_pexels_broll.py`) podem não implementar nem
                # `.status` nem `.getcode()` — nesse caso só não sabemos o
                # código (assumimos 200, já que não houve HTTPError) em vez
                # de quebrar a chamada.
                status_code = getattr(resposta, "status", None)
                if status_code is None and hasattr(resposta, "getcode"):
                    status_code = resposta.getcode()
                if status_code is None:
                    status_code = 200
                corpo_bruto = resposta.read().decode("utf-8")
        except urllib.error.HTTPError as err:
            corpo_erro = self._ler_corpo_erro(err)
            self._log_diagnostico_erro(query, err.code, corpo_erro)
            if err.code == 401:
                raise RuntimeError(
                    "❌ Pexels: API Key inválida (HTTP 401). Confirme o valor "
                    "da Secret PEXELS_API_KEY no GitHub."
                ) from err
            if err.code == 403:
                if "error code: 1010" in corpo_erro.lower():
                    raise RuntimeError(
                        f"❌ Pexels: bloqueado pelo Cloudflare (HTTP 403, "
                        f"'error code: 1010') na busca por '{query}'. Isso é um "
                        f"bloqueio por assinatura de cliente HTTP/bot, ANTES da "
                        f"API Key ser avaliada — não costuma indicar chave "
                        f"inválida nem conta suspensa. Se isso persistir mesmo "
                        f"com um User-Agent de navegador definido, o IP do "
                        f"runner (ex.: datacenter do GitHub Actions) pode estar "
                        f"numa faixa bloqueada pelo Cloudflare do Pexels; nesse "
                        f"caso não há correção possível só por código."
                    ) from err
                raise RuntimeError(
                    f"❌ Pexels: acesso negado (HTTP 403) na busca por '{query}'. "
                    f"Normalmente indica API Key inválida/revogada, conta "
                    f"suspensa ou sem plano ativo — não costuma ser um problema "
                    f"de rede. Confirme a Secret e o status da conta em "
                    f"pexels.com/api."
                ) from err
            if err.code == 429:
                raise RuntimeError(
                    f"❌ Pexels: limite de requisições excedido (HTTP 429) na "
                    f"busca por '{query}'. A cota da API (mensal/por hora) foi "
                    f"atingida."
                ) from err
            if 500 <= err.code < 600:
                raise RuntimeError(
                    f"❌ Pexels: erro no servidor do Pexels (HTTP {err.code}) na "
                    f"busca por '{query}'; provavelmente temporário."
                ) from err
            raise RuntimeError(f"❌ Pexels: busca por '{query}' falhou (HTTP {err.code}).") from err
        except (urllib.error.URLError, socket.timeout) as err:
            raise RuntimeError(f"❌ Pexels: indisponível/timeout na busca por '{query}' ({err}).") from err

        try:
            dados = json.loads(corpo_bruto)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"❌ Pexels: resposta JSON inválida na busca por '{query}' ({err})."
            ) from err

        if not isinstance(dados, dict):
            raise RuntimeError(
                f"❌ Pexels: resposta inesperada (não é um objeto JSON) na busca por '{query}'."
            )

        videos = dados.get("videos", [])
        # Diagnóstico (Fase 2.1.1, item 2/4): distingue "API respondeu 200
        # mas devolveu 0 resultados" (Caso A do diagnóstico) de qualquer
        # eliminação POSTERIOR feita pelo nosso próprio pipeline (filtros,
        # download, cache — Casos B/C/D), que é logada mais adiante em
        # `_buscar_com_fallback_de_queries()`. Sem esta linha, as duas
        # situações eram indistinguíveis no log.
        print(f"   📡 Pexels API: HTTP={status_code} resultados_recebidos={len(videos)}")
        return videos

    def _escolher_arquivo(self, video: dict) -> Optional[dict]:
        """Entre os `video_files` de um resultado do Pexels, escolhe o que
        melhor atende resolução mínima (MIN_WIDTH), priorizando fps e
        largura preferidos (PEXELS_FPS_PREFERIDO / PEXELS_RESOLUCAO_PREFERIDA).
        Ignora automaticamente qualquer arquivo com largura insuficiente."""
        candidatos = [
            arquivo
            for arquivo in video.get("video_files", [])
            if arquivo.get("width") and arquivo["width"] >= MIN_WIDTH
        ]
        if not candidatos:
            return None

        def pontuacao(arquivo: dict) -> Tuple[bool, bool, int]:
            return (
                arquivo.get("fps") == PEXELS_FPS_PREFERIDO,
                arquivo.get("width") == PEXELS_RESOLUCAO_PREFERIDA[0],
                arquivo.get("width", 0),
            )

        return max(candidatos, key=pontuacao)

    def _baixar_arquivo(self, url: str, destino: str) -> bool:
        """Baixa para um arquivo temporário `.part` e só promove para o
        destino final (`os.replace`, atômico) se o download terminar sem
        erro e com conteúdo não vazio — protege contra downloads
        interrompidos deixarem um `.mp4` corrompido no cache."""
        temporario = destino + ".part"
        try:
            # Mesmo User-Agent da busca: os arquivos de vídeo do Pexels são
            # servidos por um CDN (não necessariamente o mesmo domínio da
            # API), mas também pode estar atrás de proteção anti-bot — mais
            # seguro sempre identificar como um navegador aqui também.
            requisicao = urllib.request.Request(url, headers={"User-Agent": PEXELS_USER_AGENT})
            with urllib.request.urlopen(requisicao, timeout=PEXELS_TIMEOUT_SECONDS) as resposta:
                with open(temporario, "wb") as arquivo_saida:
                    arquivo_saida.write(resposta.read())
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, OSError) as err:
            print(f"⚠️ Pexels: falha ao baixar '{url}' ({err}).")
            if os.path.exists(temporario):
                os.remove(temporario)
            return False

        if os.path.getsize(temporario) == 0:
            os.remove(temporario)
            return False

        os.replace(temporario, destino)
        return True

    # --- interface MediaProvider -----------------------------------------
    def listar_videos(self, contexto: Optional[str] = None) -> List[MediaItem]:
        """Fluxo: reaproveita cache válido primeiro (evita gastar
        quota/banda à toa); só busca/baixa mais se ainda faltar para
        `PEXELS_MIN_DOWNLOADS_DESEJADOS`. `contexto` é o roteiro — sem ele
        não há como gerar queries visuais, então devolve só o que já está
        em cache (podendo ser uma lista vazia, o que faz o MediaManager
        cair para o fallback local).

        Orçamento (Fase 2.1.1 → correção B): o número de vídeos NOVOS
        baixados por esta chamada é limitado pelo orçamento RESIDUAL da
        execução inteira (`PEXELS_MAX_DOWNLOADS - self._downloads_nesta_execucao`),
        não mais por `PEXELS_MAX_DOWNLOADS` sozinho — este método pode ser
        chamado depois que o B-roll contextual por trecho já gastou parte
        (ou todo) o orçamento, e não deve gastar mais do que sobrou. Itens
        já em cache (`em_cache`) continuam de graça — não consomem
        orçamento nenhum, contam à parte."""
        em_cache = self._videos_em_cache_validos()
        if len(em_cache) >= PEXELS_MIN_DOWNLOADS_DESEJADOS:
            return em_cache

        if not contexto:
            return em_cache

        orcamento_restante = max(0, PEXELS_MAX_DOWNLOADS - self._downloads_nesta_execucao)
        if orcamento_restante <= 0:
            # Nada de orçamento sobrando — nem vale a pena consultar a API
            # (gastaria requisições à toa para candidatos que não
            # poderiam ser baixados de qualquer forma).
            return em_cache

        queries = self._gerar_queries_visuais(contexto)
        novos: List[MediaItem] = []

        for query in queries:
            if len(novos) >= orcamento_restante:
                break
            resultados = self._cache_busca_execucao.get(query)
            if resultados is None:
                try:
                    resultados = self._buscar(query)
                except RuntimeError as err:
                    print(f"⚠️ {err}")
                    resultados = []
                self._cache_busca_execucao[query] = resultados

            for video in resultados:
                if len(novos) >= orcamento_restante:
                    break

                duracao = video.get("duration", 0)
                if duracao < MEDIA_VIDEO_DURACAO_MINIMA:
                    continue  # ignora vídeos menores que a duração mínima

                video_id = str(video.get("id"))
                if video_id in self._rejeitados_nesta_execucao:
                    continue  # descartado por falha de renderização nesta mesma execução
                if video_id in self._metadata and self._cache_valido(self._metadata[video_id]):
                    continue  # já temos esse vídeo específico em cache

                arquivo = self._escolher_arquivo(video)
                if arquivo is None:
                    continue  # nenhum arquivo com resolução suficiente

                destino = os.path.join(self.cache_dir, f"{video_id}.mp4")
                if not self._baixar_arquivo(arquivo["link"], destino):
                    continue  # download quebrado/interrompido — já tratado e logado
                self._downloads_nesta_execucao += 1

                self._metadata[video_id] = {
                    "caminho": destino,
                    "baixado_em": time.time(),
                    "query": query,
                }
                novos.append(MediaItem(caminho=destino, categoria=f"pexels/{query}", tipo="video"))

        if novos:
            self._salvar_metadata()

        return em_cache + novos

    # --- B-roll contextual por trecho (Fase 2) ------------------------------
    def listar_videos_contextual(
        self, texto: str, duracao_alvo: float, ids_evitar: Optional[set] = None
    ) -> List[MediaItem]:
        """Sobrescreve o hook default: gera uma cascata de queries a partir
        do texto REALMENTE narrado neste trecho (via `_gerar_queries_visuais`,
        já reaproveitado sem alteração — funciona para qualquer string, não
        só o roteiro inteiro) e devolve candidatos ORDENADOS por
        `_pontuar_candidato_broll` (melhor primeiro)."""
        if not texto:
            return []
        queries = self._gerar_queries_visuais(texto)[:BROLL_MAX_QUERIES_POR_TRECHO]
        print("🎯 Queries semânticas:")
        for indice, query in enumerate(queries, start=1):
            print(f"   {indice}. {query}")
        itens = self._buscar_com_fallback_de_queries(queries, duracao_alvo, ids_evitar or set())
        if itens:
            melhor = itens[0]
            print(f"🎯 Melhor candidato: {melhor.caminho} ({melhor.categoria})")
        else:
            print("⚠️ Nenhum candidato Pexels relevante para este trecho.")
        return itens

    def _buscar_com_fallback_de_queries(
        self, queries: List[str], duracao_alvo: float, ids_evitar: set
    ) -> List[MediaItem]:
        """Cascata de queries (item 6 da Fase 2): tenta a query mais
        específica primeiro; só desce para a próxima se ainda não achou
        `BROLL_CANDIDATOS_POR_TRECHO_ALVO` candidatos bons — nunca gasta
        mais buscas do que precisa (item 19: limite de requests). Resultado
        de cada query fica em `_cache_busca_execucao` (item 7: reutilizar
        resultados — compartilhado com `listar_videos()`), então duas
        cenas com queries iguais nunca disparam uma segunda requisição.

        Cada candidato é pontuado (`_pontuar_candidato_broll`, item 9) e a
        lista final volta ORDENADA por score decrescente — quem chama (o
        `MediaManager`) só precisa pegar o primeiro que passar na
        validação completa."""
        candidatos_pontuados: List[Tuple[float, dict, dict, str, str, int]] = []
        total_queries = max(1, len(queries))

        for indice_query, query in enumerate(queries):
            if len({c[3] for c in candidatos_pontuados}) >= BROLL_CANDIDATOS_POR_TRECHO_ALVO:
                break  # já tem candidatos suficientes; não gasta mais buscas

            # --- Diagnóstico (Fase 2.1.1, itens 2/6): distingue cache HIT
            # (resultado desta query já obtido nesta execução, nenhuma
            # requisição nova) de cache MISS (vai consultar a API agora).
            # `_buscar()` já loga HTTP/resultados_recebidos no caso MISS.
            print(f"🔎 Pexels query: \"{query}\"")
            resultados = self._cache_busca_execucao.get(query)
            if resultados is None:
                print("   🌐 Cache MISS — consultando Pexels")
                try:
                    resultados = self._buscar(query)
                except RuntimeError as err:
                    print(f"⚠️ {err}")
                    resultados = []
                self._cache_busca_execucao[query] = resultados
            else:
                print(f"   🗄️ Cache HIT — busca (reaproveitando {len(resultados)} resultado(s) já obtidos nesta execução)")

            # --- Diagnóstico (itens 2/4): motivos de rejeição são
            # contados usando EXATAMENTE os filtros que existem abaixo —
            # nenhuma categoria nova foi inventada para o log. Hoje só
            # existem 3 filtros nesta cascata: duração mínima, blacklist
            # de descartados nesta execução, e ausência de um arquivo
            # (`video_files`) com largura >= MIN_WIDTH. Não há, no código
            # atual, filtro de orientação, de duplicidade ou de score
            # mínimo nesta etapa — por isso não aparecem aqui.
            rejeitados_duracao = 0
            rejeitados_ja_descartado = 0
            rejeitados_sem_arquivo_utilizavel = 0
            validos_desta_query = 0

            for video in resultados:
                duracao = video.get("duration", 0)
                if duracao < MEDIA_VIDEO_DURACAO_MINIMA:
                    rejeitados_duracao += 1
                    continue
                video_id = str(video.get("id"))
                if video_id in self._rejeitados_nesta_execucao:
                    rejeitados_ja_descartado += 1
                    continue
                arquivo = self._escolher_arquivo(video)
                if arquivo is None:
                    rejeitados_sem_arquivo_utilizavel += 1
                    continue
                score = _pontuar_candidato_broll(
                    video, arquivo, indice_query, total_queries, duracao_alvo, ids_evitar
                )
                candidatos_pontuados.append((score, video, arquivo, video_id, query, indice_query))
                validos_desta_query += 1

            print(
                "   🔍 Processamento: "
                f"candidatos_examinados={len(resultados)} "
                f"rejeitados_por_duracao={rejeitados_duracao} "
                f"rejeitados_por_ja_descartado_nesta_execucao={rejeitados_ja_descartado} "
                f"rejeitados_por_ausencia_de_arquivo_utilizavel={rejeitados_sem_arquivo_utilizavel} "
                f"-> candidatos_validos={validos_desta_query}"
            )

        candidatos_pontuados.sort(key=lambda item: item[0], reverse=True)

        # --- Diagnóstico (item 3): tabela de scores, mostrada sempre que
        # houver ao menos um candidato pontuado — ANTES da escolha final
        # (download/cache), para explicar por que cada um ganhou/perdeu.
        if candidatos_pontuados:
            print("📊 Candidatos Pexels (ordenados por score, melhor primeiro):")
            limite_exibicao = 20
            for posicao, (score, video, arquivo, video_id, query_origem, indice_q) in enumerate(
                candidatos_pontuados[:limite_exibicao], start=1
            ):
                c = _componentes_score_candidato_broll(
                    video, arquivo, indice_q, total_queries, duracao_alvo, ids_evitar
                )
                print(
                    f"   {posicao}. video_id={video_id} score={score:.1f} (query=\"{query_origem}\")"
                )
                print(
                    f"      relevância={c['relevancia']:.2f} qualidade={c['qualidade']:.2f} "
                    f"orientação={c['orientacao']:.2f} duração={c['duracao']:.2f} "
                    f"diversidade={c['diversidade']:.2f}"
                )
            if len(candidatos_pontuados) > limite_exibicao:
                print(f"   … e mais {len(candidatos_pontuados) - limite_exibicao} candidato(s) não exibido(s).")

        # --- Diagnóstico (itens 4/5/6/7): a partir daqui, candidatos
        # válidos podem ainda não virar `itens` por 3 motivos REAIS: já
        # apareceram via outra query (duplicado), o teto de downloads da
        # execução foi atingido, ou o download falhou. Cada um é contado
        # e logado separadamente, e cada item final indica sua fonte
        # (`fonte=cache` ou `fonte=api`) — sem isso, um caminho como
        # `cache/pexels/31541421.mp4` no log não permite saber se acabou
        # de ser baixado ou só reaproveitado.
        itens: List[MediaItem] = []
        vistos: set = set()
        duplicados_entre_queries = 0
        reutilizados_do_cache = 0
        baixados_agora = 0
        falhas_download = 0
        bloqueados_por_teto_download = 0

        for score, video, arquivo, video_id, query_origem, _indice_q in candidatos_pontuados:
            if video_id in vistos:
                duplicados_entre_queries += 1
                continue  # o mesmo vídeo pode ter sido encontrado por mais de uma query da cascata
            vistos.add(video_id)

            if video_id in self._metadata and self._cache_valido(self._metadata[video_id]):
                reutilizados_do_cache += 1
                print(f"🗄️ Cache HIT — vídeo já baixado (video_id={video_id}, fonte=cache)")
                itens.append(
                    MediaItem(
                        caminho=self._metadata[video_id]["caminho"],
                        categoria=f"pexels/score={score:.1f}/fonte=cache",
                        tipo="video",
                    )
                )
                continue

            if self._downloads_nesta_execucao >= PEXELS_MAX_DOWNLOADS:
                bloqueados_por_teto_download += 1
                continue  # teto de downloads da execução (item 19) — mesmo com candidatos bons sobrando

            print(f"⬇️ Download novo (video_id={video_id}, fonte=api, url={arquivo.get('link', '?')})")
            destino = os.path.join(self.cache_dir, f"{video_id}.mp4")
            if not self._baixar_arquivo(arquivo["link"], destino):
                falhas_download += 1
                print(f"❌ Download falhou (video_id={video_id})")
                continue
            print(f"✅ Download OK (video_id={video_id})")
            self._downloads_nesta_execucao += 1
            baixados_agora += 1

            self._metadata[video_id] = {
                "caminho": destino,
                "baixado_em": time.time(),
                "query": query_origem,
            }
            itens.append(
                MediaItem(caminho=destino, categoria=f"pexels/score={score:.1f}/fonte=api", tipo="video")
            )

        if itens:
            self._salvar_metadata()

        print(
            "📦 Resultado final do trecho: "
            f"candidatos_pontuados={len(candidatos_pontuados)} "
            f"duplicados_entre_queries={duplicados_entre_queries} "
            f"reutilizados_do_cache={reutilizados_do_cache} "
            f"baixados_agora={baixados_agora} "
            f"falhas_download={falhas_download} "
            f"bloqueados_por_teto_download={bloqueados_por_teto_download} "
            f"-> itens_finais={len(itens)}"
        )

        return itens

    def validar_video(self, item: MediaItem) -> bool:
        return _validar_arquivo_video(item.caminho)

    def descartar(self, item: MediaItem) -> None:
        """Remove do metadata (não do disco — isso é papel de
        `limpar_cache_expirado()`) qualquer entrada apontando para este
        arquivo, e marca o(s) ID(s) correspondente(s) na blacklist desta
        execução — para que não seja reoferecido nem nesta execução (via
        nova busca) nem na próxima (via cache, já que saiu do metadata)."""
        alterado = False
        for video_id, entrada in list(self._metadata.items()):
            if entrada.get("caminho") == item.caminho:
                self._metadata.pop(video_id, None)
                self._rejeitados_nesta_execucao.add(video_id)
                alterado = True
        if alterado:
            self._salvar_metadata()

    # --- limpeza (pensada para o runner efêmero do GitHub Actions) --------
    def limpar_cache_expirado(self) -> None:
        """Remove do disco (e do metadata) qualquer arquivo cujo cache já
        expirou (> CACHE_TTL_HOURS) e qualquer `.part` órfão de downloads
        interrompidos. Chamado ao final da execução (`main()`) para não
        acumular espaço em disco entre execuções do runner."""
        for video_id, entrada in list(self._metadata.items()):
            if not self._cache_valido(entrada):
                caminho = entrada.get("caminho")
                if caminho and os.path.exists(caminho):
                    try:
                        os.remove(caminho)
                    except OSError:
                        pass
                self._metadata.pop(video_id, None)

        if os.path.isdir(self.cache_dir):
            for nome in os.listdir(self.cache_dir):
                if nome.endswith(".part"):
                    try:
                        os.remove(os.path.join(self.cache_dir, nome))
                    except OSError:
                        pass

        self._salvar_metadata()


class MediaManager:
    """Escolhe automaticamente um vídeo de fundo, decidindo QUAL provider
    consultar e mantendo o histórico de uso entre execuções (para evitar
    repetir os últimos usados). Não conversa com disco nem com APIs
    diretamente — só com um `MediaProvider`.

    Recebe um provider primário e, opcionalmente, um provider de fallback.
    Se o primário falhar (lista vazia, exceção, validação de todos os
    candidatos falhando — ex.: Pexels indisponível), cai automaticamente
    para o fallback sem interromper o workflow (ver `escolher_video()`).
    O usuário final nunca percebe a troca — só um aviso no log.
    """

    def __init__(
        self,
        provider: Optional[MediaProvider] = None,
        provider_fallback: Optional[MediaProvider] = None,
        arquivo_historico: str = MEDIA_HISTORICO_PATH,
    ) -> None:
        self.provider = provider or LocalMediaProvider()
        self.provider_fallback = provider_fallback
        self.arquivo_historico = arquivo_historico
        self._historico: List[str] = self._carregar_historico()
        # Lembra de qual provider cada arquivo veio, para que `descartar()`
        # chame o `.descartar()` do provider certo (ex.: invalidar cache do
        # Pexels), sem o chamador precisar saber a origem.
        self._origem: Dict[str, MediaProvider] = {}

    # --- histórico / diversidade (inalterado desde a Fase 5) --------------
    def _carregar_historico(self) -> List[str]:
        if not os.path.exists(self.arquivo_historico):
            return []
        try:
            with open(self.arquivo_historico, "r", encoding="utf-8") as arquivo:
                dados = json.load(arquivo)
            if isinstance(dados, list):
                return [str(item) for item in dados]
        except (OSError, json.JSONDecodeError):
            pass
        return []

    def _salvar_historico(self) -> None:
        try:
            with open(self.arquivo_historico, "w", encoding="utf-8") as arquivo:
                json.dump(self._historico[-MEDIA_HISTORICO_TAMANHO:], arquivo, ensure_ascii=False, indent=2)
        except OSError as err:
            print(f"⚠️ Não foi possível salvar '{self.arquivo_historico}' ({err}); seguindo sem ele.")

    def _registrar_uso(self, caminho: str) -> None:
        self._historico.append(caminho)
        self._historico = self._historico[-MEDIA_HISTORICO_TAMANHO:]
        self._salvar_historico()

    def confirmar_uso(self, caminho: str) -> None:
        """Marca definitivamente um arquivo como usado no histórico
        persistente. Só deve ser chamado por quem sabe que a mídia foi
        REALMENTE aceita pelo pipeline (ex.: `criar_video()` terminou com
        sucesso) — nunca no momento em que ela é apenas sorteada/validada.

        `escolher_video()`/`escolher_imagem()` deliberadamente NÃO chamam
        isto sozinhos: escolher é só "candidatar-se"; confirmar é quem
        decide que o resultado final ficou pronto. Isso evita que uma
        mídia sorteada mas descartada por falha de renderização (ver
        `FundoInvalidoError` em `escolher_fundo_e_criar_video()`) polua o
        histórico como se tivesse sido usada de verdade."""
        self._registrar_uso(caminho)

    # --- escolha ------------------------------------------------------------
    def escolher_video(self, contexto: Optional[str] = None) -> str:
        """Escolhe um vídeo de fundo via o provider primário; se ele
        falhar por qualquer motivo (indisponibilidade, API Key inválida,
        timeout, nenhum candidato válido...), cai para o provider de
        fallback (tipicamente `LocalMediaProvider`) sem propagar o erro —
        só loga um aviso. Se não houver fallback (ou o fallback for o
        próprio primário), o erro sobe normalmente.

        IMPORTANTE: escolher NÃO marca a mídia como usada no histórico —
        isso só acontece quando quem chamou confirma o sucesso real via
        `confirmar_uso()` (ver docstring do método). Escolher é reversível
        (pode ser seguido de `descartar()`); confirmar não é."""
        try:
            return self._escolher_de(self.provider, contexto)
        except Exception as err:
            if self.provider_fallback is None or self.provider_fallback is self.provider:
                raise
            print(
                f"⚠️ Provedor de mídia primário ({type(self.provider).__name__}) falhou "
                f"({err}); usando {type(self.provider_fallback).__name__} como fallback."
            )
            return self._escolher_de(self.provider_fallback, contexto)

    def escolher_videos(self, contexto: Optional[str], quantidade: int) -> List[str]:
        """Como `escolher_video()`, mas devolve até `quantidade` caminhos
        (usado pelos múltiplos B-rolls de um mesmo Short — Fase 4; ver
        `escolher_fundo_e_criar_video()`).

        Prioriza SEMPRE um vídeo ainda não escolhido nesta mesma chamada,
        tentando primeiro o provider primário e, se ele não render nada
        inédito (esgotado ou indisponível), o fallback — só repete um
        caminho já escolhido nesta chamada quando NENHUM dos dois
        providers tem mais nada de novo a oferecer (reutilizar é o último
        recurso, nunca o primeiro). Nunca levanta exceção se já conseguiu
        pelo menos 1 vídeo válido; só levanta se não conseguir nenhum."""
        escolhidos: List[str] = []
        primario_disponivel = True

        for _ in range(max(1, quantidade)):
            ja_escolhidos = set(escolhidos)
            candidato: Optional[str] = None
            candidato_repetido: Optional[str] = None

            if primario_disponivel:
                try:
                    obtido = self._escolher_de(self.provider, contexto, excluir=ja_escolhidos)
                except Exception as err:
                    primario_disponivel = False
                    nome_fallback = (
                        type(self.provider_fallback).__name__ if self.provider_fallback else "nenhum fallback"
                    )
                    print(
                        f"⚠️ Provedor de mídia primário ({type(self.provider).__name__}) falhou "
                        f"({err}); usando {nome_fallback} para os B-rolls restantes."
                    )
                else:
                    if obtido in ja_escolhidos:
                        # Só repetição disponível no primário: guarda como
                        # última alternativa e tenta achar algo inédito no
                        # fallback antes de aceitar a repetição.
                        candidato_repetido = obtido
                    else:
                        candidato = obtido

            if candidato is None and self.provider_fallback is not None and self.provider_fallback is not self.provider:
                try:
                    obtido = self._escolher_de(self.provider_fallback, contexto, excluir=ja_escolhidos)
                except Exception:
                    obtido = None
                if obtido is not None:
                    if obtido in ja_escolhidos:
                        candidato_repetido = candidato_repetido or obtido
                    else:
                        candidato = obtido

            if candidato is None:
                candidato = candidato_repetido  # nada inédito em lugar nenhum; reaproveita

            if candidato is None:
                if escolhidos:
                    break  # já temos ao menos 1 vídeo válido; segue com o que tem
                raise RuntimeError("Nenhum provedor de mídia retornou vídeos válidos para os B-rolls.")

            escolhidos.append(candidato)

        return escolhidos

    def _escolher_de(
        self, provider: MediaProvider, contexto: Optional[str], excluir: Optional[set] = None
    ) -> str:
        candidatos = list(provider.listar_videos(contexto))
        if not candidatos:
            raise RuntimeError(f"{type(provider).__name__} não retornou nenhum vídeo candidato.")

        if excluir:
            inedito = [item for item in candidatos if item.caminho not in excluir]
            if inedito:
                # Só restringe à lista inédita se sobrar algo — se TODOS os
                # candidatos já foram escolhidos antes nesta mesma
                # chamada, é preferível reaproveitar um deles (Fase 4:
                # "reutilizar vídeos somente quando necessário") a travar
                # o pipeline por falta de mídia inédita.
                candidatos = inedito

        random.shuffle(candidatos)
        no_historico = set(self._historico)
        ordem = sorted(candidatos, key=lambda item: item.caminho in no_historico)

        tentativas = 0
        for candidato in ordem:
            if tentativas >= MEDIA_MAX_TENTATIVAS:
                break
            tentativas += 1
            if provider.validar_video(candidato):
                # Só registra QUAL provider forneceu o candidato (para
                # `descartar()` saber a quem delegar). NÃO marca como usado
                # aqui — ver `confirmar_uso()`.
                self._origem[candidato.caminho] = provider
                return candidato.caminho
            print(f"⚠️ Vídeo '{candidato.caminho}' não passou na validação; tentando outro.")

        raise RuntimeError(
            f"Nenhum dos {tentativas} vídeo(s) testado(s) via {type(provider).__name__} "
            f"passou na validação."
        )

    # --- B-roll contextual por trecho (Fase 2) ------------------------------
    def escolher_video_contextual(
        self, texto: str, duracao_alvo: float, excluir: Optional[set] = None
    ) -> Optional[str]:
        """Como `escolher_video()`, mas pede candidatos relevantes para um
        TRECHO específico (Fase 2 — B-roll contextual), não para o
        roteiro inteiro (`MediaProvider.listar_videos_contextual`).

        Ao contrário de `escolher_video()`/`escolher_videos()`, NUNCA
        levanta exceção: se nem o provider primário nem o de fallback
        conseguirem nada, devolve `None` e quem chamou decide o próximo
        passo (ver `_montar_fundo_multiplo()`, que cai para um pool de
        reserva ou reaproveita o último B-roll que funcionou — item 21
        da Fase 2: resiliência do pipeline é um requisito absoluto,
        mesmo que a busca contextual falhe por completo).

        `excluir` é só preferência de diversidade (item 10): candidatos
        nesse conjunto são despriorizados, nunca descartados — melhor
        reutilizar um vídeo bom do que usar um irrelevante."""
        excluir = excluir or set()
        candidato = self._tentar_contextual(self.provider, texto, duracao_alvo, excluir)
        if candidato is not None:
            return candidato

        if self.provider_fallback is not None and self.provider_fallback is not self.provider:
            print(
                f"⚠️ Provedor de mídia primário ({type(self.provider).__name__}) sem candidato "
                f"contextual para este trecho; tentando {type(self.provider_fallback).__name__}."
            )
            candidato = self._tentar_contextual(self.provider_fallback, texto, duracao_alvo, excluir)
            if candidato is not None:
                return candidato

        return None

    def _tentar_contextual(
        self, provider: MediaProvider, texto: str, duracao_alvo: float, excluir: set
    ) -> Optional[str]:
        try:
            candidatos = list(provider.listar_videos_contextual(texto, duracao_alvo, excluir))
        except Exception as err:
            print(f"⚠️ {type(provider).__name__} falhou ao buscar B-roll contextual ({err}).")
            return None
        if not candidatos:
            return None

        no_historico = set(self._historico)
        # Preserva a ordem de RELEVÂNCIA que o provider já calculou (score
        # — ver `_pontuar_candidato_broll`); só usa o histórico persistente
        # entre execuções + `excluir` (diversidade dentro deste Short)
        # como desempate leve e estável, nunca reordenando por score.
        ordem = sorted(
            candidatos,
            key=lambda item: item.caminho in no_historico or item.caminho in excluir,
        )

        tentativas = 0
        for candidato in ordem:
            if tentativas >= MEDIA_MAX_TENTATIVAS:
                break
            tentativas += 1
            if provider.validar_video(candidato):
                self._origem[candidato.caminho] = provider
                return candidato.caminho
            print(f"⚠️ Vídeo '{candidato.caminho}' não passou na validação; tentando outro.")

        return None

    def escolher_imagem(self) -> str:
        """Preparado para uso futuro — ainda não consumido pelo pipeline
        de vídeo. Só o `LocalMediaProvider` serve imagens por enquanto
        (Pexels desta fase é só de vídeo). Mesma regra do vídeo: NÃO marca
        como usada aqui — quem consumir precisa chamar `confirmar_uso()`
        depois de aceitar a imagem de verdade."""
        provider = self.provider if isinstance(self.provider, LocalMediaProvider) else LocalMediaProvider()
        candidatos = list(provider.listar_imagens())
        if not candidatos:
            raise RuntimeError(f"❌ Nenhuma imagem encontrada em '{provider.pasta_imagens}'.")

        random.shuffle(candidatos)
        no_historico = set(self._historico)
        ordem = sorted(candidatos, key=lambda item: item.caminho in no_historico)

        tentativas = 0
        for candidato in ordem:
            if tentativas >= MEDIA_MAX_TENTATIVAS:
                break
            tentativas += 1
            if provider.validar_imagem(candidato):
                self._origem[candidato.caminho] = provider
                return candidato.caminho

        raise RuntimeError(f"❌ Nenhuma imagem válida encontrada em '{provider.pasta_imagens}'.")

    def descartar(self, caminho: str) -> None:
        """Remove um arquivo do pool desta execução (ex: passou na
        validação leve/completa mas falhou de verdade durante a
        renderização) — delega ao provider de origem (para que ele possa,
        por exemplo, invalidar o cache) e tira das listas locais dos
        providers `LocalMediaProvider` envolvidos, para não bater no mesmo
        arquivo de novo nesta execução."""
        provider_origem = self._origem.get(caminho, self.provider)
        provider_origem.descartar(MediaItem(caminho=caminho, categoria="", tipo="video"))

        for provider in {self.provider, self.provider_fallback}:
            if isinstance(provider, LocalMediaProvider) and provider._videos is not None:
                provider._videos = [item for item in provider._videos if item.caminho != caminho]

    def limpar_cache_temporario(self) -> None:
        """Chamado ao final da execução (ver `main()`): dá a chance de
        cada provider liberar espaço em disco (cache expirado, downloads
        `.part` interrompidos) — pensado para o runner efêmero do GitHub
        Actions, que não deve acumular espaço entre execuções. Providers
        sem necessidade disso (ex.: Local) simplesmente não implementam
        `limpar_cache_expirado()` e são ignorados aqui."""
        for provider in {self.provider, self.provider_fallback}:
            limpar = getattr(provider, "limpar_cache_expirado", None)
            if callable(limpar):
                try:
                    limpar()
                except Exception as err:
                    print(f"⚠️ Falha ao limpar cache de {type(provider).__name__} ({err}).")


def _criar_media_manager() -> MediaManager:
    """Decide, a partir do ambiente, quais providers usar: se
    `PEXELS_API_KEY_ENV_VAR` estiver definida, Pexels é o primário e Local
    o fallback automático; caso contrário, só Local (comportamento
    idêntico ao da Fase 5). Centraliza essa decisão aqui para que `main()`
    não precise saber nada sobre providers."""
    api_key = os.environ.get(PEXELS_API_KEY_ENV_VAR)
    if not api_key:
        return MediaManager(provider=LocalMediaProvider())

    return MediaManager(
        provider=PexelsMediaProvider(api_key=api_key),
        provider_fallback=LocalMediaProvider(),
    )


# ===========================================================================
# 1. GERAR ROTEIRO (Gemini SDK)
# ===========================================================================
def _montar_prompt_roteiro() -> str:
    """Monta o prompt que instrui o Gemini a atuar como roteirista viral."""
    return f"""
Você é um roteirista profissional especializado em vídeos virais para YouTube
Shorts. Seu único objetivo é prender a atenção do espectador do primeiro ao
último segundo.

Gere UMA curiosidade fascinante e verdadeira sobre um dos temas abaixo,
alternando bastante entre eles e evitando assuntos já muito repetidos:
espaço, história, natureza, ciência, animais, corpo humano, oceanos ou
tecnologia.

Estruture o roteiro mentalmente seguindo este ritmo (não escreva os tempos,
apenas siga a lógica):
- 0 a 3 segundos: um gancho extremamente forte, que gere impacto imediato.
- 3 a 10 segundos: construa curiosidade, sem entregar a resposta ainda.
- 10 a 22 segundos: entregue a revelação de forma clara e surpreendente.
- 22 a 30 segundos: encerre com uma frase de impacto, sem pedir inscrição.

REGRAS OBRIGATÓRIAS:
1. Nunca cumprimente, nunca diga "olá" e nunca escreva introduções.
2. Nunca peça para se inscrever ou curtir dentro do roteiro.
3. Utilize apenas fatos amplamente aceitos e verificáveis. Nunca invente dados.
4. Use frases curtas, linguagem natural e fluida, como se estivesse contando
   algo incrível para um amigo.
5. O roteiro deve ter entre {ROTEIRO_MIN_PALAVRAS} e {ROTEIRO_MAX_PALAVRAS}
   palavras.
6. O título deve despertar curiosidade imediata, no estilo de:
   "A NASA descobriu isso por acidente", "Ninguém consegue explicar isso",
   "Esse animal desafia a ciência", "Isso realmente aconteceu".
   Evite títulos genéricos. Máximo de {TITULO_MAX_CARACTERES} caracteres.
7. Responda ESTRITAMENTE em JSON válido, sem nenhum texto fora do JSON, com
   as chaves "titulo" e "roteiro".

Exemplo de formato (não copie o conteúdo, apenas o formato):
{{"titulo": "O mistério do espaço em silêncio", "roteiro": "Você sabia que o espaço é totalmente silencioso? Como não existe ar para transportar ondas sonoras, mesmo as explosões mais gigantescas do universo acontecem em silêncio absoluto. Se você estivesse no meio de uma supernova, não ouviria nada! O som precisa de um meio físico, como o ar ou a água, para se propagar."}}
""".strip()


def _extrair_json(texto: str) -> Optional[dict]:
    """Tenta interpretar `texto` como JSON, mesmo com texto extra ao redor.

    Primeiro tenta um parse direto (caso comum: resposta é só o JSON). Se
    falhar, varre a string procurando cada ocorrência de '{' e tenta um
    `raw_decode` a partir dali — isso recupera o JSON corretamente mesmo
    quando o modelo escreve texto antes e/ou depois dele, sem depender de
    uma regex gulosa que poderia capturar chaves de trechos não relacionados.
    """
    texto = texto.strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    decodificador = json.JSONDecoder()
    for indice, caractere in enumerate(texto):
        if caractere != "{":
            continue
        try:
            dados, _ = decodificador.raw_decode(texto, indice)
            return dados
        except json.JSONDecodeError:
            continue  # essa ocorrência de '{' não inicia um JSON válido

    return None


def _validar_roteiro(dados: dict) -> Tuple[str, str]:
    """Valida e normaliza o dicionário retornado pelo Gemini."""
    if "titulo" not in dados or "roteiro" not in dados:
        raise ValueError("JSON não contém as chaves 'titulo' e 'roteiro'.")

    titulo = str(dados["titulo"]).strip()
    roteiro = str(dados["roteiro"]).strip()

    if not titulo or not roteiro:
        raise ValueError("Título ou roteiro vazio.")

    if len(titulo) > TITULO_MAX_CARACTERES:
        titulo = titulo[: TITULO_MAX_CARACTERES - 1].rstrip() + "…"

    num_palavras = len(roteiro.split())
    if not (ROTEIRO_MIN_PALAVRAS - ROTEIRO_TOLERANCIA_PALAVRAS
            <= num_palavras
            <= ROTEIRO_MAX_PALAVRAS + ROTEIRO_TOLERANCIA_PALAVRAS):
        # Fora de uma margem razoável: provavelmente o modelo ignorou a regra.
        raise ValueError(f"Roteiro fora do tamanho esperado ({num_palavras} palavras).")

    return titulo, roteiro


def gerar_roteiro() -> Tuple[str, str]:
    """Gera um roteiro inédito usando o Gemini, com fallback entre modelos."""
    print("🧠 [1/4] Gerando roteiro inédito com o Gemini...")
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = _montar_prompt_roteiro()

    ultimo_erro: Optional[Exception] = None

    # Sequência plana de tentativas (modelo, número da tentativa naquele
    # modelo), usada tanto para os logs quanto para o cálculo do backoff.
    tentativas = [
        (modelo, tentativa)
        for modelo in MODEL_PRIORITY
        for tentativa in range(1, MAX_RETRIES + 1)
    ]

    for indice, (modelo, tentativa) in enumerate(tentativas, start=1):
        try:
            print(f"🔄 Tentando gerar com '{modelo}' (tentativa {tentativa}/{MAX_RETRIES})...")
            resposta = client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            if not resposta or not resposta.text:
                raise ValueError("Resposta vazia do modelo.")

            dados = _extrair_json(resposta.text)
            if dados is None:
                raise ValueError("Não foi possível interpretar o JSON retornado.")

            titulo, roteiro = _validar_roteiro(dados)
            print(f"✅ Sucesso utilizando '{modelo}'!")
            return titulo, roteiro

        except Exception as err:
            ultimo_erro = err
            print(f"⚠️ Falha com '{modelo}' (tentativa {tentativa}/{MAX_RETRIES}): {err}")

            if indice < len(tentativas):
                espera = RETRY_BACKOFF_BASE * (2 ** (indice - 1))
                espera += random.uniform(0, RETRY_BACKOFF_JITTER)
                print(f"⏳ Tentativa {indice + 1} em {espera:.1f} segundos...")
                time.sleep(espera)

    raise Exception(
        f"Não foi possível gerar o roteiro com nenhum dos modelos disponíveis. Último erro: {ultimo_erro}"
    )


# ===========================================================================
# 2. GERAR ÁUDIO (TTS com fallback: Google Cloud -> OpenAI -> ElevenLabs)
# ===========================================================================
def preparar_texto_para_narracao(texto: str) -> str:
    """Ajusta o texto para soar mais natural quando narrado.

    Normaliza espaços, garante pontuação final e insere pequenas pausas
    (vírgulas) em conectores comuns, sem alterar o significado do texto e
    sem abusar de reticências.
    """
    texto = re.sub(r"\s+", " ", texto).strip()

    # Garante espaçamento correto após pontuação.
    texto = re.sub(r"\s+([.,!?;:])", r"\1", texto)
    texto = re.sub(r"([.,!?;:])(?=\S)", r"\1 ", texto)

    # Insere uma pequena pausa natural antes de conectores de contraste,
    # quando ainda não houver pontuação ali.
    conectores = ["mas", "porém", "então", "porque", "e por isso"]
    for conector in conectores:
        texto = re.sub(
            rf"(?<![,.!?])\s+({conector})\b",
            r", \1",
            texto,
            flags=re.IGNORECASE,
        )

    # Evita sequências de pontuação redundante (ex: ",,", ". ,").
    texto = re.sub(r"[,]{2,}", ",", texto)
    texto = re.sub(r"\.{4,}", "...", texto)  # nunca mais que reticências padrão

    if not texto.endswith((".", "!", "?")):
        texto += "."

    return texto.strip()


def _gerar_audio_google(texto: str, caminho_saida: str) -> str:
    """Provedor 2 (fallback): Google Cloud Text-to-Speech.

    Autenticação via API key (GOOGLE_TTS_API_KEY), não via credenciais de
    conta de serviço/ADC. A API REST do Cloud TTS (texttospeech.googleapis.com)
    aceita API key como método de autenticação (equivalente a `?key=...` na
    REST), e a client library oficial expõe esse mesmo mecanismo através de
    `ClientOptions(api_key=...)` — sem precisar de
    GOOGLE_APPLICATION_CREDENTIALS nem de um arquivo de service account.
    """
    client_options = ClientOptions(api_key=os.environ.get("GOOGLE_TTS_API_KEY"))
    client = texttospeech.TextToSpeechClient(client_options=client_options)

    entrada = texttospeech.SynthesisInput(text=texto)
    voz = texttospeech.VoiceSelectionParams(
        language_code=GOOGLE_TTS_LANGUAGE_CODE,
        name=GOOGLE_TTS_VOICE_NAME,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    resposta = client.synthesize_speech(
        input=entrada,
        voice=voz,
        audio_config=audio_config,
        timeout=GOOGLE_TTS_TIMEOUT_SECONDS,
    )

    if not resposta.audio_content:
        raise RuntimeError("Google Cloud TTS: a narração gerada está vazia (0 bytes).")

    with open(caminho_saida, "wb") as arquivo:
        arquivo.write(resposta.audio_content)

    return caminho_saida


def _gerar_audio_openai(texto: str, caminho_saida: str) -> str:
    """Provedor 1 (principal): OpenAI Text-to-Speech.

    A chave vem exclusivamente de OPENAI_API_KEY (variável de ambiente).
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=OPENAI_TTS_TIMEOUT_SECONDS)

    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=texto,
    ) as resposta:
        resposta.stream_to_file(caminho_saida)

    if not os.path.exists(caminho_saida) or os.path.getsize(caminho_saida) == 0:
        raise RuntimeError("OpenAI TTS: a narração gerada está vazia (0 bytes).")

    return caminho_saida


def _gerar_audio_elevenlabs(texto: str, caminho_saida: str) -> str:
    """Provedor 3 (fallback legado): ElevenLabs — mantido por compatibilidade.

    A chave existente ELEVENLABS_API_KEY continua funcionando exatamente
    como antes; se ela não existir, este provedor simplesmente não entra na
    fila (ver _provedores_tts_configurados()).
    """
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"), timeout=ELEVENLABS_TIMEOUT_SECONDS)

    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=texto,
        model_id=ELEVEN_MODEL_ID,
    )

    bytes_escritos = 0
    with open(caminho_saida, "wb") as arquivo:
        for chunk in audio_stream:
            if chunk:
                arquivo.write(chunk)
                bytes_escritos += len(chunk)

    if bytes_escritos == 0:
        raise RuntimeError("ElevenLabs: a narração gerada está vazia (0 bytes).")

    return caminho_saida


def _erro_e_timeout(err: Exception) -> bool:
    """Heurística simples para reconhecer timeout (o único caso em que se
    tenta de novo o MESMO provedor, com retry/backoff, antes de cair para o
    próximo — os demais tipos de falha de provedor vão direto para o
    próximo)."""
    if isinstance(err, (socket.timeout, TimeoutError)):
        return True
    mensagem = str(err).lower()
    return "timeout" in mensagem or "timed out" in mensagem or "deadline exceeded" in mensagem


def _provedores_tts_configurados() -> List[Tuple[str, Callable[[str, str], str]]]:
    """Monta a fila de provedores de TTS a partir das variáveis de ambiente
    presentes, já na ordem de prioridade exigida: OpenAI TTS -> Google Cloud
    TTS -> ElevenLabs. Um provedor sem credencial configurada simplesmente
    não entra na fila — a prioridade é 100% automática, sem precisar tocar
    no código para trocar de provedor."""
    provedores: List[Tuple[str, Callable[[str, str], str]]] = []

    if os.environ.get("OPENAI_API_KEY"):
        provedores.append(("OpenAI TTS", _gerar_audio_openai))

    if os.environ.get("GOOGLE_TTS_API_KEY"):
        provedores.append(("Google Cloud TTS", _gerar_audio_google))

    if os.environ.get("ELEVENLABS_API_KEY"):
        provedores.append(("ElevenLabs", _gerar_audio_elevenlabs))

    return provedores


def gerar_audio(texto: str, caminho_saida: str = NARRACAO_PATH) -> str:
    """Gera o arquivo de narração em áudio a partir do roteiro, tentando os
    provedores de TTS configurados em ordem de prioridade (OpenAI TTS ->
    Google Cloud TTS -> ElevenLabs) até um deles funcionar.

    Erro de entrada (roteiro vazio/inválido) nunca é mascarado como falha de
    provedor: é validado uma única vez, antes de qualquer tentativa. Falhas
    de provedor (quota, autenticação, indisponibilidade, timeout, erro
    inesperado) fazem cair para o próximo da fila; timeout especificamente
    tenta de novo o mesmo provedor (retry/backoff) antes de trocar.
    """
    if not texto or not texto.strip():
        raise ValueError("gerar_audio: roteiro vazio ou inválido.")

    texto_narracao = preparar_texto_para_narracao(texto)

    provedores = _provedores_tts_configurados()
    if not provedores:
        raise RuntimeError(
            "❌ TTS: nenhum provedor configurado. Defina OPENAI_API_KEY, "
            "GOOGLE_TTS_API_KEY ou ELEVENLABS_API_KEY."
        )

    erros: List[str] = []

    for nome, func in provedores:
        print(f"🎙️ [2/4] Gerando narração com {nome}...")

        for tentativa in range(1, TTS_TENTATIVAS_POR_PROVEDOR + 1):
            try:
                func(texto_narracao, caminho_saida)
                print(f"✅ Narração gerada com {nome}.")
                return caminho_saida

            except Exception as err:
                # Nunca logar API keys/credenciais — só a mensagem da
                # exceção (as libs oficiais não ecoam a chave nela).
                print(f"⚠️ Falha em {nome} (tentativa {tentativa}/{TTS_TENTATIVAS_POR_PROVEDOR}): {err}")

                if _erro_e_timeout(err) and tentativa < TTS_TENTATIVAS_POR_PROVEDOR:
                    espera = RETRY_BACKOFF_BASE * (2 ** (tentativa - 1))
                    espera += random.uniform(0, RETRY_BACKOFF_JITTER)
                    print(f"⏳ Timeout em {nome}, nova tentativa em {espera:.1f}s...")
                    time.sleep(espera)
                    continue

                erros.append(f"{nome}: {err}")
                break  # próximo provedor da fila

    raise RuntimeError(
        "❌ TTS: todos os provedores configurados falharam. Detalhes: " + " | ".join(erros)
    )


# ===========================================================================
# 2.1 TRANSCRIÇÃO E SINCRONIZAÇÃO DE PALAVRAS (faster-whisper)
# ===========================================================================
# Escolha de tecnologia: faster-whisper (CTranslate2) em vez do Whisper
# "original" (openai-whisper) ou do whisper.cpp.
#
#   • openai-whisper: mesma qualidade de transcrição, mas roda vários
#     múltiplos mais devagar em CPU (não tem backend otimizado) — arriscado
#     para o tempo de execução no GitHub Actions.
#   • whisper.cpp: rápido em CPU, mas exigiria compilar um binário C++
#     dentro do workflow (passo extra, mais superfície de manutenção/erro
#     de build) e um wrapper Python separado para extrair os timestamps.
#   • faster-whisper: usa o backend CTranslate2, é sensivelmente mais rápido
#     que o Whisper original em CPU, suporta quantização int8 (menos RAM),
#     já devolve timestamps por palavra nativamente (`word_timestamps=True`)
#     e se instala com um `pip install` normal — sem compilar nada.
#
# Por isso, para "compatibilidade com GitHub Actions + baixo consumo de
# memória + velocidade em CPU + facilidade de manutenção", faster-whisper é
# a opção com melhor equilíbrio para este cenário.

@dataclass
class PalavraLegenda:
    """Estrutura interna de cada palavra transcrita, usada em toda a
    renderização das legendas (agrupamento, cache, animação)."""
    palavra: str
    start: float
    end: float
    duration: float
    index: int
    is_keyword: bool = False
    future_effect: Optional[str] = None  # reservado para uso futuro; não usado ainda


def _normalizar_palavra(palavra: str) -> str:
    """Remove acentos e pontuação, deixando minúsculo — usado só para
    comparações internas (palavra-chave, palavras inseparáveis)."""
    sem_acento = unicodedata.normalize("NFKD", palavra).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", sem_acento.lower())


_REGEX_NUMERO = re.compile(r"^\d+([.,]\d+)?%?$")
_REGEX_ANO = re.compile(r"^(1[6-9]|20)\d{2}$")


def _eh_palavra_chave(palavra: str) -> bool:
    """Detector de palavras importantes (etapa 8): apenas marca, não aplica
    nenhum efeito visual — isso fica para uma fase futura."""
    normalizada = _normalizar_palavra(palavra)
    if normalizada in PALAVRAS_CHAVE:
        return True

    apenas_numero = re.sub(r"[^\d.,%]", "", palavra)
    if apenas_numero and (_REGEX_NUMERO.match(apenas_numero) or _REGEX_ANO.match(apenas_numero)):
        return True

    return False


def transcrever_narracao(caminho_audio: str) -> List[PalavraLegenda]:
    """Transcreve a narração com faster-whisper e devolve timestamps por
    palavra. Roda sobre a narração original (limpa), antes da mixagem com
    música, para não prejudicar a precisão da transcrição.
    """
    print(f"📝 Iniciando transcrição ({TRANSCRIPTION_ENGINE}, modelo '{WHISPER_MODEL_SIZE}')...")
    inicio = time.perf_counter()

    try:
        modelo = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
        segmentos, _info = modelo.transcribe(
            caminho_audio,
            language=WHISPER_LANGUAGE,
            word_timestamps=True,
            vad_filter=True,
        )
    except Exception as err:
        # Cobre tanto falha ao carregar/baixar os pesos do modelo quanto
        # falha ao decodificar um áudio inválido.
        raise RuntimeError(f"❌ Whisper: falha na transcrição ({err}).") from err

    palavras: List[PalavraLegenda] = []
    try:
        for segmento in segmentos:
            for palavra_w in (segmento.words or []):
                texto = palavra_w.word.strip()
                if not texto:
                    continue

                inicio_p, fim_p = float(palavra_w.start), float(palavra_w.end)
                if fim_p <= inicio_p:
                    fim_p = inicio_p + DURACAO_MINIMA_SEGURANCA  # segurança contra timestamps degenerados

                palavras.append(PalavraLegenda(
                    palavra=texto,
                    start=inicio_p,
                    end=fim_p,
                    duration=fim_p - inicio_p,
                    index=len(palavras),
                    is_keyword=_eh_palavra_chave(texto),
                ))
    except Exception as err:
        raise RuntimeError(f"❌ Whisper: falha ao processar os segmentos transcritos ({err}).") from err

    if not palavras:
        raise ValueError("A transcrição não retornou nenhuma palavra — verifique o áudio gerado.")

    print(f"🎤 Transcrição concluída em {time.perf_counter() - inicio:.2f}s")
    print(f"📚 {len(palavras)} palavras")
    return palavras


FINAIS_DE_FRASE = {".", "!", "?", "…"}


def _limitar_palavras_a_duracao(
    palavras: List[PalavraLegenda], duracao_maxima: float
) -> Tuple[List[PalavraLegenda], int]:
    """Garante que nenhuma palavra transcrita ultrapasse a duração real do
    áudio/vídeo final.

    Timestamps além da duração real acontecem de verdade com o Whisper —
    silêncio residual, comportamento do VAD, ou uma pequena diferença entre
    a duração decodificada do áudio e sua duração real. Sem esse limite, as
    legendas geradas a partir desses timestamps fazem a composição final
    "esticar" além do fundo (Ken Burns), e o `CompositeVideoClip` passa a
    exibir preto sólido (seu `bg_color` padrão) nos instantes em que o
    fundo já não está mais "tocando" — a causa raiz confirmada de uma faixa
    quase totalmente preta perto do fim do vídeo.

    Esta é a correção na fonte: valida os dados assim que chegam da
    transcrição, então tudo que é derivado deles (agrupamento, legendas,
    pontos de corte do Ken Burns) já nasce consistente — não precisa de
    nenhum ajuste ou "offset" espalhado por outras funções.

    Retorna a lista corrigida e a quantidade de palavras que precisaram de
    algum ajuste (descartadas ou só com o timestamp encurtado) — usado só
    para o log, não afeta o comportamento.
    """
    limite_inicio = max(0.0, duracao_maxima - DURACAO_MINIMA_SEGURANCA)
    resultado: List[PalavraLegenda] = []
    quantidade_ajustada = 0

    for palavra in palavras:
        novo_start = min(palavra.start, limite_inicio)
        novo_end = min(palavra.end, duracao_maxima)
        if novo_end <= novo_start:
            quantidade_ajustada += 1
            continue  # a palavra ficaria com duração zero/negativa após o corte; descarta

        if novo_start != palavra.start or novo_end != palavra.end:
            quantidade_ajustada += 1
        palavra.start = novo_start
        palavra.end = novo_end
        palavra.duration = novo_end - novo_start
        resultado.append(palavra)

    return resultado, quantidade_ajustada


def agrupar_palavras_em_blocos(palavras: List[PalavraLegenda]) -> List[List[PalavraLegenda]]:
    """Agrupa as palavras em blocos de MIN_GROUP_WORDS a MAX_GROUP_WORDS,
    fechando o grupo preferencialmente em fim de frase ou pausa natural —
    nunca no meio de uma ideia, quando é possível evitar.
    """
    grupos: List[List[PalavraLegenda]] = []
    atual: List[PalavraLegenda] = []

    for i, palavra in enumerate(palavras):
        atual.append(palavra)

        ultima_letra = palavra.palavra.rstrip()[-1:]
        fim_de_frase = ultima_letra in FINAIS_DE_FRASE
        proxima = palavras[i + 1] if i + 1 < len(palavras) else None
        pausa = (proxima.start - palavra.end) if proxima else None
        pausa_longa = pausa is not None and pausa >= PAUSA_MINIMA_QUEBRA

        atingiu_maximo = len(atual) >= MAX_GROUP_WORDS
        pode_fechar = len(atual) >= MIN_GROUP_WORDS
        eh_ultima_palavra = proxima is None

        deve_fechar = atingiu_maximo or eh_ultima_palavra or (pode_fechar and (fim_de_frase or pausa_longa))

        if deve_fechar:
            grupos.append(atual)
            atual = []

    if atual:
        # Sobra de palavras menor que o mínimo (fim do roteiro): funde no
        # grupo anterior se couber, em vez de deixar um grupo órfão.
        if grupos and len(grupos[-1]) + len(atual) <= MAX_GROUP_WORDS:
            grupos[-1].extend(atual)
        else:
            grupos.append(atual)

    print(f"🎬 {len(grupos)} grupos")
    return grupos


# ===========================================================================
# 3. EDITAR VÍDEO (MoviePy + Legendas Dinâmicas)
# ===========================================================================

def _debug_registrar(etapa: str, clip, instante: Optional[float] = None) -> None:
    """Registra o estado de um clip da pipeline no modo de diagnóstico.

    Só faz alguma coisa quando `DEBUG_RENDER=True` (o custo, desligado, é um
    único `if` por chamada). Quando ligado, imprime tipo do clip, tamanho,
    duração, fps e posição, lista os subclipes se for um
    `CompositeVideoClip`, e salva um frame PNG no mesmo instante relativo
    (por padrão, 50% da duração) em `DEBUG_RENDER_DIR/<etapa>.png` — assim
    dá pra abrir os PNGs de todas as etapas lado a lado e ver exatamente em
    qual delas o frame deixa de preencher o canvas.
    """
    if not DEBUG_RENDER:
        return

    os.makedirs(DEBUG_RENDER_DIR, exist_ok=True)

    tipo = type(clip).__name__
    largura, altura = clip.size if getattr(clip, "size", None) else (None, None)
    duracao = getattr(clip, "duration", None)
    fps = getattr(clip, "fps", None)

    if instante is None:
        instante = (duracao * DEBUG_RENDER_INSTANTE_FRACAO) if duracao else 0.0
    if duracao:
        instante = max(0.0, min(instante, duracao - 0.001))
    else:
        instante = 0.0

    pos_str = "n/d"
    pos_fn = getattr(clip, "pos", None)
    if pos_fn is not None:
        try:
            pos_str = str(pos_fn(instante))
        except Exception as err:
            pos_str = f"erro ao ler pos(): {err}"

    print(f"🔍 [DEBUG_RENDER] {etapa}")
    print(f"    tipo={tipo}  tamanho={largura}x{altura}  duracao={duracao}  fps={fps}")
    print(f"    posicao em t={instante:.2f}s: {pos_str}")

    if isinstance(clip, CompositeVideoClip):
        print(f"    subclipes ({len(clip.clips)}):")
        for i, sub in enumerate(clip.clips):
            sub_tamanho = getattr(sub, "size", None)
            sub_start = getattr(sub, "start", None)
            print(f"      [{i}] tipo={type(sub).__name__} tamanho={sub_tamanho} start={sub_start}")

    try:
        frame = clip.get_frame(instante)
        altura_real, largura_real = frame.shape[0], frame.shape[1]
        bate = (largura_real, altura_real) == (largura, altura)
        print(
            f"    frame real em t={instante:.2f}s: shape={frame.shape}  "
            f"bate com o tamanho declarado ({largura}x{altura})? {bate}"
        )

        caminho_png = os.path.join(DEBUG_RENDER_DIR, f"{etapa}.png")
        Image.fromarray(frame).save(caminho_png)
        print(f"    frame salvo em: {caminho_png}")
    except Exception as err:
        print(f"    ⚠️ não foi possível capturar/salvar o frame desta etapa: {err}")

    print()


def _debug_inspecionar_clip(indice, clip, instante: float = 0.0) -> None:
    """Imprime os atributos detalhados de UM clip: tipo, size, mask, start,
    end, position, relative_pos, ismask. Usado para vasculhar clip a clip
    dentro de uma composição, procurando qualquer um que destoe dos demais
    (tamanho diferente, posição deslocada, etc.)."""
    tipo = type(clip).__name__
    tamanho = getattr(clip, "size", None)
    tem_mask = getattr(clip, "mask", None) is not None
    start = getattr(clip, "start", None)
    end = getattr(clip, "end", None)
    duracao = getattr(clip, "duration", None)
    ismask = getattr(clip, "ismask", None)
    relative_pos = getattr(clip, "relative_pos", None)

    pos_valor = "n/d"
    pos_fn = getattr(clip, "pos", None)
    if pos_fn is not None:
        try:
            pos_valor = pos_fn(instante)
        except Exception as err:
            pos_valor = f"erro: {err}"

    print(
        f"  [{indice}] tipo={tipo} size={tamanho} mask={tem_mask} ismask={ismask} "
        f"start={start} end={end} duracao={duracao} pos={pos_valor} relative_pos={relative_pos}"
    )


def _debug_registrar_composicao(etapa: str, video_final: CompositeVideoClip, instante: Optional[float] = None) -> None:
    """Versão detalhada de `_debug_registrar`, específica para o
    CompositeVideoClip final: além do resumo padrão, inspeciona TODOS os
    subclipes com `_debug_inspecionar_clip` e imprime o `bg_color` e o
    `transparent` do composite — que é quem decide se sobra 'fundo preto'
    aparecendo atrás de algum clip."""
    if not DEBUG_RENDER:
        return

    _debug_registrar(etapa, video_final, instante)

    if instante is None:
        instante = (video_final.duration * DEBUG_RENDER_INSTANTE_FRACAO) if video_final.duration else 0.0

    print(f"🔍 [DEBUG_RENDER] {etapa} — inspeção clip a clip:")
    print(f"    CompositeVideoClip.size (o que define o canvas final) = {video_final.size}")
    print(f"    bg_color = {getattr(video_final, 'bg_color', 'n/d')}")
    for i, sub in enumerate(video_final.clips):
        _debug_inspecionar_clip(i, sub, instante)
    print()


def _debug_aplicar_borda_canvas(clip, espessura: int = 4, cor=(255, 0, 0)):
    """Desenha uma borda diretamente NOS PIXELS do frame final, exatamente
    nas bordas do canvas do clip — não é um overlay separado, é o próprio
    array de pixels do frame que recebe a borda. Usado só em investigação
    (DEBUG_RENDER=True) para visualizar onde o MoviePy acredita que o vídeo
    termina, mesmo que o conteúdo real não preencha até lá."""

    def aplicar(frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()
        e = min(espessura, frame.shape[0] // 2, frame.shape[1] // 2)
        frame[:e, :] = cor
        frame[-e:, :] = cor
        frame[:, :e] = cor
        frame[:, -e:] = cor
        return frame

    return clip.fl_image(aplicar)


def _escolher_posicao_de_corte(sobra: float) -> float:
    """Escolhe um deslocamento de corte agradável em vez de sempre centralizar.

    Recebe o espaço "sobrando" (em pixels) para cortar de um lado e escolhe um
    ponto de corte levemente fora do centro, evitando que o resultado pareça
    sempre a mesma composição.
    """
    if sobra <= 0:
        return 0.0
    # Desloca até CROP_OFFSET_MAX_FRACTION da folga disponível para um dos
    # lados, nunca cortando de forma extrema. O sinal (esquerda/direita ou
    # cima/baixo) é aleatório.
    deslocamento_max = sobra * CROP_OFFSET_MAX_FRACTION
    return random.uniform(-deslocamento_max, deslocamento_max)


def normalize_frame_size(frame: np.ndarray, altura: int, largura: int) -> np.ndarray:
    """Garante que um array de frame OU máscara tenha EXATAMENTE
    `altura` x `largura` pixels, preservando o número de canais de entrada
    (funciona com grayscale `(H,W)`, máscara `(H,W,1)`, RGB `(H,W,3)` e
    RGBA `(H,W,4)`).

    Usada como blindagem central contra a classe de bug que causava
    `operands could not be broadcast together with shapes (1921,1080,3)
    (1918,1080,1)`: duas peças da composição (ex.: um segmento de B-roll e
    a máscara/overlay construída para o canvas "oficial") terminando com
    alturas ligeiramente diferentes por causa de arredondamentos
    independentes em crops/resizes anteriores.

    Estratégia deliberadamente conservadora — NUNCA distorce (nunca dá
    resize/stretch) e nunca faz broadcast artificial: quando o array é
    MAIOR que o alvo em algum eixo, corta o excedente centralizado; quando
    é MENOR, preenche a borda replicando o pixel mais próximo
    (`mode="edge"`, sem inventar cor nova nem preto artificial). Isso só é
    seguro porque a diferença esperada aqui é de arredondamento (poucos
    pixels) — nunca uma mudança real de proporção; um desvio grande
    (ex.: dezenas de pixels) é sinal de um bug diferente e mais sério, não
    algo para este utilitário "corrigir" silenciosamente.
    """
    if frame.ndim not in (2, 3):
        raise ValueError(
            f"normalize_frame_size: array com {frame.ndim} dimensões não é "
            f"suportado (esperado 2 = grayscale/máscara ou 3 = RGB/RGBA)."
        )

    h, w = frame.shape[:2]

    # --- eixo vertical (altura) ---
    if h > altura:
        sobra = h - altura
        y0 = sobra // 2
        frame = frame[y0 : y0 + altura]
    elif h < altura:
        falta = altura - h
        topo = falta // 2
        base = falta - topo
        pad_width = [(topo, base), (0, 0)] + [(0, 0)] * (frame.ndim - 2)
        frame = np.pad(frame, pad_width, mode="edge")

    # --- eixo horizontal (largura) ---
    h_atual, w_atual = frame.shape[:2]
    if w_atual > largura:
        sobra = w_atual - largura
        x0 = sobra // 2
        frame = frame[:, x0 : x0 + largura]
    elif w_atual < largura:
        falta = largura - w_atual
        esquerda = falta // 2
        direita = falta - esquerda
        pad_width = [(0, 0), (esquerda, direita)] + [(0, 0)] * (frame.ndim - 2)
        frame = np.pad(frame, pad_width, mode="edge")

    return frame


def _redimensionar_para_vertical(clip: VideoFileClip) -> VideoFileClip:
    """Corta o vídeo de fundo para o formato vertical 9:16 com crop inteligente.

    Em vez de sempre cortar exatamente no centro, escolhe um ponto de corte
    levemente deslocado quando há folga suficiente, resultando em uma
    composição um pouco mais agradável e menos repetitiva entre execuções.
    """
    w, h = clip.size

    if w / h > ASPECT_RATIO:
        nova_largura = h * ASPECT_RATIO
        sobra = w - nova_largura
        x1 = (w - nova_largura) / 2 + _escolher_posicao_de_corte(sobra)
        x1 = max(0, min(x1, w - nova_largura))
        clip = clip.crop(x1=x1, y1=0, width=nova_largura, height=h)
    else:
        nova_altura = w / ASPECT_RATIO
        sobra = h - nova_altura
        y1 = (h - nova_altura) / 2 + _escolher_posicao_de_corte(sobra)
        y1 = max(0, min(y1, h - nova_altura))
        clip = clip.crop(x1=0, y1=y1, width=w, height=nova_altura)

    _debug_registrar("02_crop", clip)

    clip = clip.resize(width=VIDEO_WIDTH)

    _debug_registrar("03_resize", clip)

    # Blindagem definitiva (Problema 4/5 da auditoria): `clip.resize(width=
    # VIDEO_WIDTH)` deixa a altura final a cargo do arredondamento interno
    # do MoviePy a partir da altura ORIGINAL deste vídeo específico — dois
    # vídeos de fundo diferentes (aspect ratios de origem diferentes) podem
    # terminar com alturas ligeiramente diferentes (1918 vs 1921px) mesmo
    # após o mesmo crop 9:16 + resize por largura. Forçamos aqui, de uma
    # vez por todas, TODO frame emitido por este clip a ter exatamente
    # VIDEO_WIDTH x VIDEO_HEIGHT — a MESMA função `normalize_frame_size`
    # usada nos testes de regressão do bug de broadcasting. Isso garante,
    # por construção, que TODOS os segmentos de B-roll concatenados em
    # `_montar_fundo_multiplo()` compartilham exatamente o mesmo tamanho de
    # canvas — a causa raiz do `operands could not be broadcast together`.
    if clip.size != (VIDEO_WIDTH, VIDEO_HEIGHT):
        clip = clip.fl_image(lambda frame: normalize_frame_size(frame, VIDEO_HEIGHT, VIDEO_WIDTH))
        clip.size = (VIDEO_WIDTH, VIDEO_HEIGHT)  # metadado (w, h) do MoviePy — fl_image não o recalcula sozinho

    _debug_registrar("03b_normalizado", clip)

    return clip


def _ease_suave(x: float) -> float:
    """Curva de suavização (smoothstep clássico: 3x²-2x³).

    Mapeia [0,1] -> [0,1], mas com velocidade zero nas duas pontas — dá uma
    sensação de aceleração/desaceleração suave em vez de velocidade
    constante. Custo desprezível (duas multiplicações), usada no Ken Burns
    e no fade do vídeo para uma sensação mais "editada à mão".
    """
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x * x * (3.0 - 2.0 * x)


def _progresso_punch_hook(t: float, eh_hook: bool) -> float:
    """Calcula o "punch" de zoom extra (Fase 3, item 8 — hook) somado ao
    fator de zoom normal do Ken Burns, só para `eh_hook=True` e só dentro
    dos primeiros `HOOK_ZOOM_PUNCH_DURATION` segundos do trecho.

    Decai suavemente (`_ease_suave`) de `HOOK_ZOOM_PUNCH - 1.0` até 0 — uma
    entrada rápida e visualmente forte que se resolve no Ken Burns
    contínuo normal, em vez de um zoom constante o vídeo inteiro. Fora
    dessa janela (ou quando `eh_hook=False`), o retorno é sempre 0.0 e o
    Ken Burns se comporta exatamente como antes.
    """
    if not eh_hook or t >= HOOK_ZOOM_PUNCH_DURATION or t < 0:
        return 0.0
    progresso_restante = 1.0 - _ease_suave(t / HOOK_ZOOM_PUNCH_DURATION)
    return (HOOK_ZOOM_PUNCH - 1.0) * progresso_restante


def _fator_entrada_saida_grupo(t: float, duracao_grupo: float, eh_hook: bool) -> Tuple[float, float]:
    """Calcula (escala, opacidade) da animação de entrada/saída de UM grupo
    de legenda (Fase 3, item 7), em função do tempo relativo `t` (0 =
    início do grupo) e da duração total do grupo em tela.

    Entrada: fade + "scale-in" suave (escala inicial menor, sobe até 100%
    junto com a opacidade) — mais pronunciada no hook
    (`HOOK_ANIM_SCALE_START` < `LEGEND_ANIM_SCALE_START`), conforme pedido
    no item 8 (tratamento especial dos primeiros segundos). Saída: só
    fade (a escala permanece 100%) — encolher na saída lê como "erro de
    render" mais do que como uma transição intencional. Fora das duas
    janelas, devolve (1.0, 1.0): nenhum custo extra por frame no meio da
    exibição do grupo, que é a maior parte do tempo.
    """
    escala_inicial = HOOK_ANIM_SCALE_START if eh_hook else LEGEND_ANIM_SCALE_START

    if t < LEGEND_ANIM_IN_DURATION:
        progresso = _ease_suave(max(0.0, t) / LEGEND_ANIM_IN_DURATION)
        escala = escala_inicial + (1.0 - escala_inicial) * progresso
        return escala, progresso

    tempo_restante = duracao_grupo - t
    if tempo_restante < LEGEND_ANIM_OUT_DURATION:
        progresso = _ease_suave(max(0.0, tempo_restante) / LEGEND_ANIM_OUT_DURATION)
        return 1.0, progresso

    return 1.0, 1.0


def _sequencia_zoom_por_bloco(quantidade_blocos: int) -> List[float]:
    """Gera uma sequência de níveis de zoom que nunca repete o valor anterior."""
    sequencia: List[float] = []
    anterior: Optional[float] = None

    for _ in range(quantidade_blocos):
        candidatos = [z for z in BLOCK_ZOOM_LEVELS if z != anterior]
        escolha = random.choice(candidatos)
        sequencia.append(escolha)
        anterior = escolha

    return sequencia


def _aplicar_ken_burns(
    clip: VideoFileClip,
    duracao_total: float,
    pontos_de_corte: List[float],
    eh_hook: bool = False,
) -> VideoClip:
    """Aplica um efeito Ken Burns sutil: zoom lento contínuo + zoom por segmento.

    `eh_hook=True` (Fase 3, item 8) soma um "punch" de zoom extra e rápido
    nos primeiros `HOOK_ZOOM_PUNCH_DURATION` segundos — ver
    `_progresso_punch_hook()`. Usado só no primeiro trecho de B-roll do
    vídeo; em todos os outros trechos o comportamento é idêntico ao de
    antes desta fase.

    IMPLEMENTAÇÃO AUTO-CONTIDA (reescrita após um bug de renderização): em
    vez de usar `.resize()` + `.set_position()` compostos dentro de um
    `CompositeVideoClip` — que acaba embrulhado em máscaras aninhadas pelo
    MoviePy e pode dessincronizar o tamanho da imagem com a posição
    calculada — este Ken Burns usa um único `make_frame` autossuficiente:
    a cada instante, recorta uma janela do frame ORIGINAL (nas dimensões
    reais do vídeo de fundo, já em 9:16) e amplia essa janela para
    EXATAMENTE o tamanho do canvas.

    Isso garante, por construção (não por "clamp" após o fato):
    - a janela de recorte NUNCA sai dos limites do frame original (as
      coordenadas de recorte são sempre calculadas dentro de [0, folga]);
    - o frame de saída tem SEMPRE exatamente `canvas_w x canvas_h`, porque
      é o próprio alvo do resize final — nunca "menor que o canvas";
    - não há bordas pretas possíveis, pois nunca existe um "canvas vazio"
      por trás — cada frame é 100% preenchido pelo recorte redimensionado.

    O zoom contínuo e o pan usam uma curva suave (`_ease_suave`) em vez de
    velocidade linear constante, e a troca de nível de zoom entre grupos de
    legenda é uma transição curta e suave (`ZOOM_TRANSICAO_DURACAO`) em vez
    de um degrau instantâneo — dá uma sensação de câmera "respirando" em
    vez de saltando, sem nenhum custo extra de renderização (é a mesma
    quantidade de operações por quadro, só com uma curva diferente).
    """
    canvas_w, canvas_h = clip.size

    limites = sorted(set([0.0, duracao_total] + [t for t in pontos_de_corte if 0 < t < duracao_total]))
    quantidade_segmentos = max(1, len(limites) - 1)
    sequencia_zoom = _sequencia_zoom_por_bloco(quantidade_segmentos)

    def indice_do_segmento(t: float) -> int:
        idx = bisect.bisect_right(limites, t) - 1
        return min(max(idx, 0), quantidade_segmentos - 1)

    direcao_x = random.choice([-1, 1])
    direcao_y = random.choice([-1, 1])

    def fator_zoom(t: float) -> float:
        """Fator de zoom (sempre >= 1.0): zoom lento contínuo + zoom por
        segmento, somados de forma aditiva para ficar sempre sutil."""
        idx = indice_do_segmento(t)
        zoom_alvo = sequencia_zoom[idx]

        # Suaviza a troca de nível de zoom: nos primeiros
        # ZOOM_TRANSICAO_DURACAO segundos de um novo segmento, interpola do
        # nível anterior até o novo em vez de saltar instantaneamente.
        if idx > 0:
            tempo_no_segmento = t - limites[idx]
            if tempo_no_segmento < ZOOM_TRANSICAO_DURACAO:
                zoom_anterior = sequencia_zoom[idx - 1]
                progresso_transicao = _ease_suave(tempo_no_segmento / ZOOM_TRANSICAO_DURACAO)
                zoom_bloco = zoom_anterior + (zoom_alvo - zoom_anterior) * progresso_transicao
            else:
                zoom_bloco = zoom_alvo
        else:
            zoom_bloco = zoom_alvo

        progresso_geral = _ease_suave(min(t / duracao_total, 1.0)) if duracao_total > 0 else 0.0
        zoom_lento = 1.0 + (KEN_BURNS_ZOOM_MAX - 1.0) * progresso_geral
        punch_hook = _progresso_punch_hook(t, eh_hook)
        return max(1.0, 1.0 + (zoom_bloco - 1.0) + (zoom_lento - 1.0) + punch_hook)

    def make_frame(t: float) -> np.ndarray:
        frame_original = clip.get_frame(t)
        altura_original, largura_original = frame_original.shape[:2]
        fator = fator_zoom(t)

        # Janela de recorte, em pixels do frame ORIGINAL. Como fator >= 1,
        # a janela é sempre <= ao frame original (nunca precisa "inventar"
        # pixels fora dele).
        crop_w = min(largura_original, max(1, int(round(largura_original / fator))))
        crop_h = min(altura_original, max(1, int(round(altura_original / fator))))

        folga_x = largura_original - crop_w  # quanto a janela pode "andar" na horizontal
        folga_y = altura_original - crop_h   # quanto pode "andar" na vertical

        progresso_pan_linear = min(t / duracao_total, 1.0) if duracao_total > 0 else 0.0
        progresso_pan = (_ease_suave(progresso_pan_linear) - 0.5) * 2

        # Deslocamento desejado (na escala do canvas final, para manter a
        # mesma sensação de movimento configurada em PAN_MAX_FRACTION),
        # convertido para o espaço do frame original e travado dentro da
        # folga REAL disponível — nunca pode ultrapassar os limites do
        # frame original, então nunca revela "vazio".
        desloc_x_desejado = (direcao_x * PAN_MAX_FRACTION * canvas_w * progresso_pan) / fator
        desloc_y_desejado = (direcao_y * PAN_MAX_FRACTION * canvas_h * progresso_pan) / fator
        limite_x = folga_x / 2
        limite_y = folga_y / 2
        desloc_x = max(-limite_x, min(limite_x, desloc_x_desejado))
        desloc_y = max(-limite_y, min(limite_y, desloc_y_desejado))

        x0 = int(round(folga_x / 2 + desloc_x))
        y0 = int(round(folga_y / 2 + desloc_y))
        x0 = max(0, min(x0, folga_x))
        y0 = max(0, min(y0, folga_y))

        recorte = frame_original[y0:y0 + crop_h, x0:x0 + crop_w]

        if recorte.shape[1] == canvas_w and recorte.shape[0] == canvas_h:
            return recorte

        imagem_final = Image.fromarray(recorte).resize((canvas_w, canvas_h), Image.BILINEAR)
        return np.array(imagem_final)

    clip_ken_burns = VideoClip(make_frame=make_frame, duration=duracao_total)
    clip_ken_burns.fps = getattr(clip, "fps", None) or 24
    return clip_ken_burns


def _aplicar_vinheta(clip: CompositeVideoClip, intensidade: float = VIGNETTE_INTENSITY) -> CompositeVideoClip:
    """Aplica uma vinheta discreta (escurecimento sutil nos cantos)."""
    if intensidade <= 0:
        return clip

    largura, altura = clip.size
    y, x = np.ogrid[:altura, :largura]
    centro_x, centro_y = largura / 2, altura / 2
    distancia_max = math.hypot(centro_x, centro_y)

    distancia_normalizada = np.hypot(x - centro_x, y - centro_y) / distancia_max
    mascara = 1.0 - intensidade * (distancia_normalizada ** 2)
    mascara = np.clip(mascara, 1.0 - intensidade, 1.0)
    mascara = mascara[..., np.newaxis]  # broadcast para os 3 canais RGB

    def aplicar_mascara(frame: np.ndarray) -> np.ndarray:
        return np.clip(frame.astype(np.float32) * mascara, 0, 255).astype("uint8")

    return clip.fl_image(aplicar_mascara)


def _aplicar_fade_video_suave(clip: CompositeVideoClip, duracao_fade: float) -> CompositeVideoClip:
    """Fade de abertura/fechamento com curva suave (`_ease_suave`) em vez da
    rampa linear padrão do MoviePy (`vfx.fadein`/`vfx.fadeout`).

    Mesmo custo por quadro do fade padrão (uma multiplicação do frame por
    um escalar) — só muda a curva do escalar, então não há impacto
    perceptível no tempo de renderização. Feito num único `fl()` para
    entrada e saída juntas, em vez de dois `.fx()` encadeados.
    """
    duracao_total = clip.duration

    def fator_fade(t: float) -> float:
        if t < duracao_fade:
            return _ease_suave(t / duracao_fade)
        tempo_restante = duracao_total - t
        if tempo_restante < duracao_fade:
            return _ease_suave(tempo_restante / duracao_fade)
        return 1.0

    def transformar(get_frame, t):
        frame = get_frame(t)
        fator = fator_fade(t)
        if fator >= 1.0:
            return frame
        return (frame.astype(np.float32) * fator).astype("uint8")

    return clip.fl(transformar, keep_duration=True)


def _criar_barra_progresso(largura: int, duracao: float) -> VideoClip:
    """Cria uma barra de progresso fina e elegante no topo do vídeo.

    A barra só começa a avançar depois que o fade de abertura termina, e
    chega aos 100% exatamente quando o fade de fechamento começa — assim
    "progresso completo" e "tela escurecendo" acontecem como um único
    gesto, em vez de dois eventos quase simultâneos mas descoordenados.
    Reaproveita FADE_TIME, que já é conhecido nesse ponto — nenhum cálculo
    novo caro, só desloca a janela em que o avanço acontece.
    """
    inicio_avanco = FADE_TIME
    fim_avanco = max(inicio_avanco + 0.01, duracao - FADE_TIME)
    duracao_avanco = fim_avanco - inicio_avanco

    def make_frame(t: float) -> np.ndarray:
        frame = np.empty((BAR_HEIGHT, largura, 3), dtype=np.uint8)
        frame[:, :] = BAR_BACKGROUND_COLOR

        if t <= inicio_avanco:
            progresso_exato = 0.0
        elif t >= fim_avanco:
            progresso_exato = float(largura)
        else:
            progresso_exato = largura * (t - inicio_avanco) / duracao_avanco

        progresso_px = int(progresso_exato)
        fracao = progresso_exato - progresso_px  # parte sub-pixel da borda de avanço

        if progresso_px > 0:
            frame[:, :progresso_px] = BAR_COLOR

        # Mistura o pixel exatamente na borda proporcionalmente à fração —
        # sem isso, a borda "pula" um pixel inteiro por vez (visível como um
        # leve serrilhado/tremor após a compressão do vídeo). Custo: um
        # único pixel de largura calculado por quadro, desprezível.
        if fracao > 0 and progresso_px < largura:
            frame[:, progresso_px] = [
                int(bg * (1 - fracao) + fg * fracao)
                for bg, fg in zip(BAR_BACKGROUND_COLOR, BAR_COLOR)
            ]

        return frame

    barra = VideoClip(make_frame=make_frame, duration=duracao)
    barra = barra.set_position((0, 0)).set_opacity(BAR_OPACITY)
    return barra


# ---------------------------------------------------------------------------
# 3.1 CACHE E RENDERIZAÇÃO DE PALAVRAS (etapas 5, 6 e 9)
# ---------------------------------------------------------------------------
# Cada combinação única de (texto da palavra, estilo) é renderizada via
# ImageMagick/TextClip APENAS UMA VEZ por execução do bot e fica guardada
# aqui. Como palavras curtas ("que", "de", "não"...) se repetem muito em
# português, isso reduz bastante o número de chamadas ao ImageMagick — que é
# o gargalo mais caro de todo o sistema de legendas.
_CACHE_PALAVRAS_RENDERIZADAS: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
_estatisticas_cache = {"acertos": 0, "faltas": 0}


def _resetar_cache_legendas() -> None:
    """Limpa o cache de tiles de palavras (chamado uma vez por vídeo)."""
    _CACHE_PALAVRAS_RENDERIZADAS.clear()
    _estatisticas_cache["acertos"] = 0
    _estatisticas_cache["faltas"] = 0


def _obter_frame_e_alpha(clip: TextClip) -> Tuple[np.ndarray, np.ndarray]:
    """Extrai o frame RGB e o canal alpha (transparência) de um TextClip."""
    frame = clip.get_frame(0)
    if clip.mask is not None:
        alpha = clip.mask.get_frame(0).astype(np.float32)
    else:
        alpha = np.ones(frame.shape[:2], dtype=np.float32)
    return frame, alpha


def _colar_tile(
    canvas_rgb: np.ndarray,
    canvas_alpha: np.ndarray,
    tile_rgb: np.ndarray,
    tile_alpha: np.ndarray,
    x: int,
    y: int,
) -> None:
    """Cola (com alpha blending) um tile de imagem sobre um canvas, in-place."""
    altura_canvas, largura_canvas = canvas_alpha.shape
    h, w = tile_alpha.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, largura_canvas), min(y + h, altura_canvas)
    if x1 <= x0 or y1 <= y0:
        return  # tile cai inteiramente fora do canvas

    tx0, ty0 = x0 - x, y0 - y
    tx1, ty1 = tx0 + (x1 - x0), ty0 + (y1 - y0)

    alpha_fatia = tile_alpha[ty0:ty1, tx0:tx1][..., np.newaxis]
    canvas_rgb[y0:y1, x0:x1] = (
        tile_rgb[ty0:ty1, tx0:tx1] * alpha_fatia + canvas_rgb[y0:y1, x0:x1] * (1 - alpha_fatia)
    ).astype(np.uint8)
    canvas_alpha[y0:y1, x0:x1] = np.maximum(
        canvas_alpha[y0:y1, x0:x1], tile_alpha[ty0:ty1, tx0:tx1]
    )


def _obter_tile_palavra(texto: str, estilo: str) -> Tuple[np.ndarray, np.ndarray]:
    """Renderiza (ou reaproveita do cache) o tile pronto de uma palavra.

    `estilo` é "normal" (branca) ou "ativa" (amarela, com contorno preto e
    sombra discreta) — os dois únicos estilos definidos na etapa 5.
    """
    chave = (texto, estilo)
    tile_em_cache = _CACHE_PALAVRAS_RENDERIZADAS.get(chave)
    if tile_em_cache is not None:
        _estatisticas_cache["acertos"] += 1
        return tile_em_cache

    _estatisticas_cache["faltas"] += 1

    try:
        if estilo == "ativa":
            clip_texto = TextClip(
                texto, fontsize=LEGEND_FONT_SIZE, font=LEGEND_FONT, color=ACTIVE_COLOR,
                stroke_color=STROKE_COLOR, stroke_width=STROKE_WIDTH, method="label",
            )
            clip_sombra = TextClip(
                texto, fontsize=LEGEND_FONT_SIZE, font=LEGEND_FONT, color="black", method="label",
            ).set_opacity(LEGEND_SHADOW_OPACITY)

            largura = max(clip_texto.w, clip_sombra.w + LEGEND_SHADOW_OFFSET[0])
            altura = max(clip_texto.h, clip_sombra.h + LEGEND_SHADOW_OFFSET[1])
            canvas_rgb = np.zeros((altura, largura, 3), dtype=np.uint8)
            canvas_alpha = np.zeros((altura, largura), dtype=np.float32)

            sombra_rgb, sombra_alpha = _obter_frame_e_alpha(clip_sombra)
            _colar_tile(canvas_rgb, canvas_alpha, sombra_rgb, sombra_alpha, *LEGEND_SHADOW_OFFSET)

            texto_rgb, texto_alpha = _obter_frame_e_alpha(clip_texto)
            _colar_tile(canvas_rgb, canvas_alpha, texto_rgb, texto_alpha, 0, 0)

            clip_texto.close()
            clip_sombra.close()
            resultado = (canvas_rgb, canvas_alpha)
        else:
            clip_texto = TextClip(
                texto, fontsize=LEGEND_FONT_SIZE, font=LEGEND_FONT, color=NORMAL_COLOR, method="label",
        )
        resultado = tuple(a.copy() for a in _obter_frame_e_alpha(clip_texto))
        clip_texto.close()
    except Exception as err:
        raise RuntimeError(
            f"❌ ImageMagick: falha ao renderizar o texto da legenda ('{texto}'). "
            f"Verifique se o ImageMagick está instalado e se a policy.xml permite "
            f"renderização de texto (\"@*\" com rights=\"read | write\"). Erro original: {err}"
        ) from err

    _CACHE_PALAVRAS_RENDERIZADAS[chave] = resultado
    return resultado


def _escalar_tile(tile_rgb: np.ndarray, tile_alpha: np.ndarray, fator: float) -> Tuple[np.ndarray, np.ndarray]:
    """Redimensiona um tile já renderizado — usado só no "pop" da palavra
    ativa, evitando ter que re-renderizar texto via ImageMagick a cada
    quadro da animação (o cache guarda somente a versão em 100%)."""
    if abs(fator - 1.0) < 1e-3:
        return tile_rgb, tile_alpha

    altura, largura = tile_alpha.shape
    nova_largura = max(1, round(largura * fator))
    nova_altura = max(1, round(altura * fator))

    img_rgb = Image.fromarray(tile_rgb).resize((nova_largura, nova_altura), Image.BILINEAR)
    img_alpha = Image.fromarray((tile_alpha * 255).astype(np.uint8)).resize((nova_largura, nova_altura), Image.BILINEAR)
    return np.array(img_rgb), np.array(img_alpha).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 3.2 AGRUPAMENTO VISUAL E ANIMAÇÃO (etapas 3, 4, 6 e 7)
# ---------------------------------------------------------------------------
def _calcular_layout_grupo(
    grupo: List[PalavraLegenda], largura_maxima: int
) -> Tuple[List[List[Tuple[PalavraLegenda, int, int]]], int, int]:
    """Calcula a posição (x, y) central de cada palavra dentro do grupo.

    O layout é calculado UMA ÚNICA VEZ por grupo, usando a largura da
    palavra no estilo "normal" — por isso a posição de cada palavra não se
    move quando ela fica ativa (mesmo com o "pop"), o que evita o efeito de
    legenda "pulando" (etapa 7). Só quebra em mais de uma linha se o grupo
    realmente não couber na largura disponível, e evita quebrar logo depois
    de um artigo/preposição isolado.
    """
    larguras: Dict[int, int] = {}
    altura_linha = 0
    for palavra in grupo:
        _, alpha = _obter_tile_palavra(palavra.palavra, "normal")
        larguras[palavra.index] = alpha.shape[1]
        altura_linha = max(altura_linha, alpha.shape[0])

    linhas: List[List[PalavraLegenda]] = [[]]
    largura_linha_atual = 0

    for palavra in grupo:
        largura_palavra = larguras[palavra.index]
        espaco = LEGEND_WORD_SPACING if linhas[-1] else 0
        cabe_na_linha = largura_linha_atual + espaco + largura_palavra <= largura_maxima

        anterior_insep = (
            linhas[-1] and _normalizar_palavra(linhas[-1][-1].palavra) in PALAVRAS_INSEPARAVEIS
        )

        if linhas[-1] and not cabe_na_linha and not anterior_insep:
            linhas.append([])
            largura_linha_atual = 0
            espaco = 0

        linhas[-1].append(palavra)
        largura_linha_atual += espaco + largura_palavra

    layout: List[List[Tuple[PalavraLegenda, int, int]]] = []
    for indice_linha, linha in enumerate(linhas):
        largura_total = sum(larguras[p.index] for p in linha) + LEGEND_WORD_SPACING * (len(linha) - 1)
        x = (largura_maxima - largura_total) // 2
        cy = indice_linha * (altura_linha + LEGEND_LINE_SPACING) + altura_linha // 2
        posicoes = []
        for palavra in linha:
            largura_palavra = larguras[palavra.index]
            cx = x + largura_palavra // 2
            posicoes.append((palavra, cx, cy))
            x += largura_palavra + LEGEND_WORD_SPACING
        layout.append(posicoes)

    return layout, altura_linha, len(linhas)


def _desenhar_grupo(
    frame: np.ndarray,
    alpha: np.ndarray,
    layout: List[List[Tuple[PalavraLegenda, int, int]]],
    t_abs: float,
) -> None:
    """Desenha, num instante absoluto `t_abs`, todas as palavras do grupo.

    Cada palavra fica "ativa" (amarela) assim que é pronunciada e permanece
    assim pelo resto da exibição do grupo — só uma palavra muda de estilo
    a cada transição (etapa 4), e o "pop" (100% -> 108% -> 100%) toca uma
    única vez, exatamente quando a palavra se torna ativa (etapa 6).
    """
    for linha in layout:
        for palavra, cx, cy in linha:
            estilo = "ativa" if t_abs >= palavra.start else "normal"
            tile_rgb, tile_alpha = _obter_tile_palavra(palavra.palavra, estilo)

            if palavra.start <= t_abs <= palavra.start + POP_DURATION:
                progresso = (t_abs - palavra.start) / POP_DURATION
                fator = 1.0 + (POP_SCALE - 1.0) * math.sin(min(progresso, 1.0) * math.pi)
                tile_rgb, tile_alpha = _escalar_tile(tile_rgb, tile_alpha, fator)

            h, w = tile_alpha.shape
            x, y = cx - w // 2, cy - h // 2
            _colar_tile(frame, alpha, tile_rgb, tile_alpha, x, y)


def _construir_clip_grupo(
    grupo: List[PalavraLegenda], largura_video: int, altura_video: int, eh_hook: bool = False
) -> VideoClip:
    """Constrói UM único VideoClip dinâmico para o grupo inteiro.

    Em vez de empilhar vários TextClips (um por palavra por estado), o
    grupo inteiro vira um único clip cujo `make_frame` compõe os tiles já
    cacheados a cada quadro — poucos objetos MoviePy na composição final,
    menos overhead de renderização.

    `eh_hook=True` (Fase 3, item 8) só afeta o PRIMEIRO grupo do vídeo: a
    animação de entrada usa `HOOK_ANIM_SCALE_START` em vez de
    `LEGEND_ANIM_SCALE_START` — ver `_fator_entrada_saida_grupo()`.
    """
    largura_maxima = int(largura_video * LEGEND_WIDTH_FRACTION)
    layout, altura_linha, num_linhas = _calcular_layout_grupo(grupo, largura_maxima)
    altura_total = num_linhas * altura_linha + (num_linhas - 1) * LEGEND_LINE_SPACING

    inicio_grupo = grupo[0].start
    duracao_grupo = max(grupo[-1].end - inicio_grupo, DURACAO_MINIMA_SEGURANCA)

    # OTIMIZAÇÃO (gargalo real encontrado em auditoria): o MoviePy, ao
    # compor este clip sobre o vídeo (`blit_on`), chama `get_frame(t)` e
    # `mask.get_frame(t)` SEPARADAMENTE para o mesmo instante `t` — sem
    # cache, `_desenhar_grupo` (com todos os lookups de tile e o resize do
    # "pop") rodaria DUAS VEZES por frame, para cada grupo, durante o vídeo
    # inteiro. A geração de frames do MoviePy é sequencial (confirmado no
    # código-fonte: `ffmpeg_write_video` usa um `for t, frame in
    # clip.iter_frames(...)` de thread única), então um cache de 1 slot por
    # grupo é seguro e elimina o recálculo duplicado sem mudar o resultado
    # visual (a função `blit()` do MoviePy só LÊ os arrays, nunca os muta).
    _ultimo_quadro = {"t": None, "frame": None, "alpha": None}

    def _obter_quadro_do_grupo(t: float) -> Tuple[np.ndarray, np.ndarray]:
        if _ultimo_quadro["t"] != t:
            t_abs = inicio_grupo + t
            frame = np.zeros((altura_total, largura_maxima, 3), dtype=np.uint8)
            alpha = np.zeros((altura_total, largura_maxima), dtype=np.float32)
            _desenhar_grupo(frame, alpha, layout, t_abs)

            # Animação de entrada/saída do grupo inteiro (Fase 3, item 7):
            # fade + scale-in centralizado no próprio canvas do grupo, sem
            # mexer no layout/posição das palavras (que continua fixo,
            # calculado uma única vez em `_calcular_layout_grupo`). Fora
            # das janelas de entrada/saída, `escala == opacidade == 1.0` e
            # o custo extra é só a checagem — nenhum resize/blend a mais.
            escala, opacidade = _fator_entrada_saida_grupo(t, duracao_grupo, eh_hook)
            if escala < 0.999 or opacidade < 0.999:
                frame_escalado, alpha_escalado = _escalar_tile(frame, alpha, escala)
                frame = np.zeros_like(frame)
                alpha = np.zeros_like(alpha)
                h_s, w_s = alpha_escalado.shape
                x_off = (largura_maxima - w_s) // 2
                y_off = (altura_total - h_s) // 2
                _colar_tile(frame, alpha, frame_escalado, alpha_escalado, x_off, y_off)
                if opacidade < 1.0:
                    alpha = alpha * opacidade

            _ultimo_quadro["t"] = t
            _ultimo_quadro["frame"] = frame
            _ultimo_quadro["alpha"] = alpha
        return _ultimo_quadro["frame"], _ultimo_quadro["alpha"]

    def make_frame(t: float) -> np.ndarray:
        frame, _ = _obter_quadro_do_grupo(t)
        return frame

    def make_mask(t: float) -> np.ndarray:
        _, alpha = _obter_quadro_do_grupo(t)
        return alpha

    clip_video = VideoClip(make_frame=make_frame, duration=duracao_grupo)
    clip_mascara = VideoClip(make_frame=make_mask, duration=duracao_grupo, ismask=True)
    clip_video = clip_video.set_mask(clip_mascara)

    # Posição fixa (safe area, Fase 3, item 4): a base do bloco de legenda
    # sempre no mesmo lugar, longe da borda inferior — nunca cobre
    # elementos do Shorts nem "pula" de posição entre grupos.
    # `largura_maxima` já usa LEGEND_WIDTH_FRACTION < 1 - 2*SAFE_AREA_SIDE,
    # então fica centralizado dentro da margem lateral segura por
    # construção. O clamp em y_minimo é a defesa para o caso raro de um
    # grupo grande demais (3 linhas) empurrar o bloco para cima do que
    # SAFE_AREA_TOP permite.
    x_video = (largura_video - largura_maxima) // 2
    y_video = altura_video - int(altura_video * SAFE_AREA_BOTTOM) - altura_total
    y_minimo = int(altura_video * SAFE_AREA_TOP)
    y_video = max(y_minimo, y_video)
    clip_video = clip_video.set_start(inicio_grupo).set_position((x_video, y_video))

    return clip_video


def _criar_clips_de_legenda(grupos: List[List[PalavraLegenda]], largura_video: int, altura_video: int) -> List[VideoClip]:
    """Constrói o clip dinâmico de cada grupo e devolve a lista pronta para
    entrar na composição final. Só o PRIMEIRO grupo (`eh_hook=True`, Fase
    3, item 8) recebe a entrada de legenda mais pronunciada do hook."""
    return [
        _construir_clip_grupo(grupo, largura_video, altura_video, eh_hook=(indice == 0))
        for indice, grupo in enumerate(grupos)
    ]


# ---------------------------------------------------------------------------
# 3.3 MIXAGEM DE ÁUDIO (narração + música de fundo)
# ---------------------------------------------------------------------------
def _carregar_musica_de_fundo(caminho: str, duracao_alvo: float) -> Optional[AudioFileClip]:
    """Carrega a música de fundo e a ajusta à duração da narração.

    Retorna `None` (com um aviso) caso o arquivo não exista ou não possa ser
    carregado, para que o vídeo continue sendo gerado normalmente sem música.
    """
    if not os.path.exists(caminho):
        print(f"⚠️ Música de fundo '{caminho}' não encontrada. Gerando vídeo sem música.")
        return None

    try:
        musica = AudioFileClip(caminho)
    except Exception as err:
        print(f"⚠️ Não foi possível carregar a música '{caminho}' ({err}). Gerando vídeo sem música.")
        return None

    # Normaliza a música (baseado no trecho completo do arquivo) ANTES de
    # cortar/repetir, para que o ganho reflita a dinâmica real da faixa.
    # Isso faz com que músicas diferentes, com masterizações diferentes,
    # produzam um resultado final consistente ao aplicar o mesmo MUSIC_VOLUME.
    try:
        musica = afx.audio_normalize(musica)
    except Exception as err:
        # Ex.: arquivo silencioso (max_volume=0) causaria divisão por zero.
        print(f"⚠️ Não foi possível normalizar a música ({err}); usando volume original.")

    # Ajusta a duração: repete em loop se for mais curta, corta se for maior.
    if musica.duration < duracao_alvo:
        if MUSIC_LOOP:
            musica = afx.audio_loop(musica, duration=duracao_alvo)
        # Se MUSIC_LOOP for False, a música simplesmente termina antes do
        # fim do vídeo e o restante fica em silêncio — comportamento aceito
        # deliberadamente, nada a processar aqui.
    else:
        musica = musica.subclip(0, duracao_alvo)

    musica = afx.audio_fadein(musica, MUSIC_FADE_IN)
    musica = afx.audio_fadeout(musica, MUSIC_FADE_OUT)
    musica = afx.volumex(musica, MUSIC_VOLUME)

    print("🎵 Música carregada")
    return musica


def _processar_narracao(clip_narracao: AudioFileClip) -> AudioFileClip:
    """Normaliza o volume da narração e, opcionalmente, aplica um limitador leve.

    A normalização (`audio_normalize`, nativo do MoviePy) é sempre aplicada e
    evita diferenças perceptíveis de volume entre vídeos quando a narração
    vem um pouco mais baixa — isso sozinho já resolve o principal problema
    de consistência entre execuções.

    A saturação suave em "S" (tanh) é um recurso OPCIONAL, controlado por
    `ENABLE_SOFT_LIMITER` (desligado por padrão). Ela funciona mais como um
    saturador/limitador leve do que um compressor de verdade (não tem
    threshold/ratio/attack/release), e pode introduzir uma distorção
    harmônica sutil em picos que já estão no teto após a normalização — por
    isso fica desligada por padrão. Quando `ENABLE_SOFT_LIMITER=False`,
    nenhum processamento adicional além da normalização e do ganho ocorre.
    """
    narracao = afx.audio_normalize(clip_narracao)

    if ENABLE_SOFT_LIMITER:
        drive = VOICE_COMPRESSION_DRIVE
        limite = math.tanh(drive)  # normaliza a saída de volta para a escala original

        def saturar(get_frame, t):
            amostras = get_frame(t)
            return np.tanh(amostras * drive) / limite

        narracao = narracao.fl(saturar)

    narracao = afx.volumex(narracao, VOICE_VOLUME)
    return narracao


def add_sound_effect(clip_audio: AudioFileClip, caminho_efeito: Optional[str] = None) -> AudioFileClip:
    """Reservado para uso futuro: adicionar efeitos sonoros pontuais.

    Por enquanto não faz nada além de devolver o áudio original — nenhum
    efeito sonoro é utilizado nesta etapa do projeto. Mantida aqui apenas
    como ponto de extensão para quando efeitos sonoros forem implementados.
    """
    return clip_audio


def montar_audio_final(
    clip_narracao: AudioFileClip,
    clip_musica: Optional[AudioFileClip] = None,
) -> CompositeAudioClip:
    """Monta o áudio final do vídeo, misturando narração e música de fundo.

    A narração é sempre normalizada (e, opcionalmente, levemente saturada) e
    mantida como a camada mais alta da mixagem. A música (quando existir)
    entra por baixo, em volume discreto, garantindo que a narração nunca
    perca a clareza.
    """
    print("🎙 Narração carregada")
    narracao = _processar_narracao(clip_narracao)
    narracao = add_sound_effect(narracao)  # sem efeito por enquanto

    volume_musica = MUSIC_VOLUME
    if clip_musica is not None and volume_musica >= VOICE_VOLUME:
        # Segurança extra: a narração deve sempre ficar mais alta que a
        # música, mesmo que alguém ajuste as constantes de forma incorreta.
        print("⚠️ MUSIC_VOLUME estava igual ou maior que VOICE_VOLUME; reduzindo automaticamente.")
        volume_musica = VOICE_VOLUME * MUSIC_VOLUME_FALLBACK_FRACTION
        clip_musica = afx.volumex(clip_musica, volume_musica / MUSIC_VOLUME)

    camadas = [narracao] if clip_musica is None else [clip_musica, narracao]
    audio_final = CompositeAudioClip(camadas).set_duration(narracao.duration)

    # CompositeAudioClip não define `.fps` sozinho (é uma lacuna conhecida
    # do MoviePy). Sem isso, qualquer código futuro que dependa de sample
    # rate (ex.: Whisper, corte por amostra) quebraria com AttributeError.
    audio_final.fps = getattr(clip_narracao, "fps", None) or 44100

    print("🎚 Mixagem concluída")
    return audio_final


def _verificar_fundo_saudavel(clip: VideoFileClip, caminho: str, fracao_preta_maxima: float = 0.25) -> None:
    """Verifica se o vídeo de fundo já chega com uma região de preto sólido
    grande demais.

    Isso NÃO corrige nada — é uma checagem do ARQUIVO DE ENTRADA. Vídeos
    processados por ferramentas externas de conversão de proporção (que
    costumam adicionar barras pretas em vez de cortar) chegam com preto já
    embutido nos próprios pixels; nosso Ken Burns/crop então recorta e
    reproduz fielmente essa região preta, porque ela faz parte do conteúdo,
    não é um bug de composição. Sem esse aviso, o único jeito de descobrir
    isso é abrindo o arquivo final publicado e vendo a barra preta.

    Custo: lê e varre 3 linhas de UM frame — desprezível perto do render
    inteiro.
    """
    if not clip.duration:
        return

    frame = clip.get_frame(min(clip.duration / 2, max(0.0, clip.duration - 0.01)))
    altura, largura = frame.shape[:2]

    piores_fracoes = []
    for fracao_y in (0.25, 0.5, 0.75):
        linha = frame[int(altura * fracao_y)]
        pretos = int((linha.mean(axis=1) <= 10).sum())
        piores_fracoes.append(pretos / largura)

    fracao_preta = max(piores_fracoes)
    if fracao_preta > fracao_preta_maxima:
        print(
            f"⚠️ O vídeo de fundo '{caminho}' já contém uma região preta "
            f"sólida grande (~{fracao_preta * 100:.0f}% da largura, medida em 3 "
            f"linhas do frame). Isso normalmente indica que o arquivo já foi "
            f"processado por outra ferramenta ANTES de chegar aqui (ex: um "
            f"conversor de proporção que adiciona barras pretas em vez de "
            f"cortar). Nosso pipeline vai recortar/dar zoom fielmente nesse "
            f"conteúdo, incluindo a parte preta — não é um bug de composição "
            f"nosso. Recomendado: abra '{caminho}' direto num player "
            f"para confirmar antes de publicar."
        )


def _reagrupar_ate_teto(
    segmentos: List[Tuple[float, float]], teto: int
) -> List[Tuple[float, float]]:
    """Funde segmentos adjacentes (sempre preservando os limites originais
    — nunca corta no meio de um grupo de legenda) até que a quantidade
    total não ultrapasse `teto`. A cada rodada, funde o par adjacente de
    MENOR duração somada, para manter o ritmo de corte o mais equilibrado
    possível em vez de sempre fundir do mesmo lado."""
    segmentos = list(segmentos)
    while len(segmentos) > teto and len(segmentos) > 1:
        menor_indice, menor_soma = 0, float("inf")
        for indice in range(len(segmentos) - 1):
            soma = (segmentos[indice][1] - segmentos[indice][0]) + (
                segmentos[indice + 1][1] - segmentos[indice + 1][0]
            )
            if soma < menor_soma:
                menor_indice, menor_soma = indice, soma
        inicio = segmentos[menor_indice][0]
        fim = segmentos[menor_indice + 1][1]
        segmentos[menor_indice : menor_indice + 2] = [(inicio, fim)]
    return segmentos


def _definir_segmentos_broll(
    pontos_de_corte: List[float], duracao_total: float
) -> List[Tuple[float, float]]:
    """Define os intervalos `[inicio, fim)` de cada trecho de B-roll
    (Fase 4), usando como limites candidatos os MESMOS pontos onde a
    legenda já troca de grupo (`pontos_de_corte`, vindos de
    `agrupar_palavras_em_blocos()`) — assim a troca de vídeo de fundo
    nunca acontece no meio de uma legenda em exibição, sempre exatamente
    onde ela também troca. Os cortes são baseados nos segmentos da
    narração, não em palavras isoladas.

    Agrupa vários limites consecutivos em cada trecho até se aproximar de
    `BROLL_SEGMENTO_DURACAO_ALVO` segundos, nunca deixando um trecho FINAL
    menor que `BROLL_SEGMENTO_DURACAO_MINIMA` sozinho (funde com o
    anterior nesse caso). Se, mesmo assim, sobrarem mais trechos que
    `BROLL_MAX_SEGMENTOS`, funde os pares adjacentes mais curtos até caber
    no teto — sempre preservando os limites de legenda originais.
    """
    limites = sorted(set([0.0, duracao_total] + [t for t in pontos_de_corte if 0 < t < duracao_total]))
    if len(limites) <= 1:
        return [(0.0, duracao_total)]

    segmentos: List[Tuple[float, float]] = []
    inicio_atual = limites[0]
    for limite in limites[1:]:
        duracao_acumulada = limite - inicio_atual
        eh_ultimo_limite = limite == limites[-1]
        if duracao_acumulada >= BROLL_SEGMENTO_DURACAO_ALVO or eh_ultimo_limite:
            segmentos.append((inicio_atual, limite))
            inicio_atual = limite

    # Funde um último trecho remanescente curto demais com o anterior, em
    # vez de deixar um B-roll quase instantâneo no fim do vídeo.
    if len(segmentos) >= 2:
        inicio_ultimo, fim_ultimo = segmentos[-1]
        if fim_ultimo - inicio_ultimo < BROLL_SEGMENTO_DURACAO_MINIMA:
            inicio_penultimo, _fim_penultimo = segmentos[-2]
            segmentos[-2] = (inicio_penultimo, fim_ultimo)
            segmentos.pop()

    if len(segmentos) > BROLL_MAX_SEGMENTOS:
        segmentos = _reagrupar_ate_teto(segmentos, BROLL_MAX_SEGMENTOS)

    return segmentos


def _texto_do_segmento(segmento: Tuple[float, float], palavras: List["PalavraLegenda"]) -> str:
    """Reconstrói o texto REALMENTE narrado dentro de um trecho de B-roll
    `(inicio, fim)`, a partir das mesmas `PalavraLegenda` (com timestamp
    por palavra) já produzidas por `transcrever_narracao()` — a estrutura
    de segmentação que o projeto já possui (Fase 2, item 2: não duplica o
    roteiro inteiro nem cria um segundo mecanismo de divisão).

    Usa o INÍCIO da palavra (`palavra.start`) para decidir a que trecho
    ela pertence — mesmo critério já usado por `_definir_segmentos_broll()`
    para os pontos de corte, então uma palavra nunca "pertence" a um
    trecho diferente do que aparece visualmente na tela naquele
    instante."""
    inicio, fim = segmento
    palavras_do_trecho = [p.palavra for p in palavras if inicio <= p.start < fim]
    return " ".join(palavras_do_trecho).strip()


def _preparar_clip_de_fundo(
    caminho: str, duracao_alvo: float, clips_para_fechar: List
) -> VideoFileClip:
    """Abre um único vídeo de fundo e o prepara para entrar num trecho de
    B-roll: valida existência/abertura, dá loop se for mais curto que
    `duracao_alvo`, corta para exatamente essa duração (variando o ponto
    inicial quando há folga — Fase 2, item 12: evita sempre usar os
    primeiros segundos do vídeo, tipicamente intro/logo/frame estático) e
    redimensiona para 9:16. Lança `FundoInvalidoError` (nunca deixa uma
    exceção "crua" do MoviePy escapar) para qualquer arquivo
    ausente/corrompido/com duração inválida — o chamador
    (`_montar_fundo_multiplo`) decide o que fazer: descartar e tentar
    outro candidato para o mesmo trecho."""
    if not os.path.exists(caminho):
        raise FundoInvalidoError(f"❌ Vídeo de fundo '{caminho}' não encontrado.")
    try:
        clip = VideoFileClip(caminho)
    except Exception as err:
        raise FundoInvalidoError(f"❌ Vídeo de fundo '{caminho}' corrompido ou ilegível: {err}") from err
    clips_para_fechar.append(clip)

    if not clip.duration or clip.duration <= 0:
        raise FundoInvalidoError(f"❌ Vídeo de fundo '{caminho}' com duração inválida.")

    _verificar_fundo_saudavel(clip, caminho)

    if clip.duration < duracao_alvo:
        num_loops = math.ceil(duracao_alvo / clip.duration)
        clip_loop = concatenate_videoclips([clip] * num_loops)
        clips_para_fechar.append(clip_loop)
        clip = clip_loop

    ponto_inicial = _escolher_ponto_de_corte_temporal(clip.duration, duracao_alvo)
    clip = clip.subclip(ponto_inicial, ponto_inicial + duracao_alvo)
    clip = _redimensionar_para_vertical(clip)
    return clip


def _escolher_ponto_de_corte_temporal(duracao_origem: float, duracao_alvo: float) -> float:
    """Decide onde COMEÇAR o corte de `duracao_alvo` segundos dentro de um
    vídeo de origem com `duracao_origem` segundos (Fase 2, item 12).

    Se a folga (origem - alvo) for pequena (< `BROLL_FOLGA_MINIMA_PARA_VARIAR_INICIO`),
    mantém o comportamento de sempre (começa em 0s) — não vale a pena
    arriscar um recorte estranho por uma folga de frações de segundo.

    Se houver folga confortável, sorteia um ponto dentro dela, mas SEMPRE
    reservando uma margem nas duas pontas (mínimo entre metade da folga e
    8% da duração total) para reduzir a chance de cair bem em cima de uma
    transição de corte, logo/marca d'água de abertura ou frame estático
    de encerramento — comuns nos primeiros/últimos segundos de clipes de
    banco de imagens."""
    folga = duracao_origem - duracao_alvo
    if folga <= BROLL_FOLGA_MINIMA_PARA_VARIAR_INICIO:
        return 0.0

    margem = min(folga * 0.5, duracao_origem * 0.08)
    inicio_maximo = max(0.0, folga - margem)
    inicio_minimo = min(margem, inicio_maximo)
    if inicio_maximo <= inicio_minimo:
        return 0.0
    return random.uniform(inicio_minimo, inicio_maximo)


def _montar_fundo_multiplo(
    segmentos: List[Tuple[float, float]],
    caminhos_disponiveis: List[str],
    grupos: List[List["PalavraLegenda"]],
    palavras: List["PalavraLegenda"],
    roteiro_completo: str,
    clips_para_fechar: List,
    media_manager: Optional["MediaManager"],
) -> Tuple[VideoClip, List[str]]:
    """Monta o fundo dinâmico do vídeo concatenando um B-roll DIFERENTE por
    trecho (Fase 4), reaproveitando o Ken Burns já existente
    (`_aplicar_ken_burns`) DENTRO de cada trecho — cada um ganha seu
    próprio zoom/pan contínuo, então a troca de vídeo já é, por si só, uma
    "cena nova" visualmente, sem precisar de nenhuma transição adicional.

    Fase 2 (B-roll contextual): para cada trecho, primeiro tenta um
    candidato relevante para o texto REALMENTE narrado naquele trecho
    (`media_manager.escolher_video_contextual`, via `_texto_do_segmento`).
    Só se isso falhar (sem `media_manager`, sem candidato contextual, ou o
    candidato contextual não abrir de verdade) cai para o rodízio pelos
    caminhos em `caminhos_disponiveis` — o mesmo pool de reserva pré-
    buscado pela Fase 4/`escolher_fundo_e_criar_video()`, que continua
    existindo exatamente como antes para garantir que o pipeline nunca
    fique sem fundo só porque a busca contextual não achou nada (item 21
    da Fase 2: resiliência é requisito absoluto). Evita repetir o MESMO
    vídeo do trecho imediatamente anterior sempre que houver mais de uma
    opção disponível.

    Se um vídeo falhar ao abrir para um trecho específico, descarta e
    tenta o próximo candidato (contextual restante, depois pool de
    reserva — até `BROLL_MAX_TENTATIVAS_POR_SEGMENTO` alternativas) antes
    de reaproveitar o ÚLTIMO trecho que funcionou — nunca derruba o vídeo
    inteiro por causa de um único arquivo ruim que passou pela validação
    leve mas falhou de verdade aqui (Fase 5/8).

    Retorna o clip de fundo final (mesma duração da soma dos trechos) e a
    lista de caminhos REALMENTE usados — usada por
    `escolher_fundo_e_criar_video()` para só confirmar no histórico
    (`MediaManager.confirmar_uso()`) os vídeos que de fato entraram no
    Short publicado."""
    # (Fase 2.1.1 → correção B/C) `caminhos_disponiveis` deixou de ser uma
    # pré-condição obrigatória aqui: desde que `escolher_fundo_e_criar_video()`
    # não busca mais o pool de reserva antecipadamente (para não gastar o
    # orçamento global de downloads do Pexels ANTES do B-roll contextual por
    # trecho rodar), este parâmetro pode chegar vazio de propósito. Só é
    # erro fatal de fato se, quando o pool for REALMENTE necessário (algum
    # trecho sem candidato contextual), não houver como buscá-lo (sem
    # `media_manager`) nem nada já disponível — ver a busca lazy dentro do
    # loop abaixo. Quando `media_manager` é None (uso direto em teste/
    # chamada isolada, sem contextual nem forma de buscar sob demanda),
    # mantém a checagem antecipada de antes: nesse caso o chamador PRECISA
    # ter passado o pool pronto.
    if not caminhos_disponiveis and media_manager is None:
        raise FundoInvalidoError("❌ Nenhum vídeo de fundo disponível para montar os B-rolls.")

    pool_de_reserva: List[str] = list(caminhos_disponiveis)
    pool_ja_buscado = bool(caminhos_disponiveis)

    clips_segmento: List[VideoClip] = []
    caminhos_usados: List[str] = []
    ultimo_caminho_ok: Optional[str] = None
    indice_pool = 0

    for indice_segmento, (inicio, fim) in enumerate(segmentos):
        duracao_segmento = fim - inicio
        if duracao_segmento <= 0:
            continue

        # Pontos de corte LOCAIS (relativos ao início do trecho) para o
        # Ken Burns variar o zoom também dentro de um mesmo B-roll, quando
        # o trecho abrange mais de um grupo de legenda.
        pontos_locais = [
            grupo[0].start - inicio for grupo in grupos if inicio < grupo[0].start < fim
        ]

        texto_trecho = _texto_do_segmento((inicio, fim), palavras) or roteiro_completo
        print(
            f"🎞️ B-roll {indice_segmento + 1}/{len(segmentos)}\n"
            f"📝 Texto: \"{texto_trecho[:100]}{'…' if len(texto_trecho) > 100 else ''}\""
        )

        fila_candidatos: List[str] = []
        if media_manager is not None:
            candidato_contextual = media_manager.escolher_video_contextual(
                texto=texto_trecho,
                duracao_alvo=duracao_segmento,
                excluir=set(caminhos_usados),
            )
            if candidato_contextual:
                fila_candidatos.append(candidato_contextual)
            else:
                print("⚠️ Nenhum candidato contextual relevante. ↩️ Usando pool de reserva.")

        candidato_ok: Optional[str] = None
        ultima_falha: Optional[Exception] = None
        tentativas_restantes = max(1, BROLL_MAX_TENTATIVAS_POR_SEGMENTO)

        while tentativas_restantes > 0 and candidato_ok is None:
            if fila_candidatos:
                caminho = fila_candidatos.pop(0)
            else:
                if not pool_de_reserva:
                    # Só busca o pool de reserva agora, sob demanda — e só
                    # uma vez por chamada de `_montar_fundo_multiplo()`
                    # (`pool_ja_buscado`), mesmo que vários trechos caiam
                    # aqui: se a primeira busca já veio vazia, tentar de
                    # novo não mudaria nada (mesmo `media_manager`, mesmo
                    # roteiro) e só gastaria chamadas à toa.
                    if not pool_ja_buscado and media_manager is not None:
                        pool_ja_buscado = True
                        print(
                            "↩️ Pool de reserva vazio — buscando fallback agora "
                            "(media_manager.escolher_videos(), sob demanda)."
                        )
                        pool_de_reserva = media_manager.escolher_videos(
                            contexto=roteiro_completo, quantidade=BROLL_MAX_SEGMENTOS
                        )
                    if not pool_de_reserva:
                        raise FundoInvalidoError(
                            "❌ Nenhum vídeo de fundo disponível para montar os B-rolls "
                            f"(trecho {indice_segmento + 1}/{len(segmentos)})."
                        )
                caminho = pool_de_reserva[indice_pool % len(pool_de_reserva)]
                indice_pool += 1
                if caminho == ultimo_caminho_ok and len(pool_de_reserva) > 1:
                    caminho = pool_de_reserva[indice_pool % len(pool_de_reserva)]
                    indice_pool += 1

            try:
                clip_bruto = _preparar_clip_de_fundo(caminho, duracao_segmento, clips_para_fechar)
            except FundoInvalidoError as err:
                ultima_falha = err
                tentativas_restantes -= 1
                if media_manager is not None:
                    media_manager.descartar(caminho)
                print(
                    f"⚠️ B-roll do trecho {indice_segmento + 1}/{len(segmentos)} "
                    f"('{caminho}') falhou ({err}); tentando outro vídeo."
                )
                continue

            candidato_ok = caminho
            clip_segmento = _aplicar_ken_burns(
                clip_bruto, duracao_segmento, pontos_locais, eh_hook=(indice_segmento == 0)
            )
            clips_para_fechar.append(clip_segmento)
            clips_segmento.append(clip_segmento)
            caminhos_usados.append(caminho)
            ultimo_caminho_ok = caminho

        if candidato_ok is None:
            if ultimo_caminho_ok is None:
                raise FundoInvalidoError(
                    f"❌ Nenhum vídeo de fundo válido para o trecho "
                    f"{indice_segmento + 1}/{len(segmentos)}. Último erro: {ultima_falha}"
                )
            print(
                f"⚠️ Reaproveitando o último B-roll válido ('{ultimo_caminho_ok}') "
                f"para o trecho {indice_segmento + 1}/{len(segmentos)}."
            )
            clip_bruto = _preparar_clip_de_fundo(ultimo_caminho_ok, duracao_segmento, clips_para_fechar)
            clip_segmento = _aplicar_ken_burns(
                clip_bruto, duracao_segmento, pontos_locais, eh_hook=(indice_segmento == 0)
            )
            clips_para_fechar.append(clip_segmento)
            clips_segmento.append(clip_segmento)
            caminhos_usados.append(ultimo_caminho_ok)

    if not clips_segmento:
        raise FundoInvalidoError("❌ Nenhum trecho de B-roll pôde ser montado.")

    if len(clips_segmento) == 1:
        fundo_dinamico = clips_segmento[0]
    else:
        fundo_dinamico = concatenate_videoclips(clips_segmento)
        clips_para_fechar.append(fundo_dinamico)

    print(f"🎞️ Fundo montado com {len(clips_segmento)} trecho(s) de B-roll.")
    return fundo_dinamico, caminhos_usados


def criar_video(
    roteiro_completo: str,
    caminho_audio: str,
    caminhos_video_fundo: List[str],
    caminho_saida: str = VIDEO_SAIDA_PATH,
    media_manager: Optional["MediaManager"] = None,
) -> Tuple[str, List[str]]:
    """Monta o vídeo final: fundo dinâmico com MÚLTIPLOS B-rolls (Fase 4 —
    um vídeo diferente por trecho da narração, cada um com seu próprio Ken
    Burns; Fase 2 — cada trecho busca B-roll relevante ao texto REALMENTE
    narrado ali, via `media_manager`), legendas, barra de progresso,
    vinheta e fades, cortado em 9:16.

    `caminhos_video_fundo` é o POOL DE RESERVA (Fase 4/Fase 2): candidatos
    já escolhidos/validados por `escolher_fundo_e_criar_video()` a partir
    do roteiro inteiro, usados como fallback quando a busca contextual por
    trecho (Fase 2, via `media_manager`) não encontra nada relevante para
    aquele trecho especificamente — nunca a fonte primária quando
    `media_manager` está disponível. A quantidade de B-rolls REALMENTE
    usados depende da duração da narração (ver `_definir_segmentos_broll()`)
    e pode ser menor que `len(caminhos_video_fundo)` (sobra não é
    desperdiçada, só não é necessária) ou reaproveitar algum candidato
    mais de uma vez (se a narração tiver mais trechos que vídeos
    disponíveis).

    Retorna `(caminho_do_arquivo_final, caminhos_de_fundo_realmente_usados)`
    — o segundo item é repassado a `MediaManager.confirmar_uso()` para só
    marcar no histórico os vídeos que de fato entraram no Short publicado.
    """
    print("🎬 [3/4] Montando vídeo final com legendas dinâmicas e design pro...")

    # (Fase 2.1.1 → correção B/C, 2ª parte) Este guard é INDEPENDENTE do
    # guard equivalente dentro de `_montar_fundo_multiplo()` — os dois
    # existiam antes da mudança C, quando `caminhos_video_fundo` nunca
    # chegava vazio de propósito (era sempre pré-buscado eager por
    # `escolher_fundo_e_criar_video()`). Agora que o pool é lazy, uma lista
    # vazia É o caso normal sempre que há `media_manager` (o pool só é
    # buscado sob demanda, de dentro de `_montar_fundo_multiplo()`, se
    # algum trecho realmente precisar dele) — só é erro fatal aqui, ainda
    # ANTES de qualquer transcrição/segmentação, quando não há
    # `media_manager` nenhum para eventualmente suprir isso mais adiante.
    if not caminhos_video_fundo and media_manager is None:
        raise FundoInvalidoError("❌ Nenhum vídeo de fundo disponível para montar o Short.")

    # Referências guardadas à parte para garantir o fechamento de todos os
    # clips abertos, mesmo em caso de erro no meio do processo.
    clips_para_fechar: List = []

    try:
        clip_audio = AudioFileClip(caminho_audio)
        clips_para_fechar.append(clip_audio)

        duracao_final = clip_audio.duration

        if not duracao_final or duracao_final <= 0:
            raise ValueError(
                f"Narração com duração inválida ({duracao_final}s) — "
                f"verifique se '{caminho_audio}' foi gerado corretamente."
            )

        clip_musica = _carregar_musica_de_fundo(MUSICA_PATH, duracao_final)
        if clip_musica is not None:
            clips_para_fechar.append(clip_musica)

        audio_final = montar_audio_final(clip_audio, clip_musica)
        clips_para_fechar.append(audio_final)

        # Transcreve a narração (sempre a voz limpa, antes da mixagem com
        # música) e agrupa as palavras em blocos de legenda sincronizados
        # ANTES de montar o fundo: os mesmos limites de grupo
        # (`pontos_de_corte`) definem tanto onde a legenda troca quanto
        # onde o B-roll troca (Fase 4) — nunca uma troca de vídeo no meio
        # de uma legenda em exibição.
        _resetar_cache_legendas()
        palavras = transcrever_narracao(caminho_audio)
        palavras, _quantidade_ajustada = _limitar_palavras_a_duracao(palavras, duracao_final)
        if _quantidade_ajustada:
            print(
                f"⚠️ {_quantidade_ajustada} palavra(s) com timestamp além da duração real "
                f"do áudio ({duracao_final:.2f}s) foram ajustadas/descartadas."
            )
        grupos = agrupar_palavras_em_blocos(palavras)
        pontos_de_corte = [grupo[0].start for grupo in grupos]

        segmentos = _definir_segmentos_broll(pontos_de_corte, duracao_final)
        print(
            f"🎞️ {len(segmentos)} trecho(s) de B-roll planejado(s) para "
            f"{duracao_final:.1f}s (de até {len(caminhos_video_fundo)} vídeo(s) candidato(s))."
        )

        # Fundo dinâmico: um B-roll por trecho (Fase 4), cada um com zoom
        # lento contínuo + variação de zoom a cada grupo de legenda + leve
        # pan, sempre dentro dos limites da imagem (sem bordas pretas).
        # Fase 2: cada trecho busca B-roll relevante ao texto REALMENTE
        # narrado ali (`palavras`), com o pool `caminhos_video_fundo`
        # (pré-buscado a partir do roteiro inteiro) como reserva.
        fundo_dinamico, caminhos_usados = _montar_fundo_multiplo(
            segmentos, caminhos_video_fundo, grupos, palavras, roteiro_completo, clips_para_fechar, media_manager
        )
        _debug_registrar("05_ken_burns", fundo_dinamico)

        fundo_com_vinheta = _aplicar_vinheta(fundo_dinamico)
        largura_final, altura_final = fundo_com_vinheta.size
        _debug_registrar("06_vinheta", fundo_com_vinheta)

        # Verificação estrutural (custo de alguns frames — 1 por trecho de
        # B-roll —, desprezível perto do render inteiro): confirma que o
        # fundo realmente preenche 100% do canvas em CADA trecho, não só no
        # meio do vídeo. Isso não "esconde" um eventual bug de tamanho —
        # pelo contrário, faz a renderização falhar alto e claro, com a
        # causa e o trecho exatos, em vez de estourar minutos depois dentro
        # de `write_videofile()` com um `operands could not be broadcast
        # together` genérico (a causa raiz encontrada na auditoria: um
        # único ponto de amostragem no meio do vídeo só cobria o PRIMEIRO
        # trecho de B-roll, deixando inconsistências em trechos
        # posteriores passarem despercebidas até o render de verdade).
        for _indice_seg, (_inicio_seg, _fim_seg) in enumerate(segmentos):
            _t_amostra = min((_inicio_seg + _fim_seg) / 2, duracao_final - 0.01)
            _t_amostra = max(0.0, _t_amostra)
            _frame_teste = fundo_com_vinheta.get_frame(_t_amostra)
            _altura_real, _largura_real = _frame_teste.shape[:2]
            if (_largura_real, _altura_real) != (largura_final, altura_final):
                raise RuntimeError(
                    f"❌ Inconsistência de tamanho no fundo do vídeo: o trecho de "
                    f"B-roll {_indice_seg + 1}/{len(segmentos)} (t={_t_amostra:.2f}s) "
                    f"tem {_largura_real}x{_altura_real}px, mas o canvas esperado é "
                    f"{largura_final}x{altura_final}px. Isso indicaria uma regressão "
                    f"em `_redimensionar_para_vertical()`/`_aplicar_ken_burns()` — "
                    f"pare e investigue antes de publicar."
                )

        text_clips = _criar_clips_de_legenda(grupos, largura_final, altura_final)
        clips_para_fechar.extend(text_clips)
        if DEBUG_RENDER and text_clips:
            # Legendas variam ao longo do vídeo (cada grupo é um clip
            # diferente, com seu próprio intervalo de tempo); registra o
            # primeiro grupo como amostra representativa da etapa.
            _debug_registrar("07_legendas", text_clips[0])

        barra_progresso = _criar_barra_progresso(largura_final, duracao_final)
        clips_para_fechar.append(barra_progresso)

        # Segunda verificação estrutural (defesa em profundidade): confirma
        # que NENHUM subclipe (legenda, barra) se estende além da duração
        # real do áudio. Essa é exatamente a classe de bug encontrada e
        # corrigida nesta rodada — timestamps de transcrição que ultrapassam
        # a duração real faziam o CompositeVideoClip final "esticar" além do
        # fundo, sobrando tela preta no final. `_limitar_palavras_a_duracao`
        # já corrige isso na fonte; esta verificação existe para pegar
        # qualquer regressão futura (nesta função ou em qualquer nova
        # funcionalidade) antes que ela vire um vídeo publicado quebrado.
        _tolerancia_duracao = 0.05
        for _clip_verificado in [fundo_com_vinheta, barra_progresso] + text_clips:
            _fim_clip = (getattr(_clip_verificado, "start", 0.0) or 0.0) + _clip_verificado.duration
            if _fim_clip > duracao_final + _tolerancia_duracao:
                raise RuntimeError(
                    f"❌ Inconsistência de duração: um subclipe (tipo "
                    f"{type(_clip_verificado).__name__}) termina em {_fim_clip:.2f}s, além "
                    f"da duração real do áudio ({duracao_final:.2f}s). Isso faria o vídeo "
                    f"final mostrar tela preta no trecho excedente — pare e investigue "
                    f"antes de publicar."
                )
        _debug_registrar("08_barra_progresso", barra_progresso)

        total_tiles = len(_CACHE_PALAVRAS_RENDERIZADAS)
        total_consultas = _estatisticas_cache["acertos"] + _estatisticas_cache["faltas"]
        taxa_cache = (_estatisticas_cache["acertos"] / total_consultas * 100) if total_consultas else 0.0
        print(f"✨ {len(palavras)} animações")
        print(f"⚡ Cache reutilizado: {taxa_cache:.0f}% ({total_tiles} tiles únicos renderizados)")
        print(
            "🎨 Visual\n"
            f"- resolução: {largura_final}x{altura_final}\n"
            f"- legendas: OK ({len(grupos)} grupo(s))\n"
            f"- safe area: OK (topo {SAFE_AREA_TOP:.0%} / base {SAFE_AREA_BOTTOM:.0%} / lateral {SAFE_AREA_SIDE:.0%})\n"
            f"- B-roll: {len(segmentos)} cena(s)\n"
            f"- animações: {len(palavras)}"
        )

        video_final = CompositeVideoClip(
            [fundo_com_vinheta] + text_clips + [barra_progresso],
            size=(largura_final, altura_final),
        )
        clips_para_fechar.append(video_final)
        _debug_registrar_composicao("09_composite_final", video_final)

        # Abertura e fechamento suaves — aplicados tanto no vídeo quanto no
        # áudio, para que a mixagem não corte abruptamente enquanto a tela
        # já está escurecendo (bug encontrado em auditoria: antes, apenas o
        # vídeo tinha fade, e o áudio continuava no volume máximo até o
        # último frame).
        video_final = _aplicar_fade_video_suave(video_final, FADE_TIME)
        audio_final = afx.audio_fadein(audio_final, FADE_TIME)
        audio_final = afx.audio_fadeout(audio_final, FADE_TIME)
        video_final = video_final.set_audio(audio_final)
        _debug_registrar("10_antes_do_write_videofile", video_final)

        if DEBUG_RENDER:
            # Desenha uma borda vermelha diretamente nos pixels do frame,
            # exatamente nas bordas do canvas que o MoviePy acredita que o
            # vídeo tem — não é um overlay: se o conteúdo real não chegar
            # até a borda vermelha, o preto "sobrando" fica visualmente
            # isolado entre o conteúdo e a moldura, mostrando exatamente
            # onde o problema está.
            os.makedirs(DEBUG_RENDER_DIR, exist_ok=True)
            _clip_com_borda = _debug_aplicar_borda_canvas(video_final)
            _caminho_borda = os.path.join(DEBUG_RENDER_DIR, "11_com_borda_vermelha.png")
            Image.fromarray(_clip_com_borda.get_frame(video_final.duration / 2)).save(_caminho_borda)
            print(f"🔍 [DEBUG_RENDER] frame com borda vermelha salvo em: {_caminho_borda}")

        print("⏳ Renderizando vídeo...")
        inicio_render = time.perf_counter()
        video_final.write_videofile(
            caminho_saida,
            codec="libx264",
            audio_codec="aac",
            preset=RENDER_PRESET,
            bitrate=RENDER_BITRATE,
            threads=RENDER_THREADS,
            logger=None,
        )
        print(f"🎬 Render concluído em {time.perf_counter() - inicio_render:.2f}s")

        return caminho_saida, caminhos_usados

    finally:
        # Libera todos os recursos abertos, evitando processos/arquivos presos.
        for clip in reversed(clips_para_fechar):
            try:
                clip.close()
            except Exception:
                pass  # nunca deixa um erro de limpeza mascarar o erro original


def escolher_fundo_e_criar_video(roteiro: str, caminho_audio: str, media_manager: MediaManager) -> str:
    """Ponte entre a biblioteca de mídia e `criar_video()`.

    Desde a correção B/C da Fase 2.1.1, NÃO busca mais antecipadamente o
    pool de reserva (`media_manager.escolher_videos(...)`) — essa busca
    consumia o orçamento GLOBAL de downloads do Pexels
    (`PEXELS_MAX_DOWNLOADS`) inteiro antes mesmo do B-roll contextual por
    trecho (Fase 2/2.1.1, a arquitetura REAL de seleção) rodar, deixando
    todo trecho contextual bloqueado pelo teto (`bloqueados_por_teto_download`)
    mesmo quando havia orçamento de sobra. Agora `criar_video()` recebe uma
    lista de reserva vazia, e é `_montar_fundo_multiplo()` quem busca esse
    pool sob demanda (lazy) — só quando algum trecho específico não tem
    candidato contextual e realmente precisa do fallback. `listar_videos()`
    (chamada dentro dessa busca lazy) agora respeita o orçamento RESIDUAL
    (`PEXELS_MAX_DOWNLOADS - self._downloads_nesta_execucao`), então o
    contextual — que roda primeiro, trecho a trecho — sempre tem prioridade
    sobre a reserva.

    O histórico só é atualizado (`media_manager.confirmar_uso()`) DEPOIS
    que `criar_video()` retorna com sucesso, e SÓ para os vídeos que
    `criar_video()` reporta como REALMENTE usados (`caminhos_usados`) —
    nunca no momento da escolha, e nunca para candidatos que sobraram sem
    uso. Se NENHUM dos candidatos funcionar durante a renderização de
    verdade (`FundoInvalidoError` — o que já é raro, já que
    `_montar_fundo_multiplo()` tenta várias alternativas — contextual e de
    reserva — e reaproveita o último B-roll que funcionou por trecho),
    descarta os candidatos desta tentativa e tenta de novo, até
    `MEDIA_MAX_TENTATIVAS` — nunca derruba o workflow por causa de vídeos
    ruins.
    """
    ultimo_erro: Optional[Exception] = None

    for tentativa in range(1, MEDIA_MAX_TENTATIVAS + 1):
        # Lista vazia de propósito: o pool de reserva agora é responsabilidade
        # de `_montar_fundo_multiplo()`, buscado sob demanda (ver docstring
        # acima e a Fase 2.1.1/correção C). Nada é pré-buscado aqui.
        caminhos_fundo: List[str] = []
        try:
            caminho_saida, caminhos_usados = criar_video(
                roteiro, caminho_audio, caminhos_fundo, media_manager=media_manager
            )
        except FundoInvalidoError as err:
            ultimo_erro = err
            for caminho in caminhos_fundo:
                media_manager.descartar(caminho)
            print(
                f"⚠️ Nenhum fundo candidato funcionou durante a renderização ({err}); "
                f"descartados desta execução. Tentativa {tentativa}/{MEDIA_MAX_TENTATIVAS}."
            )
        else:
            # Só agora, com o vídeo final de verdade pronto, cada mídia
            # REALMENTE usada (pode ser menos que `caminhos_fundo`, ou
            # repetir alguma) é considerada "usada" e entra no histórico
            # persistente.
            for caminho in caminhos_usados:
                media_manager.confirmar_uso(caminho)
            return caminho_saida

    raise RuntimeError(
        f"❌ Nenhum vídeo de fundo funcionou após {MEDIA_MAX_TENTATIVAS} tentativas. "
        f"Último erro: {ultimo_erro}"
    )


# ===========================================================================
# 4. UPLOAD YOUTUBE (Descrição Otimizada)
# ===========================================================================
def upload_youtube(caminho_video: str, titulo: str, roteiro: str) -> str:
    """Publica o vídeo final no YouTube Shorts e retorna o ID do vídeo."""
    print("🚀 [4/4] Enviando o vídeo para o YouTube Shorts...")

    token_json_str = os.environ.get("YOUTUBE_TOKEN_JSON")
    if not token_json_str:
        raise RuntimeError("❌ Upload: variável de ambiente YOUTUBE_TOKEN_JSON não configurada.")

    try:
        info_token = json.loads(token_json_str)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"❌ Upload: YOUTUBE_TOKEN_JSON não é um JSON válido ({err}).") from err

    try:
        credentials = Credentials.from_authorized_user_info(info_token)
        youtube = build("youtube", "v3", credentials=credentials)

        descricao = YOUTUBE_DESCRICAO_TEMPLATE.format(roteiro=roteiro)

        body = {
            "snippet": {
                "title": titulo,
                "description": descricao,
                "tags": YOUTUBE_TAGS,
                "categoryId": YOUTUBE_CATEGORY_ID,
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(caminho_video, chunksize=-1, resumable=True)
        request = youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media)
        resposta = request.execute()
    except Exception as err:
        # Cobre credenciais inválidas/expiradas, quota excedida, falha de
        # rede durante o upload resumível, etc.
        raise RuntimeError(f"❌ Upload: falha ao publicar no YouTube ({err}).") from err

    video_id = resposta["id"]
    print(f"✅ Sucesso! Vídeo publicado com o ID: {video_id}")
    return video_id


# ===========================================================================
# 5. PRÓXIMAS FASES (TODO) — apenas documentado, NADA implementado
# ===========================================================================
# Os itens abaixo são pontos de extensão conhecidos para as próximas fases
# do projeto. Nenhum deles tem código funcional ainda — são só notas de
# onde cada um se encaixaria na arquitetura atual, para acelerar o design
# quando a fase correspondente começar.
#
# TODO (Diretor Cinematográfico): uma camada de decisão que escolheria
#   dinamicamente ritmo de corte, enquadramento e intensidade do Ken Burns
#   com base no conteúdo do roteiro (ex: mais estático em trechos
#   explicativos, mais dinâmico em ganchos). Se encaixaria entre
#   `agrupar_palavras_em_blocos()` e `_aplicar_ken_burns()`, informando os
#   `pontos_de_corte` e os níveis de zoom em vez de usar `BLOCK_ZOOM_LEVELS`
#   fixo.
#
# FEITO (Banco de vídeos): implementado na Fase 5 via `MediaManager` — ver
#   seção 0. Escaneia `assets/videos/` (com subpastas por tema), valida
#   levemente, escolhe evitando repetir os últimos usados
#   (`media_history.json`), com fallback automático para o antigo
#   `VIDEO_FUNDO_PATH` se a biblioteca estiver vazia.
#
# FEITO (Provedor remoto Pexels): implementado na Fase 6 via
#   `PexelsMediaProvider` — ver seção 0. Busca vídeos no Pexels a partir
#   de palavras-chave extraídas do roteiro, baixa entre `PEXELS_MIN_DOWNLOADS_DESEJADOS`
#   e `PEXELS_MAX_DOWNLOADS`, cacheia em `cache/pexels/` por `CACHE_TTL_HOURS`,
#   e cai automaticamente para `LocalMediaProvider` se falhar por qualquer
#   motivo. `MediaManager` não sabe mais de onde a mídia vem — só conversa
#   com a interface `MediaProvider`.
#
# TODO (Pixabay / Unsplash / outros MediaProviders): a interface
#   `MediaProvider` já está pronta para isso — bastaria uma nova classe
#   implementando `listar_videos()`/`validar_video()`, sem tocar em
#   `MediaManager`. Fora de escopo da Fase 6 por decisão explícita.
#
# TODO (Banco de músicas / Banco de efeitos sonoros / Banco de
#   thumbnails): mesmo princípio do `MediaManager`, ainda não aplicado a
#   `MUSICA_PATH` nem a efeitos sonoros/thumbnails. `add_sound_effect()`
#   (seção 3.3) já existe como stub exatamente para o dia em que o banco de
#   efeitos sonoros for implementado. `MediaManager.listar_imagens()` /
#   `escolher_imagem()` já existem e estão prontos para o banco de
#   thumbnails, só ainda não são consumidos pelo pipeline.
#
# TODO (IA escolhendo música / fundo / efeitos / thumbnail / voz): uma vez
#   que os "bancos" acima existam, a escolha entre os itens de cada banco
#   poderia ser feita por um modelo (ou heurística) em vez de aleatória —
#   plugaria como uma função `escolher_*(roteiro, banco) -> item`, chamada
#   antes de `criar_video()`/`upload_youtube()`.
#
# TODO (Cache inteligente): já existe um cache local por execução
#   (`_CACHE_PALAVRAS_RENDERIZADAS`, seção 3.1) para tiles de legenda. Uma
#   fase futura poderia persistir esse cache entre execuções (ex: palavras
#   muito comuns do canal) — precisaria de um armazenamento persistente,
#   que é exatamente o próximo TODO.
#
# TODO (Banco SQLite / Histórico de vídeos / Banco de prompts / Sistema
#   anti-repetição): hoje cada execução é independente e sem memória do que
#   já foi publicado, então nada impede o Gemini de gerar um tema muito
#   parecido com um vídeo recente. Um banco SQLite local guardaria
#   título/tema/roteiro de cada vídeo publicado; `_montar_prompt_roteiro()`
#   (seção 1) passaria a receber um resumo dos últimos N temas para orientar
#   o Gemini a evitar repetição, e `upload_youtube()` gravaria o resultado
#   no banco após o sucesso do upload.
#
# Nenhum dos itens acima tem qualquer dependência nova instalada nem
# qualquer código funcional nesta versão — são apenas TODOs documentados.


# ===========================================================================
# EXECUÇÃO PRINCIPAL
# ===========================================================================
def _formatar_linha_log(nome: str, status: str, duracao: float) -> str:
    """Formata uma linha de log alinhada por pontos, ex.: '🧠 Gemini ... OK (2.30s)'."""
    pontos = max(3, LARGURA_LOG_TEMPOS - len(nome))
    return f"{nome} {'.' * pontos} {status} ({duracao:.2f}s)"


def _ram_aproximada_mb() -> float:
    """RAM aproximada (pico de uso do processo) via stdlib, sem dependências
    novas. `ru_maxrss` vem em KB no Linux."""
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _rodar_etapa(nome: str, tempos: Dict[str, float], func, *args, **kwargs):
    """Executa uma etapa do pipeline, cronometrando e logando OK/❌ no
    formato padronizado. Em caso de falha, relança o erro (com o nome da
    etapa no início da mensagem, ex. '❌ Gemini: ...') para que `main()`
    interrompa o pipeline — as etapas são sequenciais e dependentes."""
    inicio = time.perf_counter()
    try:
        resultado = func(*args, **kwargs)
    except Exception as err:
        duracao = time.perf_counter() - inicio
        tempos[nome] = duracao
        print(_formatar_linha_log(nome, "FALHOU", duracao))
        mensagem_err = str(err)
        if mensagem_err.startswith("❌"):
            # A etapa interna (ex: Whisper, ImageMagick) já identificou a si
            # mesma — evita duplicar o marcador, só adiciona o contexto.
            raise RuntimeError(f"❌ {nome} (via {mensagem_err[2:]})") from err
        raise RuntimeError(f"❌ {nome}: {err}") from err

    duracao = time.perf_counter() - inicio
    tempos[nome] = duracao
    print(_formatar_linha_log(nome, "OK", duracao))
    return resultado


def _log_tempos(tempos: Dict[str, float]) -> None:
    """Imprime o resumo final: tempo total e RAM aproximada utilizada."""
    print("\n⏱️  Resumo de tempos:")
    for etapa, duracao in tempos.items():
        pontos = max(3, LARGURA_LOG_TEMPOS - len(etapa))
        print(f"{etapa} {'.' * pontos} {duracao:.2f}s")
    rotulo_ram = "💾 RAM aproximada (pico)"
    print(f"{rotulo_ram} {'.' * max(3, LARGURA_LOG_TEMPOS - len(rotulo_ram))} {_ram_aproximada_mb():.0f} MB")


def main() -> None:
    """Executa o pipeline completo: roteiro -> áudio -> vídeo -> upload."""
    inicio_total = time.perf_counter()
    tempos: Dict[str, float] = {}
    media_manager: Optional[MediaManager] = None

    try:
        media_manager = _criar_media_manager()
        titulo, roteiro = _rodar_etapa("🧠 Gemini", tempos, gerar_roteiro)
        audio_path = _rodar_etapa("🎙 Narração (TTS)", tempos, gerar_audio, roteiro)
        video_path = _rodar_etapa(
            "🎬 Render", tempos, escolher_fundo_e_criar_video, roteiro, audio_path, media_manager
        )
        _rodar_etapa("🚀 Upload", tempos, upload_youtube, video_path, titulo, roteiro)

    except Exception as err:
        print(f"\n{err}")
        tempos["🏁 Total (com erro)"] = time.perf_counter() - inicio_total
        _log_tempos(tempos)
        raise SystemExit(1)

    else:
        tempos["🏁 Total"] = time.perf_counter() - inicio_total
        _log_tempos(tempos)

    finally:
        # Remove arquivos temporários independentemente do resultado.
        if os.path.exists(NARRACAO_PATH):
            os.remove(NARRACAO_PATH)
        # Libera espaço em disco (cache expirado, downloads interrompidos)
        # — importante no runner efêmero do GitHub Actions, que não deve
        # acumular arquivos entre execuções.
        if media_manager is not None:
            media_manager.limpar_cache_temporario()


if __name__ == "__main__":
    main()