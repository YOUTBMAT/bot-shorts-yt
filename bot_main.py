"""
Bot Automático de Publicação de YouTube Shorts.

Pipeline: gera roteiro (Gemini) -> gera narração (ElevenLabs) -> transcreve
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
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
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
MODEL_PRIORITY: List[str] = ["gemini-2.5-flash", "gemini-3.6-flash"]
MAX_RETRIES: int = 2  # tentativas por modelo antes de passar para o próximo
ROTEIRO_MIN_PALAVRAS: int = 65
ROTEIRO_MAX_PALAVRAS: int = 85
ROTEIRO_TOLERANCIA_PALAVRAS: int = 15  # margem aceita além do min/max antes de rejeitar o roteiro
TITULO_MAX_CARACTERES: int = 90
RETRY_BACKOFF_BASE: float = 1.0  # segundos; dobra a cada nova tentativa (1, 2, 4...)
RETRY_BACKOFF_JITTER: float = 0.5  # variação aleatória (0 a X segundos) somada à espera

# --- Narração (ElevenLabs) ---
VOICE_ID: str = "pNInz6obpgDQGcFmaJgB"  # Voz "Adam"
ELEVEN_MODEL_ID: str = "eleven_multilingual_v2"
ELEVENLABS_TIMEOUT_SECONDS: float = 120.0  # evita que uma chamada travada prenda o job do GitHub Actions
NARRACAO_PATH: str = "narracao.mp3"

# --- Vídeo (MoviePy) ---
VIDEO_FUNDO_PATH: str = "fundo.mp4"  # fallback legado: usado só se a biblioteca de mídia estiver vazia
VIDEO_SAIDA_PATH: str = "video_final.mp4"
VIDEO_WIDTH: int = 1080
ASPECT_RATIO: float = 9 / 16  # formato vertical (Shorts)
CROP_OFFSET_MAX_FRACTION: float = 0.35  # quanto do espaço sobrando o corte pode usar p/ não centralizar
DURACAO_MINIMA_SEGURANCA: float = 0.05  # segundos; piso usado contra timestamps/durações degeneradas

# --- Biblioteca de mídia (MediaManager) ---
MEDIA_VIDEOS_DIR: str = "assets/videos"  # pode ter subpastas por tema (minecraft/, parkour/, ...)
MEDIA_IMAGENS_DIR: str = "assets/imagens"
MEDIA_EXTENSOES_VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
MEDIA_EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_VIDEO_DURACAO_MINIMA: float = 5.0  # segundos; vídeos mais curtos são ignorados
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
LEGEND_WIDTH_FRACTION: float = 0.82  # largura máxima da área de legenda
LEGEND_WORD_SPACING: int = 18  # espaço horizontal (px) entre palavras
LEGEND_LINE_SPACING: int = 12  # espaço vertical (px) entre linhas, se o grupo quebrar em 2
SAFE_AREA_BOTTOM: float = 0.20  # fração da tela reservada na base (fora da legenda)
POP_SCALE: float = 1.08  # escala máxima do "pop" da palavra ativa (100% -> 108% -> 100%)
POP_DURATION: float = 0.22  # duração do pop, em segundos, ao ativar uma palavra
MIN_GROUP_WORDS: int = 3
MAX_GROUP_WORDS: int = 5
PAUSA_MINIMA_QUEBRA: float = 0.35  # segundos; pausa que sugere quebra natural de grupo

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


def _validar_arquivo_video(caminho: str) -> bool:
    """Validação completa de um arquivo de vídeo, compartilhada por todos
    os MediaProviders: consegue abrir e tem pelo menos
    MEDIA_VIDEO_DURACAO_MINIMA segundos. Extraída como função solta (em
    vez de método duplicado em cada provider) porque a lógica é idêntica
    independente de o arquivo ter vindo do disco local ou de um download
    do Pexels — só muda de onde o arquivo chegou até aqui."""
    clip = None
    try:
        clip = VideoFileClip(caminho)
        return bool(clip.duration and clip.duration >= MEDIA_VIDEO_DURACAO_MINIMA)
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
        # Blacklist em memória, válida só para esta execução (não é
        # persistida): guarda os IDs de vídeos descartados por
        # `descartar()` (ex.: passaram na validação mas falharam de
        # verdade na renderização), para que um retry na mesma execução
        # não busque/baixe o mesmo vídeo de novo — sem isso, como
        # `descartar()` remove a entrada do metadata, o vídeo deixaria de
        # ser reconhecido como "já visto" e poderia ser rebaixado e
        # escolhido de novo no mesmo `escolher_fundo_e_criar_video()`.
        self._rejeitados_nesta_execucao: set = set()

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

    # --- palavras-chave -------------------------------------------------
    def _extrair_palavras_chave(self, roteiro: str) -> List[str]:
        """Extrai até MAX_PALAVRAS_CHAVE_PESQUISA palavras-chave do
        roteiro por frequência simples, ignorando STOPWORDS_PESQUISA e
        palavras curtas demais. Deliberadamente NÃO usa IA/embeddings para
        isso (fora de escopo desta fase) — é uma heurística leve, sem
        dependência nova, que já é suficiente para gerar termos de busca
        razoáveis a partir de um roteiro em português."""
        contagem: Dict[str, int] = {}
        for palavra in re.findall(r"[A-Za-zÀ-ÿ]+", roteiro):
            normalizada = _normalizar_palavra(palavra)
            if len(normalizada) < PEXELS_MIN_PALAVRA_LEN or normalizada in STOPWORDS_PESQUISA:
                continue
            contagem[normalizada] = contagem.get(normalizada, 0) + 1

        ordenadas = sorted(contagem.items(), key=lambda par: (-par[1], par[0]))
        palavras = [palavra for palavra, _freq in ordenadas[:MAX_PALAVRAS_CHAVE_PESQUISA]]
        return palavras or ["nature"]  # fallback genérico se o roteiro não render nenhum termo válido

    # --- busca / download ---------------------------------------------------
    def _buscar(self, query: str) -> List[dict]:
        """Consulta a API oficial do Pexels. Erros de rede/HTTP viram
        RuntimeError com prefixo "❌ Pexels: ..." — capturados por
        `listar_videos()`, que segue para a próxima palavra-chave em vez
        de interromper a busca inteira por causa de uma falha pontual."""
        parametros = urllib.parse.urlencode(
            {"query": query, "per_page": PEXELS_MAX_RESULTS, "orientation": PEXELS_ORIENTACAO}
        )
        url = f"{PEXELS_API_URL}?{parametros}"
        requisicao = urllib.request.Request(url, headers={"Authorization": self.api_key})
        try:
            with urllib.request.urlopen(requisicao, timeout=PEXELS_TIMEOUT_SECONDS) as resposta:
                dados = json.loads(resposta.read().decode("utf-8"))
            return dados.get("videos", [])
        except urllib.error.HTTPError as err:
            if err.code == 401:
                raise RuntimeError("❌ Pexels: API Key inválida (HTTP 401).") from err
            raise RuntimeError(f"❌ Pexels: busca por '{query}' falhou (HTTP {err.code}).") from err
        except (urllib.error.URLError, socket.timeout) as err:
            raise RuntimeError(f"❌ Pexels: indisponível/timeout na busca por '{query}' ({err}).") from err

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
            requisicao = urllib.request.Request(url)
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
        não há como gerar palavras-chave, então devolve só o que já está
        em cache (podendo ser uma lista vazia, o que faz o MediaManager
        cair para o fallback local)."""
        em_cache = self._videos_em_cache_validos()
        if len(em_cache) >= PEXELS_MIN_DOWNLOADS_DESEJADOS:
            return em_cache

        if not contexto:
            return em_cache

        palavras_chave = self._extrair_palavras_chave(contexto)
        novos: List[MediaItem] = []

        for palavra in palavras_chave:
            if len(em_cache) + len(novos) >= PEXELS_MAX_DOWNLOADS:
                break
            try:
                resultados = self._buscar(palavra)
            except RuntimeError as err:
                print(f"⚠️ {err}")
                continue

            for video in resultados:
                if len(em_cache) + len(novos) >= PEXELS_MAX_DOWNLOADS:
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

                self._metadata[video_id] = {
                    "caminho": destino,
                    "baixado_em": time.time(),
                    "palavra_chave": palavra,
                }
                novos.append(MediaItem(caminho=destino, categoria=f"pexels/{palavra}", tipo="video"))

        if novos:
            self._salvar_metadata()

        return em_cache + novos

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

    def _escolher_de(self, provider: MediaProvider, contexto: Optional[str]) -> str:
        candidatos = list(provider.listar_videos(contexto))
        if not candidatos:
            raise RuntimeError(f"{type(provider).__name__} não retornou nenhum vídeo candidato.")

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
# 2. GERAR ÁUDIO (ElevenLabs)
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


def gerar_audio(texto: str, caminho_saida: str = NARRACAO_PATH) -> str:
    """Gera o arquivo de narração em áudio a partir do roteiro."""
    print("🎙️ [2/4] Gerando voz ultra-realista com a ElevenLabs...")
    texto_narracao = preparar_texto_para_narracao(texto)

    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"), timeout=ELEVENLABS_TIMEOUT_SECONDS)

    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=texto_narracao,
            model_id=ELEVEN_MODEL_ID,
        )

        bytes_escritos = 0
        with open(caminho_saida, "wb") as arquivo:
            for chunk in audio_stream:
                if chunk:
                    arquivo.write(chunk)
                    bytes_escritos += len(chunk)

    except Exception as err:
        # Cobre tanto erro de rede/timeout (a lib já usa um timeout padrão de
        # ELEVENLABS_TIMEOUT_SECONDS, então "demora demais" também cai aqui)
        # quanto qualquer outra falha da API.
        raise RuntimeError(f"❌ ElevenLabs: falha ao gerar a narração ({err}).") from err

    if bytes_escritos == 0:
        # A API "funcionou" (não lançou erro) mas devolveu um áudio vazio.
        raise RuntimeError("❌ ElevenLabs: a narração gerada está vazia (0 bytes).")

    return caminho_saida


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
) -> VideoClip:
    """Aplica um efeito Ken Burns sutil: zoom lento contínuo + zoom por segmento.

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
        return max(1.0, 1.0 + (zoom_bloco - 1.0) + (zoom_lento - 1.0))

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
    grupo: List[PalavraLegenda], largura_video: int, altura_video: int
) -> VideoClip:
    """Constrói UM único VideoClip dinâmico para o grupo inteiro.

    Em vez de empilhar vários TextClips (um por palavra por estado), o
    grupo inteiro vira um único clip cujo `make_frame` compõe os tiles já
    cacheados a cada quadro — poucos objetos MoviePy na composição final,
    menos overhead de renderização.
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

    # Posição fixa (safe area): a base do bloco de legenda sempre no mesmo
    # lugar, longe da borda inferior — nunca cobre elementos do Shorts nem
    # "pula" de posição entre grupos.
    x_video = (largura_video - largura_maxima) // 2
    y_video = altura_video - int(altura_video * SAFE_AREA_BOTTOM) - altura_total
    clip_video = clip_video.set_start(inicio_grupo).set_position((x_video, y_video))

    return clip_video


def _criar_clips_de_legenda(grupos: List[List[PalavraLegenda]], largura_video: int, altura_video: int) -> List[VideoClip]:
    """Constrói o clip dinâmico de cada grupo e devolve a lista pronta para
    entrar na composição final."""
    return [_construir_clip_grupo(grupo, largura_video, altura_video) for grupo in grupos]


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


def criar_video(
    roteiro_completo: str,
    caminho_audio: str,
    caminho_video_fundo: str,
    caminho_saida: str = VIDEO_SAIDA_PATH,
) -> str:
    """Monta o vídeo final: fundo dinâmico (Ken Burns), legendas, barra de
    progresso, vinheta e fades, cortado em 9:16."""
    print("🎬 [3/4] Montando vídeo final com legendas dinâmicas e design pro...")

    # Referências guardadas à parte para garantir o fechamento de todos os
    # clips abertos, mesmo em caso de erro no meio do processo.
    clips_para_fechar: List = []

    try:
        if not os.path.exists(caminho_video_fundo):
            raise FundoInvalidoError(
                f"❌ Vídeo de fundo '{caminho_video_fundo}' não encontrado."
            )
        try:
            clip_video = VideoFileClip(caminho_video_fundo)
        except Exception as err:
            raise FundoInvalidoError(
                f"❌ Vídeo de fundo '{caminho_video_fundo}' corrompido ou ilegível: {err}"
            ) from err
        clips_para_fechar.append(clip_video)
        _debug_registrar("01_abertura", clip_video)
        _verificar_fundo_saudavel(clip_video, caminho_video_fundo)

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

        # Loop do vídeo de fundo caso o áudio seja mais longo que o vídeo.
        if clip_video.duration < duracao_final:
            num_loops = math.ceil(duracao_final / clip_video.duration)
            clip_video_loop = concatenate_videoclips([clip_video] * num_loops)
            clips_para_fechar.append(clip_video_loop)
            clip_video = clip_video_loop
            _debug_registrar("04_loop", clip_video)
        else:
            _debug_registrar("04_loop_nao_necessario", clip_video)

        clip_video = clip_video.subclip(0, duracao_final)
        clip_video = _redimensionar_para_vertical(clip_video)

        # Transcreve a narração (sempre a voz limpa, antes da mixagem com
        # música) e agrupa as palavras em blocos de legenda sincronizados.
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

        # Fundo dinâmico: zoom lento contínuo + variação de zoom a cada
        # grupo de legenda + leve pan, sempre dentro dos limites da imagem
        # (sem bordas pretas).
        fundo_dinamico = _aplicar_ken_burns(clip_video, duracao_final, pontos_de_corte)
        clips_para_fechar.append(fundo_dinamico)
        _debug_registrar("05_ken_burns", fundo_dinamico)

        fundo_com_vinheta = _aplicar_vinheta(fundo_dinamico)
        largura_final, altura_final = fundo_com_vinheta.size
        _debug_registrar("06_vinheta", fundo_com_vinheta)

        # Verificação estrutural (custo de 1 frame, desprezível perto do
        # render inteiro): confirma que o fundo realmente preenche 100% do
        # canvas antes de prosseguir. Isso não "esconde" um eventual bug de
        # faixa preta — pelo contrário, faz a renderização falhar alto e
        # claro, com a causa exata, em vez de produzir silenciosamente um
        # vídeo com borda preta que só seria percebido ao assistir.
        _frame_teste = fundo_com_vinheta.get_frame(min(0.5, duracao_final / 2))
        _altura_real, _largura_real = _frame_teste.shape[:2]
        if (_largura_real, _altura_real) != (largura_final, altura_final):
            raise RuntimeError(
                f"❌ Inconsistência de tamanho no fundo do vídeo: o frame renderizado "
                f"tem {_largura_real}x{_altura_real}px, mas o canvas esperado é "
                f"{largura_final}x{altura_final}px. Isso indicaria uma regressão no "
                f"Ken Burns/vinheta — pare e investigue antes de publicar."
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

        return caminho_saida

    finally:
        # Libera todos os recursos abertos, evitando processos/arquivos presos.
        for clip in reversed(clips_para_fechar):
            try:
                clip.close()
            except Exception:
                pass  # nunca deixa um erro de limpeza mascarar o erro original


def escolher_fundo_e_criar_video(roteiro: str, caminho_audio: str, media_manager: MediaManager) -> str:
    """Ponte entre a biblioteca de mídia e `criar_video()`.

    Antes, o caminho do fundo era sempre a constante `VIDEO_FUNDO_PATH`.
    Agora é `media_manager.escolher_video(contexto=roteiro)` — o roteiro é
    passado como contexto para que um provider remoto (Pexels) possa gerar
    palavras-chave de busca a partir dele; providers locais o ignoram. O
    resto do pipeline não muda nada.

    O histórico só é atualizado (`media_manager.confirmar_uso()`) DEPOIS
    que `criar_video()` retorna com sucesso — nunca no momento da escolha.
    Se o vídeo escolhido falhar durante a renderização de verdade
    (`FundoInvalidoError`), descarta esse arquivo do pool desta execução
    (sem jamais ter entrado no histórico) e tenta outro, até
    `MEDIA_MAX_TENTATIVAS` — nunca derruba o workflow por causa de um único
    arquivo ruim que passou pela validação leve mas falhou na hora de
    renderizar.
    """
    ultimo_erro: Optional[Exception] = None

    for tentativa in range(1, MEDIA_MAX_TENTATIVAS + 1):
        caminho_fundo = media_manager.escolher_video(contexto=roteiro)
        try:
            caminho_saida = criar_video(roteiro, caminho_audio, caminho_fundo)
        except FundoInvalidoError as err:
            ultimo_erro = err
            media_manager.descartar(caminho_fundo)
            print(
                f"⚠️ Fundo '{caminho_fundo}' falhou durante a renderização ({err}); "
                f"descartado desta execução. Tentativa {tentativa}/{MEDIA_MAX_TENTATIVAS}."
            )
        else:
            # Só agora, com o vídeo final de verdade pronto, a mídia é
            # considerada "usada" e entra no histórico persistente.
            media_manager.confirmar_uso(caminho_fundo)
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
# TODO (Múltiplos provedores de TTS): `gerar_audio()` está fortemente
#   acoplada à ElevenLabs. Generalizar exigiria extrair uma interface comum
#   (texto -> arquivo de áudio) e uma constante tipo `TTS_PROVIDER` para
#   escolher entre provedores, com fallback entre eles (mesmo padrão já
#   usado para `MODEL_PRIORITY` do Gemini).
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
        audio_path = _rodar_etapa("🎙 ElevenLabs", tempos, gerar_audio, roteiro)
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