"""Composição, sincronização audiovisual e renderização dos vídeos."""

import bisect
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
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

from models.media import FundoInvalidoError
from services.media_service import (
    ASPECT_RATIO,
    MEDIA_MAX_TENTATIVAS,
    MediaManager,
    PALAVRAS_INSEPARAVEIS,
)

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# --- Vídeo (MoviePy) ---
VIDEO_SAIDA_PATH: str = "video_final.mp4"
VIDEO_WIDTH: int = 1080
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
            clip_sombra = None
            try:
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
                resultado = (canvas_rgb, canvas_alpha)
            finally:
                clip_texto.close()
                if clip_sombra is not None:
                    clip_sombra.close()
        else:
            clip_texto = TextClip(
                texto, fontsize=LEGEND_FONT_SIZE, font=LEGEND_FONT, color=NORMAL_COLOR, method="label",
            )
            try:
                resultado = tuple(a.copy() for a in _obter_frame_e_alpha(clip_texto))
            finally:
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
    ultimo_t: Optional[float] = None
    ultimo_frame: Optional[np.ndarray] = None
    ultimo_alpha: Optional[np.ndarray] = None

    def _obter_quadro_do_grupo(t: float) -> Tuple[np.ndarray, np.ndarray]:
        nonlocal ultimo_t, ultimo_frame, ultimo_alpha
        if ultimo_t != t:
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

            ultimo_t = t
            ultimo_frame = frame
            ultimo_alpha = alpha
        assert ultimo_frame is not None and ultimo_alpha is not None
        return ultimo_frame, ultimo_alpha

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
    except Exception:
        try:
            musica.close()
        except Exception as err:
            print(f"⚠️ Não foi possível fechar a música após uma falha ({err}).")
        raise

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


def escolher_fundo_e_criar_video(
    roteiro: str, caminho_audio: str, media_manager: MediaManager
) -> str:
    """Renderiza com mídia contextual e confirma apenas os fundos utilizados."""
    ultimo_erro: Optional[Exception] = None

    for tentativa in range(1, MEDIA_MAX_TENTATIVAS + 1):
        try:
            caminho_saida, caminhos_usados = criar_video(
                roteiro, caminho_audio, [], media_manager=media_manager
            )
        except FundoInvalidoError as err:
            ultimo_erro = err
            print(
                f"⚠️ Nenhum fundo candidato funcionou durante a renderização ({err}); "
                f"tentativa {tentativa}/{MEDIA_MAX_TENTATIVAS}."
            )
        else:
            for caminho in caminhos_usados:
                media_manager.confirmar_uso(caminho)
            return caminho_saida

    raise RuntimeError(
        f"❌ Nenhum vídeo de fundo funcionou após {MEDIA_MAX_TENTATIVAS} tentativas. "
        f"Último erro: {ultimo_erro}"
    )


def validar_video_final(caminho_video: str) -> bool:
    """Confirma os requisitos técnicos do arquivo de vídeo renderizado."""
    if not os.path.isfile(caminho_video):
        print(f"❌ Falha na validação: arquivo vazio ou inexistente em {caminho_video}")
        return False

    try:
        if os.path.getsize(caminho_video) <= 0:
            print(f"❌ Falha na validação: arquivo vazio ou inexistente em {caminho_video}")
            return False
    except OSError as err:
        print(f"❌ Falha na validação: não foi possível verificar o arquivo {caminho_video} ({err})")
        return False

    clip = None
    valido = False
    try:
        clip = VideoFileClip(caminho_video)
        if clip.size != (1080, 1920):
            print(f"❌ Falha na validação: dimensões incorretas {clip.size} (esperado 1080x1920)")
        elif clip.audio is None:
            print("❌ Falha na validação: o vídeo não possui faixa de áudio (clip.audio is None)")
        elif clip.duration is None or clip.duration <= 0:
            print(f"❌ Falha na validação: duração inválida {clip.duration} (esperado > 0)")
        else:
            valido = True
    except Exception as err:
        print(f"❌ Falha na validação: erro ao inspecionar o vídeo {caminho_video} ({err})")
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as err:
                print(f"❌ Falha na validação: erro ao fechar o clip de vídeo ({err})")
                valido = False

    return valido


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
