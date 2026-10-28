"""Serviços de obtenção, seleção e cache de mídia para os vídeos."""

import json
import os
import random
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from PIL import Image
from moviepy.editor import VideoFileClip

from models.media import MediaItem

ASPECT_RATIO: float = 9 / 16
VIDEO_FUNDO_PATH: str = "fundo.mp4"
PALAVRAS_INSEPARAVEIS = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "ao", "aos", "pelo", "pela", "pelos", "pelas",
    "num", "numa", "nesse", "nessa", "neste", "nesta",
    "desse", "dessa", "deste", "desta", "de", "em", "com", "sem",
}


def _normalizar_termo_visual(palavra: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", palavra).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", sem_acento.lower())

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
MEDIA_VIDEOS_DIR: str = os.path.join("assets", "videos")  # pode ter subpastas por tema (minecraft/, parkour/, ...)
MEDIA_IMAGENS_DIR: str = os.path.join("assets", "imagens")
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
PEXELS_MAX_DOWNLOADS: int = 15  # teto global de downloads novos por execução
PEXELS_MAX_DOWNLOADS_POR_TRECHO: int = 2  # evita consumir a cota em um único trecho contextual
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
CACHE_DIR: str = os.path.join("cache", "pexels")
CACHE_METADATA_PATH: str = os.path.join(CACHE_DIR, "metadata.json")
CACHE_TTL_HOURS: float = 24.0  # depois disso, o cache expira e pode baixar de novo
MAX_PALAVRAS_CHAVE_PESQUISA: int = 4  # quantas palavras-chave extrair do roteiro
PEXELS_MIN_PALAVRA_LEN: int = 4  # ignora palavras menores que isso na extração de palavras-chave

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
    try:
        if not os.path.isfile(caminho) or os.path.getsize(caminho) <= 0:
            return False
    except OSError:
        return False

    clip = None
    valido = False
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
        valido = True
    except Exception:
        valido = False
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                valido = False

    return valido


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
            if not isinstance(dados, dict):
                return {}
            return {
                video_id: entrada
                for video_id, entrada in dados.items()
                if isinstance(entrada, dict)
            }
        except (OSError, json.JSONDecodeError):
            return {}

    def _salvar_metadata(self) -> None:
        temporario = self.metadata_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.metadata_path) or ".", exist_ok=True)
            with open(temporario, "w", encoding="utf-8") as arquivo:
                json.dump(self._metadata, arquivo, ensure_ascii=False, indent=2)
            os.replace(temporario, self.metadata_path)
        except OSError as err:
            # Nunca derruba a execução por causa do cache — é só uma
            # otimização de banda/quota, não uma etapa crítica.
            print(f"⚠️ Pexels: não foi possível salvar cache ({err}); seguindo sem persistir.")
            if os.path.exists(temporario):
                try:
                    os.remove(temporario)
                except OSError as cleanup_err:
                    print(f"⚠️ Pexels: não foi possível limpar '{temporario}' ({cleanup_err}).")

    def _caminho_dentro_do_cache(self, caminho: object) -> Optional[str]:
        if not isinstance(caminho, str) or not caminho:
            return None
        caminho_absoluto = os.path.abspath(caminho)
        raiz_cache = os.path.realpath(self.cache_dir)
        try:
            if os.path.commonpath((raiz_cache, os.path.realpath(caminho_absoluto))) != raiz_cache:
                return None
        except ValueError:
            return None
        return caminho_absoluto

    def _cache_valido(self, entrada: dict) -> bool:
        if not isinstance(entrada, dict):
            return False
        caminho = entrada.get("caminho")
        baixado_em = entrada.get("baixado_em", 0)
        caminho_cache = self._caminho_dentro_do_cache(caminho)
        if (
            caminho_cache is None
            or not isinstance(baixado_em, (int, float))
            or not os.path.isfile(caminho_cache)
        ):
            return False
        idade_horas = (time.time() - baixado_em) / 3600
        return 0 <= idade_horas < CACHE_TTL_HOURS

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
            normalizada = _normalizar_termo_visual(palavra)
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
                    while True:
                        bloco = resposta.read(1024 * 1024)
                        if not bloco:
                            break
                        arquivo_saida.write(bloco)
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
        ids_evitar_video = {
            os.path.splitext(os.path.basename(str(identificador)))[0]
            for identificador in ids_evitar
        }

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
                    video, arquivo, indice_query, total_queries, duracao_alvo, ids_evitar_video
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
                    video,
                    arquivo,
                    indice_q,
                    total_queries,
                    duracao_alvo,
                    ids_evitar_video,
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
        bloqueados_por_teto_trecho = 0
        downloads_neste_trecho = 0

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
                        pontuacao=score,
                    )
                )
                continue

            if self._downloads_nesta_execucao >= PEXELS_MAX_DOWNLOADS:
                bloqueados_por_teto_download += 1
                continue  # teto de downloads da execução (item 19) — mesmo com candidatos bons sobrando

            if downloads_neste_trecho >= PEXELS_MAX_DOWNLOADS_POR_TRECHO:
                bloqueados_por_teto_trecho += 1
                continue  # mantém os downloads novos deste trecho dentro do orçamento local

            print(f"⬇️ Download novo (video_id={video_id}, fonte=api, url={arquivo.get('link', '?')})")
            destino = os.path.join(self.cache_dir, f"{video_id}.mp4")
            downloads_neste_trecho += 1
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
                MediaItem(
                    caminho=destino,
                    categoria=f"pexels/score={score:.1f}/fonte=api",
                    tipo="video",
                    pontuacao=score,
                )
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
            f"bloqueados_por_teto_trecho={bloqueados_por_teto_trecho} "
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
                caminho_cache = self._caminho_dentro_do_cache(entrada.get("caminho"))
                if caminho_cache and os.path.exists(caminho_cache):
                    try:
                        os.remove(caminho_cache)
                    except OSError as err:
                        print(f"⚠️ Pexels: não foi possível remover cache expirado '{caminho_cache}' ({err}).")
                self._metadata.pop(video_id, None)

        if os.path.isdir(self.cache_dir):
            caminhos_referenciados = {
                caminho_cache
                for entrada in self._metadata.values()
                if (caminho_cache := self._caminho_dentro_do_cache(entrada.get("caminho")))
            }
            for nome in os.listdir(self.cache_dir):
                caminho = os.path.abspath(os.path.join(self.cache_dir, nome))
                if os.path.isfile(caminho) and (
                    nome.endswith((".part", ".tmp"))
                    or (nome.endswith(".mp4") and caminho not in caminhos_referenciados)
                ):
                    try:
                        os.remove(caminho)
                    except OSError as err:
                        print(f"⚠️ Pexels: não foi possível limpar arquivo órfão '{caminho}' ({err}).")

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
            obtido: Optional[str] = None

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
            key=lambda item: (
                item.pontuacao if item.pontuacao is not None else float("-inf"),
                item.caminho not in no_historico and item.caminho not in excluir,
            ),
            reverse=True,
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
    api_key = os.getenv(PEXELS_API_KEY_ENV_VAR)
    provider_local = LocalMediaProvider()
    if not api_key:
        return MediaManager(provider=provider_local)

    return MediaManager(
        provider=PexelsMediaProvider(api_key=api_key),
        provider_fallback=provider_local,
    )
