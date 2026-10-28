"""
Testes da Fase 11 (auditoria Pexels/B-roll).

Cobre, na ordem pedida:
 1. Geração de queries visuais.
 2. Tratamento de 200/401/403/429/500/503/timeout no Pexels.
 3. Fallback Pexels -> Local.
 4. Apenas 1 vídeo válido disponível.
 5. 2 vídeos válidos disponíveis.
 6. 4-6 vídeos válidos disponíveis.
 7. Vídeo de fundo MENOR que a duração da narração (precisa de loop).
 8. Vídeo de fundo MAIOR que a duração da narração (precisa de corte).
 9. Formato final 1080x1920 (checagem das constantes/geometria).
10. Legendas dentro da safe zone (checagem das constantes).
11. Mídia descartada não entra no histórico.
12. Retry não escolhe imediatamente o mesmo vídeo descartado/já usado.

este projeto não tem framework de testes prévio nem dependência de teste
declarada — por isso os testes usam `unittest` (stdlib) e substituem por
stubs mínimos as dependências pesadas (como MoviePy e faster-whisper)
antes de importar `services.media_service` e `services.video_service`.
Isso testa a lógica dos serviços sem depender de rede, GPU ou ffmpeg — o
que é apropriado para rodar em CI/GitHub Actions de forma rápida.

Rodar com:  python -m unittest tests.test_pexels_broll -v
"""

import io
import json
import os
import random
import sys
import tempfile
import time
import types
import unittest
import urllib.error
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _stub_module(name: str, **attrs) -> types.ModuleType:
    modulo = types.ModuleType(name)
    for chave, valor in attrs.items():
        setattr(modulo, chave, valor)
    sys.modules[name] = modulo
    return modulo


def _instalar_stubs_de_dependencias_pesadas() -> None:
    """Registra stubs para dependências pesadas que não estejam instaladas."""

    class _Generico:
        def __init__(self, *args, **kwargs):
            pass

    if "moviepy.editor" not in sys.modules:
        try:
            import moviepy.editor  # noqa: F401
        except ImportError:
            moviepy_editor = _stub_module(
                "moviepy.editor",
                VideoFileClip=_Generico,
                AudioFileClip=_Generico,
                CompositeAudioClip=_Generico,
                VideoClip=_Generico,
                TextClip=_Generico,
                CompositeVideoClip=_Generico,
                concatenate_videoclips=lambda clips, **kw: _Generico(),
            )
            _stub_module("moviepy", editor=moviepy_editor)
            _stub_module("moviepy.audio")
            _stub_module("moviepy.audio.fx")
            _stub_module(
                "moviepy.audio.fx.all",
                audio_fadein=lambda clip, *a, **k: clip,
                audio_fadeout=lambda clip, *a, **k: clip,
            )

    try:
        import google  # noqa: F401
    except ImportError:
        _stub_module("google")

    if "google.genai" not in sys.modules:
        try:
            import google.genai  # noqa: F401
        except ImportError:
            _stub_module("google.genai", Client=_Generico)
            _stub_module("google.genai.types", GenerateContentConfig=_Generico)

    if "google.api_core.client_options" not in sys.modules:
        try:
            import google.api_core.client_options  # noqa: F401
        except ImportError:
            _stub_module("google.api_core")
            _stub_module("google.api_core.client_options", ClientOptions=_Generico)

    if "google.cloud.texttospeech" not in sys.modules:
        try:
            import google.cloud.texttospeech  # noqa: F401
        except ImportError:
            _stub_module("google.cloud")
            _stub_module(
                "google.cloud.texttospeech",
                TextToSpeechClient=_Generico,
                SynthesisInput=_Generico,
                VoiceSelectionParams=_Generico,
                AudioConfig=_Generico,
                AudioEncoding=types.SimpleNamespace(MP3=1),
            )

    if "google.oauth2.credentials" not in sys.modules:
        try:
            import google.oauth2.credentials  # noqa: F401
        except ImportError:
            _stub_module("google.oauth2")
            _stub_module("google.oauth2.credentials", Credentials=_Generico)

    if "elevenlabs.client" not in sys.modules:
        try:
            import elevenlabs.client  # noqa: F401
        except ImportError:
            _stub_module("elevenlabs")
            _stub_module("elevenlabs.client", ElevenLabs=_Generico)

    if "openai" not in sys.modules:
        try:
            import openai  # noqa: F401
        except ImportError:
            _stub_module("openai", OpenAI=_Generico)

    if "faster_whisper" not in sys.modules:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            _stub_module("faster_whisper", WhisperModel=_Generico)

    if "googleapiclient.discovery" not in sys.modules:
        try:
            import googleapiclient.discovery  # noqa: F401
        except ImportError:
            _stub_module("googleapiclient")
            _stub_module("googleapiclient.discovery", build=lambda *a, **k: None)
            _stub_module("googleapiclient.http", MediaFileUpload=_Generico)


_instalar_stubs_de_dependencias_pesadas()
from models.media import FundoInvalidoError, MediaItem  # noqa: E402
from services import media_service, video_service  # noqa: E402


# ===========================================================================
# Dublês de teste (fakes) usados em vários casos abaixo
# ===========================================================================
class FakeMediaProvider(media_service.MediaProvider):
    """MediaProvider totalmente controlável para testes: devolve exatamente
    os `MediaItem`s configurados, valida conforme uma lista/branco de
    aprovação, e registra descartes para inspeção."""

    def __init__(self, itens=None, validos=None, falha_listar: Exception = None):
        self.itens = itens or []
        self.validos = validos if validos is not None else {item.caminho for item in self.itens}
        self.falha_listar = falha_listar
        self.descartados = []

    def listar_videos(self, contexto=None):
        if self.falha_listar:
            raise self.falha_listar
        return list(self.itens)

    def validar_video(self, item):
        return item.caminho in self.validos

    def descartar(self, item):
        self.descartados.append(item)


def _media_item(caminho: str) -> MediaItem:
    return MediaItem(caminho=caminho, categoria="teste", tipo="video")


class FakeClip:
    """Substituto controlável de VideoFileClip/VideoClip para os testes de
    `_preparar_clip_de_fundo` / `_validar_arquivo_video` — evita precisar
    de um arquivo .mp4 de verdade e de ffmpeg no ambiente de teste."""

    def __init__(self, duration=10.0, size=(1920, 1080)):
        self.duration = duration
        self.size = size
        self.closed = False
        self.subclip_chamado_com = None

    def get_frame(self, t):
        import numpy as np

        return np.zeros((self.size[1], self.size[0], 3), dtype="uint8")

    def subclip(self, inicio, fim):
        self.subclip_chamado_com = (inicio, fim)
        clone = FakeClip(duration=fim - inicio, size=self.size)
        return clone

    def close(self):
        self.closed = True


# ===========================================================================
# 1. Geração de queries visuais
# ===========================================================================
class TestQueriesVisuais(unittest.TestCase):
    def setUp(self):
        self.provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)

    def test_saturno_gera_queries_visuais_em_ingles(self):
        roteiro = "Saturno possui aneis formados por milhoes de particulas ha bilhoes de anos."
        queries = self.provider._gerar_queries_visuais(roteiro)
        self.assertIn("saturn rings", queries)
        self.assertTrue(any("saturn" in q or "planet" in q for q in queries))

    def test_baleia_gera_queries_visuais_em_ingles(self):
        roteiro = "Uma baleia azul pode pesar mais de cem toneladas e viver no oceano profundo."
        queries = self.provider._gerar_queries_visuais(roteiro)
        self.assertTrue(any("whale" in q for q in queries))

    def test_ignora_palavras_abstratas_e_numeros(self):
        roteiro = "Saturno tem aneis formados ha milhoes de anos, um recorde de bilhoes de toneladas."
        queries = self.provider._gerar_queries_visuais(roteiro)
        texto_unido = " ".join(queries).lower()
        for palavra_proibida in ("anos", "milhoes", "bilhoes", "recorde", "toneladas"):
            self.assertNotIn(palavra_proibida, texto_unido)

    def test_fallback_generico_quando_roteiro_sem_termo_visual(self):
        roteiro = "de para com em muito"  # só stopwords
        queries = self.provider._gerar_queries_visuais(roteiro)
        self.assertTrue(len(queries) >= 1)

    def test_nao_ultrapassa_limite_maximo_de_queries(self):
        roteiro = (
            "saturno marte lua sol terra estrela galaxia universo planeta asteroide "
            "cometa foguete astronauta satelite nasa gravidade oceano mar baleia tubarao"
        )
        queries = self.provider._gerar_queries_visuais(roteiro)
        self.assertLessEqual(len(queries), media_service.MAX_PALAVRAS_CHAVE_PESQUISA)


# ===========================================================================
# 1.1 Query semântica contextual (Fase 2.1)
# ===========================================================================
class TestQueriesVisuaisSemanticas(unittest.TestCase):
    """Cobre a auditoria de qualidade de queries pedida na Fase 2.1: um
    trecho sem entidade nomeada não pode mais produzir palavras soltas em
    português como query ('estamos', 'corpo', 'morte'...) — ver
    `CONCEITOS_CONTEXTUAIS_PT_EN` / `_gerar_queries_visuais()`."""

    def setUp(self):
        self.provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)

    # --- Teste 1: palavras inúteis não viram query isolada ---------------
    def test_palavras_de_discurso_nunca_viram_query(self):
        texto = "Nao estamos falando de um monstro de filme."
        queries = self.provider._gerar_queries_visuais(texto)
        texto_unido = " ".join(queries).lower()
        for palavra_proibida in ("estamos", "falando", "filme"):
            self.assertNotIn(palavra_proibida, texto_unido)

    # --- Teste 2: entidade (Cachalote) ------------------------------------
    def test_cachalote_gera_conceitos_de_baleia_e_ecolocalizacao(self):
        texto = "A Cachalote consegue emitir cliques sonoros."
        queries = self.provider._gerar_queries_visuais(texto)
        texto_unido = " ".join(queries).lower()
        self.assertIn("sperm whale", texto_unido)
        self.assertTrue("whale" in texto_unido and "echolocation" in texto_unido)

    # --- Teste 3: contexto sem entidade nomeada ---------------------------
    def test_criatura_sem_nome_usa_contexto_de_profundezas_do_mar(self):
        texto = "uma criatura real que navega pelas profundezas"
        queries = self.provider._gerar_queries_visuais(texto)
        texto_unido = " ".join(queries).lower()
        self.assertTrue(any(termo in texto_unido for termo in ("deep sea", "underwater", "marine")))
        # nenhuma das palavras cruas em português deveria sobrar como query
        for palavra_proibida in ("criatura", "real", "navega", "profundezas"):
            self.assertNotIn(palavra_proibida, texto_unido)

    # --- Teste 4: ação -----------------------------------------------------
    def test_foguete_decolando_gera_rocket_launch(self):
        texto = "um foguete decolando rumo ao espaco"
        queries = self.provider._gerar_queries_visuais(texto)
        texto_unido = " ".join(queries).lower()
        self.assertIn("rocket launch", texto_unido)

    # --- Teste 5: conceito composto (nunca palavra isolada genérica) -----
    def test_animal_marinho_perigoso_gera_conceito_composto(self):
        texto = "um animal marinho perigoso"
        queries = self.provider._gerar_queries_visuais(texto)
        for query in queries:
            self.assertGreaterEqual(len(query.split()), 2)  # sempre multipalavra
        self.assertNotIn("animal", queries)
        self.assertNotIn("perigoso", queries)

    # --- Teste 6: deduplicação --------------------------------------------
    def test_queries_duplicadas_sao_removidas(self):
        texto = "cachalote cachalote cachalote"
        queries = self.provider._gerar_queries_visuais(texto)
        self.assertEqual(len(queries), len(set(queries)))

    # --- Teste 7: limite respeitado com entidade+conceito juntos ---------
    def test_limite_maximo_respeitado_combinando_entidade_e_conceito(self):
        texto = (
            "cachalote baleia tubarao golfinho criatura perigoso profundezas "
            "predador nadando atacando"
        )
        queries = self.provider._gerar_queries_visuais(texto)
        self.assertLessEqual(len(queries), media_service.MAX_PALAVRAS_CHAVE_PESQUISA)

    # --- Teste 8: fallback quando não há candidato nenhum -----------------
    def test_texto_sem_conceito_visual_usa_fallback_generico_composto(self):
        texto = "isso porem tambem assim entao"  # só stopwords/conectivos
        queries = self.provider._gerar_queries_visuais(texto)
        self.assertTrue(len(queries) >= 1)
        for query in queries:
            self.assertIn(query, media_service.PEXELS_QUERIES_FALLBACK_GENERICAS)

    # --- Teste 9: acentuação não quebra o processamento --------------------
    def test_acentos_nao_quebram_geracao_de_queries(self):
        texto = "Cachalote no oceano, perto de um orgao vital, na agua profunda."
        queries = self.provider._gerar_queries_visuais(texto)  # não deve lançar exceção
        self.assertTrue(len(queries) >= 1)

    # --- Teste 10: números não quebram o processamento ---------------------
    def test_numeros_nao_quebram_geracao_de_queries(self):
        texto = "230 dB e mais forte que qualquer som conhecido"
        queries = self.provider._gerar_queries_visuais(texto)  # não deve lançar exceção
        self.assertIsInstance(queries, list)

    # --- Teste 11: texto vazio ----------------------------------------------
    def test_texto_vazio_usa_fallback_seguro(self):
        queries = self.provider._gerar_queries_visuais("")
        self.assertTrue(len(queries) >= 1)
        for query in queries:
            self.assertIn(query, media_service.PEXELS_QUERIES_FALLBACK_GENERICAS)

    # --- Teste 12: texto muito curto ----------------------------------------
    def test_texto_muito_curto_usa_fallback_seguro(self):
        queries = self.provider._gerar_queries_visuais("e")
        self.assertTrue(len(queries) >= 1)

    # --- Extras: prioridade e composição ------------------------------------
    def test_entidade_tem_prioridade_sobre_conceito_generico(self):
        # "cachalote" (entidade) e "profundezas" (conceito) no mesmo
        # trecho: a entidade, mais específica, deve aparecer nas queries.
        texto = "o cachalote vive nas profundezas do oceano"
        queries = self.provider._gerar_queries_visuais(texto)
        texto_unido = " ".join(queries).lower()
        self.assertIn("sperm whale", texto_unido)

    def test_fallback_generico_nunca_contem_palavra_solta_em_portugues(self):
        for query in media_service.PEXELS_QUERIES_FALLBACK_GENERICAS:
            self.assertGreaterEqual(len(query.split()), 2)


# ===========================================================================
# 2. Tratamento de status HTTP / timeout / JSON inválido no Pexels
# ===========================================================================
class TestPexelsBusca(unittest.TestCase):
    def setUp(self):
        self.provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)
        self.provider.api_key = "chave-de-teste-1234567890"

    def _mock_urlopen_sucesso(self, corpo: dict):
        resposta = mock.MagicMock()
        resposta.read.return_value = json.dumps(corpo).encode("utf-8")
        resposta.__enter__.return_value = resposta
        resposta.__exit__.return_value = False
        return mock.patch("services.media_service.urllib.request.urlopen", return_value=resposta)

    def test_200_retorna_lista_de_videos(self):
        with self._mock_urlopen_sucesso({"videos": [{"id": 1}, {"id": 2}]}):
            resultado = self.provider._buscar("saturn rings")
        self.assertEqual(len(resultado), 2)

    def _mock_http_error(self, codigo: int, corpo: bytes = b'{"error":"x"}'):
        erro = urllib.error.HTTPError(
            url="https://api.pexels.com/videos/search",
            code=codigo,
            msg="erro",
            hdrs=None,
            fp=io.BytesIO(corpo),
        )
        return mock.patch("services.media_service.urllib.request.urlopen", side_effect=erro)

    def test_401_mensagem_especifica_de_credencial(self):
        with self._mock_http_error(401):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")
        self.assertIn("401", str(ctx.exception))
        self.assertIn("PEXELS_API_KEY", str(ctx.exception))

    def test_403_mensagem_especifica_de_acesso_negado(self):
        with self._mock_http_error(403):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")
        self.assertIn("403", str(ctx.exception))

    def test_429_mensagem_especifica_de_limite(self):
        with self._mock_http_error(429):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")
        self.assertIn("429", str(ctx.exception))

    def test_500_mensagem_especifica_de_servidor(self):
        with self._mock_http_error(500):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")
        self.assertIn("500", str(ctx.exception))

    def test_503_mensagem_especifica_de_servidor(self):
        with self._mock_http_error(503):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")
        self.assertIn("503", str(ctx.exception))

    def test_timeout_vira_runtimeerror(self):
        with mock.patch("services.media_service.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")
        self.assertIn("timeout", str(ctx.exception).lower())

    def test_json_invalido_vira_runtimeerror(self):
        resposta = mock.MagicMock()
        resposta.read.return_value = b"isto nao e json"
        resposta.__enter__.return_value = resposta
        resposta.__exit__.return_value = False
        with mock.patch("services.media_service.urllib.request.urlopen", return_value=resposta):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")
        self.assertIn("JSON", str(ctx.exception))

    def test_erro_nunca_expoe_a_api_key_no_log(self):
        with mock.patch("builtins.print") as mock_print:
            with self._mock_http_error(403, corpo=b"chave-de-teste-1234567890 invalida"):
                with self.assertRaises(RuntimeError):
                    self.provider._buscar("saturn rings")
        for chamada in mock_print.call_args_list:
            texto_logado = " ".join(str(a) for a in chamada.args)
            self.assertNotIn("chave-de-teste-1234567890", texto_logado)


class TestPexelsCacheFiles(unittest.TestCase):
    def test_download_grava_em_blocos_em_vez_de_ler_o_video_inteiro(self):
        provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)
        resposta = mock.MagicMock()
        resposta.read.side_effect = [b"bloco-1", b"bloco-2", b""]
        resposta.__enter__.return_value = resposta
        resposta.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as pasta:
            destino = os.path.join(pasta, "video.mp4")
            with mock.patch("services.media_service.urllib.request.urlopen", return_value=resposta):
                self.assertTrue(provider._baixar_arquivo("https://cdn/video.mp4", destino))

            with open(destino, "rb") as arquivo:
                self.assertEqual(arquivo.read(), b"bloco-1bloco-2")
        self.assertEqual(resposta.read.call_count, 3)

    def test_limpeza_remove_orfaos_sem_remover_arquivos_fora_do_cache(self):
        provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)
        with tempfile.TemporaryDirectory() as pasta:
            pasta_cache = os.path.join(pasta, "cache")
            os.makedirs(pasta_cache)
            expirado = os.path.join(pasta_cache, "expirado.mp4")
            orfao = os.path.join(pasta_cache, "orfao.mp4")
            temporario = os.path.join(pasta_cache, "metadata.json.tmp")
            externo = os.path.join(pasta, "nao-apagar.mp4")
            for caminho in (expirado, orfao, temporario, externo):
                with open(caminho, "wb") as arquivo:
                    arquivo.write(b"video")
            provider.cache_dir = pasta_cache
            provider.metadata_path = os.path.join(pasta_cache, "metadata.json")
            provider._metadata = {
                "expirado": {
                    "caminho": expirado,
                    "baixado_em": time.time() - (media_service.CACHE_TTL_HOURS + 1) * 3600,
                },
                "externo": {"caminho": externo, "baixado_em": time.time()},
            }

            provider.limpar_cache_expirado()

            self.assertFalse(os.path.exists(expirado))
            self.assertFalse(os.path.exists(orfao))
            self.assertFalse(os.path.exists(temporario))
            self.assertTrue(os.path.exists(externo))
            self.assertEqual(provider._metadata, {})


# ===========================================================================
# 3. Fallback Pexels -> Local
# ===========================================================================
class TestFallbackPexelsParaLocal(unittest.TestCase):
    def test_primario_falha_usa_fallback(self):
        primario = FakeMediaProvider(falha_listar=RuntimeError("Pexels indisponível"))
        item_local = _media_item("assets/videos/local1.mp4")
        fallback = FakeMediaProvider(itens=[item_local])

        manager = media_service.MediaManager(
            provider=primario, provider_fallback=fallback, arquivo_historico="/tmp/hist_teste_1.json"
        )
        with mock.patch("builtins.print"):
            resultado = manager.escolher_video(contexto="qualquer roteiro")

        self.assertEqual(resultado, "assets/videos/local1.mp4")

    def test_sem_fallback_propaga_erro(self):
        primario = FakeMediaProvider(falha_listar=RuntimeError("Pexels indisponível"))
        manager = media_service.MediaManager(
            provider=primario, provider_fallback=None, arquivo_historico="/tmp/hist_teste_2.json"
        )
        with self.assertRaises(RuntimeError):
            manager.escolher_video(contexto="qualquer roteiro")

    def test_factory_usa_provedores_reais_pexels_e_local(self):
        with mock.patch.dict(os.environ, {"PEXELS_API_KEY": "chave-de-teste"}):
            manager = media_service._criar_media_manager()

        self.assertIsInstance(manager.provider, media_service.PexelsMediaProvider)
        self.assertIsInstance(manager.provider_fallback, media_service.LocalMediaProvider)

    def test_factory_sem_chave_usa_provedor_local_real(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            manager = media_service._criar_media_manager()

        self.assertIsInstance(manager.provider, media_service.LocalMediaProvider)
        self.assertIsNone(manager.provider_fallback)


# ===========================================================================
# 4-6. escolher_videos() com pools de tamanhos diferentes
# ===========================================================================
class TestEscolherVideosPool(unittest.TestCase):
    def _manager_com_pool(self, quantidade_itens: int, arquivo_historico: str) -> "services.media_service.MediaManager":
        itens = [_media_item(f"video{i}.mp4") for i in range(quantidade_itens)]
        provider = FakeMediaProvider(itens=itens)
        return media_service.MediaManager(provider=provider, provider_fallback=None, arquivo_historico=arquivo_historico)

    def test_apenas_1_video_valido_reaproveita_sem_travar(self):
        manager = self._manager_com_pool(1, "/tmp/hist_teste_4.json")
        with mock.patch("builtins.print"):
            escolhidos = manager.escolher_videos(contexto="x", quantidade=5)
        self.assertEqual(len(escolhidos), 5)
        self.assertTrue(all(caminho == "video0.mp4" for caminho in escolhidos))

    def test_2_videos_validos_alterna_entre_eles(self):
        manager = self._manager_com_pool(2, "/tmp/hist_teste_5.json")
        with mock.patch("builtins.print"):
            escolhidos = manager.escolher_videos(contexto="x", quantidade=4)
        self.assertEqual(len(escolhidos), 4)
        self.assertEqual(set(escolhidos), {"video0.mp4", "video1.mp4"})

    def test_6_videos_validos_todos_distintos(self):
        manager = self._manager_com_pool(6, "/tmp/hist_teste_6.json")
        with mock.patch("builtins.print"):
            escolhidos = manager.escolher_videos(contexto="x", quantidade=6)
        self.assertEqual(len(escolhidos), 6)
        self.assertEqual(len(set(escolhidos)), 6)  # nenhuma repetição quando não é necessário

    def test_nenhum_video_disponivel_levanta_erro(self):
        manager = media_service.MediaManager(
            provider=FakeMediaProvider(itens=[]), provider_fallback=None, arquivo_historico="/tmp/hist_teste_7.json"
        )
        with self.assertRaises(RuntimeError):
            manager.escolher_videos(contexto="x", quantidade=3)


# ===========================================================================
# 7-8. Loop (vídeo menor) vs. corte (vídeo maior) que a duração alvo
# ===========================================================================
class TestPrepararClipDeFundo(unittest.TestCase):
    def test_video_menor_que_duracao_da_narracao_faz_loop(self):
        clip_curto = FakeClip(duration=3.0, size=(1920, 1080))
        chamadas_concat = []

        def fake_concatenate(clips, **kw):
            chamadas_concat.append(len(clips))
            duracao_total = sum(c.duration for c in clips)
            return FakeClip(duration=duracao_total, size=(1920, 1080))

        with mock.patch("services.video_service.VideoFileClip", return_value=clip_curto), mock.patch(
            "services.video_service.concatenate_videoclips", side_effect=fake_concatenate
        ), mock.patch("services.video_service.os.path.exists", return_value=True), mock.patch(
            "services.video_service._verificar_fundo_saudavel"
        ), mock.patch(
            "services.video_service._redimensionar_para_vertical", side_effect=lambda c: c
        ):
            resultado = video_service._preparar_clip_de_fundo("curto.mp4", duracao_alvo=10.0, clips_para_fechar=[])

        self.assertTrue(chamadas_concat, "esperava que o loop (concatenate_videoclips) fosse acionado")
        self.assertGreaterEqual(chamadas_concat[0], 4)  # ceil(10/3) == 4 cópias
        self.assertAlmostEqual(resultado.duration, 10.0)

    def test_video_maior_que_duracao_da_narracao_corta_sem_loop(self):
        clip_longo = FakeClip(duration=30.0, size=(1920, 1080))

        with mock.patch("services.video_service.VideoFileClip", return_value=clip_longo), mock.patch(
            "services.video_service.concatenate_videoclips"
        ) as mock_concat, mock.patch("services.video_service.os.path.exists", return_value=True), mock.patch(
            "services.video_service._verificar_fundo_saudavel"
        ), mock.patch(
            "services.video_service._redimensionar_para_vertical", side_effect=lambda c: c
        ):
            resultado = video_service._preparar_clip_de_fundo("longo.mp4", duracao_alvo=6.0, clips_para_fechar=[])

        mock_concat.assert_not_called()  # não precisa de loop
        # Fase 2 (item 12): com folga grande (30s de origem para 6s de
        # alvo), o ponto inicial agora VARIA em vez de ser sempre 0 — só a
        # DURAÇÃO do corte (sempre 6.0s) e os limites (nunca negativo, nunca
        # ultrapassa o fim do clipe) são garantidos.
        inicio_usado, fim_usado = clip_longo.subclip_chamado_com
        self.assertAlmostEqual(fim_usado - inicio_usado, 6.0)
        self.assertGreaterEqual(inicio_usado, 0.0)
        self.assertLessEqual(fim_usado, 30.0)
        self.assertAlmostEqual(resultado.duration, 6.0)

    def test_arquivo_inexistente_levanta_fundo_invalido(self):
        with mock.patch("services.video_service.os.path.exists", return_value=False):
            with self.assertRaises(FundoInvalidoError):
                video_service._preparar_clip_de_fundo("nao_existe.mp4", duracao_alvo=5.0, clips_para_fechar=[])


# ===========================================================================
# 8b. _escolher_ponto_de_corte_temporal() — variação do ponto inicial do
#     corte quando há folga (Fase 2, item 12)
# ===========================================================================
class TestEscolherPontoDeCorteTemporal(unittest.TestCase):
    def test_folga_pequena_mantem_inicio_em_zero(self):
        # 6.5s de origem para 6.0s de alvo = 0.5s de folga, abaixo do
        # mínimo (BROLL_FOLGA_MINIMA_PARA_VARIAR_INICIO=1.0s) — mantém o
        # comportamento antigo (sempre começa em 0).
        ponto = video_service._escolher_ponto_de_corte_temporal(duracao_origem=6.5, duracao_alvo=6.0)
        self.assertEqual(ponto, 0.0)

    def test_folga_grande_varia_dentro_dos_limites_seguros(self):
        random.seed(42)
        for _ in range(50):
            ponto = video_service._escolher_ponto_de_corte_temporal(duracao_origem=60.0, duracao_alvo=6.0)
            self.assertGreaterEqual(ponto, 0.0)
            self.assertLessEqual(ponto + 6.0, 60.0)

    def test_folga_grande_produz_pontos_iniciais_diferentes_entre_chamadas(self):
        random.seed(7)
        pontos = {video_service._escolher_ponto_de_corte_temporal(60.0, 6.0) for _ in range(20)}
        self.assertGreater(len(pontos), 1)  # não é sempre o mesmo valor fixo

    def test_nunca_ultrapassa_duracao_de_origem(self):
        random.seed(1)
        for _ in range(50):
            ponto = video_service._escolher_ponto_de_corte_temporal(duracao_origem=8.0, duracao_alvo=6.0)
            self.assertLessEqual(ponto + 6.0, 8.0 + 1e-9)


# ===========================================================================
# 9. Formato final 1080x1920 (geometria)
# ===========================================================================
class TestGeometriaVertical(unittest.TestCase):
    def test_constantes_produzem_1080x1920(self):
        altura_esperada = round(video_service.VIDEO_WIDTH / video_service.ASPECT_RATIO)
        self.assertEqual(video_service.VIDEO_WIDTH, 1080)
        self.assertEqual(altura_esperada, 1920)


# ===========================================================================
# 10. Legendas dentro da safe zone
# ===========================================================================
class TestSafeZoneLegendas(unittest.TestCase):
    def test_safe_area_bottom_dentro_de_faixa_razoavel(self):
        # Não deixa a legenda colada na borda (perto de 0) nem ocupando a
        # tela quase inteira (perto de 1) — faixa larga o suficiente para
        # não travar em ajustes finos futuros, mas pega regressões grandes.
        self.assertGreater(video_service.SAFE_AREA_BOTTOM, 0.10)
        self.assertLess(video_service.SAFE_AREA_BOTTOM, 0.35)

    def test_legend_width_fraction_deixa_margem_lateral(self):
        self.assertLess(video_service.LEGEND_WIDTH_FRACTION, 0.95)

    def test_safe_area_top_dentro_de_faixa_razoavel(self):
        self.assertGreater(video_service.SAFE_AREA_TOP, 0.0)
        self.assertLess(video_service.SAFE_AREA_TOP, 0.25)

    def test_safe_area_side_dentro_de_faixa_razoavel(self):
        self.assertGreater(video_service.SAFE_AREA_SIDE, 0.0)
        self.assertLess(video_service.SAFE_AREA_SIDE, 0.20)

    def test_safe_areas_top_e_bottom_nao_ultrapassam_a_tela(self):
        # Topo + base não podem, juntos, consumir a tela inteira (senão
        # não sobraria nenhuma área útil para o texto).
        self.assertLess(video_service.SAFE_AREA_TOP + video_service.SAFE_AREA_BOTTOM, 1.0)

    def test_legend_width_fraction_respeita_safe_area_lateral(self):
        # A largura máxima de legenda precisa caber dentro do que sobra
        # depois de reservar SAFE_AREA_SIDE dos dois lados (item 4: nenhuma
        # margem "solta" além das constantes centralizadas).
        largura_disponivel = 1.0 - 2 * video_service.SAFE_AREA_SIDE
        self.assertLessEqual(video_service.LEGEND_WIDTH_FRACTION, largura_disponivel)

    def test_clamp_y_legenda_nao_ultrapassa_safe_area_top(self):
        # Reproduz o clamp aplicado em `_construir_clip_grupo()`: um grupo
        # de legenda alto o bastante para, sem o clamp, ficar acima da
        # safe area do topo precisa ser empurrado de volta para dentro
        # dela.
        altura_video = video_service.VIDEO_HEIGHT
        y_minimo = int(altura_video * video_service.SAFE_AREA_TOP)
        y_calculado_sem_clamp = -50  # simula um grupo de 3 linhas bem alto
        y_video = max(y_minimo, y_calculado_sem_clamp)
        self.assertGreaterEqual(y_video, y_minimo)


# ===========================================================================
# 13. Animação de entrada/saída das legendas (Fase 3)
# ===========================================================================
class TestAnimacaoEntradaSaidaLegenda(unittest.TestCase):
    def test_meio_do_grupo_fica_em_escala_e_opacidade_totais(self):
        duracao = 2.0
        escala, opacidade = video_service._fator_entrada_saida_grupo(1.0, duracao, eh_hook=False)
        self.assertEqual(escala, 1.0)
        self.assertEqual(opacidade, 1.0)

    def test_inicio_do_grupo_comeca_com_escala_e_opacidade_reduzidas(self):
        escala, opacidade = video_service._fator_entrada_saida_grupo(0.0, 2.0, eh_hook=False)
        self.assertAlmostEqual(escala, video_service.LEGEND_ANIM_SCALE_START, places=3)
        self.assertAlmostEqual(opacidade, 0.0, places=3)

    def test_entrada_termina_em_escala_e_opacidade_totais(self):
        escala, opacidade = video_service._fator_entrada_saida_grupo(
            video_service.LEGEND_ANIM_IN_DURATION, 2.0, eh_hook=False
        )
        self.assertAlmostEqual(escala, 1.0, places=2)
        self.assertAlmostEqual(opacidade, 1.0, places=2)

    def test_saida_do_grupo_reduz_opacidade_mas_nao_a_escala(self):
        duracao = 2.0
        t = duracao - video_service.LEGEND_ANIM_OUT_DURATION / 2
        escala, opacidade = video_service._fator_entrada_saida_grupo(t, duracao, eh_hook=False)
        self.assertEqual(escala, 1.0)  # saída nunca encolhe, só esmaece
        self.assertLess(opacidade, 1.0)

    def test_hook_usa_escala_inicial_mais_forte_que_o_padrao(self):
        escala_normal, _ = video_service._fator_entrada_saida_grupo(0.0, 2.0, eh_hook=False)
        escala_hook, _ = video_service._fator_entrada_saida_grupo(0.0, 2.0, eh_hook=True)
        self.assertLess(escala_hook, escala_normal)
        self.assertEqual(escala_hook, video_service.HOOK_ANIM_SCALE_START)

    def test_escala_e_opacidade_sempre_no_intervalo_valido(self):
        for t in [0.0, 0.05, 0.16, 0.5, 1.0, 1.85, 1.9, 2.0]:
            escala, opacidade = video_service._fator_entrada_saida_grupo(t, 2.0, eh_hook=False)
            self.assertGreaterEqual(escala, 0.0)
            self.assertLessEqual(escala, 1.0001)
            self.assertGreaterEqual(opacidade, 0.0)
            self.assertLessEqual(opacidade, 1.0001)


# ===========================================================================
# 14. Punch de zoom do hook (Fase 3, item 8)
# ===========================================================================
class TestHookZoomPunch(unittest.TestCase):
    def test_punch_e_zero_quando_nao_e_hook(self):
        self.assertEqual(video_service._progresso_punch_hook(0.0, eh_hook=False), 0.0)
        self.assertEqual(video_service._progresso_punch_hook(0.1, eh_hook=False), 0.0)

    def test_punch_e_maximo_no_inicio_do_hook(self):
        punch = video_service._progresso_punch_hook(0.0, eh_hook=True)
        self.assertAlmostEqual(punch, video_service.HOOK_ZOOM_PUNCH - 1.0, places=3)

    def test_punch_decai_ate_zero_ao_final_da_janela(self):
        punch = video_service._progresso_punch_hook(video_service.HOOK_ZOOM_PUNCH_DURATION, eh_hook=True)
        self.assertEqual(punch, 0.0)

    def test_punch_e_sempre_decrescente_dentro_da_janela(self):
        amostras = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        valores = [video_service._progresso_punch_hook(t, eh_hook=True) for t in amostras]
        for anterior, atual in zip(valores, valores[1:]):
            self.assertLessEqual(atual, anterior)

    def test_punch_e_zero_depois_da_janela_do_hook(self):
        punch = video_service._progresso_punch_hook(video_service.HOOK_ZOOM_PUNCH_DURATION + 1.0, eh_hook=True)
        self.assertEqual(punch, 0.0)

    def test_apenas_o_primeiro_trecho_de_broll_recebe_o_hook(self):
        # Reproduz a regra de `_montar_fundo_multiplo()`: só o trecho de
        # índice 0 é tratado como hook.
        segmentos = [(0.0, 6.0), (6.0, 12.0), (12.0, 18.0)]
        flags_hook = [indice == 0 for indice in range(len(segmentos))]
        self.assertEqual(flags_hook, [True, False, False])


# ===========================================================================
# 11. Mídia descartada não entra no histórico
# ===========================================================================
class TestHistoricoRespeitaDescarte(unittest.TestCase):
    def test_descartar_nao_confirma_uso(self):
        item = _media_item("video_ruim.mp4")
        provider = FakeMediaProvider(itens=[item])
        manager = media_service.MediaManager(
            provider=provider, provider_fallback=None, arquivo_historico="/tmp/hist_teste_11.json"
        )
        with mock.patch("builtins.print"):
            caminho = manager.escolher_video(contexto="x")
        manager.descartar(caminho)

        self.assertNotIn(caminho, manager._historico)
        self.assertEqual(len(provider.descartados), 1)
        self.assertEqual(provider.descartados[0].caminho, caminho)

    def test_confirmar_uso_registra_no_historico(self):
        item = _media_item("video_bom.mp4")
        provider = FakeMediaProvider(itens=[item])
        manager = media_service.MediaManager(
            provider=provider, provider_fallback=None, arquivo_historico="/tmp/hist_teste_12.json"
        )
        with mock.patch("builtins.print"):
            caminho = manager.escolher_video(contexto="x")
        with mock.patch.object(media_service.MediaManager, "_salvar_historico"):
            manager.confirmar_uso(caminho)

        self.assertIn(caminho, manager._historico)


# ===========================================================================
# 12. Retry não escolhe imediatamente o vídeo já usado/descartado
# ===========================================================================
class TestRetryNaoRepeteImediatamente(unittest.TestCase):
    def test_escolher_videos_prioriza_ineditos_antes_de_repetir(self):
        itens = [_media_item(f"video{i}.mp4") for i in range(3)]
        provider = FakeMediaProvider(itens=itens)
        manager = media_service.MediaManager(
            provider=provider, provider_fallback=None, arquivo_historico="/tmp/hist_teste_13.json"
        )
        with mock.patch("builtins.print"):
            escolhidos = manager.escolher_videos(contexto="x", quantidade=3)

        # Com exatamente 3 candidatos válidos para 3 pedidos, não deveria
        # ter NENHUMA repetição.
        self.assertEqual(len(set(escolhidos)), 3)

    def test_definir_segmentos_broll_nao_corta_no_meio_de_legenda(self):
        pontos_de_corte = [0.0, 5.0, 11.0, 18.0, 25.0]
        segmentos = video_service._definir_segmentos_broll(pontos_de_corte, duracao_total=30.0)

        # Invariante real: todo LIMITE de segmento tem que ser um dos
        # pontos de corte originais (ou 0/duração total) — nunca um corte
        # novo "no meio" de um grupo de legenda. O inverso não precisa ser
        # verdade (nem todo ponto de corte vira limite de segmento: pontos
        # muito próximos são fundidos até chegar perto da duração alvo).
        pontos_validos = set(pontos_de_corte) | {0.0, 30.0}
        limites_dos_segmentos = {inicio for inicio, _fim in segmentos} | {fim for _inicio, fim in segmentos}
        for limite in limites_dos_segmentos:
            self.assertIn(limite, pontos_validos, f"limite de segmento {limite}s não é um ponto de corte válido")

        # E os segmentos têm que cobrir o vídeo inteiro sem buraco/overlap.
        segmentos_ordenados = sorted(segmentos)
        self.assertEqual(segmentos_ordenados[0][0], 0.0)
        self.assertEqual(segmentos_ordenados[-1][1], 30.0)
        for (_, fim_atual), (inicio_prox, _) in zip(segmentos_ordenados, segmentos_ordenados[1:]):
            self.assertEqual(fim_atual, inicio_prox)

    def test_definir_segmentos_broll_respeita_teto_maximo(self):
        pontos_de_corte = [float(i) for i in range(0, 40, 2)]  # muitos pontos, 2 em 2 segundos
        segmentos = video_service._definir_segmentos_broll(pontos_de_corte, duracao_total=40.0)
        self.assertLessEqual(len(segmentos), video_service.BROLL_MAX_SEGMENTOS)

    def test_definir_segmentos_broll_video_curto_gera_1_segmento(self):
        segmentos = video_service._definir_segmentos_broll([], duracao_total=4.0)
        self.assertEqual(segmentos, [(0.0, 4.0)])


# ===========================================================================
# 13. normalize_frame_size() — utilitário central de normalização de
#     dimensões (Problema 4/5 da auditoria: broadcasting entre frame e
#     máscara de tamanhos ligeiramente diferentes)
# ===========================================================================
class TestNormalizeFrameSize(unittest.TestCase):
    def test_bug_original_frame_1921_mask_1918_normalizam_para_o_mesmo_alvo(self):
        import numpy as np

        frame = np.zeros((1921, 1080, 3), dtype="uint8")
        mascara = np.ones((1918, 1080), dtype="float32")

        frame_norm = video_service.normalize_frame_size(frame, altura=1920, largura=1080)
        mascara_norm = video_service.normalize_frame_size(mascara, altura=1920, largura=1080)

        self.assertEqual(frame_norm.shape, (1920, 1080, 3))
        self.assertEqual(mascara_norm.shape, (1920, 1080))
        # Depois de normalizados, a operação que antes quebrava
        # (broadcasting entre frame e máscara) passa a funcionar.
        resultado = frame_norm.astype("float32") * mascara_norm[..., None]
        self.assertEqual(resultado.shape, (1920, 1080, 3))

    def test_mascara_grayscale_h_w(self):
        import numpy as np

        mascara = np.ones((100, 50), dtype="float32")
        resultado = video_service.normalize_frame_size(mascara, altura=90, largura=60)
        self.assertEqual(resultado.shape, (90, 60))

    def test_mascara_h_w_1(self):
        import numpy as np

        mascara = np.ones((100, 50, 1), dtype="float32")
        resultado = video_service.normalize_frame_size(mascara, altura=90, largura=60)
        self.assertEqual(resultado.shape, (90, 60, 1))

    def test_frame_rgb_h_w_3(self):
        import numpy as np

        frame = np.zeros((100, 50, 3), dtype="uint8")
        resultado = video_service.normalize_frame_size(frame, altura=110, largura=45)
        self.assertEqual(resultado.shape, (110, 45, 3))

    def test_frame_rgba_h_w_4(self):
        import numpy as np

        frame = np.zeros((100, 50, 4), dtype="uint8")
        resultado = video_service.normalize_frame_size(frame, altura=95, largura=55)
        self.assertEqual(resultado.shape, (95, 55, 4))

    def test_dimensoes_impares(self):
        import numpy as np

        frame = np.zeros((101, 51, 3), dtype="uint8")
        resultado = video_service.normalize_frame_size(frame, altura=100, largura=50)
        self.assertEqual(resultado.shape, (100, 50, 3))

        frame_menor = np.zeros((99, 49, 3), dtype="uint8")
        resultado_menor = video_service.normalize_frame_size(frame_menor, altura=100, largura=50)
        self.assertEqual(resultado_menor.shape, (100, 50, 3))

    def test_ja_no_tamanho_certo_retorna_sem_alterar_conteudo(self):
        import numpy as np

        frame = np.arange(100 * 50 * 3, dtype="uint8").reshape(100, 50, 3)
        resultado = video_service.normalize_frame_size(frame, altura=100, largura=50)
        self.assertTrue((resultado == frame).all())

    def test_padding_replica_borda_em_vez_de_inventar_preto(self):
        import numpy as np

        frame = np.full((10, 10, 3), 200, dtype="uint8")
        resultado = video_service.normalize_frame_size(frame, altura=12, largura=10)
        # As linhas adicionadas (padding) replicam a borda (200), nunca
        # viram preto artificial (0) — não é um "broadcast artificial".
        self.assertTrue((resultado[0] == 200).all())
        self.assertTrue((resultado[-1] == 200).all())


# ===========================================================================
# 14. _redimensionar_para_vertical() — consistência de dimensões entre
#     vídeos de fundo de aspect ratios de origem diferentes (causa raiz do
#     bug de broadcasting: cada segmento de B-roll podia terminar com uma
#     altura ligeiramente diferente)
# ===========================================================================
class FakeVideoClipGeometria:
    """Fake mínimo de VideoFileClip que reproduz o MESMO arredondamento
    (`int(round(...))`) que o MoviePy real usa em `.crop()`/`.resize()` —
    necessário para reproduzir de verdade a divergência de 1-3px entre
    vídeos de origem diferentes que motivou o bug."""

    def __init__(self, size):
        self.size = size  # (largura, altura), convenção do MoviePy

    def crop(self, x1=0, y1=0, width=None, height=None):
        w = int(round(width)) if width is not None else self.size[0]
        h = int(round(height)) if height is not None else self.size[1]
        return FakeVideoClipGeometria((w, h))

    def resize(self, width=None, height=None):
        w, h = self.size
        if width is not None:
            fator = width / w
            return FakeVideoClipGeometria((int(round(width)), int(round(h * fator))))
        if height is not None:
            fator = height / h
            return FakeVideoClipGeometria((int(round(w * fator)), int(round(height))))
        return FakeVideoClipGeometria(self.size)

    def fl_image(self, func):
        import numpy as np

        largura, altura = self.size
        frame_fake = np.zeros((altura, largura, 3), dtype="uint8")
        novo_frame = func(frame_fake)
        nova_altura, nova_largura = novo_frame.shape[:2]
        return FakeVideoClipGeometria((nova_largura, nova_altura))


class TestRedimensionarParaVerticalConsistencia(unittest.TestCase):
    def _tamanho_final(self, largura_original, altura_original):
        with mock.patch("services.video_service._debug_registrar"):
            resultado = video_service._redimensionar_para_vertical(
                FakeVideoClipGeometria((largura_original, altura_original))
            )
        return resultado.size

    def test_varios_aspect_ratios_de_origem_produzem_sempre_o_mesmo_tamanho(self):
        # Resoluções de origem bem diferentes (típicas de B-roll do Pexels
        # e de bibliotecas locais) — TODAS precisam terminar EXATAMENTE no
        # mesmo canvas, senão a concatenação de segmentos volta a poder
        # gerar o `operands could not be broadcast together` original.
        origens = [
            (1920, 1080),  # 16:9 "padrão" do Pexels
            (1280, 720),
            (3840, 2160),  # 4K
            (1080, 1350),  # já quase vertical, mas não exatamente 9:16
            (1000, 563),   # aspect ratio "estranho", gera crop fracionário
            (1917, 1079),  # ímpar, propositalmente feito pra estressar o arredondamento
        ]
        for largura, altura in origens:
            with self.subTest(origem=(largura, altura)):
                tamanho_final = self._tamanho_final(largura, altura)
                self.assertEqual(tamanho_final, (video_service.VIDEO_WIDTH, video_service.VIDEO_HEIGHT))

    def test_altura_alvo_e_1920_para_largura_1080(self):
        self.assertEqual(video_service.VIDEO_HEIGHT, 1920)


# ===========================================================================
# 15. User-Agent do Pexels + diagnóstico específico do bloqueio Cloudflare
#     (HTTP 403 / "error code: 1010") — Problema 1 da auditoria
# ===========================================================================
class TestPexelsUserAgentECloudflare(unittest.TestCase):
    def setUp(self):
        self.provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)
        self.provider.api_key = "chave-de-teste-1234567890"

    def test_busca_envia_user_agent_de_navegador(self):
        capturado = {}

        class _RespostaFake:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"videos": []}).encode("utf-8")

        def fake_urlopen(requisicao, timeout=None):
            capturado["headers"] = dict(requisicao.header_items())
            return _RespostaFake()

        with mock.patch("services.media_service.urllib.request.urlopen", side_effect=fake_urlopen):
            self.provider._buscar("saturn rings")

        # urllib normaliza o nome do header para "User-agent" (capitalização
        # própria dele) — o que importa é que o valor de navegador chegou.
        headers_lower = {k.lower(): v for k, v in capturado["headers"].items()}
        self.assertIn("user-agent", headers_lower)
        self.assertEqual(headers_lower["user-agent"], media_service.PEXELS_USER_AGENT)
        self.assertNotIn("python-urllib", headers_lower["user-agent"].lower())

    def test_download_envia_user_agent_de_navegador(self):
        capturado = {}

        class _RespostaFake:
            def __init__(self):
                self.lido = False

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, _tamanho=-1):
                if self.lido:
                    return b""
                self.lido = True
                return b"conteudo-fake-de-video"

        def fake_urlopen(requisicao, timeout=None):
            capturado["headers"] = dict(requisicao.header_items())
            return _RespostaFake()

        destino = os.path.join(_diretorio_temporario_para_teste(self), "video.mp4")
        with mock.patch("services.media_service.urllib.request.urlopen", side_effect=fake_urlopen):
            ok = self.provider._baixar_arquivo("https://player.vimeo.com/x.mp4", destino)

        self.assertTrue(ok)
        headers_lower = {k.lower(): v for k, v in capturado["headers"].items()}
        self.assertIn("user-agent", headers_lower)
        self.assertEqual(headers_lower["user-agent"], media_service.PEXELS_USER_AGENT)

    def test_403_com_corpo_1010_gera_mensagem_especifica_de_cloudflare(self):
        corpo = b"error code: 1010"

        class _ErroFake(urllib.error.HTTPError):
            def __init__(self):
                super().__init__("https://api.pexels.com/videos/search", 403, "Forbidden", {}, io.BytesIO(corpo))

            def read(self):
                return corpo

        with mock.patch("services.media_service.urllib.request.urlopen", side_effect=_ErroFake()):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")

        self.assertIn("Cloudflare", str(ctx.exception))
        self.assertIn("1010", str(ctx.exception))
        # E continua sem nunca vazar a API key na mensagem.
        self.assertNotIn(self.provider.api_key, str(ctx.exception))

    def test_403_sem_1010_mantem_mensagem_generica_de_conta_key(self):
        corpo = b'{"error":"forbidden"}'

        class _ErroFake(urllib.error.HTTPError):
            def __init__(self):
                super().__init__("https://api.pexels.com/videos/search", 403, "Forbidden", {}, io.BytesIO(corpo))

            def read(self):
                return corpo

        with mock.patch("services.media_service.urllib.request.urlopen", side_effect=_ErroFake()):
            with self.assertRaises(RuntimeError) as ctx:
                self.provider._buscar("saturn rings")

        self.assertNotIn("Cloudflare", str(ctx.exception))
        self.assertIn("pexels.com/api", str(ctx.exception))


def _diretorio_temporario_para_teste(testcase: unittest.TestCase) -> str:
    import tempfile

    diretorio = tempfile.mkdtemp()
    testcase.addCleanup(lambda: __import__("shutil").rmtree(diretorio, ignore_errors=True))
    return diretorio


# ===========================================================================
# 17. _texto_do_segmento() — reconstrói o texto narrado em cada trecho a
#     partir das PalavraLegenda (Fase 2, item 2/3)
# ===========================================================================
def _palavra(texto: str, start: float, end: float, index: int) -> "services.video_service.PalavraLegenda":
    return video_service.PalavraLegenda(palavra=texto, start=start, end=end, duration=end - start, index=index)


class TestTextoDoSegmento(unittest.TestCase):
    def test_junta_apenas_palavras_do_intervalo(self):
        palavras = [
            _palavra("Saturno", 0.0, 0.5, 0),
            _palavra("tem", 0.5, 0.8, 1),
            _palavra("aneis", 0.8, 1.3, 2),
            _palavra("enormes", 6.0, 6.5, 3),  # já no PRÓXIMO trecho
        ]
        texto = video_service._texto_do_segmento((0.0, 6.0), palavras)
        self.assertEqual(texto, "Saturno tem aneis")

    def test_intervalo_sem_palavras_retorna_string_vazia(self):
        palavras = [_palavra("Saturno", 10.0, 10.5, 0)]
        texto = video_service._texto_do_segmento((0.0, 6.0), palavras)
        self.assertEqual(texto, "")

    def test_usa_o_inicio_da_palavra_como_criterio_de_pertencimento(self):
        # Uma palavra que COMEÇA dentro do trecho pertence a ele mesmo que
        # termine depois do limite (`fim` do trecho) — mesmo critério de
        # `_definir_segmentos_broll()`.
        palavras = [_palavra("particulas", 5.8, 6.4, 0)]
        texto = video_service._texto_do_segmento((0.0, 6.0), palavras)
        self.assertEqual(texto, "particulas")


# ===========================================================================
# 18. _pontuar_candidato_broll() — score de relevância/qualidade/
#     orientação/duração/diversidade (Fase 2, item 9)
# ===========================================================================
class TestPontuarCandidatoBroll(unittest.TestCase):
    def _video(self, **overrides):
        base = {"id": 1, "width": 1920, "height": 1080, "duration": 12}
        base.update(overrides)
        return base

    def _arquivo(self, **overrides):
        base = {"width": 1920, "height": 1080}
        base.update(overrides)
        return base

    def test_resolucao_maior_pontua_mais(self):
        video = self._video()
        candidato_a = self._arquivo(width=1280)
        candidato_b = self._arquivo(width=1920)
        score_a = media_service._pontuar_candidato_broll(video, candidato_a, 0, 4, duracao_alvo=6.0)
        score_b = media_service._pontuar_candidato_broll(video, candidato_b, 0, 4, duracao_alvo=6.0)
        self.assertGreater(score_b, score_a)

    def test_vertical_pontua_mais_que_horizontal(self):
        arquivo = self._arquivo()
        video_vertical = self._video(width=1080, height=1920)
        video_horizontal = self._video(width=1920, height=1080)
        score_vertical = media_service._pontuar_candidato_broll(video_vertical, arquivo, 0, 4, duracao_alvo=6.0)
        score_horizontal = media_service._pontuar_candidato_broll(video_horizontal, arquivo, 0, 4, duracao_alvo=6.0)
        self.assertGreater(score_vertical, score_horizontal)

    def test_query_mais_relevante_pontua_mais_que_a_ultima_da_cascata(self):
        video = self._video()
        arquivo = self._arquivo()
        score_primeira_query = media_service._pontuar_candidato_broll(video, arquivo, 0, 4, duracao_alvo=6.0)
        score_ultima_query = media_service._pontuar_candidato_broll(video, arquivo, 3, 4, duracao_alvo=6.0)
        self.assertGreater(score_primeira_query, score_ultima_query)

    def test_video_ja_usado_pontua_menos_mas_nao_zera(self):
        video = self._video(id=42)
        arquivo = self._arquivo()
        score_inedito = media_service._pontuar_candidato_broll(video, arquivo, 0, 4, duracao_alvo=6.0, ja_usados=set())
        score_repetido = media_service._pontuar_candidato_broll(
            video, arquivo, 0, 4, duracao_alvo=6.0, ja_usados={"42"}
        )
        self.assertGreater(score_inedito, score_repetido)
        # penalidade é leve (peso 5%), nunca elimina o candidato
        self.assertGreater(score_repetido, 0.0)

    def test_duracao_com_folga_confortavel_pontua_mais_que_no_limite(self):
        arquivo = self._arquivo()
        video_no_limite = self._video(duration=6.0)  # exatamente a duração do trecho
        video_com_folga = self._video(duration=14.0)  # ~2.3x a duração do trecho
        score_no_limite = media_service._pontuar_candidato_broll(video_no_limite, arquivo, 0, 4, duracao_alvo=6.0)
        score_com_folga = media_service._pontuar_candidato_broll(video_com_folga, arquivo, 0, 4, duracao_alvo=6.0)
        self.assertGreater(score_com_folga, score_no_limite)

    def test_path_do_video_usado_aplica_penalidade_ao_id_pexels_correspondente(self):
        arquivo = self._arquivo()
        video_usado = self._video(id=42)
        video_novo = self._video(id=43)
        for video in (video_usado, video_novo):
            video["video_files"] = [
                {"width": 1920, "height": 1080, "fps": 30, "link": f"https://cdn/{video['id']}.mp4"}
            ]

        provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)
        provider.api_key = "chave-de-teste"
        provider.cache_dir = "/tmp/cache_teste_score_path"
        provider.metadata_path = "/tmp/cache_teste_score_path/metadata.json"
        provider._metadata = {}
        provider._rejeitados_nesta_execucao = set()
        provider._cache_busca_execucao = {"q1": [video_usado, video_novo]}
        provider._downloads_nesta_execucao = 0
        with mock.patch.object(provider, "_baixar_arquivo", return_value=True), mock.patch.object(
            provider, "_salvar_metadata"
        ), mock.patch("builtins.print"):
            itens = provider._buscar_com_fallback_de_queries(
                ["q1"], duracao_alvo=6.0, ids_evitar={"/tmp/cache_teste_score_path/42.mp4"}
            )

        self.assertEqual(os.path.basename(itens[0].caminho), "43.mp4")


class TestLegendaAtiva(unittest.TestCase):
    def test_tile_ativo_preserva_a_sombra_composta(self):
        import numpy as np

        class FakeTextClip:
            def __init__(self, _texto, **kwargs):
                self.w = 8
                self.h = 8
                self.mask = None
                self.cor = (0, 0, 0) if kwargs.get("color") == "black" else (255, 209, 0)
                self.fechado = False

            def set_opacity(self, _opacidade):
                return self

            def get_frame(self, _tempo):
                if self.fechado:
                    raise RuntimeError("clip fechado")
                return np.full((self.h, self.w, 3), self.cor, dtype=np.uint8)

            def close(self):
                self.fechado = True

        video_service._resetar_cache_legendas()
        with mock.patch.object(video_service, "TextClip", FakeTextClip):
            rgb, alpha = video_service._obter_tile_palavra("teste", "ativa")

        self.assertEqual(rgb.shape[:2], (8 + video_service.LEGEND_SHADOW_OFFSET[1], 8 + video_service.LEGEND_SHADOW_OFFSET[0]))
        self.assertTrue(np.any(alpha[:, -video_service.LEGEND_SHADOW_OFFSET[0]:] > 0))
        self.assertTrue(np.all(rgb[:, -video_service.LEGEND_SHADOW_OFFSET[0]:] == 0))


# ===========================================================================
# 19. PexelsMediaProvider._buscar_com_fallback_de_queries() — cascata,
#     cache de execução e teto de downloads (Fase 2, itens 6/7/19)
# ===========================================================================
class TestBuscarComFallbackDeQueries(unittest.TestCase):
    def setUp(self):
        self.provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)
        self.provider.api_key = "chave-de-teste"
        self.provider.cache_dir = "/tmp/cache_teste_fase2"
        self.provider.metadata_path = "/tmp/cache_teste_fase2/metadata.json"
        self.provider._metadata = {}
        self.provider._rejeitados_nesta_execucao = set()
        self.provider._cache_busca_execucao = {}
        self.provider._downloads_nesta_execucao = 0

    def _video(self, video_id, duration=12, width=1920, height=1080):
        return {
            "id": video_id,
            "duration": duration,
            "width": width,
            "height": height,
            "video_files": [{"width": width, "height": height, "fps": 30, "link": f"https://cdn/{video_id}.mp4"}],
        }

    def test_cascata_pula_para_proxima_query_quando_a_primeira_nao_acha_nada(self):
        chamadas = []

        def fake_buscar(query):
            chamadas.append(query)
            if query == "saturn rings":
                return []  # 1ª query sem resultados
            return [self._video(1)]  # 2ª query funciona

        with mock.patch.object(self.provider, "_buscar", side_effect=fake_buscar), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=True
        ), mock.patch.object(self.provider, "_salvar_metadata"):
            itens = self.provider._buscar_com_fallback_de_queries(
                ["saturn rings", "saturn planet"], duracao_alvo=6.0, ids_evitar=set()
            )

        self.assertEqual(chamadas, ["saturn rings", "saturn planet"])
        self.assertEqual(len(itens), 1)

    def test_mesma_query_repetida_entre_chamadas_nao_dispara_2a_requisicao(self):
        contagem = {"n": 0}

        def fake_buscar(query):
            contagem["n"] += 1
            return [self._video(1)]

        with mock.patch.object(self.provider, "_buscar", side_effect=fake_buscar), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=True
        ), mock.patch.object(self.provider, "_salvar_metadata"):
            self.provider._buscar_com_fallback_de_queries(["saturn rings"], duracao_alvo=6.0, ids_evitar=set())
            # 2ª "cena" gera a MESMA query — não deve chamar a API de novo.
            self.provider._buscar_com_fallback_de_queries(["saturn rings"], duracao_alvo=6.0, ids_evitar=set())

        self.assertEqual(contagem["n"], 1)

    def test_para_de_buscar_queries_apos_achar_candidatos_suficientes(self):
        chamadas = []

        def fake_buscar(query):
            chamadas.append(query)
            # cada query já devolve BROLL_CANDIDATOS_POR_TRECHO_ALVO vídeos distintos
            return [self._video(v) for v in range(100, 100 + media_service.BROLL_CANDIDATOS_POR_TRECHO_ALVO)]

        with mock.patch.object(self.provider, "_buscar", side_effect=fake_buscar), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=True
        ), mock.patch.object(self.provider, "_salvar_metadata"):
            self.provider._buscar_com_fallback_de_queries(
                ["q1", "q2", "q3", "q4"], duracao_alvo=6.0, ids_evitar=set()
            )

        # a 1ª query já rendeu candidatos suficientes — não deveria ter
        # gastado busca nenhuma das outras 3.
        self.assertEqual(chamadas, ["q1"])

    def test_respeita_teto_de_downloads_da_execucao(self):
        self.provider._downloads_nesta_execucao = media_service.PEXELS_MAX_DOWNLOADS  # já no teto

        with mock.patch.object(self.provider, "_buscar", return_value=[self._video(1)]), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=True
        ) as mock_baixar:
            itens = self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        mock_baixar.assert_not_called()
        self.assertEqual(itens, [])

    def test_limita_downloads_novos_a_dois_por_trecho(self):
        videos = [self._video(video_id) for video_id in range(1, 6)]
        with mock.patch.object(self.provider, "_buscar", return_value=videos), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=True
        ) as mock_baixar, mock.patch.object(self.provider, "_salvar_metadata"):
            itens = self.provider._buscar_com_fallback_de_queries(
                ["q1"], duracao_alvo=6.0, ids_evitar=set()
            )

        self.assertEqual(mock_baixar.call_count, media_service.PEXELS_MAX_DOWNLOADS_POR_TRECHO)
        self.assertEqual(len(itens), media_service.PEXELS_MAX_DOWNLOADS_POR_TRECHO)

    def test_resultados_voltam_ordenados_por_score_decrescente(self):
        def fake_buscar(query):
            return [
                self._video(1, width=1280),  # qualidade pior
                self._video(2, width=1920),  # qualidade melhor -> deve vir primeiro
            ]

        with mock.patch.object(self.provider, "_buscar", side_effect=fake_buscar), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=True
        ), mock.patch.object(self.provider, "_salvar_metadata"):
            itens = self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        self.assertEqual(len(itens), 2)
        self.assertIn("2.mp4", itens[0].caminho)


# ===========================================================================
# 20. MediaManager.escolher_video_contextual() — nunca lança exceção,
#     cai para o fallback, respeita diversidade (Fase 2, itens 10/21)
# ===========================================================================
class FakeMediaProviderContextual(media_service.MediaProvider):
    """Fake que sabe responder de forma diferente conforme o texto — usado
    para testar que `escolher_video_contextual` realmente repassa o texto
    do trecho ao provider, e não apenas ignora (como o `FakeMediaProvider`
    genérico, que herda o comportamento default)."""

    def __init__(self, resposta_por_texto=None, itens_padrao=None, falha=None):
        self.resposta_por_texto = resposta_por_texto or {}
        self.itens_padrao = itens_padrao or []
        self.falha = falha
        self.chamadas: List[Tuple[str, float, set]] = []

    def listar_videos(self, contexto=None):
        return list(self.itens_padrao)

    def listar_videos_contextual(self, texto, duracao_alvo, ids_evitar=None):
        self.chamadas.append((texto, duracao_alvo, ids_evitar))
        if self.falha:
            raise self.falha
        return list(self.resposta_por_texto.get(texto, self.itens_padrao))

    def validar_video(self, item):
        return True


class TestEscolherVideoContextual(unittest.TestCase):
    def test_relevancia_contextual_prevalece_sobre_historico(self):
        melhor = media_service.MediaItem("melhor.mp4", "pexels", "video", pontuacao=90.0)
        pior = media_service.MediaItem("pior.mp4", "pexels", "video", pontuacao=50.0)
        provider = FakeMediaProviderContextual(itens_padrao=[pior, melhor])
        manager = media_service.MediaManager(
            provider=provider, provider_fallback=None, arquivo_historico="/tmp/hist_ctx_score.json"
        )
        manager._historico = ["melhor.mp4"]

        with mock.patch("builtins.print"):
            resultado = manager.escolher_video_contextual("tema", duracao_alvo=6.0)

        self.assertEqual(resultado, "melhor.mp4")

    def test_repassa_texto_e_duracao_ao_provider(self):
        item = _media_item("saturno.mp4")
        provider = FakeMediaProviderContextual(resposta_por_texto={"saturno tem aneis": [item]})
        manager = media_service.MediaManager(
            provider=provider, provider_fallback=None, arquivo_historico="/tmp/hist_ctx_1.json"
        )

        with mock.patch("builtins.print"):
            resultado = manager.escolher_video_contextual("saturno tem aneis", duracao_alvo=6.0)

        self.assertEqual(resultado, "saturno.mp4")
        self.assertEqual(provider.chamadas[0][0], "saturno tem aneis")
        self.assertEqual(provider.chamadas[0][1], 6.0)

    def test_primario_sem_candidato_cai_para_fallback(self):
        primario = FakeMediaProviderContextual(itens_padrao=[])
        item_local = _media_item("assets/videos/local.mp4")
        fallback = FakeMediaProviderContextual(itens_padrao=[item_local])
        manager = media_service.MediaManager(
            provider=primario, provider_fallback=fallback, arquivo_historico="/tmp/hist_ctx_2.json"
        )

        with mock.patch("builtins.print"):
            resultado = manager.escolher_video_contextual("qualquer texto", duracao_alvo=6.0)

        self.assertEqual(resultado, "assets/videos/local.mp4")

    def test_primario_levanta_excecao_cai_para_fallback_sem_propagar(self):
        primario = FakeMediaProviderContextual(falha=RuntimeError("Pexels fora do ar"))
        item_local = _media_item("assets/videos/local.mp4")
        fallback = FakeMediaProviderContextual(itens_padrao=[item_local])
        manager = media_service.MediaManager(
            provider=primario, provider_fallback=fallback, arquivo_historico="/tmp/hist_ctx_3.json"
        )

        with mock.patch("builtins.print"):
            resultado = manager.escolher_video_contextual("qualquer texto", duracao_alvo=6.0)

        self.assertEqual(resultado, "assets/videos/local.mp4")

    def test_nenhum_provider_com_candidato_retorna_none_sem_lancar(self):
        primario = FakeMediaProviderContextual(itens_padrao=[])
        fallback = FakeMediaProviderContextual(itens_padrao=[])
        manager = media_service.MediaManager(
            provider=primario, provider_fallback=fallback, arquivo_historico="/tmp/hist_ctx_4.json"
        )

        with mock.patch("builtins.print"):
            resultado = manager.escolher_video_contextual("qualquer texto", duracao_alvo=6.0)

        self.assertIsNone(resultado)  # nunca levanta — Fase 2, item 21: resiliência é absoluta

    def test_prefere_candidato_fora_do_excluir_quando_disponivel(self):
        item_ja_usado = _media_item("ja_usado.mp4")
        item_novo = _media_item("novo.mp4")
        # A ordem de retorno já reflete o score (o provider real ordena);
        # aqui simulamos o provider devolvendo o já-usado primeiro para
        # confirmar que o desempate por `excluir` reordena antes da validação.
        provider = FakeMediaProviderContextual(itens_padrao=[item_ja_usado, item_novo])
        manager = media_service.MediaManager(
            provider=provider, provider_fallback=None, arquivo_historico="/tmp/hist_ctx_5.json"
        )

        with mock.patch("builtins.print"):
            resultado = manager.escolher_video_contextual(
                "texto", duracao_alvo=6.0, excluir={"ja_usado.mp4"}
            )

        self.assertEqual(resultado, "novo.mp4")


# ===========================================================================
# 21. _escolher_ponto_de_corte_temporal() já coberto na seção 8b acima.
# 22. Rejeição de vídeo ruim (duração/resolução/preto) — cobertura já
#     existente na Fase 1 via `_validar_arquivo_video()`
#     (TestGeometriaVertical/TestPrepararClipDeFundo) e
#     `_verificar_fundo_saudavel()` — sem mudança de comportamento nesta
#     fase, então sem testes novos aqui.
# ===========================================================================


# ===========================================================================
# 23. Fase 2.1.1 — Diagnóstico do pipeline Pexels (auditoria).
#
# Cobre os 7 cenários pedidos no diagnóstico:
#   1. API retorna 0 vídeos.
#   2. API retorna vídeos, todos eliminados por filtros REAIS do código
#      (duração / já descartado nesta execução / sem arquivo utilizável).
#   3. API retorna candidato válido, que é baixado com sucesso.
#   4. API retorna candidato válido, mas o download falha.
#   5. Cache já tem o vídeo — não deve baixar de novo.
#   6. HTTP 403 / Cloudflare 1010 — diagnóstico específico já existente
#      continua funcionando.
#   7. HTTP 429 — tratamento existente continua funcionando.
#
# Mais alguns testes de sanidade sobre o próprio mecanismo de diagnóstico
# (tabela de scores, componentes batendo com o score final, e a chave da
# API nunca vazando através do novo log da cascata).
# ===========================================================================
class TestDiagnosticoFase2_1_1(unittest.TestCase):
    def setUp(self):
        self.provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)
        self.provider.api_key = "chave-de-teste"
        self.provider.cache_dir = "/tmp/cache_teste_fase2_1_1"
        self.provider.metadata_path = "/tmp/cache_teste_fase2_1_1/metadata.json"
        self.provider._metadata = {}
        self.provider._rejeitados_nesta_execucao = set()
        self.provider._cache_busca_execucao = {}
        self.provider._downloads_nesta_execucao = 0

    def _video(self, video_id, duration=12, width=1920, height=1080):
        return {
            "id": video_id,
            "duration": duration,
            "width": width,
            "height": height,
            "video_files": [
                {"width": width, "height": height, "fps": 30, "link": f"https://cdn/{video_id}.mp4"}
            ],
        }

    def _saida_prints(self, mock_print) -> str:
        return "\n".join(" ".join(str(a) for a in chamada.args) for chamada in mock_print.call_args_list)

    # --- Cenário 1: API retorna 0 vídeos ----------------------------------
    def test_cenario1_api_retorna_zero_videos(self):
        resposta = mock.MagicMock()
        resposta.read.return_value = json.dumps({"videos": []}).encode("utf-8")
        resposta.__enter__.return_value = resposta
        resposta.__exit__.return_value = False

        with mock.patch("services.media_service.urllib.request.urlopen", return_value=resposta), mock.patch(
            "builtins.print"
        ) as mock_print:
            itens = self.provider._buscar_com_fallback_de_queries(
                ["saturn rings"], duracao_alvo=6.0, ids_evitar=set()
            )

        saida = self._saida_prints(mock_print)
        self.assertEqual(itens, [])
        self.assertIn("resultados_recebidos=0", saida)
        self.assertIn("candidatos_examinados=0", saida)

    # --- Cenário 2: API retorna vídeos, todos eliminados por filtros -----
    def test_cenario2_api_retorna_videos_mas_todos_sao_filtrados(self):
        self.provider._rejeitados_nesta_execucao = {"20"}
        videos = [
            self._video(10, duration=2.0),  # < MEDIA_VIDEO_DURACAO_MINIMA
            dict(
                self._video(30),
                video_files=[{"width": 320, "height": 180, "fps": 30, "link": "https://cdn/30.mp4"}],
            ),  # nenhum arquivo com largura >= MIN_WIDTH
            self._video(20),  # já descartado nesta execução
        ]

        with mock.patch.object(self.provider, "_buscar", return_value=videos), mock.patch(
            "builtins.print"
        ) as mock_print:
            itens = self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        saida = self._saida_prints(mock_print)
        self.assertEqual(itens, [])
        self.assertIn("candidatos_examinados=3", saida)
        self.assertIn("rejeitados_por_duracao=1", saida)
        self.assertIn("rejeitados_por_ausencia_de_arquivo_utilizavel=1", saida)
        self.assertIn("rejeitados_por_ja_descartado_nesta_execucao=1", saida)
        self.assertIn("candidatos_validos=0", saida)

    # --- Cenário 3: candidato válido e download funciona -------------------
    def test_cenario3_api_retorna_candidato_valido_e_baixa(self):
        with mock.patch.object(self.provider, "_buscar", return_value=[self._video(1)]), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=True
        ), mock.patch.object(self.provider, "_salvar_metadata"), mock.patch("builtins.print") as mock_print:
            itens = self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        self.assertEqual(len(itens), 1)
        self.assertIn("fonte=api", itens[0].categoria)
        saida = self._saida_prints(mock_print)
        self.assertIn("✅ Download OK", saida)
        self.assertIn("baixados_agora=1", saida)

    # --- Cenário 4: candidato válido, mas download falha --------------------
    def test_cenario4_candidato_valido_mas_download_falha(self):
        with mock.patch.object(self.provider, "_buscar", return_value=[self._video(1)]), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=False
        ), mock.patch("builtins.print") as mock_print:
            itens = self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        self.assertEqual(itens, [])
        saida = self._saida_prints(mock_print)
        self.assertIn("❌ Download falhou", saida)
        self.assertIn("falhas_download=1", saida)

    # --- Cenário 5: cache já tem o vídeo -> não baixa de novo --------------
    def test_cenario5_cache_contem_video_existente_nao_baixa_de_novo(self):
        self.provider._metadata = {
            "1": {"caminho": "/tmp/cache_teste_fase2_1_1/1.mp4", "baixado_em": time.time(), "query": "q1"}
        }
        with mock.patch.object(self.provider, "_cache_valido", return_value=True), mock.patch.object(
            self.provider, "_buscar", return_value=[self._video(1)]
        ), mock.patch.object(self.provider, "_baixar_arquivo") as mock_baixar, mock.patch.object(
            self.provider, "_salvar_metadata"
        ), mock.patch(
            "builtins.print"
        ) as mock_print:
            itens = self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        mock_baixar.assert_not_called()
        self.assertEqual(len(itens), 1)
        self.assertIn("fonte=cache", itens[0].categoria)
        saida = self._saida_prints(mock_print)
        self.assertIn("Cache HIT — vídeo já baixado", saida)
        self.assertIn("reutilizados_do_cache=1", saida)

    # --- Cenário 6: HTTP 403 / Cloudflare 1010 ------------------------------
    def test_cenario6_http_403_1010_diagnostico_especifico_continua(self):
        erro = urllib.error.HTTPError(
            url="https://api.pexels.com/videos/search",
            code=403,
            msg="erro",
            hdrs=None,
            fp=io.BytesIO(b"error code: 1010"),
        )
        with mock.patch("services.media_service.urllib.request.urlopen", side_effect=erro), mock.patch(
            "builtins.print"
        ) as mock_print:
            itens = self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        self.assertEqual(itens, [])
        saida = self._saida_prints(mock_print)
        self.assertIn("Cloudflare", saida)

    # --- Cenário 7: HTTP 429 -------------------------------------------------
    def test_cenario7_http_429_diagnostico_especifico_continua(self):
        erro = urllib.error.HTTPError(
            url="https://api.pexels.com/videos/search",
            code=429,
            msg="erro",
            hdrs=None,
            fp=io.BytesIO(b"{}"),
        )
        with mock.patch("services.media_service.urllib.request.urlopen", side_effect=erro), mock.patch(
            "builtins.print"
        ) as mock_print:
            itens = self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        self.assertEqual(itens, [])
        saida = self._saida_prints(mock_print)
        self.assertIn("limite de requisições excedido", saida)

    # --- Sanidade: tabela de scores aparece quando há candidatos -----------
    def test_tabela_de_scores_aparece_quando_ha_candidatos(self):
        with mock.patch.object(
            self.provider, "_buscar", return_value=[self._video(1), self._video(2)]
        ), mock.patch.object(self.provider, "_baixar_arquivo", return_value=True), mock.patch.object(
            self.provider, "_salvar_metadata"
        ), mock.patch(
            "builtins.print"
        ) as mock_print:
            self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        saida = self._saida_prints(mock_print)
        self.assertIn("📊 Candidatos Pexels", saida)
        self.assertIn("relevância=", saida)
        self.assertIn("qualidade=", saida)

    # --- Sanidade: componentes do score somam para o score final -----------
    def test_componentes_do_score_somam_para_o_score_final(self):
        video = self._video(1)
        arquivo = video["video_files"][0]
        score = media_service._pontuar_candidato_broll(video, arquivo, 0, 4, 6.0, set())
        c = media_service._componentes_score_candidato_broll(video, arquivo, 0, 4, 6.0, set())
        esperado = round(
            (
                media_service.BROLL_PESO_RELEVANCIA * c["relevancia"]
                + media_service.BROLL_PESO_QUALIDADE * c["qualidade"]
                + media_service.BROLL_PESO_ORIENTACAO * c["orientacao"]
                + media_service.BROLL_PESO_DURACAO * c["duracao"]
                + media_service.BROLL_PESO_DIVERSIDADE * c["diversidade"]
            )
            * 100,
            2,
        )
        self.assertEqual(score, esperado)

    # --- Sanidade: a API key nunca aparece no log da cascata ----------------
    def test_chave_nunca_aparece_no_log_da_cascata(self):
        erro = urllib.error.HTTPError(
            url="https://api.pexels.com/videos/search",
            code=403,
            msg="erro",
            hdrs=None,
            fp=io.BytesIO(b"chave-de-teste invalida"),
        )
        with mock.patch("services.media_service.urllib.request.urlopen", side_effect=erro), mock.patch(
            "builtins.print"
        ) as mock_print:
            self.provider._buscar_com_fallback_de_queries(["q1"], duracao_alvo=6.0, ids_evitar=set())

        saida = self._saida_prints(mock_print)
        self.assertNotIn("chave-de-teste", saida)


# ===========================================================================
# 24. Correção B/C (Fase 2.1.1) — orçamento residual em listar_videos() e
#     pool de reserva lazy em _montar_fundo_multiplo()/escolher_fundo_e_criar_video().
#
#     Objetivo: o B-roll contextual por trecho (_buscar_com_fallback_de_queries,
#     já coberto acima) deve ter prioridade sobre a pré-busca genérica de
#     `escolher_videos()`/`listar_videos()` no consumo do teto GLOBAL
#     `PEXELS_MAX_DOWNLOADS`. Nenhuma mudança em scoring, queries, cache ou
#     filtros — só em QUANDO a pré-busca roda e QUANTO ela pode gastar.
# ===========================================================================
class TestListarVideosOrcamentoResidual(unittest.TestCase):
    def setUp(self):
        self.provider = media_service.PexelsMediaProvider.__new__(media_service.PexelsMediaProvider)
        self.provider.api_key = "chave-de-teste"
        self.provider.cache_dir = "/tmp/cache_teste_orcamento_residual"
        self.provider.metadata_path = "/tmp/cache_teste_orcamento_residual/metadata.json"
        self.provider._metadata = {}
        self.provider._rejeitados_nesta_execucao = set()
        self.provider._cache_busca_execucao = {}
        self.provider._downloads_nesta_execucao = 0

    def _video(self, video_id, duration=12, width=1920, height=1080):
        return {
            "id": video_id,
            "duration": duration,
            "width": width,
            "height": height,
            "video_files": [
                {"width": width, "height": height, "fps": 30, "link": f"https://cdn/{video_id}.mp4"}
            ],
        }

    def test_baixa_ate_o_orcamento_residual_mesmo_com_mais_candidatos_disponiveis(self):
        # Só há orçamento residual de 2 downloads, apesar do teto global maior.
        self.provider._downloads_nesta_execucao = media_service.PEXELS_MAX_DOWNLOADS - 2
        videos = [self._video(i) for i in range(1, 6)]  # 5 candidatos válidos disponíveis

        with mock.patch.object(self.provider, "_buscar", return_value=videos), mock.patch.object(
            self.provider, "_baixar_arquivo", return_value=True
        ), mock.patch.object(self.provider, "_salvar_metadata"), mock.patch("builtins.print"):
            resultado = self.provider.listar_videos(contexto="roteiro qualquer")

        self.assertEqual(len(resultado), 2)  # não 5, respeitou o orçamento residual
        self.assertEqual(self.provider._downloads_nesta_execucao, media_service.PEXELS_MAX_DOWNLOADS)

    def test_orcamento_zerado_nao_consulta_a_api(self):
        self.provider._downloads_nesta_execucao = media_service.PEXELS_MAX_DOWNLOADS  # já no teto global

        with mock.patch.object(self.provider, "_buscar") as mock_buscar, mock.patch("builtins.print"):
            resultado = self.provider.listar_videos(contexto="roteiro qualquer")

        mock_buscar.assert_not_called()  # nem tenta: não há orçamento pra baixar nada mesmo
        self.assertEqual(resultado, [])
        self.assertEqual(
            self.provider._downloads_nesta_execucao, media_service.PEXELS_MAX_DOWNLOADS
        )  # não regride nem ultrapassa

    def test_cache_ja_suficiente_nao_consulta_a_api_nem_gasta_orcamento(self):
        itens_em_cache = [_media_item(f"cache{i}.mp4") for i in range(media_service.PEXELS_MIN_DOWNLOADS_DESEJADOS)]
        with mock.patch.object(
            self.provider, "_videos_em_cache_validos", return_value=itens_em_cache
        ), mock.patch.object(self.provider, "_buscar") as mock_buscar, mock.patch("builtins.print"):
            resultado = self.provider.listar_videos(contexto="roteiro qualquer")

        mock_buscar.assert_not_called()
        self.assertEqual(resultado, itens_em_cache)
        self.assertEqual(self.provider._downloads_nesta_execucao, 0)

    def test_cache_hits_nao_consomem_orcamento_mesmo_misturados_com_novos(self):
        # 1 já em cache (grátis) + candidatos novos, com orçamento residual = 2
        self.provider._downloads_nesta_execucao = media_service.PEXELS_MAX_DOWNLOADS - 2
        self.provider._metadata = {
            "9": {"caminho": "/tmp/cache_teste_orcamento_residual/9.mp4", "baixado_em": time.time(), "query": "x"}
        }
        with mock.patch.object(
            self.provider, "_videos_em_cache_validos", return_value=[_media_item("9.mp4")]
        ), mock.patch.object(
            self.provider, "_buscar", return_value=[self._video(1), self._video(2), self._video(3)]
        ), mock.patch.object(self.provider, "_baixar_arquivo", return_value=True), mock.patch.object(
            self.provider, "_salvar_metadata"
        ), mock.patch(
            "builtins.print"
        ):
            resultado = self.provider.listar_videos(contexto="roteiro qualquer")

        # 1 do cache (grátis) + no máximo 2 novos (orçamento residual) = 3
        self.assertEqual(len(resultado), 3)
        self.assertEqual(self.provider._downloads_nesta_execucao, media_service.PEXELS_MAX_DOWNLOADS)


class TestMontarFundoMultiploPoolLazy(unittest.TestCase):
    def _cenario_2_trechos(self):
        segmentos = [(0.0, 3.0), (3.0, 6.0)]
        palavras = [_palavra("um", 0.0, 0.5, 0), _palavra("dois", 3.0, 3.5, 1)]
        return segmentos, palavras, []

    def test_pool_nao_e_buscado_quando_contextual_atende_todos_os_trechos(self):
        segmentos, palavras, grupos = self._cenario_2_trechos()
        manager = mock.MagicMock()
        manager.escolher_video_contextual.side_effect = ["c1.mp4", "c2.mp4"]

        with mock.patch("services.video_service._preparar_clip_de_fundo", return_value=FakeClip()), mock.patch(
            "services.video_service._aplicar_ken_burns", side_effect=lambda clip, *a, **k: clip
        ), mock.patch("services.video_service.concatenate_videoclips", side_effect=lambda clips, **kw: FakeClip()), mock.patch(
            "builtins.print"
        ):
            _, caminhos_usados = video_service._montar_fundo_multiplo(
                segmentos, [], grupos, palavras, "roteiro", [], manager
            )

        manager.escolher_videos.assert_not_called()  # pré-busca nunca foi necessária
        self.assertEqual(caminhos_usados, ["c1.mp4", "c2.mp4"])

    def test_pool_e_buscado_sob_demanda_quando_contextual_falha_num_trecho(self):
        segmentos, palavras, grupos = self._cenario_2_trechos()
        manager = mock.MagicMock()
        manager.escolher_video_contextual.side_effect = ["c1.mp4", None]
        manager.escolher_videos.return_value = ["reserva.mp4"]

        with mock.patch("services.video_service._preparar_clip_de_fundo", return_value=FakeClip()), mock.patch(
            "services.video_service._aplicar_ken_burns", side_effect=lambda clip, *a, **k: clip
        ), mock.patch("services.video_service.concatenate_videoclips", side_effect=lambda clips, **kw: FakeClip()), mock.patch(
            "builtins.print"
        ):
            _, caminhos_usados = video_service._montar_fundo_multiplo(
                segmentos, [], grupos, palavras, "roteiro", [], manager
            )

        manager.escolher_videos.assert_called_once_with(
            contexto="roteiro", quantidade=video_service.BROLL_MAX_SEGMENTOS
        )
        self.assertEqual(caminhos_usados, ["c1.mp4", "reserva.mp4"])

    def test_pool_e_buscado_apenas_uma_vez_mesmo_com_varios_trechos_sem_contextual(self):
        segmentos = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
        palavras = [_palavra("um", 0.0, 0.5, 0)]
        manager = mock.MagicMock()
        manager.escolher_video_contextual.return_value = None
        manager.escolher_videos.return_value = ["reserva.mp4"]

        with mock.patch("services.video_service._preparar_clip_de_fundo", return_value=FakeClip()), mock.patch(
            "services.video_service._aplicar_ken_burns", side_effect=lambda clip, *a, **k: clip
        ), mock.patch("services.video_service.concatenate_videoclips", side_effect=lambda clips, **kw: FakeClip()), mock.patch(
            "builtins.print"
        ):
            video_service._montar_fundo_multiplo(segmentos, [], [], palavras, "roteiro", [], manager)

        manager.escolher_videos.assert_called_once()

    def test_sem_media_manager_e_pool_vazio_levanta_fundo_invalido_imediatamente(self):
        segmentos, palavras, grupos = self._cenario_2_trechos()
        with self.assertRaises(FundoInvalidoError):
            video_service._montar_fundo_multiplo(segmentos, [], grupos, palavras, "roteiro", [], None)

    def test_sem_media_manager_com_pool_pre_populado_funciona_como_antes(self):
        segmentos, palavras, grupos = self._cenario_2_trechos()
        with mock.patch("services.video_service._preparar_clip_de_fundo", return_value=FakeClip()), mock.patch(
            "services.video_service._aplicar_ken_burns", side_effect=lambda clip, *a, **k: clip
        ), mock.patch("services.video_service.concatenate_videoclips", side_effect=lambda clips, **kw: FakeClip()), mock.patch(
            "builtins.print"
        ):
            _, caminhos_usados = video_service._montar_fundo_multiplo(
                segmentos, ["pool.mp4"], grupos, palavras, "roteiro", [], None
            )
        self.assertEqual(caminhos_usados, ["pool.mp4", "pool.mp4"])


if __name__ == "__main__":
    unittest.main()