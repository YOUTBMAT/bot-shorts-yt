"""Orquestração do pipeline automático de publicação de YouTube Shorts."""

import asyncio
import os
import sys
import time
from types import ModuleType
from typing import Dict, Optional

from services.media_service import _criar_media_manager
from services.script_service import gerar_roteiro_gemini
from services.tts_service import gerar_audio_edge_tts
from services.video_service import escolher_fundo_e_criar_video, validar_video_final
from services.youtube_service import fazer_upload_youtube

resource: Optional[ModuleType]
try:
    import resource as _resource
except ImportError:
    resource = None
else:
    resource = _resource

NARRACAO_PATH = "narracao.mp3"
LARGURA_LOG_TEMPOS = 22


def _formatar_linha_log(nome: str, status: str, duracao: float) -> str:
    pontos = max(3, LARGURA_LOG_TEMPOS - len(nome))
    return f"{nome} {'.' * pontos} {status} ({duracao:.2f}s)"


def _ram_aproximada_mb() -> float:
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _rodar_etapa(nome: str, tempos: Dict[str, float], func, *args, **kwargs):
    inicio = time.perf_counter()
    try:
        resultado = func(*args, **kwargs)
    except Exception as err:
        duracao = time.perf_counter() - inicio
        tempos[nome] = duracao
        print(_formatar_linha_log(nome, "FALHOU", duracao))
        mensagem_err = str(err)
        if mensagem_err.startswith("❌"):
            raise RuntimeError(f"❌ {nome} (via {mensagem_err[2:]})") from err
        raise RuntimeError(f"❌ {nome}: {err}") from err

    duracao = time.perf_counter() - inicio
    tempos[nome] = duracao
    print(_formatar_linha_log(nome, "OK", duracao))
    return resultado


def _log_tempos(tempos: Dict[str, float]) -> None:
    print("\n⏱️  Resumo de tempos:")
    for etapa, duracao in tempos.items():
        pontos = max(3, LARGURA_LOG_TEMPOS - len(etapa))
        print(f"{etapa} {'.' * pontos} {duracao:.2f}s")
    rotulo_ram = "💾 RAM aproximada (pico)"
    pontos = max(3, LARGURA_LOG_TEMPOS - len(rotulo_ram))
    print(f"{rotulo_ram} {'.' * pontos} {_ram_aproximada_mb():.0f} MB")


def main() -> None:
    """Executa em sequência roteiro, TTS, renderização e publicação."""
    inicio_total = time.perf_counter()
    tempos: Dict[str, float] = {}
    media_manager = None

    try:
        titulo, roteiro = _rodar_etapa("🧠 Gemini", tempos, gerar_roteiro_gemini)

        def gerar_audio():
            sucesso = asyncio.run(gerar_audio_edge_tts(roteiro, NARRACAO_PATH))
            if not sucesso:
                raise RuntimeError("❌ TTS: falha ao gerar a narração com Edge TTS.")
            return NARRACAO_PATH

        audio_path = _rodar_etapa("🎙 Narração (TTS)", tempos, gerar_audio)
        media_manager = _criar_media_manager()
        video_path = _rodar_etapa(
            "🎬 Render", tempos, escolher_fundo_e_criar_video, roteiro, audio_path, media_manager
        )
        if not validar_video_final(video_path):
            print("❌ CRÍTICO: o vídeo não passou nos critérios de qualidade; upload cancelado.")
            sys.exit(1)
        _rodar_etapa("🚀 Upload", tempos, fazer_upload_youtube, video_path, titulo, roteiro)
    except Exception as err:
        print(f"\n{err}")
        tempos["🏁 Total (com erro)"] = time.perf_counter() - inicio_total
        _log_tempos(tempos)
        raise SystemExit(1)
    else:
        tempos["🏁 Total"] = time.perf_counter() - inicio_total
        _log_tempos(tempos)
    finally:
        if os.path.exists(NARRACAO_PATH):
            os.remove(NARRACAO_PATH)
        if media_manager is not None:
            media_manager.limpar_cache_temporario()


if __name__ == "__main__":
    main()
