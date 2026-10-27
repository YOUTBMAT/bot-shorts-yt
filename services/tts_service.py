import edge_tts


async def gerar_audio_edge_tts(
    texto: str, caminho_saida: str, voz: str = "pt-BR-AntonioNeural"
) -> bool:
    try:
        comunicacao = edge_tts.Communicate(texto, voz)
        await comunicacao.save(caminho_saida)
        return True
    except Exception:
        return False
