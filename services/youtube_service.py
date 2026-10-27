"""Autenticação e publicação de vídeos no YouTube."""

import hashlib
import json
import os

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

YOUTUBE_CATEGORY_ID: str = "27"
YOUTUBE_TAGS = ["shorts", "curiosidades", "fatos interessantes", "ciencia", "viral"]
YOUTUBE_DESCRICAO_TEMPLATE: str = (
    "{roteiro}\n\n"
    "💡 Gostou dessa curiosidade? Inscreva-se no canal para acompanhar mais vídeos diários!\n\n"
    "#shorts #curiosidades #fatos #ciencia #conhecimento"
)
ESTADO_PUBLICACAO_PATH = "estado_publicacao.json"


def _carregar_estado_publicacao() -> dict:
    if not os.path.exists(ESTADO_PUBLICACAO_PATH):
        return {"publicacoes": {}}

    try:
        with open(ESTADO_PUBLICACAO_PATH, "r", encoding="utf-8") as arquivo:
            estado = json.load(arquivo)
    except (OSError, json.JSONDecodeError) as err:
        raise RuntimeError(f"❌ Upload: não foi possível ler {ESTADO_PUBLICACAO_PATH} ({err}).") from err

    if not isinstance(estado, dict) or not isinstance(estado.get("publicacoes"), dict):
        raise RuntimeError(f"❌ Upload: formato inválido em {ESTADO_PUBLICACAO_PATH}.")
    return estado


def _salvar_estado_publicacao(estado: dict) -> None:
    caminho_temporario = f"{ESTADO_PUBLICACAO_PATH}.tmp"
    try:
        with open(caminho_temporario, "w", encoding="utf-8") as arquivo:
            json.dump(estado, arquivo, ensure_ascii=False, indent=2)
            arquivo.write("\n")
        os.replace(caminho_temporario, ESTADO_PUBLICACAO_PATH)
    except OSError as err:
        raise RuntimeError(
            f"❌ Upload concluído, mas não foi possível salvar {ESTADO_PUBLICACAO_PATH} ({err})."
        ) from err


def fazer_upload_youtube(caminho_video: str, titulo: str, roteiro: str) -> str:
    """Publica o vídeo final no YouTube Shorts e retorna seu ID."""
    print("🚀 [4/4] Enviando o vídeo para o YouTube Shorts...")

    roteiro_hash = hashlib.sha256(roteiro.encode("utf-8")).hexdigest()
    estado = _carregar_estado_publicacao()
    publicacao = estado["publicacoes"].get(roteiro_hash)
    if isinstance(publicacao, dict) and publicacao.get("status") == "PUBLICADO":
        video_id = publicacao.get("video_id")
        if not video_id:
            raise RuntimeError(
                f"❌ Upload: estado PUBLICADO sem ID de vídeo em {ESTADO_PUBLICACAO_PATH}."
            )
        print(f"⏭️ Roteiro já publicado (ID: {video_id}); upload ignorado.")
        return str(video_id)

    if not os.path.isfile(caminho_video):
        raise RuntimeError(f"❌ Upload: arquivo de vídeo não encontrado: {caminho_video}")
    if os.path.getsize(caminho_video) == 0:
        raise RuntimeError(f"❌ Upload: arquivo de vídeo vazio: {caminho_video}")

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
        video_id = resposta["id"]
    except Exception as err:
        raise RuntimeError(f"❌ Upload: falha ao publicar no YouTube ({err}).") from err

    print(f"✅ Sucesso! Vídeo publicado com o ID: {video_id}")
    estado["publicacoes"][roteiro_hash] = {
        "status": "PUBLICADO",
        "video_id": video_id,
    }
    _salvar_estado_publicacao(estado)
    return video_id
