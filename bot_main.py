import os
import json
import random
import math
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
from moviepy.editor import (
    VideoFileClip, 
    AudioFileClip, 
    TextClip, 
    CompositeVideoClip, 
    CompositeAudioClip
)
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

# ---------------------------------------------------------------------------
# 1. GERAR ROTEIRO (Gemini SDK Nova)
# ---------------------------------------------------------------------------
def gerar_roteiro():
    print("🧠 [1/4] Gerando roteiro inédito com o Gemini...")
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Inicializa o cliente moderno da SDK google-genai
    client = genai.Client(api_key=api_key)

    prompt = """
    Gere UMA curiosidade fascinante, rápida e impressionante sobre ciência, espaço, história ou natureza para um vídeo curto.
    
    REGRAS OBRIGATÓRIAS:
    1. Texto direto, sem saudações.
    2. Entre 25 e 35 palavras no máximo.
    3. Comece com um gancho forte ("Você sabia que...", "Existe algo incrível...").
    4. Responda ESTRITAMENTE em formato JSON válido com as chaves "titulo" e "roteiro".

    Exemplo:
    {"titulo": "O silêncio do espaço #shorts", "roteiro": "Você sabia que o espaço é totalmente silencioso? Como não existe ar para transportar as ondas sonoras, o som não consegue se propagar!"}
    """

    resposta = client.models.generate_content(
        model='gemini-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    dados = json.loads(resposta.text.strip())
    return dados["titulo"], dados["roteiro"]

# ---------------------------------------------------------------------------
# 2. GERAR ÁUDIO (ElevenLabs)
# ---------------------------------------------------------------------------
def gerar_audio(texto, caminho_saida="narracao.mp3"):
    print("🎙️ [2/4] Gerando voz ultra-realista com a ElevenLabs...")
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
    
    audio_stream = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Voz Adam (Padrao universal no plano free)
        text=texto,
        model_id="eleven_multilingual_v2"
    )
    
    with open(caminho_saida, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)
    return caminho_saida

# ---------------------------------------------------------------------------
# 3. EDITAR VÍDEO (MoviePy + Legendas Dinâmicas)
# ---------------------------------------------------------------------------
def criar_video(roteiro_completo, caminho_audio, caminho_video_fundo, caminho_saida="video_final.mp4"):
    print(f"🎬 [3/4] Montando vídeo final com legendas dinâmicas e design pro...")
    
    # Usa DejaVu-Sans-Bold, garantida e nativa no Linux do GitHub Actions
    FONT_NAME = "DejaVu-Sans-Bold"
    FONT_SIZE = 65
    COLOR_TEXT = "yellow"
    STROKE_COLOR = "black"
    STROKE_WIDTH = 2
    
    clip_video = VideoFileClip(caminho_video_fundo)
    clip_audio = AudioFileClip(caminho_audio)
    
    final_duration = clip_audio.duration
    clip_video = clip_video.subclip(0, final_duration)
    
    # Redimensionamento e Crop 9:16 sem esticar
    w, h = clip_video.size
    target_ratio = 9/16
    
    if w/h > target_ratio:
        new_w = h * target_ratio
        clip_video = clip_video.crop(x1=(w-new_w)/2, y1=0, width=new_w, height=h)
    else:
        new_h = w / target_ratio
        clip_video = clip_video.crop(x1=0, y1=(h-new_h)/2, width=w, height=new_h)
        
    clip_video = clip_video.resize(width=1080)
    final_w, final_h = clip_video.size

    # Legendas em blocos de 5 palavras
    palavras = roteiro_completo.split()
    tamanho_bloco = 5
    blocos_texto = []
    for i in range(0, len(palavras), tamanho_bloco):
        bloco = " ".join(palavras[i : i + tamanho_bloco])
        blocos_texto.append(bloco)

    num_blocos = len(blocos_texto)
    duration_per_block = final_duration / num_blocos
    
    text_clips = []
    
    for i, texto in enumerate(blocos_texto):
        start_time = i * duration_per_block
        
        txt_clip = TextClip(
            texto,
            fontsize=FONT_SIZE,
            font=FONT_NAME,
            color=COLOR_TEXT,
            stroke_color=STROKE_COLOR,
            stroke_width=STROKE_WIDTH,
            method='caption',
            size=(final_w * 0.8, None),
            align='center',
            interline=10
        )
        
        txt_clip = txt_clip.set_start(start_time).set_duration(duration_per_block)
        txt_clip = txt_clip.set_position(('center', 0.55), relative=True)
        txt_clip = txt_clip.crossfadein(0.15).crossfadeout(0.15)
        
        text_clips.append(txt_clip)

    video_final = CompositeVideoClip([clip_video] + text_clips)
    video_final = video_final.set_audio(clip_audio)
    
    print("⏳ Renderizando vídeo...")
    video_final.write_videofile(
        caminho_saida,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        logger=None
    )
    
    return caminho_saida

# ---------------------------------------------------------------------------
# 4. UPLOAD YOUTUBE
# ---------------------------------------------------------------------------
def upload_youtube(caminho_video, titulo):
    print("🚀 [4/4] Enviando o vídeo para o YouTube Shorts...")
    token_json_str = os.environ.get("YOUTUBE_TOKEN_JSON")
    info_token = json.loads(token_json_str)
    
    credentials = Credentials.from_authorized_user_info(info_token)
    youtube = build('youtube', 'v3', credentials=credentials)

    body = {
        'snippet': {
            'title': titulo,
            'description': 'Curiosidade rápida gerada automaticamente! #shorts #curiosidades',
            'tags': ['shorts', 'curiosidades', 'viral'],
            'categoryId': '27'
        },
        'status': {
            'privacyStatus': 'public',
            'selfDeclaredMadeForKids': False
        }
    }

    media = MediaFileUpload(caminho_video, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part=','.join(body.keys()), body=body, media_body=media)
    resposta = request.execute()
    print(f"✅ Sucesso! Vídeo publicado com o ID: {resposta['id']}")

# ---------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        titulo, roteiro = gerar_roteiro()
        audio_path = gerar_audio(roteiro)
        video_fundo = "fundo.mp4" 
        
        video_path = criar_video(roteiro, audio_path, video_fundo)
        upload_youtube(video_path, titulo)
        
        if os.path.exists("narracao.mp3"):
            os.remove("narracao.mp3")
            
    except Exception as e:
        print(f"❌ Ocorreu um erro na execução: {e}")
        exit(1)