import os
import json
import random
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from moviepy.editor import (
    VideoFileClip, 
    AudioFileClip, 
    TextClip, 
    CompositeVideoClip, 
    CompositeAudioClip
)
from moviepy.audio.fx.all import audio_fadeout, volumex
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

# 1. ROTEIRO COM GEMINI (Usa alias dinâmico + Modo JSON nativo)
def gerar_roteiro():
    print("🧠 [1/4] Gerando roteiro inédito com o Gemini...")
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        'gemini-flash-latest',
        generation_config={"response_mime_type": "application/json"}
    )

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

    resposta = model.generate_content(prompt)
    dados = json.loads(resposta.text.strip())
    return dados["titulo"], dados["roteiro"]

# 2. ÁUDIO COM ELEVENLABS (SDK 1.x / Generator)
def gerar_audio(texto, caminho_saida="narracao.mp3"):
    print("🎙️ [2/4] Gerando voz ultra-realista com a ElevenLabs...")
    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
    
    audio_stream = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Voz Adam (Padrao / PT-BR)
        text=texto,
        model_id="eleven_multilingual_v2"
    )
    
    with open(caminho_saida, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)
    return caminho_saida

# 3. RENDERIZAR VÍDEO COM MOVIEPY + MIXAGEM DE ÁUDIO
def criar_video(texto, arquivo_audio, fundo_mp4="fundo.mp4", musica_mp3="musica.mp3", saida="video_final.mp4"):
    print("🎬 [3/4] Renderizando o vídeo vertical, legendas e mixagem de áudio...")
    
    # --- ÁUDIOS ---
    narracao_clip = AudioFileClip(arquivo_audio)
    duracao = narracao_clip.duration

    if os.path.exists(musica_mp3):
        musica_clip = AudioFileClip(musica_mp3)
        
        max_inicio_musica = max(0, musica_clip.duration - duracao)
        inicio_musica = random.uniform(0, max_inicio_musica) if max_inicio_musica > 0 else 0
        
        if musica_clip.duration < duracao:
            musica_clip = musica_clip.loop(duration=duracao)
        else:
            musica_clip = musica_clip.subclip(inicio_musica, inicio_musica + duracao)
            
        # Reduz o volume da música para 12% e aplica fade-out nos últimos 1.5s
        musica_clip = volumex(musica_clip, 0.12)
        musica_clip = audio_fadeout(musica_clip, 1.5)
        
        audio_final = CompositeAudioClip([narracao_clip, musica_clip])
    else:
        print("⚠️ [Aviso] musica.mp3 não encontrada. Gerando vídeo apenas com a narração.")
        audio_final = narracao_clip

    # --- VÍDEO DE FUNDO (Corte Aleatório) ---
    video_fundo = VideoFileClip(fundo_mp4)
    
    max_inicio_video = max(0, video_fundo.duration - duracao)
    inicio_video = random.uniform(0, max_inicio_video) if max_inicio_video > 0 else 0
    
    if video_fundo.duration < duracao:
        video_fundo = video_fundo.loop(duration=duracao)
    else:
        video_fundo = video_fundo.subclip(inicio_video, inicio_video + duracao)

    video_fundo = video_fundo.resize((1080, 1920))

    # --- LEGENDAS ---
    txt_clip = TextClip(
        texto,
        fontsize=48,
        color='yellow',
        font='DejaVu-Sans-Bold',
        method='caption',
        size=(850, None),
        align='center',
        bg_color='black'
    ).set_position(('center', 0.4), relative=True).set_duration(duracao)

    # --- MONTAGEM FINAL ---
    video_final = CompositeVideoClip([video_fundo, txt_clip]).set_audio(audio_final)
    video_final.write_videofile(saida, fps=30, codec="libx264", audio_codec="aac")

    narracao_clip.close()
    if os.path.exists(musica_mp3):
        musica_clip.close()
    video_fundo.close()
    
    return saida

# 4. UPLOAD PARA YOUTUBE
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
            'categoryId': '27' # Categoria Educação
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

if __name__ == "__main__":
    try:
        titulo, roteiro = gerar_roteiro()
        audio_path = gerar_audio(roteiro)
        video_path = criar_video(roteiro, audio_path)
        upload_youtube(video_path, titulo)
        
        if os.path.exists("narracao.mp3"):
            os.remove("narracao.mp3")
            
    except Exception as e:
        print(f"❌ Ocorreu um erro na execução: {e}")
        exit(1)