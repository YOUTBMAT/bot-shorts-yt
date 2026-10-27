import json
import os
import random
import time
from typing import Optional, Tuple

from google import genai
from google.genai import types


MODEL_PRIORITY = ["gemini-3.6-flash"]
MAX_RETRIES = 2
ROTEIRO_MIN_PALAVRAS = 65
ROTEIRO_MAX_PALAVRAS = 85
ROTEIRO_TOLERANCIA_PALAVRAS = 15
TITULO_MAX_CARACTERES = 90
RETRY_BACKOFF_BASE = 1.0
RETRY_BACKOFF_JITTER = 0.5


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
            continue

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
        raise ValueError(f"Roteiro fora do tamanho esperado ({num_palavras} palavras).")

    return titulo, roteiro


def gerar_roteiro_gemini() -> Tuple[str, str]:
    """Gera um roteiro inédito usando o Gemini, com fallback entre modelos."""
    print("🧠 [1/4] Gerando roteiro inédito com o Gemini...")
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = _montar_prompt_roteiro()

    ultimo_erro: Optional[Exception] = None
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
                config=types.GenerateContentConfig(response_mime_type="application/json"),
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
