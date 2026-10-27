# Automação de YouTube Shorts

Bot que cria e publica vídeos curtos: gera um roteiro, produz a narração, monta um vídeo vertical com legendas sincronizadas e B-roll, valida o arquivo final e publica pela YouTube Data API.

## Arquitetura

`bot_main.py` é o orquestrador do pipeline. A lógica de cada etapa está isolada em serviços Python:

- `services/script_service.py` gera e valida roteiros com Gemini, incluindo retentativas para falhas temporárias.
- `services/tts_service.py` gera a narração com `edge-tts`, sem depender dos provedores pagos OpenAI, Google Cloud TTS ou ElevenLabs. A biblioteca é gratuita e usa as vozes online do Microsoft Edge; não requer uma chave de API de TTS.
- `services/media_service.py` seleciona mídias locais ou busca vídeos contextuais no Pexels, com cache, controle de downloads e histórico para reduzir repetições.
- `services/video_service.py` transcreve a narração, sincroniza legendas e compõe/renderiza o vídeo vertical.
- `services/youtube_service.py` verifica o estado de publicação, autentica na YouTube Data API e publica o vídeo.

O pipeline tenta encontrar B-roll relevante para cada trecho da narração no Pexels antes de recorrer às mídias locais. Se a busca contextual não encontrar material adequado, o serviço usa a biblioteca local e, como último recurso, o vídeo genérico `fundo.mp4`.

Antes do upload, o vídeo renderizado passa por uma validação técnica: precisa existir, não estar vazio, ter resolução exata de **1080 × 1920**, conter áudio e ter duração positiva. Vídeos que não atendem a esses critérios não são publicados.

## Persistência de estado

O serviço do YouTube registra em `estado_publicacao.json` o hash SHA-256 do roteiro publicado, seu status (`PUBLICADO`) e o ID retornado pela API. Se o mesmo roteiro for processado novamente, o upload é ignorado.

O workflow `.github/workflows/agendador.yml` tem permissão de escrita no repositório e, ao final da execução, faz commit e push de alterações em `media_history.json`, `estado_publicacao.json` e `cache/`. Assim, o histórico de mídia e o estado de publicação sobrevivem entre execuções em runners efêmeros.

## Requisitos

- Python 3.14 e FFmpeg.
- ImageMagick e fontes DejaVu para renderização das legendas.
- Dependências Python listadas em `requirements.txt`:

```sh
python -m pip install -r requirements.txt
```

## Configuração

Configure as variáveis de ambiente necessárias antes de executar o bot:

- `GEMINI_API_KEY`: chave de acesso ao Gemini.
- `YOUTUBE_TOKEN_JSON`: credenciais autorizadas para a YouTube Data API.
- `PEXELS_API_KEY`: opcional; sem ela, o serviço utiliza a biblioteca local de vídeos.

Não é necessária chave de API para o Edge TTS. Nunca adicione credenciais ao repositório.

## Execução e testes

Execute o bot com:

```sh
python bot_main.py
```

O workflow agenda a publicação diariamente às 18:00 UTC e também permite execução manual pelo GitHub Actions. Execute os testes unitários com:

```sh
python -m unittest discover -s tests -v
```
