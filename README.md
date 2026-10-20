# YouTube Shorts Automation Bot

Automatiza a criação e publicação diária de YouTube Shorts: gera um roteiro com Gemini, produz narração por TTS, monta vídeo vertical com B-roll e legendas sincronizadas e publica pela YouTube Data API.

## Requisitos

- Python 3.14 e FFmpeg.
- ImageMagick com as fontes DejaVu instaladas (necessário para renderizar legendas).
- Dependências Python: `python -m pip install -r requirements.txt`.

## Configuração

O programa lê credenciais diretamente das variáveis de ambiente; não carrega arquivos `.env`. No GitHub Actions, configure estes GitHub Secrets:

- `GEMINI_API_KEY` e `YOUTUBE_TOKEN_JSON` (obrigatórios).
- Pelo menos um provedor de voz: `OPENAI_API_KEY`, `GOOGLE_TTS_API_KEY` ou `ELEVENLABS_API_KEY`.
- `PEXELS_API_KEY` é opcional; sem ela, o bot usa os vídeos locais.

Para execução local, exporte as mesmas variáveis no shell antes de iniciar `python bot_main.py`. Não adicione credenciais ao repositório.

## Execução e testes

O workflow em `.github/workflows/agendador.yml` publica diariamente às 18:00 UTC e também pode ser iniciado manualmente no GitHub Actions. O upload está configurado como público.

Execute os testes unitários com:

```sh
python -m unittest discover -s tests -v
```
