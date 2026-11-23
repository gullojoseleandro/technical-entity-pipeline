# Technical Entity Extraction Pipeline

An asynchronous Python 3.12 LCEL pipeline that extracts a validated technical summary from one paragraph. It supports OpenAI or Anthropic, checks for truncated responses, and retries malformed or incomplete structured output once.

## Quick start

Clone the entire repository; `requirements.txt` and `.env.example` are included at its root.
From this repository on Windows, with [uv](https://docs.astral.sh/uv/) installed:

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

Put **one** real provider key in `.env` (never commit it), then run:

```powershell
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe main.py --text "FastAPI uses Redis. The production API is down."
```

Without uv, create a Python 3.12 virtual environment with `python -m venv .venv`, activate it, and install `pip install -r requirements.txt`. On macOS/Linux, use `cp .env.example .env` and `.venv/bin/python main.py`.

## Configuration

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `openai` (default) or `anthropic` |
| `OPENAI_API_KEY` | Required for OpenAI |
| `ANTHROPIC_API_KEY` | Required for Anthropic |
| `OPENAI_MODEL` | Defaults to `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Defaults to `claude-haiku-4-5-20251001` |

The key is read from the environment or ignored `.env` file. A live run makes a provider API request and may incur usage charges.

## Expected shape

For a paragraph about a FastAPI service using Redis and PostgreSQL with connection pressure, an illustrative response is:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

`tecnologias` may be empty if the input names none. Its entries and `resumen_tecnico` cannot be blank. `nivel_de_criticidad` is restricted to `baja`, `media`, or `alta`.

## How the pipeline works

`chain.py` composes `ChatPromptTemplate | model.with_structured_output(TechnicalEntities, include_raw=True) | validation` and calls the resulting chain with `.ainvoke()` from `process_text()`. `include_raw=True` exposes the parsed value, parser error, and response metadata. The validator rejects parser failures, missing fields, and `finish_reason=length` or `stop_reason=max_tokens` before accepting the Pydantic object. `.with_retry(stop_after_attempt=2)` makes one additional model call only for structured-output failures; provider/network errors are not retried by this policy. Logs report validation and retry-relevant events without logging the source paragraph, raw model response, or credentials.

## Offline checks

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q schemas.py chain.py main.py tests
```

The tests use a fake model and require no API key or network access. The live demo was not run against a paid provider as part of repository verification.
