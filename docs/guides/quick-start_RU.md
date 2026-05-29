# Быстрый старт

## Требования

- Python 3.11+
- OpenAI API key (embeddings через `text-embedding-3-small` + LLM judge `gpt-4o-mini`)
- Gemini API key (генерация: Gemini 2.5 Flash / Gemini 3 Flash)

## Установка

```bash
git clone https://github.com/spqr-86/regulatory-rag.git
cd regulatory-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Настройка .env

Скопировать `.env.example` в `.env` и заполнить ключи:

```env
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

Необязательные переопределения (значения по умолчанию заданы в `config/settings.py` и `src/v7/config.py`):

```env
# LLM провайдеры (по умолчанию: gemini)
# SIMPLE_LLM_PROVIDER=gemini
# COMPLEX_LLM_PROVIDER=gemini

# Путь ChromaDB (по умолчанию: ./chroma_db)
# CHROMA_DB_PATH=./chroma_db

# LangSmith трейсинг (необязательно)
# LANGSMITH_API_KEY=your_key
# LANGSMITH_TRACING_V2=true
# LANGSMITH_PROJECT=regulatory-rag
```

## Индексация документов

Поместить PDF/DOCX файлы в `source_docs/` и запустить:

```bash
python index.py
```

> ВНИМАНИЕ: `index.py` — деструктивная операция: удаляет всю коллекцию ChromaDB перед переиндексацией.

Индексатор использует HybridChunker (docling_core, max_tokens=400) для разбиения документов по структурным заголовкам и пунктам. Текущий корпус: 11 PDF, индекс после v3.0-hybrid ещё не пересобран.

## Запуск UI

```bash
streamlit run app.py --server.port 8502
```

Открыть `http://localhost:8502`. Запрос проходит через граф V7:
`intent_gate → router → rag_simple → evaluate_triage → [llm_verifier/rag_complex] → generate_answer`

## Запуск API

```bash
uvicorn api:app --port 8503
```

Эндпоинты:
- `POST /query` — пайплайн V7, тело: `{"query": "..."}`
- `GET /health` — проверка работоспособности

## Eval

```bash
python eval/run_v7_eval.py --skip-judge    # только пайплайн, без LLM judge (~$0)
python eval/run_v7_eval.py                 # полный eval с gpt-4o-mini judge
python eval/run_v7_eval.py --limit 5       # smoke test
```

Результаты записываются в `benchmarks/eval_v7_{date}.jsonl`. См. [evaluation/README_RU.md](../evaluation/README_RU.md).

## Тесты

```bash
pytest -m unit          # 237 юнит-тестов
pytest                  # все тесты (3 предсуществующих падения в test_evaluate_triage.py)
```
