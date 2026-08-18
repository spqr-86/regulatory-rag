---
paths:
  - "src/**/*.py"
  - "eval/**/*.py"
  - "scripts/**/*.py"
  - "mcp_server.py"
  - "api.py"
  - "app.py"
  - "index.py"
---

<!--
Создано 27.07.2026. Повод: замер — глобальные шаблоны ~/.claude/rules/llm-agents.md
матчили здесь 6 файлов из ~40 (по именам *agent*, *llm*, *prompt*, *eval*, *backend*).
Ядро RAG-графа — graph.py, router.py, abstain.py, domain_gate.py, generate_answer.py —
проходило мимо. Раскладка проекта старше правила, поэтому правило локальное.
-->

# Этот проект — RAG/LLM целиком

`src/` — агентный и LLM-код, независимо от имени файла: `graph.py`, `router.py`,
`abstain.py`, `domain_gate.py`, `hard_gates.py`, `generate_answer.py`, `rag_simple.py`,
`rag_complex.py` — узлы графа, а не утилиты.

**Читать `~/.claude/rules/llm-agents.md`** перед изменением любого файла здесь — глобальные
шаблоны подхватывают его в этом проекте лишь частично.

## Владение (граница с prombez-agent, зафиксирована 18.07.2026)

Этот проект — **владелец корпуса и ВСЕГО доступа к текстам**: `retrieve_chunks`,
адресный `get_norm`, `get_chunk`, ingest, переиндексация. Экспертиза (реестр НПА,
верификация цитат, applicability, кейсы) живёт в prombez-agent и сюда не переезжает.

## Локальное

- `mcp_server.py` — публичный контракт для prombez-agent. Смена имён тулов, схемы аргументов
  или формата `chunk_id` (`source::N`) ломает потребителя — правится согласованно с обоими.
- **Логи MCP-сервера только на stderr** (`utils/logging.py`, `configure_logging()` в начале
  `mcp_server.py`). structlog в stdout ломает stdio-протокол MCP — уже ловили.
- Переиндексация требует явного `CHROMA_COLLECTION_NAME` (`fire_safety` для корпуса ПБ);
  забытая переменная один раз залила чанки в чужую коллекцию.
- Чанк на строку таблицы — не «оптимизировать» склейкой: в нормативке требование живёт
  в строке, склейка усредняет эмбеддинг и ломает адресные кейсы.
