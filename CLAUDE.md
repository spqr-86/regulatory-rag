# CLAUDE.md

## Project Overview

**AI Safety Compliance Assistant** — RAG system for analyzing Russian workplace safety regulations (ГОСТ, СНиП, СП). Uses hybrid retrieval (semantic + BM25), FlashRank reranking, and multi-agent LangGraph workflows.

**Stack**: Python 3.11+, LangChain, LangGraph, ChromaDB, Docling, FlashRank, Streamlit, Google Gemini 3

## Virtual Environment

**IMPORTANT**: Always use the project's venv. Never install packages globally.

```bash
source venv/bin/activate   # before any command
```

## Deployment (VPS)

- **URL:** http://213.176.64.237:8502
- **Port:** 8502 (UFW opened), tmux session `sia` (attach: `tmux a -t sia`)
- **Start:** `cd /home/petr/projects/safety-incident-analyzer && source venv/bin/activate && streamlit run app.py --server.port 8502`
- **Indexed docs:** 12 PDFs → 1973 chunks (добавлен полный ТК РФ, v2.3-noise-clean, 2026-05-16).
- **Машина — это VPS.** Локальная директория `/home/petr/projects/safety-incident-analyzer` и есть прод-инстанс. Никакого scp/git pull не нужно — правки сразу на сервере, остаётся только перезапустить tmux `sia`.

**VPS env requirements:**
- `HTTPS_PROXY=socks5h://localhost:40000` + `HTTP_PROXY=socks5h://localhost:40000` — WARP proxy для Gemini (VPS AEZA заблокирован по ASN)
- PySocks установлен (`pip install httpx[socks] requests[socks]`) — нужен для SOCKS в httpx/requests

## Known Issues

- ~~**Gemini 503 → stub fallback**~~ ✅ Fixed 2026-05-08: tenacity retry (3 attempts, exp backoff 2→4→8s), fallback to stub only after all retries.
- ~~**P1** Баг чанкинга в `src/file_handler.py`~~ ✅ Fixed 2026-05-15: MIN_BBOX_HEIGHT больше не роняет текст, MAX_CHUNK_SIZE из settings, update_bbox flush. 830 → 1069 чанков. PIPELINE_VERSION v2.2-grouped. — см. backlog.md
- ~~**P1** `app.py` не читает `result["answer"]`~~ ✅ Fixed ранее (строка 241 читает result["answer"])
- ~~**P2** FlashRank score inflation в evaluate_complex~~ ✅ Fixed 2026-05-16: vector_score сохраняется в bridge.py, MMR/gates используют его вместо FlashRank score. correctness 6.86 → 7.9.

## Commands

```bash
streamlit run app.py --server.port 8502          # run app (VPS port)
python index.py                                  # reindex (DESTRUCTIVE: drops ChromaDB)
pytest -v                                        # tests (-m unit / integration / "not slow")
black . && ruff check . --fix                    # lint (run before commits)
python scripts/validate_prompts.py               # validate after prompt changes
python scripts/trace_v7.py "вопрос"             # E2E smoke test с трассировкой
python scripts/measure_cps.py --n 10             # Cost-per-sample замер (token usage + $$)
```

## Eval Datasets

- **`tests/dataset_original.csv`** — 50 вопросов + ground_truth. Основной золотой датасет для оценки качества RAG. Формат: `question, ground_truth`.
- **`tests/dataset_clean.csv`** — 38 вопросов (очищенная версия).
- **`eval/gosts_gold_v1.json`** — 5 вопросов по ГОСТ с источниками (id, question, gold_answer, sources).
- **`eval/gosts_gold_v2.json`** — 5 вопросов по ГОСТ с chunk_hint (question, gold_answer, source, chunk_hint).
- **`benchmarks/`** — результаты прогонов eval (JSONL) + история метрик (`results_history.jsonl`).
- **`benchmarks/cps_YYYY-MM-DD.json`** — Cost-Per-Sample замеры (token usage + $$ per query). Baseline 2026-05-22: $0.0102/query на N=10.

## Code Style

- **Line length**: 88 (Black/Ruff, see `pyproject.toml`)
- **Imports**: absolute (`from src.utils import logger`). Order: stdlib → third-party → local.
- **Type hints**: `from __future__ import annotations` at top of files.
- **Language**: Domain logic/comments in Russian; infrastructure in English. Follow convention of the file being edited.
- **Testing**: `unittest.mock`, markers `@pytest.mark.unit / integration / slow`, config in `pyproject.toml`.

## Architecture

### Multi-Agent RAG (`agents/multiagent_rag.py`) — legacy fallback (V7 Graph — основной)

```
glossary expansion → regex filter → rag_agent (ReAct) → verifier → format_final
                                   → direct_response (chitchat/out_of_scope)
Revision: verifier (needs_revision) → rag_agent (max 1)
```

- **Regex Filter** (`_classify_query`): deterministic regex, no LLM. Classifies chitchat / out_of_scope / rag.
- **RAG Agent** (flash, thinking: 8192, max 16 steps): ReAct with `search_documents` + `visual_proof` tools.
- **Verifier** (flash, thinking: 1024): JSON fact-check, 6 criteria.
- **Revision**: agent receives previous `draft_answer` + verifier feedback.
- **Shared rules**: `prompts/common/base_rules.j2` — macro imported via `{% import "common/base_rules.j2" as rules %}`.
- **Term Glossary**: `config/term_glossary.yaml` — deterministic expansion of domain abbreviations before query enters the graph. To add terms: edit YAML, no code changes.

**Simple RAG Chain** (`src/final_chain.py`): Legacy fallback. Hybrid retrieval → FlashRank rerank → LLM.

### Prompt Management

Versioned Jinja2 templates. Registry: `prompts/registry.yaml`. Override via env: `PROMPT_RESEARCH_AGENT_VERSION=v2`.

### LLM Factory (`src/llm_factory.py`)

`get_llm()`, `get_gemini_llm(thinking_budget)`, `get_vision_llm()`. Provider via `LLM_PROVIDER` env var. **Note**: Gemini SDK patched to disable AFC (Automatic Function Calling) — prevents duplicate tool calls.

## Common Patterns

### Adding a New Agent
1. Create `prompts/agents/<name>_v1.j2`
2. Register in `prompts/registry.yaml`
3. Add logic in `agents/`
4. Run `python scripts/validate_prompts.py`

### Adding a New Prompt Version
1. Create new file (e.g., `research_v3.j2`)
2. Add to `prompts/registry.yaml`, update `active_version` or test via env override

## Important Notes

- **Russian Language**: System optimized for Russian regulatory text.
- **`index.py` is destructive**: Drops entire ChromaDB before reindexing.
- **Config**: pydantic-settings in `config/settings.py`, copy `.env.example` → `.env`.
- **Debugging**: `DEBUG_PROMPTS=true`, LangSmith traces (`LANGSMITH_TRACING_V2=true`), error reports in `analysis/error_reports/`.
- **Commits**: Do NOT add `Co-Authored-By` lines. Run `black . && ruff check . --fix` before committing.

## Active Work: V7 Migration

**Design:** `docs/plans/2026-02-16-v7-migration-design.md` — полная архитектура, модули, roadmap, инструкции для агента.
**Spec:** `docs/feature/migration-v7` — исходная спецификация v7 (1729 строк, все типы и функции).
**Implementation plans:** `docs/plans/2026-02-16-v7-stage0-implementation.md` — пример формата плана для этапа.

### Ключевые принципы v7 (секция 5 дизайн-документа)
- **5.5 Не изобретать велосипед** — использовать библиотеки (`rank_bm25`, `pymorphy3`, `razdel`, `pydantic`), не писать свои реализации
- **5.6 Строгий граф зависимостей** — `state_types ← config ← nlp_core ← hard_gates ← nodes ← graph`, обратных импортов нет
- **5.7 Тонкие ноды** — ноды только оркестрируют (read state → call function → write state), вся логика в `nlp_core`/`hard_gates`

### Прогресс
| Этап | Модуль | Статус | Ветка |
|------|--------|--------|-------|
| 0 | `state_types` + `config_v7` | ✅ Done | `feature/v7-migration-stage0` |
| 1 | `nlp_core` | ✅ Done | `feature/v7-migration-stage1` |
| 2 | `hard_gates` | ✅ Done | `feature/v7-migration-stage1` |
| 3 | `nodes/*` | ✅ Done | `feature/v7-migration-stage1` |
| 4 | `graph.py` | ✅ Done | `feature/v7-migration-stage1` |
| 5 | Миграция + cleanup | ✅ Done | `feature/v7-migration-stage1` |
| 6 | Production readiness | ✅ Done | `main` |

**Этап 6 включает** (все на `main`):
- `generate_answer` node + `answer: str` в RAGState — LLM-синтез через bridge DI
- `make_generate_fn()` в bridge (Gemini, thinking_budget=4096, fallback=stub)
- `intent_gate`: regex вместо frozenset (корректная фильтрация шума)
- Тарировка порогов: `HARD_GATE_THRESHOLD=0.50`, `TRIAGE_SOFT_THRESHOLD=0.38` — открыта зона borderline→llm_verifier
- Domain gate: `V7_DOMAIN_GATE_THRESHOLD=0.25` — pre-retrieval OOS filter по cosine similarity к corpus centroid (src/v7/domain_gate.py)
- Фикс `top_score` в rag_simple: использует только vector scores (не BM25-inflated)
- E2E smoke test: `python scripts/trace_v7.py` (цветная трассировка пути + answer)
- 151 unit-тест, все зелёные

### E2E smoke test
```bash
python scripts/trace_v7.py "для кого проводится повторный инструктаж?"
python scripts/trace_v7.py --no-chroma "привет как дела"   # stub mode
```

### Backlog

См. `docs/plans/backlog.md`.

**Синхронизация:** Таблица прогресса дублируется в `CLAUDE.md`, `AGENTS.md` и `GEMINI.md`. При обновлении одного — обнови остальные два.

---

## Session Log

### 2026-05-22 (сессия 37)

- **Сделано:** CPS baseline для V7 pipeline. Создан `scripts/measure_cps.py` (инструментация через `langchain_core.callbacks.UsageMetadataCallbackHandler`). Прогон на 10 вопросах из `dataset_original.csv`: **$0.0102/query, $10.21/1k**, avg 8 416 токенов (4 920 in + 3 495 out incl. thinking), avg latency 21.9 сек. Сырые цифры — `benchmarks/cps_2026-05-22.json` (gitignored). README дополнен секцией "Экономика (CPS baseline)". Разбор для интервью GenAI Lab — `~/career/shad/notes/sia_cps_baseline.md`.
- **Решения:** Pricing assumption — Gemini 2.5 Flash ($0.30/M in, $2.50/M out) как прокси для preview-модели (тот же tier). N=10 принято как достаточно для интервью; conditional пути (verifier/rewriter) не покрыты — отмечено в ограничениях.
- **Наблюдения:** На 10 sampled queries verifier/rewriter не сработали — все пошли по simple path (1 LLM call/query). Разброс стоимости ×5 ($0.004–$0.022) — главный драйвер output длина + thinking tokens. Главный рычаг оптимизации — Model Router.

### 2026-05-22 (сессия 36)

- **Сделано:** README переработан — добавлены Two-Stage Retrieval с Cross-Encoder, отдельный пункт Guardrails (intent_gate + hard gates + abstain), расшифровка fast path / slow path для rag_simple/rag_complex. Вводные разделы переписаны в технический стиль (убран рекламный тон + перечисление "охрана труда, пожарная безопасность"). ASCII-диаграмма пайплайна заменена на Mermaid flowchart.
- **Решения:** README позиционирует SIA как Regulatory Compliance Q&A, не привязанный к ОТ. Mermaid вместо ASCII — рендерится на десктоп-GitHub, на мобильном остаётся как код (приемлемо).
- **Наблюдения:** Создан внешний документ ~/career/shad/notes/project_sia_deep_dive.md с подробным разбором архитектуры (все 10 нод, ссылки на конкретные файлы GitHub) — для интервью и портфолио.

### 2026-05-17 (сессия 24)

- **Сделано:**
  - **DeepSeek в gosts_pipeline**: `_generate()` переключён с Gemini на DeepSeek (`deepseek-chat`, openai SDK, temperature=0, max_tokens=2048). `DEEPSEEK_API_KEY` добавлен в `.env`. Основной SIA pipeline (ОТ) — Gemini без изменений.
  - **Gold dataset v2** (`eval/gosts_gold_v2.json`): chunk-based, 5 вопросов по реальным чанкам с числовыми данными. Eval: 3/5 правильных. 2 провала — нужный чанк есть в базе, но не попал в top-15 retrieval.
- **Решения:** Gold dataset для RAG строить bottom-up (от чанков), не top-down (от тем). Тема документа ≠ данные в индексе.
- **Наблюдения:** Sources в passages пустые — metadata["source"] не сохраняется при индексировании (мелкий баг в index_gosts.py).

### 2026-05-17 (сессия 23)

- **Сделано:**
  - **GOST RAG** (`index_gosts.py`): проиндексировано 108 DOCX ГОСТов/СНиПов ЕРС из Google Drive. Баг: word/_rels/document.xml.rels использует абсолютные пути (`Target="word/settings.xml"`) → python-docx падает. Фикс: `_fix_docx_rels()` патчит ZIP в памяти через `re.sub`. Итог: **9344 чанков** в `chroma_db_gosts/` (коллекция `wta_gosts`).
  - **`src/gosts_pipeline.py`**: retrieve (ChromaDB wta_gosts, top_k=15) → FlashRank rerank (top_n=8) → DeepSeek (`deepseek-chat`, openai SDK, temperature=0) generate. `_load_store()` через `lru_cache`. *(Переключён с Gemini на DeepSeek в сессии 24)*
  - **`api.py`**: новый эндпоинт `POST /query/gosts`. В lifespan: `_gosts_load()` (non-fatal — 503 если не готово). SIA API запущен как `nohup uvicorn api:app --port 8503` (PID 929502, не в tmux — не переживёт рестарт).
  - **`eval/gosts_gold_v1.json`**: gold dataset 5 QA-пар (насосы ГОСТ 32601, трубопроводы ГОСТ 32569, фланцы DN100 ГОСТ 33259, ЗСО СП 30, степень защиты IP ГОСТ 14254).
- **Решения:** Для WTA-ГОСТов отдельная ChromaDB-коллекция вместо второго экземпляра SIA; GOST pipeline — Gemini (не переключали на DeepSeek).
- **Наблюдения:** SIA API (порт 8503) нужно перенести в tmux `sia` — иначе умрёт при рестарте VPS.

### 2026-05-16 (сессия 22)

- **Сделано:**
  - **Regex noise cleanup** (`src/file_handler.py`): добавлена `_clean_noise()` — удаляет URL-watermarks (`https://...`), page markers (`14/34`), timestamps (`25.01.2026, 20:10`) из текста чанков. `PIPELINE_VERSION` → `v2.3-noise-clean`. Переиндексация: 1069 → 976 чанков (−93 пустых мусорных).
  - **FlashRank score inflation fix** (P2): `bridge.py` сохраняет `vector_score` до перезаписи FlashRank-ом; `nlp_core.py` `mmr_select` использует `vector_score` для relevance; `rag_complex.py` `top_score` по `vector_score`. Эффект: top_score 0.998 (раздутый) → 0.577 (реальный), вопросы про Программу А теперь отвечаются корректно.
  - **Eval**: correctness 6.86 → **7.9** ✅ (цель 7.5 достигнута). Faith=0.988, false-sufficiency=0.15.
  - **Добавлен полный ТК РФ** (Trudovoj_kodeks_RF.pdf, ~1MB) → `source_docs/`. Переиндексация: 976 → **1973 чанков** (12 документов). Ст. 63, 229, 229.3 и весь раздел о расследовании НС теперь в индексе.
  - **Parent-context chunking** (#3 плана) реализован и протестирован: child 400 chars для embedding, parent 1500 для LLM. Eval: correctness 7.9→7.49, false-sufficiency 0.15→0.25. **Откачен** — регрессия из-за завышенных similarity scores малых чанков.
  - Backlog обновлён (план 5 пунктов, пункт #1 DONE; #3 tested-reverted).
- **Решения:** FlashRank не откалиброван на русском домене — его score нельзя использовать для порогов. Parent-context chunking требует перекалибровки порогов под новый диапазон scores.
- **Наблюдения:** False-sufficiency=0.15 выше цели 10%. vocabulary mismatch: "самостоятельно" vs "возглавляет" в ст.229.3 ТК РФ — чанк есть, но не поднимается. Следующий рычаг — eval cost optimization (кэш + smoke test 10 вопросов), затем contextual retrieval (#2 плана).

### 2026-05-15 (сессия 21)

- **Сделано:**
  - **Code review** (субагент code-reviewer): 25 находок (1 Critical, 10 High, 11 Medium, 5 Low). Отчёт: `REVIEW_2026-05-15.md`.
  - **Фикс P1 — баг чанкинга** (`src/file_handler.py`): root cause подтверждён диагностическим скриптом (`scripts/diagnose_chunking.py`) — `MIN_BBOX_HEIGHT=7` роняло 37% items на 2464.pdf включая «Повторный инструктаж … 6 месяцев» (bbox_h=5.71). Фикс: bbox-фильтр зануляет bbox, но оставляет текст; `MAX_CHUNK_SIZE` → `settings.CHUNK_SIZE`; `update_bbox=False` flush-ит чанк; `PIPELINE_VERSION` → v2.2-grouped. Переиндексация: 830 → **1069 чанков**. trace_v7 top score 1.000 на целевом вопросе.
  - **Фикс state-рассинхрона** (`agents/multiagent_rag.py`): `escalated_from_simple` и `subquestions` добавлены в `RAGState` TypedDict — ревизия после escalation теперь корректно идёт в rag_complex.
  - **Hardening visual_proof** (`src/agent_tools.py`): `_validate_file_name` (path traversal: `../`, abs path, subdir) + `_validate_bbox` (finite, range [0..10000], non-zero area) — защита от prompt-injection из PDF и DoS fitz.
  - **Cache invalidation** (`index.py`): destructive reindex теперь удаляет `document_cache/` и `.bm25_cache.pkl`.
  - **17 новых тестов**, 396 всего.
  - README.md + backlog.md + CLAUDE.md знакомлены с изменениями.
- **Решения:** Машина — это VPS (hostname ideologicalmage.aeza.network, 213.176.64.237). Деплой = рестарт tmux sia, никакого scp/SSH.
- **Наблюдения:** После фикса чанкинга ожидается рост correctness — corpus стал полнее на 239 чанков. Eval под фикс не прогонялся — следующий шаг перед VPS деплоем eval.

### 2026-05-14 (сессия 20)

- **Сделано:**
  - **Gemini quota разблокирован.** Причина блокировки eval (free-tier 20 RPD) — Google-side billing misclassification (известный баг: AI Studio показывает Tier 1 · Postpay, а API метрит по `generate_content_free_tier_requests` limit 20). НЕ гео/IP/конфиг — billing-аккаунт здоров (KZ, Paid 1, история списаний). Баг отпустило ~14.05. Eval гоняется на `gemini-3-flash-preview` + ключ проекта sia-prod, 0×429.
  - **Фикс #0 — max_output_tokens** (`src/llm_factory.py`): gemini-3 считает reasoning-токены ВНУТРИ max_output_tokens; захардкоженный 2048 + thinking_budget=4096 → ответы обрывались на полуслове (28/50). Фикс: `max_output_tokens = thinking_budget + 4096`. TDD. Обрывы 28→1.
  - **Фикс #2 — глоссарий в V7**: term_glossary.yaml был подключён только к легаси `multiagent_rag.py`, V7-граф его не применял. Вынес в `src/glossary.py`, подключил в `src/v7/nodes/router.py` (расширяет `active_query`). Попутно починен пре-существующий баг `_make_term_pattern`: 4-буквенные аббревиатуры («соут») стеммились в `со\w*` и ловили «состоять» — теперь ≤4 букв матчатся целым словом. TDD.
  - **Фикс #3 — срез генератора** (`src/v7/bridge.py`): `_generate` резал `final_passages[:15]`, хотя апстрим капит на 24 (`merge_all_passages`). Норма ответа выпадала (наблюдалась на #22). Фикс: `passages[:15]→[:24]`. TDD.
  - **Сверка docs/ с V7-реальностью** (коммиты 1e92d97, 624da5b, e066a54): аудит 13 доков, переписаны quick-start.md + evaluation/README.md, обновлены README + architecture-доки + passport + prompt-management + DATA_PIPELINE. `ROADMAP.md` → `docs/plans/2025-12-roadmap-historical.md`, `evaluation/examples.md` удалён. Вычищены сквозные ошибки (K=40→12, чанки 1200/150→1500/400, мёртвые eval-скрипты).
- **Решения:** Тариф Gemini привязан к billing-аккаунту/проекту, НЕ к IP — IPv6-патч и WARP оба упирались в free-tier (опровергает сессию 19). При сравнении eval-прогонов учитывать run-to-run вариативность LLM-судьи — мелкие per-question дельты на OOS-вопросах = шум.
- **Наблюдения:** Прогрессия eval (`gemini-3-flash-preview`, 3 прогона): faithfulness 0.927→0.976→**0.996**, correctness 6.04→6.61→**6.86**, in-scope correctness 5.38→6.14→**6.44**, обрывы 28→2→1. correctness всё ещё < цели 7.5 — основной рычаг теперь **Cause #1**: баг чанкинга в `src/file_handler.py` (`_process_docling_document`) выроняет целые пункты норм («Повторный инструктаж проводится не реже 1 раза в 6 месяцев» есть в PDF 2464 и чисто извлекается Docling, но отсутствует в индексе). Требует дебага группировки + переиндексации (index.py destructive). Возможный побочный эффект глоссария на вопросах с терминами (программа А 8→5, СОУТ 5→0) — проверить повторным прогоном. Бенчмарки: `benchmarks/eval_v7_2026-05-14*.jsonl`.

### 2026-05-13 (сессия 19)

- **Сделано:** VPS апгрейд NLs-3 (8GB/4c/120GB/€19.77). Domain gate реализован: `src/v7/domain_gate.py` (corpus centroid cosine similarity), добавлен в `intent_gate.py`. `V7_DOMAIN_GATE_THRESHOLD=0.25` откалиброван. IPv6 monkey-patch в `llm_factory.py` — обход ASN-блокировки Gemini без WARP. CLAUDE.md обновлён (порог тоже).
- **Решения:** ~~WARP proxy → Google применяет free tier 20 RPD. IPv6 прямое соединение — работает (200 OK), платная квота.~~ **[ИСПРАВЛЕНО в сессии 20: неверно. Тариф не зависит от IP — и WARP, и IPv6 упирались в free-tier 20. Причина была Google-side billing misclassification, не прокси.]** Квота per-project, не per-key — новый ключ в том же проекте не помогает.
- **Наблюдения:** Все 20 RPD free tier исчерпаны за сессию тестами. Eval с domain_gate пока не запущен. Ожидаемый эффект: correct_abstain 0.1 → ~0.8+ (OOS вопросы будут отсекаться до retrieval по косинусному сходству).

### 2026-05-13 (сессия 18)

- **Сделано:** Ориентация по состоянию проекта. Подтверждено: Приказ 223н + КоАП ст.5.27.1 уже в source_docs локально. ПП 1479 — сервер rostransnadzor.gov.ru не отвечает с нашего IP, пока без него.
- **Решения:** Corpus переиндексировать через scp + python index.py на VPS (SSH с этой машины не работает).
- **Наблюдения:** Indexed docs на VPS не обновлены — 8 PDFs без новых документов. SSH с VPS-сессии Claude → другой VPS не работает (разные ключи).

### 2026-05-12 (сессия 17)

- **Сделано:** V8 Epic 4 полностью завершён. Task 1c: `must_not_contain` в датасете (14 строк), `compute_inversion_detected/rate` в metrics.py. Task 1e: latency p50/p95/max/count_over_10s. Task 1a: `parse_citations`, `compute_citation_rate/in_retrieval/doc_match`, `run_query` возвращает `final_passages`. Task 1b: датасет 36→50 (oos_type: 10 out_of_scope + 5 false_premise). Fix: generate prompt передаёт `[Источник: ...]` для каждого пассажа (bridge.py `_short_source`). 33 новых теста, 221 всего.
- **Решения:** inversion_detected = substring match — false positives возможны, принято как baseline. Для false_premise вопросов is_oos=False (система должна корректировать, не молчать). Стратегия ruff imports: usage-код добавлять ДО импортов.
- **Наблюдения:** citation_doc_match был 0 т.к. LLM не знал источников — исправлено. Corpus gaps: Приказ 223н (НС), КоАП ст.5.27.1, ПП 1479 — нужны для completeness > 0.65. Eval с реальным LLM ещё не запущен.

### 2026-05-11 (сессия 14)

- **Сделано:** Eval на gemini-2.5-flash: V7=0.614, V8_promptv2=0.611. Промпт v2 нейтрален по completeness но снизил false_abstain 0.057→0.029 (-50%). FastAPI api.py создан (порт 8503, POST /query + GET /health, lifespan init). fastapi+uvicorn+structlog добавлены в requirements.txt. V8 Epic 4 план сохранён: docs/plans/2026-05-11-eval-improvements.md.
- **Решения:** completeness 0.61 = потолок corpus coverage (нет Приказа 223н по НС, нет КоАП). Eval через леммы не ловит инверсии — нужен negative_keywords. Citation validity не измеряется — разрыв между промптом v2 и eval.
- **Наблюдения:** OOS detection = 1.0 (идеально). False abstain = 0.029 (хорошо). Система работает, bottleneck — corpus и eval metrics, не retrieval/generation.

### 2026-05-09 (сессия 13)

- **Сделано:** V8 Epic 3 Multi-Query Expand (make_expand_fn, set_expand_fn DI, RRF слияние, 7 тестов). Eval: +0.000 completeness — retrieval не bottleneck. Добавлен 776н (СУОТ) в corpus, переиндексировано 749→830 чанков, completeness 0.567→0.589. Датасет очищен 38→36 (удалены Q34 ПБ склад, Q35 вентиляция как OOS). Промпт генерации v2: 14 проблем исправлено (роль убрана, 3 варианта ответа, HIGH/MED/LOW score, дословные цитаты, обязательные ссылки, противоречия без приоритизации). Decision log: docs/plans/2026-05-09-generate-prompt-v2.md.
- **Решения:** Retrieval работает (16+ фрагментов находятся) — bottleneck в generation. Multi-query expand нейтрален на stub LLM. Corpus gaps важнее всего — 776н дал +0.023. Роль "эксперта" в промпте провоцирует достройку из общих знаний — заменена на "отвечаешь СТРОГО на основе фрагментов".
- **Наблюдения:** gemini-3-flash → 404 (нестабильна). Все eval — на stub. Нужна gemini-2.5-flash для честного eval. Промпт v2 не оценён с реальным LLM. Corpus: 830 чанков, dataset: 36 вопросов.

### 2026-05-08 (сессия 12)

- **Сделано:** V8 Epic 1 (eval suite: metrics.py, run_eval.py, compare.py, 29 тестов) + V8 Epic 2 (evidence_assess: 3-verdict флаг, FlashRank в rag_simple, 314 тестов). Датасет очищен: 49→38 вопросов (удалены out-of-corpus). Модель: gemini-3-flash (нестабильна). V7 baseline stub-загрязнён — нужен перегон.
- **Решения:** Смена модели в .env = невалидация baseline. Перед сменой модели предупреждать пользователя. Gemini dashboard "Gemini 3 Flash" ≠ API ID `gemini-3-flash` (404). Работающий: `gemini-3-flash-preview` (20 RPD) или `gemini-2.5-flash` (10K RPD).
- **Наблюдения:** completeness=0.49 на 49 вопросах — 11 вопросов были out-of-corpus → реальный baseline на 38 вопросах ожидается 0.60-0.65. V8 evidence_assess даёт +0.002 — нужен Epic 3 (Multi-Query Expand) для реального улучшения.

### 2026-05-08 (сессия 11)

- **Сделано:** visual_enrichment нода добавлена в V7 pipeline. make_visual_proof_fn() + inject в bridge.py. 270 тестов, все зелёные.
- **Решения:** Ruff-хук снимает неиспользуемые импорты после каждого Edit — добавлять import и использование в одном edit.
- **Наблюдения:** visual_enrichment — no-op без inject → безопасно деплоить.

### 2026-05-07

- **Сделано:** Запущено на VPS порт 8502 (UFW opened). Переиндексировано 7 PDF → 749 чанков (ранее был только 1 документ — кеш из failed SOCKS runs). Исправлен `evaluate_complex`: `top_k` 12 → 24 (ответы стали полными, 8 категорий вместо 2-3 для "программа А"). Установлен PySocks для WARP proxy в Docling. В `.env` прописаны `HTTP(S)_PROXY=socks5h://localhost:40000`.
- **Решения:** WARP proxy через `.env` (а не setenv в процессе) — Docling читает `.env` при старте. ChromaDB `document_cache/*.pkl` был corrupted (0 chunks) — удалили, переиндексировали. Gemini model = `gemini-3-flash-preview` (подтверждено через API list).
- **Наблюдения:** Gemini 503 при перегрузке → bridge.py падает в stub → сырой текст вместо ответа. Нужен retry. `trace_v7.py` показывает `fallback_passages: 12` — это NOT означает использование fallback, это поле существует в state (от rag_simple), misleading.
