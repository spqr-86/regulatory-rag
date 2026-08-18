# Regulatory MAS Agent — план

Дата: 2026-06-05

## Архитектура

```
User query
    ↓
input_guard (on-topic фильтр)
    ↓
Coordinator
  ├── create_plan() → CoordinatorPlan (Pydantic)
  ├── RegulationsAgent (SIA pipeline + HyDE промпт)
  ├── WebAgent (tavily)
  └── CalcAgent (eval())
      ↓
  synthesize()
      ↓
  CriticAgent → score < 7 → revision
      ↓
  Final answer
```

## Память

- **Short-term:** SqliteSaver (`data/memory.db`) — история диалога в рамках сессии
- **Long-term:**
  - SQLite таблица `user_profiles` — организация, отрасль
  - ChromaDB коллекция `user_history` — семантический поиск по прошлым запросам

Переключение на Postgres/Redis — одна строчка (LangGraph адаптеры).

## Файловая структура

```
regulatory-rag/agents/react_agent/
├── PLAN.md
├── tools.py          — 3 @tool: search_regulations, web_search, calculator
├── specialists.py    — 3 SpecializedAgent на create_react_agent()
├── coordinator.py    — CoordinatorAgent + Pydantic schemas
├── critic.py         — CriticAgent + revision loop
├── memory.py         — SqliteSaver + user_profiles + ChromaDB history
├── agent_app.py      — Streamlit UI (TAO steps)
└── eval/
    ├── task_basket.py — 10-15 нормативных сценариев
    ├── graders.py     — deterministic + LLM-as-Judge
    └── run_eval.py    — Iron User + benchmark vs baseline SIA
```

## Этапы

### Этап 1 — Core MAS (из seminar_3, Part 2)
- [ ] tools.py: 3 @tool с docstring
- [ ] specialists.py: RegulationsAgent, WebAgent, CalcAgent
- [ ] coordinator.py: CoordinatorAgent + Pydantic (SubTask, AgentResult, CoordinatorPlan)
- [ ] Smoke-test в терминале: 3 вопроса

### Этап 1.5 — Memory (из seminar_2, Step 1)
- [ ] memory.py: SqliteSaver для short-term
- [ ] Таблица user_profiles в SQLite
- [ ] ChromaDB коллекция user_history
- [ ] Coordinator читает профиль и инжектирует в system prompt

### Этап 2 — Качество (из seminar_2 + seminar_3)
- [ ] HyDE промпт для RegulationsAgent (seminar_2, Step 2)
- [ ] input_guard: LLM on-topic классификатор (seminar_2, Step 3)
- [ ] critic.py: CriticAgent + revision loop (seminar_3, Part 3)

### Этап 3 — Eval (из seminar_4)
- [ ] task_basket.py: 10 сценариев (простые / multi-domain / tricky)
- [ ] graders.py: deterministic + LLM-as-Judge (usefulness, groundedness, efficiency)
- [ ] Iron User для диалоговых сценариев
- [ ] Benchmark: MAS vs baseline SIA RAG на dataset_original.csv

### Этап 4 — UI + деплой
- [ ] agent_app.py: Streamlit чат + раскрывающиеся TAO шаги
- [ ] Деплой VPS порт 8504

## Источники из ноутбуков

| Компонент | Ноутбук | Секция |
|-----------|---------|--------|
| @tool паттерн | lecture_1_2 | 1.3–1.4 |
| create_react_agent() | seminar_3 | Part 1 |
| MemorySaver / SqliteSaver | seminar_2 | Step 1 |
| HyDE промпт | seminar_2 | Step 2 |
| input_guard | seminar_2 | Step 3 |
| CoordinatorAgent + Pydantic | seminar_3 | Part 2 |
| CriticAgent + Reflexion | seminar_3 | Part 3 |
| Task basket + graders | seminar_4 | Part 2–3 |
| Iron User | seminar_4 | Part 5 |
| LLM-as-Judge | seminar_4 | Part 7 |
