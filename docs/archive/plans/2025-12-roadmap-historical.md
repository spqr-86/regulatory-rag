# 📊 План развития проекта Safety Incident Analyzer

> ⚠️ **HISTORICAL (pre-V7, archived 2026-05-14).** Этот roadmap описывает планы
> до V7-миграции и легаси Multi-Agent RAG. Сохранён как исторический документ.
> Актуальное состояние — в `CLAUDE.md` (Session Log) и `docs/plans/backlog.md`.

**Дата создания:** 2025-12-14
**Последнее обновление:** 2026-02-09
**Фокус:** Multi-Agent RAG с единым ReAct-агентом, regex-фильтр, thinking levels, Term Glossary

---

## 🎯 Текущее состояние

### ✅ Что уже реализовано

1. **Базовая система оценки:**
   - LLM-as-a-Judge в `src/custom_evaluators.py` (`check_correctness`)
   - Интеграция с LangSmith для A/B тестирования
   - Встроенные evaluators: `qa` (correctness), `context_qa` (faithfulness)

2. **Метрики Retrieval качества (`src/retrieval_metrics.py`):**
   - Hit Rate @ K — проверка наличия релевантного документа в top-K
   - Mean Reciprocal Rank (MRR) — позиция первого релевантного документа
   - Precision @ K, Recall @ K, NDCG @ K
   - Оптимизированный поиск (K=40, Rerank Top-20)

3. **Метрики Generation качества (`src/advanced_generation_metrics.py`):**
   - Faithfulness evaluation (с поддержкой CoT и логического вывода)
   - Answer Relevance, Context Relevance, Completeness
   - Citation Quality (поддержка формата `[Источник: ...]`)

4. **Многоагентная система (MAS) на базе LangGraph:**
   - 3-Agent Workflow: RelevanceChecker → ResearchAgent (с CoT) → VerificationAgent (QA)
   - Поддержка циклов доработки (Research Loop, max 3)
   - Визуализация хода мыслей (Chain-of-Thought) в UI

5. **Multi-Agent RAG с единым ReAct-агентом (Gemini 3):**
   - Regex-фильтр → RAG Agent (Flash, thinking: 8192) → Verifier (Flash, thinking: 1024)
   - Единый агент с `search_documents` + `visual_proof` инструментами и условной декомпозицией
   - Regex-классификация chitchat/out_of_scope без LLM (детерминированно, zero-cost)
   - Ревизия с передачей draft_answer + feedback верификатора
   - BASE_RULES макрос для нормативной точности (10 краевых случаев)
   - Term Glossary (`config/term_glossary.yaml`) — детерминированная расшифровка доменных сокращений с stem-based matching для русской морфологии

6. **Prompt Management System v2:**
   - Версионирование промптов (`registry.yaml`)
   - Jinja2 макрос `base_rules.j2` с интеграцией глоссария
   - Query Expansion v2 (без хардкода маппингов)
   - RAG Agent v1 (единый ReAct с условной декомпозицией), Verifier v2 (6 критериев)

7. **CI/CD и скрипты:**
   - GitHub Actions workflow для автоматического eval
   - Скрипт `run_full_evaluation.py` с поддержкой `--mode mas/rag`

### ❌ Что отсутствует

- W&B/MLflow интеграция для experiment tracking
- Production мониторинг с логированием запросов
- Drift detection (автоматическая проверка деградации)
- Систематический error analysis и failure cases
- Расширенный датасет (целевое значение: 100+ вопросов, текущее: 41)
- Retrieval dataset с аннотацией релевантности чанков
- Edge cases dataset (краевые случаи)
- Hyperparameter tuning infrastructure

---

## 🚀 План развития (Roadmap)

### **Этап 1: Фундамент метрик (Приоритет: ВЫСОКИЙ)**

#### 1.1 Метрики Retrieval качества

**Цель:** Измерить качество поиска релевантных документов до генерации ответа

**Задачи:**
```python
# Создать: src/retrieval_metrics.py

Метрики для реализации:
├── Hit Rate @ K         # Найден ли релевантный док в top-K
├── MRR (Mean Reciprocal Rank)  # На какой позиции первый релевантный
├── NDCG @ K            # Normalized Discounted Cumulative Gain
├── Precision @ K       # Точность в top-K
└── Recall @ K          # Полнота в top-K
```

**Референс:** Используйте RAGAS библиотеку или реализуйте вручную

**Файлы:**
- `src/retrieval_metrics.py` - реализация метрик
- `tests/test_retrieval_metrics.py` - unit тесты
- `eval/retrieval_evaluation.py` - скрипт для eval

**Зависимости датасета:**
- Расширить `tests/dataset.csv` → добавить поля:
  - `relevant_chunks`: список релевантных фрагментов документов
  - `irrelevant_chunks`: примеры нерелевантных фрагментов

**Критерий завершения:**
- [ ] Все метрики реализованы и протестированы
- [ ] Интеграция с `run_ab_test.py`
- [ ] Baseline замеры для текущей системы

---

#### 1.2 Метрики Generation качества

**Цель:** Комплексная оценка качества генерируемых ответов

**Задачи:**
```python
# Создать: src/generation_metrics.py

Метрики категории:
├── Correctness (уже есть check_correctness)
│   └── Улучшить: добавить Rubric-based evaluation
├── Faithfulness / Groundedness
│   ├── Context Precision  # Все ли предложения подтверждены контекстом?
│   ├── Context Recall     # Весь ли нужный контекст использован?
│   └── Hallucination Score # Детектор выдуманных фактов
├── Relevance
│   ├── Answer Relevance   # Отвечает ли на вопрос?
│   └── Context Relevance  # Релевантен ли контекст вопросу?
└── Качество ответа
    ├── Completeness       # Полнота ответа
    ├── Conciseness        # Лаконичность
    └── Citation Quality   # Корректность цитат [cite: X]
```

**Референс:**
- RAGAS framework (ragas-ai/ragas)
- TruLens Eval
- LangChain Evaluators

**Файлы:**
- `src/generation_metrics.py`
- `src/hallucination_detector.py` - отдельный детектор
- `src/citation_validator.py` - валидация цитат

**Критерий завершения:**
- [ ] Реализовано 8+ метрик
- [ ] Автоматический детектор галлюцинаций
- [ ] Валидация формата цитат [cite: X, Y, Z]

---

#### 1.3 End-to-End метрики

**Цель:** Оценка всего RAG пайплайна

**Задачи:**
```python
# Метрики пайплайна:
├── Latency
│   ├── Retrieval time
│   ├── Reranking time
│   ├── Generation time
│   └── Total response time
├── Cost
│   ├── Embedding API calls
│   ├── LLM tokens (prompt + completion)
│   └── Total cost per query
└── User Experience
    ├── Answer length distribution
    ├── Citation count statistics
    └── Agent rejection rate (RelevanceChecker)
```

**Файлы:**
- `src/pipeline_metrics.py`
- `utils/cost_tracker.py` - трекинг стоимости API

**Критерий завершения:**
- [ ] Автоматический сбор метрик в `run_ab_test.py`
- [ ] Dashboard с визуализацией (опционально: Streamlit или logs)

---

### **Этап 2: Расширение датасета (Приоритет: ВЫСОКИЙ)**

#### 2.1 Увеличение объема golden questions

**Текущее состояние:** 16 вопросов (недостаточно для robust evaluation)

**Цель:** Минимум 100 golden questions, оптимально 200+

**Задачи:**

1. **Генерация синтетических вопросов:**
```python
# Создать: scripts/generate_questions.py

Стратегии генерации:
├── От документа к вопросу (Doc → Q)
│   ├── Использовать LLM для генерации вопросов из чанков
│   └── Автоматическая генерация эталонных ответов
├── Вариации существующих вопросов
│   ├── Перефразирование
│   ├── Упрощение/Усложнение
│   └── Изменение формулировки
└── Вопросы разной сложности
    ├── Простые (факты из 1 документа)
    ├── Средние (синтез из 2-3 чанков)
    └── Сложные (multi-hop reasoning)
```

2. **Ручная валидация:**
   - Создать интерфейс для ревью сгенерированных вопросов
   - Файл: `scripts/review_questions.py` (простой CLI или Streamlit)

3. **Категоризация вопросов:**
```csv
question,ground_truth,category,difficulty,required_hops
"Кто проводит вводный инструктаж?","...","instructazh",1,1
"С какой периодичностью...","...","periodic_training",2,1
```

**Файлы:**
- `scripts/generate_questions.py`
- `scripts/review_questions.py`
- `tests/dataset_extended.csv` - расширенный датасет
- `tests/dataset_categories.json` - метаданные категорий

**Критерий завершения:**
- [ ] 100+ вопросов с эталонами
- [ ] Разметка по категориям и сложности
- [ ] Баланс по темам (инструктажи, СИЗ, обучение, стажировка, и т.д.)

---

#### 2.2 Датасет для retrieval метрик

**Цель:** Аннотировать релевантность чанков для каждого вопроса

**Формат:**
```json
{
  "question": "Кто проводит вводный инструктаж?",
  "ground_truth": "...",
  "relevant_chunks": [
    {"chunk_id": "doc_123_chunk_5", "relevance": 2},  // 2 = highly relevant
    {"chunk_id": "doc_456_chunk_12", "relevance": 1}  // 1 = somewhat relevant
  ],
  "irrelevant_chunks": ["doc_789_chunk_3"]
}
```

**Задачи:**
1. Создать скрипт для semi-automatic аннотации
2. Использовать LLM для первичной разметки → ручная коррекция
3. Хранить в `tests/retrieval_dataset.jsonl`

**Файлы:**
- `scripts/annotate_relevance.py`
- `tests/retrieval_dataset.jsonl`

**Критерий завершения:**
- [ ] Минимум 50 вопросов с аннотированными чанками
- [ ] Инструкция по аннотации в `tests/ANNOTATION_GUIDE.md`

---

### **Этап 3: Автоматизация evaluation (Приоритет: СРЕДНИЙ)**

#### 3.1 CI/CD pipeline для evaluation

**Цель:** Автоматический запуск eval при изменении кода/моделей

**Задачи:**

1. **GitHub Actions workflow:**
```yaml
# .github/workflows/evaluation.yml

Triggers:
- Push в main/develop ветки
- Pull Request
- Ручной запуск (workflow_dispatch)
- По расписанию (например, weekly)

Steps:
1. Setup Python environment
2. Load test dataset
3. Run retrieval metrics
4. Run generation metrics
5. Compare with baseline
6. Post results as PR comment
7. Block merge if metrics degraded > threshold
```

2. **Baseline tracking:**
   - Хранить baseline метрики в `benchmarks/baseline.json`
   - Автоматическое сравнение: новая версия vs baseline
   - Регрессия → блокировка merge

**Файлы:**
- `.github/workflows/evaluation.yml`
- `scripts/run_full_eval.py` - единая точка входа для eval
- `benchmarks/baseline.json`
- `benchmarks/results_history.jsonl` - история всех запусков

**Критерий завершения:**
- [ ] Working GitHub Action
- [ ] Автоматические комментарии в PR с метриками
- [ ] Документация в `docs/CI_EVAL.md`

---

#### 3.2 Experiment tracking

**Цель:** Версионирование экспериментов и гиперпараметров

**Инструменты:**
- Weights & Biases (W&B)
- MLflow
- LangSmith (уже используется)

**Рекомендация:** W&B для лучшей визуализации + LangSmith для трассировки

**Задачи:**
```python
# Интеграция W&B в run_ab_test.py

Логировать:
├── Гиперпараметры
│   ├── CHUNK_SIZE, CHUNK_OVERLAP
│   ├── VECTOR_SEARCH_K
│   ├── HYBRID_RETRIEVER_WEIGHTS
│   ├── Модели (LLM, embeddings, reranker)
│   └── Temperature, top_p и т.д.
├── Метрики
│   ├── Все retrieval метрики
│   ├── Все generation метрики
│   └── Pipeline метрики
└── Артефакты
    ├── Примеры ответов (best/worst)
    ├── Confusion matrices
    └── Error analysis
```

**Файлы:**
- `src/experiment_tracker.py` - обертка над W&B
- `config/experiment_config.yaml` - шаблон конфигурации эксперимента
- Обновить `run_ab_test.py` для логирования

**Критерий завершения:**
- [ ] Все эксперименты логируются в W&B
- [ ] Дашборды с метриками
- [ ] Воспроизводимость экспериментов (config → exact results)

---

### **Этап 4: Production мониторинг (Приоритет: СРЕДНИЙ)**

#### 4.1 Online evaluation

**Цель:** Мониторинг качества в production (Streamlit app)

**Задачи:**

1. **Implicit feedback:**
```python
# В app.py добавить:

Собирать метрики:
├── Query характеристики
│   ├── Длина вопроса
│   ├── Язык (всегда RU?)
│   └── Тип вопроса (classification)
├── Retrieval метрики
│   ├── Количество найденных доков
│   ├── Scores топ документов
│   └── Reranker scores
├── Generation метрики
│   ├── Длина ответа
│   ├── Количество цитат
│   ├── Latency
│   └── Cost
└── Agent метрики
    ├── Какие агенты сработали
    ├── Количество итераций Research/Verify
    └── Причины rejection (если RelevanceChecker отклонил)
```

2. **Explicit feedback (опционально):**
   - Кнопки 👍/👎 в UI
   - Форма для репорта неправильных ответов
   - Сохранение в `production_logs/user_feedback.jsonl`

**Файлы:**
- `src/production_monitor.py`
- `production_logs/queries.jsonl` - все запросы
- `production_logs/user_feedback.jsonl` - фидбек пользователей
- `scripts/analyze_production_logs.py` - анализ логов

**Критерий завершения:**
- [ ] Логирование всех запросов
- [ ] Дашборд для мониторинга (Streamlit отдельная страница?)
- [ ] Еженедельные автоматические отчеты

---

#### 4.2 Drift detection

**Цель:** Обнаружение деградации модели со временем

**Задачи:**
1. Периодически прогонять golden dataset через production систему
2. Сравнивать метрики с baseline
3. Алерты при снижении качества > threshold

**Файлы:**
- `scripts/run_drift_detection.py`
- `config/drift_thresholds.yaml`
- Интеграция с GitHub Actions (scheduled weekly run)

**Критерий завершения:**
- [ ] Автоматический drift check раз в неделю
- [ ] Email/Slack уведомления при детекте drift

---

### **Этап 5: Анализ ошибок и улучшения (Приоритет: НИЗКИЙ)**

#### 5.1 Failure analysis

**Цель:** Систематический анализ случаев плохой работы

**Задачи:**
```python
# Создать: scripts/error_analysis.py

Автоматически выявлять:
├── Категории ошибок
│   ├── Retrieval failures (релевантный док не найден)
│   ├── Hallucinations (ответ не из контекста)
│   ├── Incomplete answers (неполнота)
│   ├── Incorrect citations (неверные цитаты)
│   └── Refusal errors (RelevanceChecker ошибочно отклонил)
├── Паттерны
│   ├── Проблемные типы вопросов
│   ├── Проблемные документы
│   └── Краевые случаи (очень короткие/длинные вопросы)
└── Приоритизация
    └── Сортировка по частоте/критичности
```

**Файлы:**
- `scripts/error_analysis.py`
- `analysis/error_reports/YYYY-MM-DD_errors.md` - отчеты
- `analysis/failure_cases.csv` - база failure cases

**Критерий завершения:**
- [ ] Автоматическая генерация error reports
- [ ] Выявлено топ-5 паттернов ошибок
- [ ] План улучшений на основе анализа

---

#### 5.2 Edge cases dataset

**Цель:** Покрыть тестами краевые случаи

**Примеры edge cases:**
- Вопросы на смежные темы (не охрана труда)
- Очень общие вопросы ("Расскажи про инструктажи")
- Вопросы с противоречивой информацией в документах
- Вопросы, требующие актуальной информации (год издания СНиП)
- Вопросы с опечатками/грамматическими ошибками
- Multi-hop reasoning (требуют синтеза из 3+ документов)

**Файлы:**
- `tests/edge_cases_dataset.csv`
- `tests/adversarial_questions.csv` - намеренно сложные

**Критерий завершения:**
- [ ] 30+ edge cases с эталонами
- [ ] Baseline метрики на edge cases
- [ ] Идентификация слабых мест системы

---

### **Этап 6: Оптимизация на основе метрик (Приоритет: НИЗКИЙ)**

#### 6.1 Hyperparameter tuning

**Цель:** Автоматический подбор оптимальных гиперпараметров

**Параметры для оптимизации:**
```python
search_space = {
    # Retrieval
    'VECTOR_SEARCH_K': [5, 10, 15, 20],
    'HYBRID_WEIGHTS_VECTOR': [0.4, 0.5, 0.6, 0.7],
    'RERANKER_TOP_K': [3, 5, 7, 10],

    # Chunking
    'CHUNK_SIZE': [800, 1000, 1200, 1500],
    'CHUNK_OVERLAP': [100, 150, 200, 250],

    # LLM
    'TEMPERATURE': [0.0, 0.1, 0.3, 0.5],

    # Agents
    'MAX_RESEARCH_ITERATIONS': [1, 2, 3],
}
```

**Метод:** Grid Search или Bayesian Optimization (Optuna)

**Метрика оптимизации:** Weighted combination:
```python
score = 0.5 * correctness + 0.3 * faithfulness + 0.1 * latency_penalty + 0.1 * cost_penalty
```

**Файлы:**
- `scripts/hyperparameter_search.py`
- `config/tuning_results.json`

**Критерий завершения:**
- [ ] Найдена оптимальная конфигурация
- [ ] Улучшение метрик минимум на 5% от baseline

---

#### 6.2 A/B тестирование в production

**Цель:** Сравнение версий системы на реальных пользователях

**Задачи:**
1. Реализовать feature flags в `app.py`
2. Случайное распределение пользователей (A: старая версия, B: новая)
3. Сбор метрик для обеих групп
4. Статистическая значимость различий

**Файлы:**
- `src/ab_test_manager.py`
- `config/feature_flags.yaml`
- `scripts/analyze_ab_test.py` - статистический анализ

**Критерий завершения:**
- [ ] Инфраструктура для A/B тестов
- [ ] Провести минимум 1 A/B тест (например, новая модель embeddings)

---

## 📅 Рекомендуемый порядок реализации

### **Sprint 1 (1-2 недели): Фундамент** — ✅ ЗАВЕРШЕН
1. ✅ Retrieval метрики (`src/retrieval_metrics.py`) — 8 функций реализовано
2. ✅ Generation метрики (`src/advanced_generation_metrics.py`) — 6+ функций
3. ✅ Детектор галлюцинаций (`evaluate_faithfulness`)
4. 🔄 Расширение датасета до 50 вопросов — **текущее: 41 вопрос**

**Deliverables:**
- ✅ Working eval pipeline с 10+ метриками
- ✅ Baseline замеры для текущей системы
- 🔄 Отчет с анализом baseline — частично

**Статус:** Основные метрики реализованы. Требуется расширение датасета.

---

### **Sprint 2 (1-2 недели): Автоматизация** — 🔄 В ПРОЦЕССЕ
1. ✅ Prompt Management System (Versioning, Jinja2, Registry) — внедрено
2. ❌ Интеграция W&B для experiment tracking — **не реализовано**
3. ✅ CI/CD pipeline (GitHub Actions) — `.github/workflows/evaluation.yml`
3. ❌ Расширение датасета до 100 вопросов — **текущее: 41**
4. ❌ Retrieval dataset с аннотацией релевантности — **не создан**

**Deliverables:**
- ✅ Автоматический eval при каждом PR
- ❌ W&B дашборды с метриками
- 🔄 Расширенный тестовый датасет — требуется 59+ вопросов

**Статус:** CI/CD настроен. Необходимо: W&B интеграция и расширение датасета.

---

### **Sprint 3 (1-2 недели): Production и анализ** — ❌ НЕ НАЧАТ
1. ❌ Production мониторинг в Streamlit — **не реализовано**
2. ❌ Drift detection — **скрипты отсутствуют**
3. ❌ Error analysis и failure cases — **не реализовано**
4. ❌ Edge cases dataset — **не создан**

**Deliverables:**
- ❌ Production мониторинг с логированием
- ❌ Первый error analysis report
- ❌ Идентификация топ-5 проблем

**Статус:** Спринт не начат. Требуется полная реализация.

---

### **Sprint 4+ (опционально): Оптимизация**
1. Hyperparameter tuning
2. A/B testing infrastructure
3. Continuous improvement на основе production данных

---

## 📊 KPI успеха проекта

### Метрики для отслеживания:

**Retrieval качество:**
- Hit Rate @ 10 > 0.90
- MRR > 0.75
- NDCG @ 10 > 0.85

**Generation качество:**
- Correctness score (0-10) > 8.0
- Faithfulness (0-1) > 0.95
- Hallucination rate < 5%
- Citation accuracy > 95%

**Operational:**
- P95 latency < 10 секунд
- Cost per query < $0.05
- Uptime > 99%

---

## 🛠️ Инструменты и библиотеки

### Evaluation Frameworks:
- **RAGAS** - готовые метрики для RAG (context precision, faithfulness, etc.)
- **TruLens** - трассировка и eval
- **DeepEval** - LLM evaluation framework
- **LangSmith** (уже используется) - трассировка и dataset management

### Experiment Tracking:
- **Weights & Biases** - визуализация экспериментов
- **MLflow** - альтернатива W&B
- **LangSmith** - для LangChain специфичных метрик

### Тестирование:
- **pytest** - unit тесты для метрик
- **Hypothesis** - property-based testing
- **locust** - load testing для production

---

## 📚 Референсы и материалы

### Статьи и гайды:
1. [Evaluating RAG Applications](https://www.ragas.io/) - RAGAS docs
2. [LangChain Evaluation Guide](https://python.langchain.com/docs/guides/evaluation/)
3. [OpenAI Evals Framework](https://github.com/openai/evals)
4. [Google's RLHF for RAG](https://arxiv.org/abs/2310.02743)

### Best Practices:
- Минимум 100 golden questions для надежного eval
- Комбинация automated + human evaluation
- Continuous monitoring в production
- Regular drift detection (weekly/monthly)
- Версионирование датасетов и моделей

---

## ✅ Чеклист готовности к production

- [x] Retrieval метрики реализованы и stable — **src/retrieval_metrics.py**
- [ ] Generation метрики показывают > 80% correctness — **требует baseline eval**
- [ ] Golden dataset > 100 вопросов — **текущее: 41 вопрос**
- [x] CI/CD eval pipeline работает — **.github/workflows/evaluation.yml**
- [ ] Production мониторинг настроен — **не реализовано**
- [ ] Drift detection активен — **не реализовано**
- [ ] Error analysis проводится регулярно — **не реализовано**
- [x] Документация по eval обновлена — **quick-start.md, examples.md**
- [ ] Baseline метрики задокументированы — **требует полного eval запуска**
- [ ] Алерты настроены для критических метрик — **не реализовано**

**Прогресс:** 3/10 пунктов завершено (30%)

---

## 🎯 Следующие шаги (Action Items)

### ✅ Завершено:
1. ✅ Создать `src/retrieval_metrics.py` с базовыми метриками (Hit Rate, MRR, NDCG)
2. ✅ Создать `src/advanced_generation_metrics.py` (faithfulness, relevance, completeness)
3. ✅ Реализовать детектор галлюцинаций (`evaluate_faithfulness`)
4. ✅ Создать GitHub Action для автоматического eval

### Немедленно (текущий приоритет):
1. 🔄 Расширить датасет до 50+ вопросов (текущее: 41)
2. 🔄 Создать Retrieval dataset с аннотацией релевантных чанков
3. 🔄 Запустить полную baseline evaluation и сохранить результаты
4. 🔄 Настроить W&B или MLflow для experiment tracking

### Краткосрочно (этот месяц):
1. Довести датасет до 100 вопросов
2. Реализовать production мониторинг (`production_logs/`)
3. Создать скрипты drift detection
4. Провести первый error analysis

### Долгосрочно (квартал):
1. Production дашборды для мониторинга метрик
2. Edge cases dataset (30+ краевых случаев)
3. Hyperparameter optimization
4. A/B testing infrastructure

---

**Автор плана:** AI Assistant
**Дата создания:** 2025-12-14
**Последнее обновление:** 2026-02-02

_Для вопросов и предложений: создайте issue в репозитории_
