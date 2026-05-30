# RAG-промптинг для compliance/regulatory доменов — ресёрч

**Дата:** 2026-05-29
**Контекст:** Russian regulatory RAG, gemini-2.5-flash (simple) + gemini-3-flash-preview (complex). 30 чанков на запрос. Промпт v2: faithfulness 0.82, correctness 6.1. v3 с усиленным abstain: correctness 6.6 (+0.5), но регрессии — модель отказывается от ответа когда факт явно в чанке.

---

## Ключевые находки

### 1. Abstention paradox
"Sufficient context" работа ICLR 2025: модель не различает high/low confidence retrieval. Если в system prompt усилить "если не уверен — abstain" — модель применяет порог слишком осторожно. Conservative prompt: 60% abstention, basic: 1.8%. Industry consensus: **false negative > false positive по utility** — лучше ответ с caveat, чем отказ.

Источник: `tianpan.co/blog/2026-04-16-rag-retrieval-abstention-empty-corpus`, `arxiv.org/html/2505.13545v1` (Know-Or-Not).

### 2. COVER (ACL 2025) — context-driven over-refusal
Refusal rate сильно зависит от system prompt, количества retrieved документов. Чем больше шумных чанков (>20) — тем выше over-refusal даже на benign запросы. У нас 30 чанков — это много.

Источник: `aclanthology.org/2025.findings-acl.1243.pdf`.

### 3. Gemini 2.5 Flash и длинный контекст
- Single-factoid needle-in-haystack — LITM не подтверждается (26/26).
- Сложные задачи: после 20% утилизации контекста — "contextual memory degradation, confusing past information with current state".
- Safety filters срабатывают чаще на длинном контексте (BlockedPromptException).

Источники: `arxiv.org/html/2511.05850v1`, `introl.com/blog/long-context-llm-infrastructure...`, `databricks.com/blog/long-context-rag-capabilities...`.

### 4. False premise — модели sycophantic by default
FalseQA + arxiv 2510.10965: модели подыгрывают ложной предпосылке. Эффективный паттерн: chain-of-thought "сначала разбери предпосылку → если ложная, **исправь её, не отвечай на вопрос как задан**". Не жёсткий "начни с НЕВЕРНО" (overshoots), а градация.

Источники: `aclanthology.org/2023.acl-long.309.pdf`, `arxiv.org/pdf/2510.10965`.

### 5. Cohere/Command R+ — grounded generation парадигма
Документы как list[dict] с title/snippet. Модель: 1) предсказывает релевантные документы, 2) цитирует, 3) генерирует. Inline citations — native, не промпт. Для Gemini эмулируется в промпте, точность ниже.

Источник: `docs.cohere.com/docs/retrieval-augmented-generation-rag`.

---

## Практические паттерны для нашего промпта

### Pattern A — Tiered answer policy (3 уровня, не 2)
```
DIRECT_ANSWER: чанк содержит точный ответ → факт + цитата
PARTIAL_ANSWER: близкая информация, не точный → выпиши + явно отметь чего нет
ABSTAIN: ни прямого, ни близкого → "В фрагментах информации нет"
```
Это убирает регрессии типа "Программа В сколько раз" — модель не прыгает в ABSTAIN если факт виден.

### Pattern B — Extract-first, qualify-after ⭐
```
ШАГ 1. Выпиши ВСЕ числовые/временные/перечневые факты из чанков 
       по теме вопроса. Не оценивай — отвечают ли они на вопрос.
ШАГ 2. На основе выписанных фактов сформируй ответ.
ШАГ 3. Если факт частично отвечает — "точный ответ X, прямого 
       утверждения о Y в фрагментах нет".
```
Прямо лечит "не реже 1 раза в год" → "нет ответа": факт обязан попасть в выписку на шаге 1.

### Pattern C — False premise через chain-of-thought (градация)
```
Если вопрос содержит утверждение/предпосылку:
1. Найди в чанках факт, прямо подтверждающий или опровергающий
2. Прямо опровергает → "В вопросе допущена неточность: [правильно по Фр.N]"
3. Частично (нюанс/исключение) → "Не всегда: [нюанс по Фр.N]"
4. В чанках нет факта про предпосылку → отвечай как задан, не утверждай 
   что предпосылка верна или нет
```
Жёсткое "Это НЕВЕРНО" — только случай 2.

### Pattern D — Citations
- `[Фрагмент N]` или `[N]` достаточно для Flash, `[doc, p.X]` точнее но больше ошибок на 30 чанках
- В чанке префикс `[Фрагмент 7, ТК РФ ст.227]` — model видит source при ретриве
- Inline citation после каждого факта, не сводная в конце

### Pattern E — Exhaustive enumeration trigger ⭐
```
Если вопрос содержит "какие", "перечислите", "функции", "обязанности",
"виды", "способы", "документы" — это запрос на ПОЛНЫЙ перечень:
- ВСЕ элементы во ВСЕХ релевантных чанках
- Пронумеруй 1..N
- Перед списком: "В фрагментах найдено N элементов"
- Не сокращай, не группируй, не пиши "и т.д."
```
Trigger по keywords надёжнее общей инструкции "будь полным".

### Pattern F — XML структура чанков (Anthropic)
```
<documents>
  <document index="1" source="ТК РФ" section="Ст.228.1">
    <content>...</content>
  </document>
</documents>
```
На длинных контекстах улучшает grounding citation accuracy.

---

## Failure modes которых избегать

1. **Монолитный rule-block с 10+ правилами** — Flash на длинном контексте теряет последовательность. Разбить на нумерованные шаги алгоритма.
2. **Conservative-инструкция без баланса** — приводит к 60% abstention. Каждое "не делай X" → "при этом обязательно делай Y".
3. **30 чанков на Flash** — degradation после 20% контекста. Резать до 10-15, если recall@5 достаточен.
4. **Цитата только в конце** — теряет grounding. Inline после каждого факта.
5. **"Это НЕВЕРНО" как hardcoded префикс** — overshoots. Заменить на градацию.
6. **JSON + длинный prose в одном вызове** — Flash чаще ломает структуру.

---

## Рекомендации для нашего случая (по приоритету)

### Priority 1 — High impact, low risk
- **Pattern B (Extract-first)** — лечит регрессии "Программа В", "Плановое А и Б"
- **Pattern C (false-premise градация)** — лечит "Обеденный перерыв"
- **Pattern E (enumeration trigger)** — лечит partial synthesis (9 вопросов)

### Priority 2 — A/B нужен
- Top-k 30 → 15 на gemini-2.5-flash. Замерить recall — если падает, откатить.
- XML структура чанков — улучшение grounding.

### Priority 3 — Architectural
- Tiered answer (A) — 3 состояния, не 2. faithfulness без потери correctness.
- Двухпроходная генерация (extract → answer) — Cohere паттерн. Эмулировать в одном промпте через "ШАГ 1 / ШАГ 2".

---

## Спорные моменты

- **Long context vs RAG** — для compliance с большой базой НПА RAG однозначно. Long context актуален только для маленьких баз (<200 страниц).
- **Inline citations Gemini** — 2-3% могут быть unfaithful (не native, как у Command R+). Не блокер.
- **Temperature** — у нас 0.0, это правильно для compliance.

---

## Список источников

1. tianpan.co/blog/2026-04-16-rag-retrieval-abstention-empty-corpus — abstention paradox
2. aclanthology.org/2025.findings-acl.1243.pdf — COVER, over-refusal
3. arxiv.org/html/2505.13545v1 — Know-Or-Not, conservative vs basic
4. arxiv.org/html/2511.05850v1 — Gemini 2.5 Flash long context
5. introl.com/blog/long-context-llm-infrastructure... — degradation patterns
6. databricks.com/blog/long-context-rag-capabilities-openai-o1-and-google-gemini — failure modes
7. arxiv.org/html/2411.03538v1 — Long Context RAG Performance taxonomy
8. docs.cohere.com/docs/retrieval-augmented-generation-rag — grounded generation
9. docs.cohere.com/page/migrating-prompts — migration to inline citations
10. mbrenndoerfer.com/writing/rag-prompt-engineering-context-citations — citation strategies
11. mbrenndoerfer.com/writing/hallucination-mitigation — refusal threshold
12. aclanthology.org/2023.acl-long.309.pdf — FalseQA framework
13. arxiv.org/pdf/2510.10965 — CoT for false premise
14. aclanthology.org/2026.eacl-long.321.pdf — RefusalBench

## Применяется в

- `prompts/agents/generate_answer_v4.j2` (в работе) — Patterns B, C, E
