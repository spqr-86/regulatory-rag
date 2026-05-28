# Как работает Triage (evaluate_triage)

Triage — центральная нода принятия решений в графе V7. После `rag_simple` она определяет, достаточно ли извлечённых фрагментов для генерации ответа, или поиск нужно усилить. Решение детерминировано: только числа, никакого LLM.

---

## Метрики

После того как `rag_simple` возвращает топ-12 фрагментов со scores, triage проверяет три независимых метрики:

| Метрика | Что измеряет | Порог |
|--------|-----------------|-----------|
| `top_score` | Cosine similarity лучшего фрагмента (vector score, не FlashRank) | `HARD_GATE_THRESHOLD = 0.50` |
| `passage_count` | Количество фрагментов, прошедших reranking | `MIN_PASSAGES = 5` |
| `keyword_overlap` | Доля ключевых слов запроса, найденных в топ-фрагментах | `MIN_KEYWORD_OVERLAP = 0.15` |

Все три должны пройти одновременно — это и есть **hard gate** (`check_hard_gates`).

Дополнительно:
- `max_single_doc_ratio` — доля фрагментов из одного документа (если > 0.8 и запрос требует нескольких документов → `escalation_hint`)

---

## Три исхода

```
check_full_triage()
    │
    ├── hard_sufficient=True AND no escalation_hint
    │       → sufficient  ──→ generate_answer
    │
    ├── top_score < TRIAGE_SOFT_THRESHOLD (0.38) OR passage_count < MIN_PASSAGES
    │       → clearly_bad ──→ rag_complex
    │
    └── иначе (hard gates частично не прошли, или escalation_hint)
            → borderline  ──→ llm_verifier
```

**sufficient** — все hard gates зелёные, фрагменты разнообразны → генерируем ответ.

**borderline** — score в зоне `[0.38, 0.50)` или проблема с разнообразием → отправляем в `llm_verifier`, который решает: `sufficient` / `rewrite` / `escalate`.

**clearly_bad** — score < 0.38 или слишком мало фрагментов → сразу в `rag_complex` (топ-60 + MMR), без LLM-верификации.

---

## Почему vector score, а не FlashRank

FlashRank — cross-encoder; его scores кластеризуются около 1.0 для любых релевантных фрагментов (не откалиброваны как вероятности). Для данного домена калибровка не проводилась. Vector cosine similarity (из ChromaDB) — откалиброванная метрика, 0.0–1.0 с понятным смыслом. Поэтому `top_score` = vector score фрагмента с лучшим cosine после перестановки FlashRank.

Баг был здесь: до исправления `top_score` брался из FlashRank, что вызывало завышение (~0.95+) и все запросы шли в `sufficient` даже при плохом retrieval. После исправления: correctness 6.86 → 7.9.

---

## Почему HARD_GATE_THRESHOLD = 0.50

Откалиброван на датасете из 50 вопросов. При 0.50:
- Быстрый путь (sufficient): ~48% запросов, ~5с
- Медленный путь (borderline + clearly_bad): ~52%, ~22с

При 0.45 — слишком много идёт в sufficient, растёт false-sufficiency. При 0.55 — почти всё идёт в complex, задержка +40% без прироста correctness.

`TRIAGE_SOFT_THRESHOLD = 0.38` — нижняя граница «нашли хоть что-то». Ниже этого — retrieval явно плохой, `llm_verifier` не поможет, нужен `rag_complex`.

---

## Код

- `src/v7/hard_gates.py` — `check_hard_gates()`, `check_full_triage()`
- `src/v7/nodes/evaluate_triage.py` — тонкая нода: читает состояние → вызывает `check_full_triage()` → записывает `triage`
- `src/v7/config.py` — все пороги с префиксом `V7_` (переопределение через `.env` без изменения кода)

---

## FAQ для собеседований

> **Q:** Почему LLM не решает, достаточно ли данных?
> **A:** LLM недетерминированы: одни и те же фрагменты в разные дни дают разные решения. Числовые пороги воспроизводимы — если пользователь сообщает об abstain, я открываю лог и вижу `top_score=0.47 < 0.50`. Такой трассируемости с LLM-роутингом нет. Плюс экономия ~1с задержки на каждом запросе.

> **Q:** Как были выбраны пороги 0.50 / 0.38?
> **A:** Откалиброваны на датасете eval: сканировалась сетка значений и наблюдалось распределение correctness по путям. 0.50 — точка, где прирост correctness от медленного пути перестаёт компенсировать его задержку.
