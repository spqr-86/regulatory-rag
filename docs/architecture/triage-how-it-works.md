# Как работает тriage (evaluate_triage)

Triage — центральный узел V7-графа. После `rag_simple` он решает, достаточно ли найденных пассажей для ответа, или нужно усилить поиск. Решение принимается детерминированно: только числа, никаких LLM.

---

## Что измеряется

После `rag_simple` возвращает top-12 пассажей с scores. Triage смотрит на три независимые метрики:

| Метрика | Что считает | Порог |
|---------|-------------|-------|
| `top_score` | cosine similarity лучшего пассажа (vector score, не FlashRank) | `HARD_GATE_THRESHOLD = 0.50` |
| `passage_count` | сколько пассажей прошло reranking | `MIN_PASSAGES = 5` |
| `keyword_overlap` | доля ключевых слов запроса, найденных в топ-пассажах | `MIN_KEYWORD_OVERLAP = 0.15` |

Все три должны быть выполнены одновременно — это **hard gate** (`check_hard_gates`).

Дополнительно считается:
- `max_single_doc_ratio` — доля пассажей из одного документа (если > 0.8 и запрос требует multi-doc → `escalation_hint`)

---

## Три исхода (3-way triage)

```
check_full_triage()
    │
    ├── hard_sufficient=True AND no escalation_hint
    │       → sufficient  ──→ generate_answer
    │
    ├── top_score < TRIAGE_SOFT_THRESHOLD (0.38) OR passage_count < MIN_PASSAGES
    │       → clearly_bad ──→ rag_complex
    │
    └── иначе (hard gates частично не выполнены, или escalation_hint)
            → borderline  ──→ llm_verifier
```

**sufficient** — все hard gates зелёные, пассажи разнообразны → генерируем ответ.

**borderline** — score в зоне `[0.38, 0.50)` или diversity проблема → отдаём `llm_verifier`, который решает: `sufficient` / `rewrite` / `escalate`.

**clearly_bad** — score < 0.38 или мало пассажей → сразу `rag_complex` (top-60 + MMR), без LLM-верификации.

---

## Почему vector score, не FlashRank

FlashRank — cross-encoder, его scores близки к 1.0 для любых релевантных пассажей (не калиброваны как вероятности). Тариф calibration не делался. Vector cosine similarity (из ChromaDB) — откалиброванная метрика, 0.0–1.0 с понятным смыслом. Поэтому `top_score` = vector score пассажа с лучшим cosine после FlashRank reorder.

Баг был именно здесь: до фикса `top_score` брался из FlashRank, что давало inflation (~0.95+) и все запросы шли в `sufficient`, даже плохие. После фикса correctness 6.86 → 7.9.

---

## Почему HARD_GATE_THRESHOLD = 0.50

Откалиброван на eval-датасете из 50 вопросов. При 0.50:
- Fast path (sufficient): ~48% запросов, ~5s
- Slow path (borderline + clearly_bad): ~52%, ~22s

При 0.45 — слишком много идёт в sufficient, false-sufficiency растёт. При 0.55 — почти всё уходит в complex, latency +40% без прироста correctness.

TRIAGE_SOFT_THRESHOLD = 0.38 — нижняя граница "вообще что-то нашли". Ниже — заведомо плохой retrieval, llm_verifier не поможет, нужен rag_complex.

---

## Код

- `src/v7/hard_gates.py` — `check_hard_gates()`, `check_full_triage()`
- `src/v7/nodes/evaluate_triage.py` — тонкая нода: читает state → вызывает `check_full_triage()` → пишет `triage`
- `src/v7/config.py` — все пороги с `V7_` env prefix (можно менять через `.env` без правки кода)

---

## На интервью

> **Q:** Почему не LLM решает, достаточно ли данных?
> **A:** LLM недетерминирован: одни и те же пассажи в разные дни дают разное решение. Числовые пороги воспроизводимы — если пользователь жалуется на абстейн, я открываю лог и вижу `top_score=0.47 < 0.50`. С LLM-роутером такой трассировки нет. Плюс ~1s latency экономии на каждом запросе.

> **Q:** Как выбрали пороги 0.50 / 0.38?
> **A:** Калибровка на eval-датасете: перебирали сетку значений, смотрели на distribution correctness по path. 0.50 — точка, после которой прирост correctness от slow path перестаёт компенсировать его latency.
