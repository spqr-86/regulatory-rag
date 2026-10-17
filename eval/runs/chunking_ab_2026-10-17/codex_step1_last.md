Готово. На 117 вопросах, **hybrid**:

| Бюджет | Found A → B | Complete A → B |
|---|---:|---:|
| 8000 | 90,6% → 75,2% | **41,9% → 73,5%** |
| 16000 | 95,7% → 89,7% | **54,7% → 86,3%** |

Спорных меток — **12/254 (4,7%)**. Без родителя — **1142/6871 чанков (16,6%)**.

Ограничения: целевые G005/G047 в hybrid B остались неполными при обоих бюджетах. `complete` проверяет уникальные слова, поэтому не гарантирует буквального наличия каждого подпункта.

**22 теста и линтер проходят**; сохранённые контексты, покрытия, агрегаты и хэши входов проверены. Сети, платных вызовов и коммитов не было.

Созданы:

- [norms.py](/home/petr/projects/ai/regulatory-rag/eval/chunking_ab/norms.py), [step1.py](/home/petr/projects/ai/regulatory-rag/eval/chunking_ab/step1.py), [тесты](/home/petr/projects/ai/regulatory-rag/tests/test_chunking_ab_norms.py).
- В `eval/runs/chunking_ab_2026-10-17/`: [step1_report.md](/home/petr/projects/ai/regulatory-rag/eval/runs/chunking_ab_2026-10-17/step1_report.md), `norm_labels.jsonl`, `norm_labels_disputed.md`, `step1_norms.jsonl`, `step1_parents.jsonl`, `step1_details.jsonl`, `step1_results.json`, `step1_run.log`.