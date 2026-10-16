# v8, этап 1 — что переписать Петру для учёта

По плану (`2026-10-07-v8-stage1-langsmith-eval-plan.md`) код реализации пишет Пётр. 16.10.2026
Пётр решил: ради срока RAG код пишет Codex под ревью ассистента, а Пётр потом
реализует эти куски заново сам. Тесты остаются эталоном: удалить реализацию, оставить тест,
написать до PASS.

| # | Что | Файлы | Тест-эталон | Статус кода |
|---|---|---|---|---|
| 1 | Флаг `SIMPLE_REASONING_EFFORT`: per-path effort для быстрого пути; `none` → `reasoning.enabled=false` | `config/settings.py`, `src/infra/llm_factory.py`, `.env.example` | см. ниже | Codex, 16.10 |
| 2 | `reasoning_tokens`, `completion_tokens`, время ответа в outputs LangSmith-target; simple effort в метаданных | `eval/ls/target.py`, `eval/ls/run.py` | см. ниже | Codex, 16.10 |
| 0a | Судьи correctness/citations в LangSmith (задачи 4, 4a.3) | `eval/ls/judge.py` | `tests/eval_ls/test_judge.py` | ассистент, 15.10 (`1eb6807`) |
| 0b | Согласие судьи с разметкой (задача 5) | `eval/ls/align.py` | `tests/eval_ls/test_align.py` | ассистент, 15.10 (`60d89db`) |
| 3 | Pairwise-судья: `parse_preference`, `make_pairwise_evaluator`, CLI | `eval/ls/pairwise.py`, `eval/prompts/judge_pairwise.md` | `tests/eval_ls/test_pairwise.py` (шаг 1 задачи 6 плана) | Codex, 16.10 |

## Что потренировать на каждом

1. Как настройка доходит до запроса к провайдеру: settings → фабрика → `extra_body`; почему
   effort читается при импорте и второй эксперимент нужен в отдельном процессе.
2. Откуда берутся токены: `usage_metadata.output_token_details.reasoning` (langchain-openai) и
   почему сводка 23.09 показала 0 при ненулевых счётчиках по вызовам.
3. `evaluate_comparative(randomize_order=True)`: зачем рандомизировать A/B (позиционное смещение
   судьи) и как ничья превращается в 0.5/0.5.
