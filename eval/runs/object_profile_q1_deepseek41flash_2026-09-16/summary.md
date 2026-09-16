# q1 на DeepSeek V4.1 Flash — 2026-09-16

Модель `deepseek/deepseek-v4.1-flash` через OpenRouter; конфигурация как у
`object_profile_verify_sonnet5_2026-09-16` (v2, answer prompt v4, `department_demo_v2`, 1115 чанков),
кроме модели. Запуск: `--only 1 --model deepseek/deepseek-v4.1-flash`. 1 answer-вызов,
0 schema-повторов, 0 verify-вызовов.

Usage: 4 786 input / 6 293 output (из них 5 030 reasoning). По прайсу OpenRouter на 16.09
($0.15 / $0.60 за 1M) — около $0.005. Счёт не сверялся.

| | gpt-4o-mini | Sonnet 5 | DeepSeek V4.1 Flash |
|---|---|---|---|
| статус | `needs_review/verification_contradiction` | `needs_context/applicability_unclear` | `needs_context/applicability_unclear` |
| подответы (ручная оценка) | — | 4/4 | 3.5/4 — нет 20 м до огнетушителя |
| запрещённые выводы | 1 | 0 | 0 |
| $ за вопрос | — | ~0.075 | ~0.005 |

## Выводы

- Ловушку q1 прошла: порог 50/10 к 8 людям офиса не применила, недостающие величины
  (люди в здании, на этаже, рабочие места, категории) вынесла в clarifying_questions.
- Натянутого вывода Sonnet («2 огнетушителя в офисе = минимум на этаж») нет: факт и норма
  названы рядом без утверждения о соответствии.
- Потеря: расстояние 20 м (п. 406 / п. 5.13.4 Инструкции) не упомянуто.
- Один вопрос, один прогон — не доказательство. Нужны повторы и другие ловушки на пороги.
