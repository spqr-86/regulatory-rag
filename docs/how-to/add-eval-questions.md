# Adding Questions to the Dataset

The evaluation dataset lives in `tests/dataset.csv`. It is a plain UTF-8 CSV with two columns:

```
question,ground_truth
```

## How to Add Questions

Open `tests/dataset.csv` in any text editor and append rows. Each row needs:

- `question` — the question as a user would type it
- `ground_truth` — the correct answer, specific and grounded in the normative documents

**Good ground truth:** includes concrete requirements, deadlines, or references.  
**Bad ground truth:** vague ("depends on the situation") or too short ("yes, required").

## Example Row

```
Как часто проводится повторный инструктаж по охране труда?,Повторный инструктаж проводится не реже одного раза в 6 месяцев (п. 15 Приказа 2464).
```

## Question Categories

Aim for a mix of:
- In-scope questions (answer exists in the corpus)
- Out-of-scope questions (system should abstain)
- False-premise questions (system should correct the premise)

## After Adding Questions

Run a quick eval to check the new questions:

```bash
python eval/run_v7_eval.py --skip-judge --limit 10
```

Or test a single question manually:

```bash
python scripts/trace_v7.py "ваш вопрос здесь"
```
