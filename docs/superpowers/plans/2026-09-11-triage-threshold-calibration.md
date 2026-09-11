# Triage threshold calibration — implementation plan

**Issue:** #9  
**Status:** approved direction; implementation starts after terminal-decision contract  
**Datasets:** original 43 questions = development; later 90 practitioner questions = validation

## Goal

Choose the cheapest simple/complex routing policy that does not increase the risk of
generating from incomplete context. Calibration operates on the exact packed contexts
seen by the generator and keeps the complex acceptance threshold independent from the
simple threshold.

The primary risk metric is `unsafe_generate_rate`: a terminal `generate` decision whose
packed context does not cover every required element in the annotation. Cost proxies are
the escalation rate, number of complex retrievals, and retrieval p50/p95. Binary Hit Rate
remains a diagnostic only.

## Annotation contract

Every development record receives `required_elements`:

```json
[
  {
    "id": "general_rule",
    "label": "general rule or duty",
    "acceptable_chunk_ids": ["document.pdf#12", "document.pdf#15"],
    "critical": true
  }
]
```

Chunks inside one element are alternatives (OR); elements are cumulative (AND). This
preserves the existing meaning of `relevant_chunk_ids` for factoids while allowing an
enumeration, conditions, and exceptions to be scored separately. Coverage is measured on
the final packed context. Records without explicit element annotation are rejected by the
calibration command rather than silently treated as complete.

## Tasks

1. **Decouple the calibrated signals (TDD).**
   - Make `rag_complex` use `COMPLEX_THRESHOLD` directly instead of inheriting the simple
     threshold.
   - Put `MIN_KEYWORD_OVERLAP_ORIGINAL` into both retrieval plans and apply it when
     validating the original-query obligation.
   - Pin both changes with unit tests.

2. **Add the annotation and metric primitives (TDD).**
   - Validate the `required_elements` shape at the eval boundary.
   - Compute element coverage, critical misses, binary hit, and full coverage from packed
     passage identities.
   - Add pure tests for alternative chunks, cumulative elements, and missing annotation.

3. **Build a paired route audit (TDD).**
   - Retrieve simple and hypothetical complex candidates once per question.
   - Evaluate the current terminal contract on both exact packed contexts.
   - Report all eight `escalated × simple_hit × complex_hit` cells, element-coverage
     transitions, terminal route/reason counts, unsafe generations, cost, and p50/p95.
   - Persist per-question records so every aggregate is auditable.

4. **Create the 43-question development annotation.**
   - Factoids get one required element with alternative supporting chunks.
   - Enumerations and high-risk questions are split into categories, conditions, and
     exceptions using the indexed normative text.
   - Validate every referenced chunk against the corpus.

5. **Freeze the current-policy baseline.**
   - Run the paired audit once with production defaults.
   - Record numerator/denominator counts, not only rates.

6. **Run the calibration grid on development.**
   - Sweep only effective simple-path controls: `HARD_GATE_THRESHOLD`, `MIN_PASSAGES`,
     `MIN_KEYWORD_OVERLAP_ACTIVE`, `MIN_KEYWORD_OVERLAP_ORIGINAL`, and the comparison-query
     `require_multi_doc` policy.
   - Exclude `TRIAGE_SOFT_THRESHOLD`; it changes only diagnostic reason codes.
   - Select the lowest-cost profile with zero increase in unsafe generations and critical
     misses versus baseline. Break ties by higher full coverage, then lower p95.

7. **Validate once on the untouched 90-question set.**
   - Annotate it only after the profile is frozen.
   - Reject the candidate if paired risk worsens; do not retune on validation.
   - Apply config defaults only after the validation gate passes, then run unit, full, route,
     and generative quality gates before push.

## Stop conditions

- No threshold change is applied without explicit element annotations for both dev and
  validation.
- If no profile satisfies the risk constraint, retain current defaults and close #9 with a
  measured negative result.
- A generated answer is never used to label the context that produced it.
