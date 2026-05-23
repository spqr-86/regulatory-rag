# V7 Migration — Stage 1 Implementation Plan: NLP Core

**Date:** 2026-02-16
**Design Doc:** `docs/plans/2026-02-16-v7-migration-design.md`
**Source Spec:** `docs/feature/migration-v7` (sections `extract_keywords`, `class BM25Index`, `rrf_merge`, `mmr_select`)

---

## 1. Objective

Implement the `nlp_core` module as defined in the v7 migration design. This module will provide core NLP utilities for text processing and document ranking, completely isolated from the existing codebase.

## 2. Files to Create

1.  `src/v7/nlp_core.py` — Main module file.
2.  `tests/v7/test_nlp_core.py` — Unit tests for the module.

## 3. Dependencies to Add

The following packages need to be added to `requirements.txt`:
- `pymorphy3`
- `razdel`
- `rank_bm25`

## 4. Implementation Steps

### Step 4.1: Add Dependencies
- Add the three required libraries to `requirements.txt`.
- Run `pip install -r requirements.txt` to ensure they are installed in the environment.

### Step 4.2: Create `src/v7/nlp_core.py` Structure
- Create the file `src/v7/nlp_core.py`.
- Add necessary imports from `typing`, `pymorphy3`, `razdel`, `rank_bm25`, and other standard libraries.
- Import `ScoredDoc`, `Doc` types from `src.v7.state_types`.
- Define the function signatures based on the design document.

### Step 4.3: Implement `extract_keywords`
- **Logic:**
    - Initialize `pymorphy3.MorphAnalyzer`.
    - Use `razdel.tokenize` to get tokens.
    - Normalize each token to its normal form using the analyzer.
    - Filter out stop words (e.g., prepositions, conjunctions) and short tokens.
    - Return a list of unique, normalized keywords.
- **Testing:**
    - Write a unit test in `test_nlp_core.py` to verify correct lemmatization and stop-word removal for a sample Russian sentence.

### Step 4.4: Implement `BM25Index` Class
- **Logic:**
    - `__init__`: No initialization logic needed.
    - `build(self, docs: list[Doc])`:
        - Tokenize each document's content using `extract_keywords`.
        - Initialize `rank_bm25.BM25Okapi` with the tokenized corpus.
        - Store the `docs` list and the `bm25` model instance.
    - `query(self, tokens: list[str], top_k: int) -> list[ScoredDoc]`:
        - Use the stored `bm25` model to get document scores for the query tokens.
        - Create `ScoredDoc` objects from the results, combining the original `Doc` with the calculated score.
        - Return the top `top_k` scored documents.
- **Testing:**
    - Write a unit test to build an index from a small set of documents and verify that a query returns the most relevant document with the highest score.

### Step 4.5: Implement `rrf_merge`
- **Logic:**
    - Implement the Reciprocal Rank Fusion algorithm.
    - Iterate through multiple lists of ranked `ScoredDoc` objects.
    - Calculate the RRF score for each unique document ID.
    - Sort the documents by their final RRF score in descending order.
    - Return the merged and re-ranked list of `ScoredDoc` objects.
- **Testing:**
    - Write a unit test with two or more overlapping rankings to ensure the merge logic correctly re-ranks the documents.

### Step 4.6: Implement `mmr_select`
- **Logic:**
    - Implement the Maximal Marginal Relevance algorithm.
    - This function is a fallback and might not be used if the vector store provides native MMR.
    - The implementation will require document embeddings and a query embedding.
    - Iteratively select documents that balance similarity to the query and dissimilarity to already selected documents.
- **Testing:**
    - Write a unit test to verify that the selection promotes diversity among the returned documents.

## 5. Verification

1.  **Unit Tests:** Run `pytest tests/v7/test_nlp_core.py` and ensure all tests pass.
2.  **Linting:** Run `black . && ruff check . --fix` to ensure code formatting and quality standards are met.
3.  **No Regressions:** Confirm that no existing tests have been broken. The changes should be fully isolated.

---
