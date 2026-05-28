# Testing Guide

How to verify that documentation and code are accurate and up to date.

## What to Check

### 1. Links and References

- **Internal links:** Verify that all file references in docs (e.g., `[code](src/main.py)`) point to existing files.
- **External links:** Check that URLs open and lead to current resources.
- **Images:** Confirm that all images render correctly.

### 2. Code Examples

- **Install commands:** Run dependency installation (`pip install -r requirements.txt`) in a clean environment to confirm it works.
- **Run examples:** Execute code examples from README and guides. They should work without errors.
- **Config:** Check that example config files (e.g., `.env.example`) are current and contain all required parameters.

### 3. Versions and Dependencies

- **Version consistency:** Ensure library versions mentioned in docs match `pyproject.toml` or `requirements.txt`.
- **Compatibility:** Confirm that Python version and OS requirements are up to date.

### 4. Project Structure

- **File tree:** If docs include a directory structure, compare it with the actual file layout. Deleted or moved files should be reflected in docs.
- **Module descriptions:** Confirm that descriptions of files and directories match reality.

## Procedure

1. Read the changed or new documentation files.
2. Go through the checklist above.
3. Use automated link checkers if configured, or verify manually.
4. For any inaccuracies found, either fix them immediately or create a backlog item.
