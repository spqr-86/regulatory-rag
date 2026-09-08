"""Порядок определений в app.py.

Streamlit исполняет скрипт сверху вниз на каждом взаимодействии, поэтому
обращение к функции модуля из кода, лежащего выше её `def`, падает NameError
у пользователя, а не при импорте. Ловим это статически.
"""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _module_level_uses_before_definition(source: str) -> list[tuple[str, int, int]]:
    """Возвращает (имя, строка использования, строка определения) для нарушений."""
    tree = ast.parse(source)

    defined_at = {
        node.name: node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    violations: list[tuple[str, int, int]] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                continue  # тело функции выполняется позже, порядок там не важен
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                def_line = defined_at.get(child.id)
                if def_line is not None and child.lineno < def_line:
                    violations.append((child.id, child.lineno, def_line))
            walk(child)

    walk(tree)
    return violations


@pytest.mark.unit
def test_app_has_no_module_level_use_before_definition():
    violations = _module_level_uses_before_definition(APP.read_text(encoding="utf-8"))
    assert violations == [], "\n".join(
        f"{name} использован на строке {use}, определён на {defined}"
        for name, use, defined in violations
    )
