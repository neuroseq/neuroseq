"""Keep lower layers from importing the visualization widgets.

``visualization_support`` (shared state and helpers) and ``stimulation``
(stimulus logic) must not depend on ``neuroseq.visualization`` at runtime.
Imports inside ``if TYPE_CHECKING:`` blocks are allowed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "neuroseq"
FORBIDDEN = "neuroseq.visualization"
# Widgets that compose a visualization widget themselves.
ALLOWED = {
    # The stimulation editor embeds the StimWidget it edits.
    "stimulation/stim_editor.py",
}


def _is_type_checking(node: ast.If) -> bool:
    test: ast.expr = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node):
            for statement in node.orelse:
                visit(statement)
            return
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
            found.extend(
                (node.lineno, f"{node.module}.{alias.name}") for alias in node.names
            )
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def _is_forbidden(module: str) -> bool:
    return module == FORBIDDEN or module.startswith(FORBIDDEN + ".")


@pytest.mark.parametrize("package", ["visualization_support", "stimulation"])
def test_package_does_not_import_visualization(package: str) -> None:
    violations: list[str] = []
    for path in sorted((SRC / package).rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(
            f"{relative}:{line} imports {module}"
            for line, module in _runtime_imports(tree)
            if _is_forbidden(module)
        )
    assert not violations, "\n".join(violations)
