"""Check that graphcritic sources do not import heavyweight ML frameworks."""

import ast
from pathlib import Path


def test_graphcritic_has_no_heavy_framework_imports():
    package_root = Path(__file__).resolve().parents[1] / "graphcritic"
    forbidden = {"jax", "torch", "flax"}
    violations = []

    for source_path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module.split(".", maxsplit=1)[0],)
            else:
                continue
            violations.extend(f"{source_path.name}: {name}" for name in names if name in forbidden)

    assert not violations, "Forbidden imports found: " + ", ".join(violations)
