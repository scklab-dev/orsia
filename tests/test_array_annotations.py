"""Repository regression for explicit NumPy array annotations."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_numpy_annotations_have_explicit_dtypes() -> None:
    violations = []
    paths = [
        path
        for directory in (ROOT / "libs", ROOT / "scripts", ROOT / "tests")
        for pattern in ("*.py", "*.pyi")
        for path in directory.rglob(pattern)
    ]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        annotations = [
            node.annotation
            for node in ast.walk(tree)
            if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation is not None
        ]
        annotations.extend(
            node.returns
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.returns is not None
        )
        annotations.extend(node.value for node in ast.walk(tree) if isinstance(node, ast.TypeAlias))
        annotation_nodes = {node for annotation in annotations for node in ast.walk(annotation)}
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.alias) and node.name == "ArrayLike":
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: ArrayLike import")
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name == "ArrayLike":
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: ArrayLike use")
                if name == "NDArray" or (name == "ndarray" and node in annotation_nodes):
                    parent = parents.get(node)
                    if not isinstance(parent, ast.Subscript) or parent.value is not node:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: untyped {name}")
    assert not violations, "\n".join(violations)
