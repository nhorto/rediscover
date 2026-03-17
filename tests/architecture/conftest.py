"""Shared fixtures for architecture tests."""

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parent.parent.parent / "src"


def collect_python_files(root: Path) -> list[Path]:
    """Collect all .py files under root, skipping __pycache__ and .pyc."""
    files = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def parse_imports(filepath: Path) -> list[tuple[str, int]]:
    """Parse a Python file and return all import module paths with line numbers.

    Returns a list of (module_path, line_number) tuples.
    For 'import foo.bar' returns ('foo.bar', lineno).
    For 'from foo.bar import baz' returns ('foo.bar', lineno).
    """
    try:
        tree = ast.parse(filepath.read_text(), filename=str(filepath))
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, node.lineno))
    return imports


def classify_zone(filepath: Path) -> str:
    """Classify a file into its architecture zone based on path.

    Zones: utils, types, providers, domains, app
    Returns 'unknown' for files that don't fit a zone.
    """
    rel = filepath.relative_to(SRC_ROOT)
    parts = rel.parts

    if parts[0] == "utils":
        return "utils"
    elif parts[0] == "types":
        return "types"
    elif parts[0] == "providers":
        return "providers"
    elif parts[0] == "domains":
        return "domains"
    elif parts[0] == "app":
        return "app"
    else:
        return "unknown"


def classify_domain_layer(filepath: Path) -> tuple[str, str] | None:
    """For domain files, return (domain_name, layer).

    layer is one of: types, config, service, init
    Returns None for non-domain files.
    """
    rel = filepath.relative_to(SRC_ROOT)
    parts = rel.parts

    if parts[0] != "domains" or len(parts) < 3:
        return None

    domain_name = parts[1]
    filename = parts[2]

    if filename == "__init__.py":
        return (domain_name, "init")
    elif filename == "types.py":
        return (domain_name, "types")
    elif filename == "config.py":
        return (domain_name, "config")
    elif filename == "service.py":
        return (domain_name, "service")
    else:
        return (domain_name, filename.replace(".py", ""))


def import_zone(module_path: str) -> str | None:
    """Determine the zone of an imported module path.

    Only classifies src.* imports. Returns None for external packages.
    """
    if not module_path.startswith("src."):
        return None

    parts = module_path.split(".")
    if len(parts) < 2:
        return None

    zone = parts[1]
    if zone in ("utils", "types", "providers", "domains", "app"):
        return zone
    return None


def import_domain(module_path: str) -> str | None:
    """For src.domains.X imports, return the domain name X. Otherwise None."""
    if not module_path.startswith("src.domains."):
        return None
    parts = module_path.split(".")
    if len(parts) >= 3:
        return parts[2]
    return None


def import_domain_layer(module_path: str) -> str | None:
    """For src.domains.X.Y imports, return the layer Y. Otherwise None."""
    if not module_path.startswith("src.domains."):
        return None
    parts = module_path.split(".")
    if len(parts) >= 4:
        layer = parts[3]
        return layer
    return None


@pytest.fixture(scope="session")
def all_src_files():
    """All Python files under src/."""
    return collect_python_files(SRC_ROOT)


@pytest.fixture(scope="session")
def src_root():
    return SRC_ROOT
