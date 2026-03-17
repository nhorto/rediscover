"""Architecture tests: coding conventions and structural rules.

Validates file size limits, naming conventions, domain structure,
and absence of hardcoded secrets.

Error messages use VIOLATION / FIX / RULE format for clarity.
"""

import re

import pytest

from tests.architecture.conftest import (
    SRC_ROOT,
    classify_zone,
    collect_python_files,
    parse_imports,
)

# Maximum lines for domain files (types, config, service)
DOMAIN_FILE_LINE_LIMIT = 300

# Patterns that suggest hardcoded API keys or secrets
SECRET_PATTERNS = [
    (r"""['"]sk-[a-zA-Z0-9]{20,}['"]""", "OpenAI-style API key (sk-...)"),
    (r"""['"]key-[a-zA-Z0-9]{20,}['"]""", "API key pattern (key-...)"),
    (r"""(?:api_key|apikey|api_token)\s*=\s*['"][a-zA-Z0-9]{10,}['"]""", "Hardcoded api_key assignment"),
    (r"""(?:secret|password|token)\s*=\s*['"][a-zA-Z0-9]{10,}['"]""", "Hardcoded secret/password/token"),
]


class TestFileSizeLimits:
    """Validate that domain files stay under the line limit."""

    def test_domain_files_under_limit(self):
        """Domain files (types, config, service) should be under 300 lines."""
        files = collect_python_files(SRC_ROOT)
        domain_files = [
            f for f in files
            if classify_zone(f) == "domains" and f.name != "__init__.py"
        ]

        violations = []
        for filepath in domain_files:
            line_count = len(filepath.read_text().splitlines())
            if line_count > DOMAIN_FILE_LINE_LIMIT:
                rel_path = filepath.relative_to(SRC_ROOT)
                violations.append(
                    f"  {rel_path}: {line_count} lines (limit: {DOMAIN_FILE_LINE_LIMIT})"
                )

        if violations:
            viol_text = "\n".join(violations)
            pytest.fail(
                f"VIOLATION: Domain files exceed {DOMAIN_FILE_LINE_LIMIT}-line limit:\n"
                f"{viol_text}\n"
                f"FIX: Split large files. Extract helpers into separate modules within the domain.\n"
                f"RULE: Domain files (types.py, config.py, service.py) must be under "
                f"{DOMAIN_FILE_LINE_LIMIT} lines."
            )


class TestDomainStructure:
    """Validate that each domain directory has the required files."""

    def test_each_domain_has_init(self):
        """Every domain directory must have an __init__.py."""
        domains_dir = SRC_ROOT / "domains"
        if not domains_dir.exists():
            pytest.skip("No domains directory")

        violations = []
        for child in sorted(domains_dir.iterdir()):
            if child.is_dir() and not child.name.startswith("_"):
                init_file = child / "__init__.py"
                if not init_file.exists():
                    violations.append(f"  {child.relative_to(SRC_ROOT)}/")

        if violations:
            viol_text = "\n".join(violations)
            pytest.fail(
                f"VIOLATION: Domain directories missing __init__.py:\n"
                f"{viol_text}\n"
                f"FIX: Add an empty __init__.py to each domain directory.\n"
                f"RULE: Each domain must have an __init__.py for proper Python packaging."
            )


class TestEvaluationPurity:
    """Validate that evaluation/scoring functions don't import from providers."""

    def test_no_provider_imports_in_evaluation_files(self):
        """Files containing scoring/evaluation functions should not import providers."""
        files = collect_python_files(SRC_ROOT)

        violations = []
        for filepath in files:
            # Check files that are likely evaluation/scoring
            content = filepath.read_text()

            # Heuristic: file contains functions with "score", "evaluate", "metric" in name
            has_eval_functions = bool(
                re.search(r"def\s+(?:\w*(?:score|evaluate|metric)\w*)\s*\(", content)
            )
            if not has_eval_functions:
                continue

            # Check if it imports from providers
            imports = parse_imports(filepath)
            rel_path = filepath.relative_to(SRC_ROOT)
            for module_path, lineno in imports:
                if module_path.startswith("src.providers"):
                    violations.append(
                        f"  {rel_path}:{lineno} defines evaluation functions "
                        f"but imports {module_path}"
                    )

        if violations:
            viol_text = "\n".join(violations)
            pytest.fail(
                f"VIOLATION: Evaluation functions import from providers:\n"
                f"{viol_text}\n"
                f"FIX: Evaluation functions must be pure. Pass data as parameters "
                f"instead of importing providers.\n"
                f"RULE: Evaluation functions are pure — no API calls, no I/O, no provider imports."
            )


class TestNoHardcodedSecrets:
    """Validate no hardcoded API keys or secrets in source code."""

    def test_no_secret_patterns_in_source(self):
        """Source files should not contain hardcoded API keys or tokens."""
        files = collect_python_files(SRC_ROOT)

        violations = []
        for filepath in files:
            content = filepath.read_text()
            rel_path = filepath.relative_to(SRC_ROOT)

            for pattern, description in SECRET_PATTERNS:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    # Find the line number
                    line_num = content[:match.start()].count("\n") + 1
                    # Show a snippet (redacted)
                    snippet = match.group()[:20] + "..."
                    violations.append(
                        f"  {rel_path}:{line_num} — {description}: {snippet}"
                    )

        if violations:
            viol_text = "\n".join(violations)
            pytest.fail(
                f"VIOLATION: Hardcoded secrets found in source code:\n"
                f"{viol_text}\n"
                f"FIX: Use environment variables via .env file. "
                f"Never commit secrets to source control.\n"
                f"RULE: No hardcoded API keys, tokens, or passwords in source files."
            )
