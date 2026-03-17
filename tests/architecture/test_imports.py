"""Architecture tests: dependency direction and import rules.

These tests use Python's ast module to parse all .py files in src/ and validate
that imports follow the architectural dependency flow:

    Utils -> Types -> Providers -> Domains -> App

Error messages use VIOLATION / FIX / RULE format for clarity.
"""

import pytest

from tests.architecture.conftest import (
    SRC_ROOT,
    classify_domain_layer,
    classify_zone,
    collect_python_files,
    import_domain,
    import_domain_layer,
    import_zone,
    parse_imports,
)

# Zone hierarchy: lower number = lower in the stack, cannot import higher
ZONE_LEVEL = {
    "utils": 0,
    "types": 1,
    "providers": 2,
    "domains": 3,
    "app": 4,
}

# Domain layer hierarchy: lower number = lower in the stack
LAYER_LEVEL = {
    "types": 0,
    "config": 1,
    "service": 2,
}


def _all_src_files():
    """Collect all Python source files, excluding __init__.py (barrel exports exempt)."""
    files = collect_python_files(SRC_ROOT)
    return [f for f in files if f.name != "__init__.py"]


class TestDependencyDirection:
    """Validate that imports respect the zone hierarchy."""

    @pytest.mark.parametrize(
        "filepath",
        _all_src_files(),
        ids=lambda p: str(p.relative_to(SRC_ROOT)),
    )
    def test_zone_imports_respect_hierarchy(self, filepath):
        """Each zone can only import from zones at the same level or below."""
        source_zone = classify_zone(filepath)
        if source_zone == "unknown":
            pytest.skip("File not in a recognized zone")

        source_level = ZONE_LEVEL[source_zone]
        imports = parse_imports(filepath)
        rel_path = filepath.relative_to(SRC_ROOT)

        violations = []
        for module_path, lineno in imports:
            target_zone = import_zone(module_path)
            if target_zone is None:
                continue  # external package, skip

            target_level = ZONE_LEVEL.get(target_zone)
            if target_level is None:
                continue

            if target_level > source_level:
                violations.append(
                    f"  Line {lineno}: imports {module_path} "
                    f"({source_zone} level {source_level} -> {target_zone} level {target_level})"
                )

        if violations:
            viol_text = "\n".join(violations)
            pytest.fail(
                f"VIOLATION: {rel_path} ({source_zone}) imports from higher-level zones.\n"
                f"{viol_text}\n"
                f"FIX: Move the imported code to a lower-level zone, or restructure the dependency.\n"
                f"RULE: Dependency flow is Utils -> Types -> Providers -> Domains -> App. "
                f"A zone can only import from zones at its level or below."
            )


class TestDomainLayerImports:
    """Validate that within a domain, imports respect the layer hierarchy."""

    @pytest.mark.parametrize(
        "filepath",
        [f for f in _all_src_files() if classify_domain_layer(f) is not None],
        ids=lambda p: str(p.relative_to(SRC_ROOT)),
    )
    def test_domain_layer_hierarchy(self, filepath):
        """Within a domain: types cannot import config/service, config cannot import service."""
        domain_info = classify_domain_layer(filepath)
        if domain_info is None:
            pytest.skip("Not a domain file")

        domain_name, source_layer = domain_info
        if source_layer not in LAYER_LEVEL:
            pytest.skip(f"Layer '{source_layer}' not in hierarchy")

        source_level = LAYER_LEVEL[source_layer]
        imports = parse_imports(filepath)
        rel_path = filepath.relative_to(SRC_ROOT)

        violations = []
        for module_path, lineno in imports:
            target_domain = import_domain(module_path)
            if target_domain != domain_name:
                continue  # cross-domain import, handled elsewhere

            target_layer = import_domain_layer(module_path)
            if target_layer is None or target_layer not in LAYER_LEVEL:
                continue

            target_level = LAYER_LEVEL[target_layer]
            if target_level > source_level:
                violations.append(
                    f"  Line {lineno}: imports {module_path} "
                    f"({source_layer} -> {target_layer})"
                )

        if violations:
            viol_text = "\n".join(violations)
            pytest.fail(
                f"VIOLATION: {rel_path} ({domain_name}/{source_layer}) imports from "
                f"a higher layer within the same domain.\n"
                f"{viol_text}\n"
                f"FIX: Move shared code to a lower layer (types < config < service).\n"
                f"RULE: Within a domain, types.py cannot import config.py or service.py; "
                f"config.py cannot import service.py."
            )


class TestNoCrossDomainCircular:
    """Validate no circular dependencies between domains."""

    def test_no_circular_domain_dependencies(self):
        """Build a domain dependency graph and check for cycles."""
        files = collect_python_files(SRC_ROOT)
        domain_files = [f for f in files if classify_zone(f) == "domains"]

        # Build adjacency: domain -> set of domains it imports from
        graph: dict[str, set[str]] = {}
        for filepath in domain_files:
            info = classify_domain_layer(filepath)
            if info is None:
                continue
            source_domain = info[0]
            if source_domain not in graph:
                graph[source_domain] = set()

            imports = parse_imports(filepath)
            for module_path, _ in imports:
                target_domain = import_domain(module_path)
                if target_domain and target_domain != source_domain:
                    graph[source_domain].add(target_domain)

        # Check for cycles using DFS
        def has_cycle(node: str, visited: set, rec_stack: set, path: list) -> list | None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    result = has_cycle(neighbor, visited, rec_stack, path)
                    if result is not None:
                        return result
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            path.pop()
            rec_stack.discard(node)
            return None

        visited: set[str] = set()
        for domain in graph:
            if domain not in visited:
                cycle = has_cycle(domain, visited, set(), [])
                if cycle is not None:
                    cycle_str = " -> ".join(cycle)
                    pytest.fail(
                        f"VIOLATION: Circular dependency between domains: {cycle_str}\n"
                        f"FIX: Break the cycle by extracting shared types to src/types/ "
                        f"or using dependency injection.\n"
                        f"RULE: No circular dependencies between domains."
                    )


class TestProviderIsolation:
    """Validate that providers don't import from domains or app."""

    def test_providers_do_not_import_domains(self):
        """Providers should not depend on domain code."""
        files = collect_python_files(SRC_ROOT)
        provider_files = [
            f for f in files
            if classify_zone(f) == "providers" and f.name != "__init__.py"
        ]

        violations = []
        for filepath in provider_files:
            imports = parse_imports(filepath)
            rel_path = filepath.relative_to(SRC_ROOT)
            for module_path, lineno in imports:
                target_zone = import_zone(module_path)
                if target_zone in ("domains", "app"):
                    violations.append(
                        f"  {rel_path}:{lineno} imports {module_path} (zone: {target_zone})"
                    )

        if violations:
            viol_text = "\n".join(violations)
            pytest.fail(
                f"VIOLATION: Provider files import from domains or app:\n"
                f"{viol_text}\n"
                f"FIX: Move shared types to src/types/. Providers should only import "
                f"from utils, types, or external packages.\n"
                f"RULE: Providers cannot import from domains or app."
            )
