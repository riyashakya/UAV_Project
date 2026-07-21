"""ADR-001 firewall: no module under ``src/sim`` may reach ``ultralytics`` (or torch/sahi).

Walks the static import graph of the whole ``src`` package (resolving relative imports) and
fails if any module under ``src.sim`` can reach a forbidden dependency by *any* import path —
enforced, not merely documented (Phase 3 acceptance).
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN = {"ultralytics", "torch", "sahi"}


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports_of(path: Path) -> set[str]:
    mod = _module_name(path)
    pkg_parts = mod.split(".")[:-1]  # package that contains this module
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module)
            else:  # relative import → resolve to an absolute dotted name
                up = node.level - 1
                target = pkg_parts[: len(pkg_parts) - up] if up > 0 else pkg_parts
                if node.module:
                    target = target + node.module.split(".")
                found.add(".".join(target))
    return found


def _build_graph() -> dict[str, set[str]]:
    return {_module_name(p): _imports_of(p) for p in SRC.rglob("*.py")}


def _reachable_forbidden(start: str, graph: dict[str, set[str]]) -> tuple[str, str] | None:
    seen: set[str] = set()
    stack = [start]
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        for imp in graph.get(mod, set()):
            if imp.split(".")[0] in FORBIDDEN:
                return mod, imp
            if imp in graph:  # follow first-party edges only
                stack.append(imp)
    return None


def test_sim_never_imports_ultralytics():
    graph = _build_graph()
    sim_modules = [m for m in graph if m.startswith("src.sim")]
    assert sim_modules, "no src.sim modules found — test wiring is wrong"

    offenders = {}
    for mod in sim_modules:
        hit = _reachable_forbidden(mod, graph)
        if hit:
            offenders[hit[0]] = hit[1]
    assert not offenders, f"src.sim reaches forbidden deps (ADR-001 violated): {offenders}"


def test_forbidden_list_is_actually_detected():
    """Sanity: the graph walk would catch a violation if one existed."""
    graph = {"src.sim.fake": {"ultralytics"}}
    assert _reachable_forbidden("src.sim.fake", graph) == ("src.sim.fake", "ultralytics")
