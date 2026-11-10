"""The package layering of `rojev` (docs/CODEBASE.md §2): every import points down the stack, never up or sideways.

Fails when a module imports a package it is not allowed to depend on, so the boundaries between the shared protocol,
the robot layer, control, System 1, System 2, the owner and evaluation stay clean as the code grows.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "rojev"
ALLOWED = {
    "protocol": set(),
    "robot": set(),
    "transport": set(),
    "control": {"protocol", "robot"},
    "system1": {"protocol", "robot", "control", "transport"},
    "system2": {"protocol", "observability", "transport"},
    "owner": {"protocol", "control", "system2"},
    "eval": {"protocol", "system2"},
    "observability": set(),
    "legacy_v2": {"protocol", "system2", "eval"},
    "app": {"protocol", "robot", "control", "system1", "system2", "owner"},
}


def rojev_imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith("rojev."):
            out.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            out |= {a.name.split(".")[1] for a in node.names if a.name.startswith("rojev.")}
    return out


def test_every_package_is_declared():
    packages = {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert packages == set(ALLOWED)
    assert {p.name for p in ROOT.glob("*.py")} == {"__init__.py"}


def test_imports_only_point_down_the_stack():
    violations = []
    for package, allowed in ALLOWED.items():
        for path in (ROOT / package).rglob("*.py"):
            bad = rojev_imports(path) - allowed - {package}
            if bad:
                violations.append(f"{path.relative_to(ROOT)} imports {sorted(bad)}")
    assert not violations, violations


def test_nothing_depends_on_the_app_layer():
    users = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")
             if "app" in rojev_imports(p) and p.parts[len(ROOT.parts)] != "app"]
    assert not users


def test_nothing_depends_on_legacy_v2():
    users = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")
             if "legacy_v2" in rojev_imports(p) and p.parts[len(ROOT.parts)] != "legacy_v2"]
    assert not users
