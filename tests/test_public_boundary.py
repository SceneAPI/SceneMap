from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {Path("tests/test_public_boundary.py")}
TEXT_SUFFIXES = {"", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
BANNED = (
    ("from " + "app.", "server internals"),
    ("import " + "app.", "server internals"),
    ("app." + "adapters", "server internals"),
    ("app." + "core.errors", "server internals"),
    ("app." + "main", "server entrypoint"),
    ("app." + "db.session", "database internals"),
    ("sfmapi" + "_client", "client SDK"),
    ("@" + "sfmapi/client", "client SDK"),
    ("sfmapi" + "-sdk", "client SDK repository"),
    ("sfmapi" + "-client", "client SDK package"),
    ("clients/" + "python", "removed in-repo SDK path"),
    ("clients/" + "typescript", "removed in-repo SDK path"),
    ("clients/" + "cpp", "removed in-repo SDK path"),
)


def _files() -> list[Path]:
    roots = [ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "src", ROOT / "tests"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix in TEXT_SUFFIXES
                and "__pycache__" not in path.parts
            )
    return files


def test_plugin_uses_public_sfmapi_boundary() -> None:
    failures: list[str] = []
    for path in _files():
        rel = path.relative_to(ROOT)
        if rel in SKIP:
            continue
        text = path.read_text(encoding="utf-8")
        for marker, label in BANNED:
            if marker in text:
                failures.append(f"{rel}: {marker!r} ({label})")

    assert not failures, "\n".join(failures)
