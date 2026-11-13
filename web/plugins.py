"""The plugin catalogue, read from the repository itself: every folder with a FLIKA info.xml is one plugin.

For each plugin: its info.xml (name, author, version, dependencies, menu entries), its README, and every Python file
parsed with `ast` (line count, module docstring, top-level classes and functions). Nothing is imported or run.
Status comes from the folder the plugin sits in (top level, EXPERIMENTAL/, NOT_WORKING/, ...); topics from topics.py.
"""
import ast
import os
import re
import warnings
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from .topics import TOPICS, topic_of

ROOT = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/gddickinson/flika_plugins/tree/master/"
README_MAX = 60_000
BOILERPLATE = ("Download this repository", "To install")
SKIP_DIRS = {"__pycache__", ".git", "site", "web", "node_modules"}

GROUPS = [
    ("main", "Plugins", "Folders at the top level of the repository: the plugins to copy into ~/.FLIKA/plugins."),
    ("experimental", "Experimental", "EXPERIMENTAL/: newer plugins still being developed."),
    ("py311", "Not working with Python 3.11", "NOT_WORKING_WITH_PY311/: plugins that do not run on Python 3.11+."),
    ("broken", "Not working", "NOT_WORKING/: plugins marked as not working."),
    ("old", "Old versions", "OLD/: superseded versions kept for reference."),
    ("backup", "Backups", "backups/: snapshots of plugins."),
]
_GROUP_DIRS = {"EXPERIMENTAL": "experimental", "NOT_WORKING_WITH_PY311": "py311", "NOT_WORKING": "broken",
               "OLD": "old", "backups": "backup"}


def slug(rel: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", rel.replace("/", ".")).strip("-.")


def _text(el) -> str:
    return " ".join((el.text or "").split()) if el is not None else ""


def parse_info(path: Path) -> dict:
    """Fields of a FLIKA info.xml; tolerant of the variations in this repo."""
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError as e:
        return {"name": "", "xml_error": str(e)}
    url = _text(root.find("url"))
    return {
        "name": (root.get("name") or "").strip(),
        "author": _text(root.find("author")),
        "version": _text(root.find("version")),
        "url": url if url.startswith("http") and "username" not in url else "",
        "dependencies": sorted({d.get("name").strip() for d in root.iter("dependency") if d.get("name")}),
        "menu": [" ".join(a.itertext()).strip() for a in root.iter("action") if " ".join(a.itertext()).strip()],
        "description": _text(root.find("description")) or _text(root.find("long_description")),
    }


def first_paragraph(md: str) -> str:
    """First prose paragraph of a README (headings, badges, images and lists skipped)."""
    para = []
    for line in md.splitlines():
        s = line.strip()
        if not s:
            if para:
                break
            continue
        if s.startswith(("#", "!", "[!", "|", "```", "-", "*", "<", "=")) or set(s) <= set("-=_*"):
            if para:
                break
            continue
        para.append(s)
        if len(para) == 1 and s.startswith(BOILERPLATE):
            para = []
    text = " ".join(para).replace("\\_", "_")
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[*_`]{1,2}([^*_`]+)[*_`]{1,2}", r"\1", text)


def parse_python(path: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    info = {"lines": text.count("\n") + (not text.endswith("\n") and bool(text))}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError as e:
        info["error"] = f"{type(e).__name__}: {e.msg} (line {e.lineno})"
        return info
    doc = ast.get_docstring(tree) or ""
    info["doc"] = doc.strip().split("\n\n")[0].replace("\n", " ")[:300]
    info["classes"] = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    info["functions"] = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            mods.add(n.module.split(".")[0])
    info["imports"] = sorted(mods)
    return info


def _walk(top: Path):
    """All files under `top`, skipping SKIP_DIRS (os.walk prunes them, unlike rglob which would read .git)."""
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            yield Path(dirpath) / f


def _plugin_dirs() -> list[Path]:
    return sorted(p.parent for p in _walk(ROOT) if p.name == "info.xml")


def _files(d: Path, nested: set) -> list[Path]:
    return [p for p in _walk(d) if not any(p.is_relative_to(n) for n in nested)]


def _readme(d: Path):
    for name in ("README.md", "README.rst", "README.txt", "README"):
        if (d / name).is_file():
            text = (d / name).read_text(encoding="utf-8", errors="replace")
            return name, text[:README_MAX], len(text) > README_MAX
    return None, "", False


def build_plugin(d: Path, all_dirs: list[Path]) -> dict:
    rel = d.relative_to(ROOT).as_posix()
    nested = {o for o in all_dirs if o != d and o.is_relative_to(d)}
    info = parse_info(d / "info.xml")
    readme_name, readme, cut = _readme(d)
    files = _files(d, nested)
    py = []
    for f in files:
        if f.suffix == ".py":
            py.append({"path": f.relative_to(d).as_posix(), **parse_python(f)})
    top = rel.split("/")[0]
    group = _GROUP_DIRS.get(top, "main")
    imports = set().union(*[set(f.get("imports", [])) for f in py]) if py else set()
    docs = [f.relative_to(d).as_posix() for f in files
            if f.suffix.lower() in (".md", ".rst", ".txt", ".html", ".pdf") and f.name != readme_name]
    description = info.get("description") or first_paragraph(readme)
    return {
        "id": slug(rel), "path": rel, "folder": d.name.strip(), "group": group, "topic": topic_of(d.name),
        **info, "name": info.get("name") or d.name.strip(), "description": description,
        "readme_name": readme_name, "readme": readme, "readme_cut": cut, "docs": docs,
        "files": py, "n_files": len(files), "lines": sum(f["lines"] for f in py),
        "n_classes": sum(len(f.get("classes", [])) for f in py),
        "n_functions": sum(len(f.get("functions", [])) for f in py),
        "parse_errors": [f["path"] for f in py if "error" in f],
        "uses_flika": "flika" in imports, "uses_qt": bool(imports & {"qtpy", "PyQt5", "PyQt6", "PySide2", "PySide6"}),
        "nested_in": next((slug(o.relative_to(ROOT).as_posix()) for o in all_dirs if o != d and d.is_relative_to(o)),
                          None),
        "github": GITHUB + rel,
    }


@lru_cache(maxsize=1)
def catalogue() -> dict:
    dirs = _plugin_dirs()
    plugins = {}
    for d in dirs:
        p = build_plugin(d, dirs)
        plugins[p["id"]] = p
    by_folder = {}
    for p in plugins.values():
        by_folder.setdefault(p["folder"], []).append(p["id"])
    for p in plugins.values():
        p["same_folder"] = [i for i in by_folder[p["folder"]] if i != p["id"]]
    gorder = [g for g, _, _ in GROUPS]
    order = sorted(plugins, key=lambda i: (gorder.index(plugins[i]["group"]), plugins[i]["name"].lower(), i))
    return {"plugins": plugins, "order": order,
            "groups": [{"id": g, "title": t, "blurb": b} for g, t, b in GROUPS],
            "topics": [{"id": t, "title": n, "blurb": b} for t, n, b in TOPICS]}


INDEX_FIELDS = ("id", "path", "name", "folder", "group", "topic", "author", "version", "description", "lines",
                "n_files", "uses_flika", "dependencies", "menu")


def index_entry(p: dict) -> dict:
    e = {k: p[k] for k in INDEX_FIELDS}
    e["names"] = sorted({n for f in p["files"] for n in f.get("classes", []) + f.get("functions", [])})
    return e


def stats(cat: dict) -> dict:
    ps = list(cat["plugins"].values())
    count = lambda key: {g["id"]: sum(p[key] == g["id"] for p in ps) for g in cat[key + "s"]}  # noqa: E731
    return {"plugins": len(ps), "python_files": sum(len(p["files"]) for p in ps),
            "lines": sum(p["lines"] for p in ps), "classes": sum(p["n_classes"] for p in ps),
            "functions": sum(p["n_functions"] for p in ps), "with_readme": sum(bool(p["readme"]) for p in ps),
            "groups": count("group"), "topics": count("topic"),
            "parse_errors": sum(len(p["parse_errors"]) for p in ps)}
