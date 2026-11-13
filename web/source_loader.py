"""Load named functions, classes or methods from a plugin file without importing the plugin.

Most plugins import flika and Qt at module level, so they cannot be imported outside FLIKA. This parses the file,
executes only the import statements that the requested code needs (skipping any that fail, e.g. flika or qtpy) and
the requested top-level `def` / `class` blocks or class methods, verbatim. Tests, fixtures and the web demos therefore
run the plugin's own code.

Names:  "func"          a top-level function
        "Class"         a top-level class (its bases must be importable, so plain helper classes only)
        "Class.method"  one method of a class, returned as a plain function (decorators such as @staticmethod dropped;
                        call it with a stand-in `self` if it uses one)
"""
import ast
from pathlib import Path


def _names_used(nodes) -> set:
    used = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                used.add(n.id)
            elif isinstance(n, ast.Attribute):
                base = n
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    used.add(base.id)
    return used


def _bound_names(node) -> set:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def _is_import_block(node) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Try):
        return all(isinstance(b, (ast.Import, ast.ImportFrom, ast.Pass)) for b in node.body)
    return False


def load(path, names, extra: dict | None = None) -> dict:
    """{name: object} for the requested names in `path`; `extra` adds globals (e.g. stand-ins for flika names)."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    picked = []
    for name in names:
        cls, _, meth = name.partition(".")
        if cls not in top:
            raise KeyError(f"{path}: no top-level {cls!r}")
        if not meth:
            picked.append((name, top[cls]))
            continue
        fn = next((n for n in top[cls].body if isinstance(n, ast.FunctionDef) and n.name == meth), None)
        if fn is None:
            raise KeyError(f"{path}: class {cls!r} has no method {meth!r}")
        fn = ast.FunctionDef(name=f"{cls}__{meth}", args=fn.args, body=fn.body, decorator_list=[],
                             returns=fn.returns, type_comment=None, type_params=[])
        picked.append((name, ast.copy_location(fn, top[cls])))
    used = _names_used([n for _, n in picked])
    ns = {"__name__": f"flika_plugin_{path.stem}", "__file__": str(path), **(extra or {})}
    needed = used - set(extra or {})   # names given in `extra` (stand-ins) are never imported
    for node in tree.body:
        if _is_import_block(node) and _bound_names(node) & needed:
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            except Exception:  # noqa: BLE001 - flika, qtpy and other GUI-only imports are expected to fail
                pass
    ns.update(extra or {})
    for _, node in picked:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), ns)
    return {name: ns[node.name] for name, node in picked}
