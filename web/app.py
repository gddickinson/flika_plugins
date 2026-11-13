"""FastAPI web app for the FLIKA plugins collection: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8198, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.
Everything is read from the repository's own files (plugins.py, demos.py); no plugin is imported or run here.
The demos compute in the browser (static/*_engine.js).
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .demos import demo_list, localization_data, track_data
from .plugins import catalogue, index_entry, stats

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "FLIKA Plugins"

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Counts for the home page: plugins per status and topic, lines, demos."""
    cat = catalogue()
    s = stats(cat)
    return {"title": TITLE, "repo": "flika_plugins", "stats": s,
            "groups": [g | {"count": s["groups"][g["id"]]} for g in cat["groups"]],
            "topics": [t | {"count": s["topics"][t["id"]]} for t in cat["topics"]], "demos": demo_list()}


@app.get("/api/catalogue")
def catalogue_index():
    """Every plugin's searchable summary, in status-group order."""
    cat = catalogue()
    return {"groups": cat["groups"], "topics": cat["topics"],
            "plugins": [index_entry(cat["plugins"][i]) for i in cat["order"]]}


@app.get("/api/plugin/{pid}")
def plugin(pid: str):
    """One plugin: info.xml fields, README, Python files with their classes and functions, demos."""
    cat = catalogue()
    if pid not in cat["plugins"]:
        raise HTTPException(404, f"no plugin {pid!r}")
    p = dict(cat["plugins"][pid])
    names = {i: cat["plugins"][i]["path"] for i in cat["plugins"]}
    p["same_folder"] = [{"id": i, "path": names[i]} for i in p["same_folder"]]
    p["demos"] = [{"id": d["id"], "title": d["title"]} for d in demo_list() if p["path"] in d["plugins"]]
    pos = cat["order"].index(pid)
    p["prev"] = cat["order"][pos - 1] if pos else None
    p["next"] = cat["order"][pos + 1] if pos + 1 < len(cat["order"]) else None
    return p


@app.get("/api/demos")
def demos():
    return demo_list()


@app.get("/api/demo-data/localization")
def demo_localization():
    """Frames (real and synthetic test data) with their reference positions, for the localization demo."""
    return localization_data()


@app.get("/api/demo-data/tracks")
def demo_tracks():
    """Tracks from the experimental advanced_particle_tracking plugin's committed results."""
    return track_data()


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = (["/info", "/catalogue", "/demos", "/demo-data/localization", "/demo-data/tracks"]
                    + [f"/plugin/{i}" for i in catalogue()["order"]])
