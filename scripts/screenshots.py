"""Smoke test and README screenshots: open each route in Chrome (Playwright), fail on console errors, save PNGs.

Static site (no server needed):  python scripts/screenshots.py --static [route ...]
Running app:                     python scripts/screenshots.py --base http://127.0.0.1:PORT [route ...]
Routes are hash paths without '#/' ('' is the home page). PNGs go to docs/img/<name>.png with --shots.
"""
import argparse
import functools
import http.server
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def serve(directory: Path) -> str:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=str(directory))
    http.server.ThreadingHTTPServer.request_queue_size = 128  # many ES modules load at once
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("routes", nargs="*", default=[""])
    ap.add_argument("--base")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--shots", action="store_true", help="save PNGs to docs/img")
    ap.add_argument("--wait", type=int, default=1500, help="ms to wait after load (animations)")
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    if a.static:
        import build_static
        site = Path(tempfile.mkdtemp()) / "site"
        build_static.build(site)
        base = serve(site)
    else:
        base = (a.base or "http://127.0.0.1:8000").rstrip("/") + "/"
    out = ROOT / "docs" / "img"
    if a.shots:
        out.mkdir(parents=True, exist_ok=True)
    bad = 0
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:
            browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": a.width, "height": 900})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        for r in a.routes:
            errors.clear()
            page.goto(base + "#/" + r)
            page.wait_for_timeout(a.wait)
            text = page.inner_text("#app")
            failed = errors or "Something went wrong" in text
            bad += bool(failed)
            print(f"{'FAIL' if failed else 'ok  '} #/{r} {' | '.join(errors)[:300]}")
            if a.shots:
                page.screenshot(path=str(out / f"{(r or 'home').replace('/', '_')}.png"), full_page=False)
        # phone width: no horizontal scroll on the home page
        page.set_viewport_size({"width": 390, "height": 800})
        page.goto(base + "#/")
        page.wait_for_timeout(800)
        over = page.evaluate("document.documentElement.scrollWidth - innerWidth")
        print(f"{'FAIL' if over > 1 else 'ok  '} phone width overflow {over}px")
        bad += over > 1
        browser.close()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
