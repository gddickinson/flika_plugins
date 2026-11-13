# flika_plugins

A collection of FLIKA plugins, plus a web catalogue of them in `web/`.

@INTERFACE.md

- Plugins import flika and Qt at module level; load a function/class/method with `web/source_loader.load`.
- Plugin tests: use the flika_new interpreter (see the flika-plugin skill). Web tests: `python -m pytest -q tests`
  (needs requirements-web.txt). Local app: `python run_web.py` (port 8198).
- After changing a ported function or its JS port: `python scripts/make_demo_fixture.py`, then the node tests.
- Update `SESSION_LOG.md` each session.
