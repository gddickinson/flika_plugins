# INTERFACE: flika_plugins

A collection of FLIKA plugins. Each folder with an `info.xml` is one plugin (64 of them); copy a folder into
`~/.FLIKA/plugins` to install it. Plugins with their own maps: `spt_batch_analysis/INTERFACE.md`,
`locsAndTracksPlotter/INTERFACE.md`.

## Plugins (the repository's content)
| Path | What |
| --- | --- |
| `<plugin>/` at the top level | the current plugins (28): spt_batch_analysis, thunderstorm_flika, locsAndTracksPlotter, detect_puffs_GDedit, kymograph_analyzer, fft_Chunker, flash_remover, volumeSlider, ... |
| `EXPERIMENTAL/` | plugins in development (18) |
| `NOT_WORKING_WITH_PY311/`, `NOT_WORKING/` | plugins that do not run on Python 3.11+ (6) / marked not working (3) |
| `OLD/`, `backups/` | superseded versions (8) and a snapshot of spt_batch_analysis (1) |

## Web catalogue (`web/`)
| File | Contents |
| --- | --- |
| `web/app.py` | FastAPI app: `/api/info`, `/api/catalogue`, `/api/plugin/{id}`, `/api/demos`, `/api/demo-data/localization`, `/api/demo-data/tracks`; `STATIC_ENDPOINTS` |
| `web/plugins.py` | `catalogue()`: finds info.xml folders, `parse_info()`, `first_paragraph()` (README), `parse_python()` (ast: lines, docstring, classes, functions, imports); `index_entry()`, `stats()`, `slug()` |
| `web/topics.py` | `TOPICS`, `topic_of(folder)`: the only hand-made grouping |
| `web/demos.py` | `DEMOS` (which plugin functions each JS engine ports), `localization_data()` (test-data frames + reference positions), `track_data()`, `real_camera()` |
| `web/source_loader.py` | `load(path, names, extra)`: run a plugin's own function / class / method without importing flika or Qt |
| `web/static/` | front end: `main.js` (router), `home.js`, `browse.js`, `plugin.js`, `demos.js`, `demo_localize.js`, `demo_tracks.js`, `demo_fft.js`, `about.js`, `ui.js` (chart, Markdown), `static_api.js`, `core.js` / `style.css` (shared kit), `style_app.css` |
| `web/static/*_engine.js` | JS ports: `localize_engine.js` (+ `erf.js`), `track_engine.js`, `fft_engine.js`; `*.test.mjs` node tests, `fixture.test.mjs` helper |

Connections: `app.py` → `plugins.py` → `topics.py`; `app.py` → `demos.py` (reads `spt_batch_analysis/test_data`,
`EXPERIMENTAL/advanced_particle_tracking/*tracking_results.csv`). `scripts/make_demo_fixture.py` → `thunderstorm_python`
and `source_loader.py` → `tests/fixtures/demo_cases.json` + `localization_frames.json` ← node tests.

## Tooling
| File | What |
| --- | --- |
| `run_web.py` | local server on 127.0.0.1:8198 |
| `scripts/build_static.py` | export the static site (every `STATIC_ENDPOINTS` path as JSON) |
| `scripts/screenshots.py` | Playwright smoke test / README screenshots (`docs/img/`) |
| `scripts/make_demo_fixture.py` | Python outputs of the ported functions for the node tests |
| `tests/` | `test_web.py`, `test_catalogue.py`, `test_demos.py`, `test_fix_*.py` (regressions for plugin fixes) |
| `.github/workflows/pages.yml` | build + publish the static site on push to master |
