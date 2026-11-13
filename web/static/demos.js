// Demos index and the #/demo/<id> router: each demo lives in its own module and exports render(host, ctx).
import { api, app, esc, list, route } from './core.js';

const MODULES = { localization: './demo_localize.js', tracks: './demo_tracks.js', fft: './demo_fft.js', kymograph: './demo_kymo.js' };
const sources = d => `<ul class="src-list">${d.functions.map(f => `<li><a href="https://github.com/gddickinson/flika_plugins/blob/master/${esc(f.path)}" target="_blank" rel="noopener">${esc(f.path)}</a>: <code>${f.names.map(esc).join('</code>, <code>')}</code></li>`).join('')}</ul>`;

route('demos', async () => {
  const demos = await api('/demos');
  app().innerHTML = `<h1>Interactive demos</h1>
    <p class="lead">Pure-numpy analyses from the plugins, ported line for line to JavaScript and run in your browser.
      Each port is tested in node against the plugin's own Python on the same inputs (to 1e-6 or better).</p>
    ${list(demos, d => `<div class="panel"><h2 style="margin-top:0"><a href="#/demo/${d.id}">${esc(d.title)}</a></h2>
      <p>${esc(d.blurb)}</p><p class="small mute">Ports:</p>${sources(d)}</div>`)}
    <p class="small mute">Not ported (they need FLIKA, Qt, a GPU or large libraries): U-Track linking, the SVM classifier,
      multi-emitter and MLE fitting, puff detection (detect_puffs), kymograph extraction from a movie, light-sheet reshaping, the experimental deep-learning
      plugins. Their pages in the catalogue describe them.</p>`;
});

route('demo', async ([id]) => {
  const demos = await api('/demos');
  const d = demos.find(x => x.id === id);
  if (!d || !MODULES[id]) { app().innerHTML = '<h1>No such demo</h1><p><a href="#/demos">All demos</a></p>'; return; }
  app().innerHTML = `<p class="kicker"><a href="#/demos">Demo</a></p><h1>${esc(d.title)}</h1><p class="lead">${esc(d.blurb)}</p>
    <div id="demo"></div><h2>Ported from</h2>${sources(d)}
    <p class="small mute">Plugin: ${d.plugins.map(p => `<a href="#/plugin/${esc(p)}">${esc(p)}</a>`).join(', ')}. Engine:
      <code>static/${esc(d.engine)}</code>, checked by its node test against <code>tests/fixtures/demo_cases.json</code>
      (written by <code>scripts/make_demo_fixture.py</code> from the plugin code).</p>`;
  (await import(MODULES[id])).render(document.getElementById('demo'), { demo: d });
});
