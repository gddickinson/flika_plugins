// Home: what the repository is, real counts, status groups, topics and demos.
import { api, app, esc, list, route } from './core.js';

route('home', async () => {
  const info = await api('/info'), s = info.stats;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">gddickinson / flika_plugins</p>
    <h1 class="glow">FLIKA Plugins</h1>
    <p class="lead">The plugins written at UC Irvine (Parker and Pathak labs) for
      <a href="https://flika-org.github.io/" target="_blank" rel="noopener">FLIKA</a>, the image-processing program for
      biologists: calcium-puff detection, single-particle tracking and ThunderSTORM-style localization, light-sheet
      volumes, ROI tools and movie export. Each page is read from the plugin's own files (info.xml, README and Python
      source). Three of the analyses are ported to JavaScript and run here in your browser.</p>
    <a class="btn" href="#/plugins">Browse the plugins</a><a class="btn alt" href="#/demos">Interactive demos</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${s.plugins}</b><span>plugin folders (info.xml)</span></div>
    <div class="tile"><b>${s.groups.main}</b><span>at the top level</span></div>
    <div class="tile"><b>${s.python_files}</b><span>Python files</span></div>
    <div class="tile"><b>${s.lines.toLocaleString()}</b><span>lines of Python</span></div>
    <div class="tile"><b>${info.demos.length}</b><span>interactive demos</span></div>
  </div>
  <h2>Interactive demos</h2>
  <div class="grid">${list(info.demos, d => `<a class="card" href="#/demo/${d.id}"><b>${esc(d.title)}</b><span>${esc(d.blurb)}</span></a>`)}</div>
  <h2>By status</h2>
  <p class="small mute">Status is the folder a plugin sits in; the repository keeps experimental, broken and old versions beside the working ones.</p>
  <div class="grid">${list(info.groups, g => `<a class="card" href="#/plugins?group=${g.id}"><b>${esc(g.title)}</b>
    <small>${g.count} plugin${g.count === 1 ? '' : 's'}</small><span>${esc(g.blurb)}</span></a>`)}</div>
  <h2>By topic</h2>
  <div class="grid">${list(info.topics, t => `<a class="card" href="#/plugins?topic=${t.id}"><b>${esc(t.title)}</b>
    <small>${t.count} plugin${t.count === 1 ? '' : 's'}</small><span>${esc(t.blurb)}</span></a>`)}</div>`;
});
