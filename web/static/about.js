// About: what the repository is, how to install the plugins and run this site, how the catalogue is made, credits.
import { api, app, route } from './core.js';

route('about', async () => {
  const s = (await api('/info')).stats;
  app().innerHTML = `<h1>About</h1>
  <p class="lead">flika_plugins is a collection of Python plugins developed at UC Irvine to extend
    <a href="https://github.com/flika-org/flika" target="_blank" rel="noopener">FLIKA</a>, an interactive image
    processing program for biologists, for research in the Parker and Pathak labs (from the repository README).</p>
  <h2>Installing a plugin</h2>
  <ol><li>Download the plugin folder.</li><li>Move it to your <code>~/.FLIKA/plugins</code> directory.</li>
    <li>Restart FLIKA: the plugin appears in FLIKA's Plugins menu.</li></ol>
  <p>See the <a href="http://flika-org.github.io/plugins.html" target="_blank" rel="noopener">FLIKA documentation</a>.
    Folders under EXPERIMENTAL/, NOT_WORKING/, NOT_WORKING_WITH_PY311/, OLD/ and backups/ are kept for reference.</p>
  <h2>How this catalogue is made</h2>
  <ul>
    <li>Every folder with an <code>info.xml</code> is a plugin (${s.plugins} of them). Name, author, version, dependencies
      and menu entries come from info.xml; the description from info.xml or the first paragraph of the README.</li>
    <li>Every Python file is parsed with Python's <code>ast</code>, not imported: ${s.python_files} files,
      ${s.lines.toLocaleString()} lines, ${s.classes} classes and ${s.functions} top-level functions.
      ${s.parse_errors} files do not parse with Python 3.12 (Python 2 syntax) and are flagged on their plugin's page.</li>
    <li>Status is the folder a plugin sits in; topics are the only hand-made grouping.</li>
    <li>The demos are JavaScript ports of pure-numpy code from spt_batch_analysis (ThunderSTORM localization, track
      features, directional autocorrelation) and fft_Chunker, tested in node against the plugin code itself.</li>
  </ul>
  <h2>Run this site locally</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py            # http://127.0.0.1:8198</pre>
  <p>The static copy on GitHub Pages is built by <code>scripts/build_static.py</code>.</p>
  <h2>Credits</h2>
  <p>Plugins by George Dickinson, Kyle Ellefsen and Brett Settle; several were adapted or extended from original code by
    Kyle Ellefsen and Brett Settle (README). thunderstorm_python is a Python reimplementation of ThunderSTORM
    (Ovesn&yacute; et al., Bioinformatics 2014). The directional autocorrelation follows Gorelik &amp; Gautreau (2014).
    Licence: MIT.</p>`;
});
