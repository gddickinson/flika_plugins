// Spot detection and localization demo: runs localize_engine.js (the thunderstorm_python port) on frames from the
// repository's test data and compares the result with the reference positions shipped with them.
import { api, esc } from './core.js';
import { analyzeFrame, matchToReference } from './localize_engine.js';
import { fmt, vals } from './ui.js';

const median = a => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };

export async function render(host) {
  const data = await api('/demo-data/localization');
  host.innerHTML = `<div class="demo-grid">
    <div>
      <div class="stage"><canvas id="cv" width="512" height="512" aria-label="frame with detections"></canvas></div>
      <div class="legend"><span style="--c:#3db8ff">reference</span><span style="--c:#ff6bd6">local maximum</span><span style="--c:#3dffb0">fit</span></div>
      <p class="small mute" id="note"></p>
    </div>
    <div>
      <div class="ctl2">
        <label for="src">Data</label><select id="src">${data.sources.map(s => `<option value="${s.id}">${esc(s.title)}</option>`).join('')}</select>
        <label for="fr">Frame</label><select id="fr"></select>
        <label for="thr">Threshold <b id="thrv"></b> x std(Wave.F1)</label><input id="thr" type="range" min="0.5" max="4" step="0.1">
        <label for="con">Local maximum</label><select id="con"><option value="8">8-neighbourhood</option><option value="4">4-neighbourhood</option></select>
        <label for="fit">Fitter</label><select id="fit"><option value="lsq">least squares (LSQ)</option><option value="wlsq">weighted least squares (WLSQ)</option></select>
        <label for="sig">Initial sigma <b id="sigv"></b> px</label><input id="sig" type="range" min="0.8" max="2.5" step="0.1">
        <label for="rad">Fit radius <b id="radv"></b> px</label><input id="rad" type="range" min="2" max="5" step="1">
        <label for="ppa">Photons per ADU</label><input id="ppa" type="number" step="any" min="0">
        <label for="base">Baseline (ADU)</label><input id="base" type="number" step="any">
        <label class="chk"><input id="em" type="checkbox"> EMCCD (excess noise factor 2)</label>
        <label class="chk"><input id="showF" type="checkbox"> show the wavelet-filtered image</label>
      </div>
      <div id="vals"></div>
    </div>
  </div>
  <ul class="small" id="proto"></ul>
  <h3>Localizations</h3><div class="tblwrap"><table class="tbl" id="tbl"></table></div>
  <p class="small mute">Positions are reported like the plugin: (x + 0.5) x pixel size, i.e. ThunderSTORM's corner origin.
    Uncertainty is the Thompson formula on the fitted photons and sqrt(background level - baseline) as the background SD.
    "Matched" pairs each localization with the nearest unused reference point within 1 px.</p>`;
  const $ = s => host.querySelector(s), cv = $('#cv'), g = cv.getContext('2d');
  let src, timer = 0;

  const setSource = () => {
    src = data.sources.find(s => s.id === $('#src').value);
    $('#fr').innerHTML = src.frames.map((f, i) => `<option value="${i}">${esc(f.name ? `${f.name}: ${f.description}` : `frame ${f.index + 1}`)}</option>`).join('');
    const d = src.defaults;
    $('#thr').value = 1; $('#con').value = d.connectivity; $('#fit').value = d.fitter; $('#sig').value = d.initial_sigma; $('#rad').value = 3;
    $('#ppa').value = +src.camera.photons_per_adu.toPrecision(6); $('#base').value = src.camera.baseline; $('#em').checked = src.camera.is_emccd;
    $('#note').textContent = src.note;
    $('#proto').innerHTML = src.protocol.map(p => `<li>${esc(p)}</li>`).join('');
    run();
  };

  function run() {
    $('#thrv').textContent = (+$('#thr').value).toFixed(1); $('#sigv').textContent = (+$('#sig').value).toFixed(1); $('#radv').textContent = $('#rad').value;
    const f = src.frames[+$('#fr').value], img = { width: f.width, height: f.height, data: Float64Array.from(f.image) };
    const camera = { photons_per_adu: +$('#ppa').value, baseline: +$('#base').value, is_emccd: $('#em').checked };
    const t0 = performance.now();
    const r = analyzeFrame(img, { connectivity: +$('#con').value, fitter: $('#fit').value, initialSigma: +$('#sig').value,
      threshold: +$('#thr').value, fitRadius: +$('#rad').value, pixelSize: src.pixel_size, camera });
    const ms = performance.now() - t0;
    draw(img, r, f);
    const m = matchToReference(r.fits.map(q => [q.x, q.y]), f.reference, 1);
    const rms = k => Math.sqrt(m.matched.reduce((s, q) => s + q[k] ** 2, 0) / (m.matched.length || NaN)) * src.pixel_size;
    $('#vals').innerHTML = vals([['threshold', r.threshold], ['local maxima', r.detected.length], ['after border removal', r.positions.length],
      ['localizations', r.fits.length], ['reference points', f.reference.length], ['matched (<= 1 px)', m.matched.length],
      ['RMS x error', rms('dx'), 'nm'], ['RMS y error', rms('dy'), 'nm'],
      ['median uncertainty', median(r.result.uncertainty), 'nm'], ['median sigma', median(r.result.sigma_x), 'nm'],
      ['median intensity', median(r.fits.map(q => q.intensity * camera.photons_per_adu)), 'photons'], ['time', ms, 'ms']]);
    $('#tbl').innerHTML = '<tr><th>#</th><th>x [nm]</th><th>y [nm]</th><th>sigma [nm]</th><th>intensity [ADU]</th><th>offset [ADU]</th><th>uncertainty [nm]</th><th>LM steps</th></tr>'
      + r.fits.slice(0, 60).map((q, i) => `<tr><td>${i + 1}</td><td>${fmt(r.result.x[i], 1)}</td><td>${fmt(r.result.y[i], 1)}</td><td>${fmt(r.result.sigma_x[i], 1)}</td>
        <td>${fmt(q.intensity, 1)}</td><td>${fmt(q.background, 1)}</td><td>${fmt(r.result.uncertainty[i], 2)}</td><td>${q.iterations}</td></tr>`).join('')
      + (r.fits.length > 60 ? `<tr><td colspan="8" class="mute">... ${r.fits.length - 60} more</td></tr>` : '');
  }

  function draw(img, r, f) {
    const A = $('#showF').checked ? r.F : img.data, s = [...A].sort((a, b) => a - b);
    const lo = s[Math.floor(s.length * 0.01)], hi = s[Math.floor(s.length * 0.995)];
    const sc = Math.min(cv.width / img.width, cv.height / img.height);
    const off = document.createElement('canvas'); off.width = img.width; off.height = img.height;
    const og = off.getContext('2d'), id = og.createImageData(img.width, img.height);
    for (let i = 0; i < A.length; i++) {
      const v = Math.max(0, Math.min(255, (A[i] - lo) / (hi - lo || 1) * 255));
      id.data.set([v, v, v, 255], i * 4);
    }
    og.putImageData(id, 0, 0);
    g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height); g.imageSmoothingEnabled = false;
    g.drawImage(off, 0, 0, img.width * sc, img.height * sc);
    const P = (x, y) => [(x + 0.5) * sc, (y + 0.5) * sc];
    g.lineWidth = 1.5; g.strokeStyle = '#3db8ff';
    for (const [x, y] of f.reference) { const [a, b] = P(x, y); g.beginPath(); g.arc(a, b, 1.6 * sc, 0, 7); g.stroke(); }
    g.strokeStyle = '#ff6bd6';
    for (const [row, col] of r.detected) { const [a, b] = P(col, row); g.beginPath(); g.moveTo(a - 3, b); g.lineTo(a + 3, b); g.moveTo(a, b - 3); g.lineTo(a, b + 3); g.stroke(); }
    g.fillStyle = '#3dffb0';
    for (const q of r.fits) { const [a, b] = P(q.x, q.y); g.beginPath(); g.arc(a, b, 2.2, 0, 7); g.fill(); }
  }

  const later = () => { clearTimeout(timer); timer = setTimeout(run, 60); };
  $('#src').addEventListener('change', setSource);
  ['#fr', '#con', '#fit', '#em', '#showF'].forEach(s => $(s).addEventListener('change', run));
  ['#thr', '#sig', '#rad', '#ppa', '#base'].forEach(s => $(s).addEventListener('input', later));
  setSource();
}
