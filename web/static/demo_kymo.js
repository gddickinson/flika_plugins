// Kymograph demo: kymo_engine.js (the kymograph_analyzer port) on a synthetic kymograph with a known speed.
import { esc, rng } from './core.js';
import { detectEvents, measureVelocities, synthKymo } from './kymo_engine.js';
import { fmt, vals } from './ui.js';

export function render(host) {
  host.innerHTML = `<div class="demo-grid">
    <div><div class="stage"><canvas id="cv" width="480" height="320" aria-label="kymograph: time down, position across"></canvas></div>
      <div class="legend"><span style="--c:#ffd23d">true path</span><span style="--c:#ff6bd6">estimated speed</span><span style="--c:#3dffb0">detected event start</span></div>
      <p class="small mute">Synthetic kymograph (seeded; not data): rows are time (down), columns are position along the line.</p></div>
    <div class="ctl2">
      <label for="sp">True speed <b id="spv"></b> samples/row</label><input id="sp" type="range" min="-2" max="2" step="0.05" value="0.8">
      <label for="nz">Noise SD <b id="nzv"></b></label><input id="nz" type="range" min="0" max="20" step="0.5" value="3">
      <label for="wd">Spot width (SD) <b id="wdv"></b> samples</label><input id="wd" type="range" min="1.5" max="10" step="0.5" value="4">
      <label for="ns">Spots <b id="nsv"></b></label><input id="ns" type="range" min="1" max="4" step="1" value="2">
      <label for="spc">Distance per sample (spacing, e.g. µm)</label><input id="spc" type="number" step="any" min="0" value="1">
      <label for="rt">Time per row (row_time, e.g. s)</label><input id="rt" type="number" step="any" min="0" value="1">
      <label for="em">Events</label><select id="em"><option value="threshold">threshold: mean + k x SD</option><option value="zscore">z-score > k</option></select>
      <label for="k">k <b id="kv"></b></label><input id="k" type="range" min="0.5" max="4" step="0.1" value="1.5">
      <button type="button" id="seed" class="alt">new noise</button>
      <div id="vals"></div>
    </div></div>
  <p class="small mute">velocity is the least-squares optical-flow estimate -sum(dI/dt dI/dx)/sum(dI/dx^2) over the whole
    kymograph; mean/median are of the per-pixel map -(dI/dt)/(dI/dx) where |dI/dx| is at least 10% of its maximum.
    Noise in the gradients pulls both towards zero, so with noisy data the estimate is a lower bound on the speed.
    Positive = towards the end of the line. Units are samples per row unless you set spacing and time per row.</p>`;
  const $ = s => host.querySelector(s), cv = $('#cv'), g = cv.getContext('2d');
  let seed = 1;
  function run() {
    const sp = +$('#sp').value, o = { speed: sp, noise: +$('#nz').value, width: +$('#wd').value, spots: +$('#ns').value };
    [['#spv', sp], ['#nzv', o.noise], ['#wdv', o.width], ['#nsv', o.spots], ['#kv', +$('#k').value]].forEach(([s, v]) => { $(s).textContent = v; });
    const k = synthKymo(rng(seed), o), spacing = +$('#spc').value || 1, rowTime = +$('#rt').value || 1;
    const v = measureVelocities(k, { spacing, rowTime }), ev = detectEvents(k, { method: $('#em').value, threshold: +$('#k').value });
    const s = [...k.data].sort((a, b) => a - b), lo = s[0], hi = s[s.length - 1];
    const off = document.createElement('canvas'); off.width = k.cols; off.height = k.rows;
    const og = off.getContext('2d'), id = og.createImageData(k.cols, k.rows);
    k.data.forEach((x, i) => { const c = (x - lo) / (hi - lo || 1) * 255; id.data.set([c, c, c, 255], i * 4); });
    og.putImageData(id, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(off, 0, 0, cv.width, cv.height);
    const sx = cv.width / k.cols, sy = cv.height / k.rows, est = v.velocity / spacing * rowTime;
    for (let s0 = 0; s0 < o.spots; s0++) {
      const x0 = (s0 + 0.5) * k.cols / o.spots - sp * k.rows / 2;
      g.setLineDash([5, 5]); g.strokeStyle = '#ffd23d'; g.beginPath(); g.moveTo((x0 + 0.5) * sx, 0); g.lineTo((x0 + sp * k.rows + 0.5) * sx, k.rows * sy); g.stroke();
      g.setLineDash([]); g.strokeStyle = '#ff6bd6'; g.beginPath(); g.moveTo((x0 + 0.5) * sx, 0); g.lineTo((x0 + est * k.rows + 0.5) * sx, k.rows * sy); g.stroke();
    }
    g.fillStyle = '#3dffb0';
    for (const e of ev) g.fillRect(e.position * sx, e.time * sy, Math.max(sx, 2), Math.max(sy, 2));
    const u = spacing === 1 && rowTime === 1 ? 'samples/row' : 'distance/time';
    $('#vals').innerHTML = vals([['true speed', sp * spacing / rowTime, u], ['velocity (least squares)', v.velocity, u],
      ['median of map', v.median_velocity, u], ['mean of map', v.mean_velocity, u], ['SD of map', v.std_velocity, u],
      ['events', ev.length]]) + `<p class="small">${esc($('#em').value)} events: ${ev.slice(0, 6).map(e => `x=${e.position} t=${e.time} (${e.duration} rows, max ${fmt(e.amplitude, 1)})`).join('; ')}${ev.length > 6 ? '; ...' : ''}</p>`;
  }
  ['#sp', '#nz', '#wd', '#ns', '#k', '#spc', '#rt'].forEach(s => $(s).addEventListener('input', run));
  $('#em').addEventListener('change', run);
  $('#seed').addEventListener('click', () => { seed++; run(); });
  run();
}
