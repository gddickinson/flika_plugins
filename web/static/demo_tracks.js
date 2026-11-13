// Track features demo: track_engine.js (the spt_batch_analysis port) on committed tracks, seeded random walks, a track
// you draw, or one you paste; plus the directional autocorrelation over a set of tracks.
import { api, esc, rng } from './core.js';
import { analyseTrack, autocorrelation, parseTrack, simulateTrack } from './track_engine.js';
import { chart, vals } from './ui.js';

const W = 520, H = 400;

export async function render(host) {
  const data = await api('/demo-data/tracks');
  const real = data.tracks.filter(t => t.frame.length >= 10);
  host.innerHTML = `<div class="demo-grid">
    <div>
      <div class="stage"><canvas id="cv" width="${W}" height="${H}" aria-label="track plot: click to add points in draw mode"></canvas></div>
      <p class="small mute" id="cap"></p>
      <div class="seg" role="group" aria-label="track source">${['real', 'free', 'confined', 'directed', 'draw'].map(k =>
        `<button type="button" data-k="${k}">${k === 'real' ? 'committed tracks' : k}</button>`).join('')}</div>
      <div id="realctl"><label for="tid">Track (longest first)</label><select id="tid">${real.map((t, i) =>
        `<option value="${i}">track ${t.id}: ${t.frame.length} points, ${esc(t.movement_type)}</option>`).join('')}</select></div>
      <div id="simctl"><button type="button" id="new" class="alt">new random walk</button>
        <label for="n">Points <b id="nv">60</b></label><input id="n" type="range" min="4" max="300" value="60"></div>
      <details class="box"><summary>Paste a track</summary>
        <label for="txt">frame,x,y (or x,y) per line, in pixels</label><textarea id="txt"></textarea>
        <button type="button" id="use">analyse pasted track</button><p id="err" class="err small"></p></details>
    </div>
    <div><div id="vals"></div></div>
  </div>
  <div class="two"><div id="c1"></div><div id="c2"></div></div>
  <h2>Directional autocorrelation over many tracks</h2>
  <p class="small">AutocorrelationAnalyzer.process_track_data on the committed tracks with at least
    <input id="minlen" type="number" min="3" max="60" value="10" aria-label="minimum track length"> points; lags of
    <input id="nint" type="number" min="2" max="40" value="15" aria-label="number of lags"> steps.</p>
  <div id="c3"></div><p class="small mute" id="acnote"></p>
  <p class="small mute">${esc(data.note)} Source: <code>${esc(data.source)}</code>. Random walks are generated here with a
    seeded generator and are not data. Persistence is 1 - mean |turn| / 180: 1 for a straight line, about 0.5 for
    uncorrelated directions, 0 when every step reverses. Skewness and kurtosis are of the step projections on the
    dominant axis of the gyration tensor (dashed line).</p>`;
  const $ = s => host.querySelector(s), cv = $('#cv'), g = cv.getContext('2d');
  let kind = 'real', seed = 1, track = null, drawn = [];

  const make = () => {
    $('#realctl').classList.toggle('hide', kind !== 'real'); $('#simctl').classList.toggle('hide', !['free', 'confined', 'directed'].includes(kind));
    if (kind === 'real') { const t = real[+$('#tid').value]; track = t; $('#cap').textContent = `Track ${t.id} from ${data.source} (positions in pixels).`; }
    else if (kind === 'draw') { track = drawn.length ? { frame: drawn.map((_, i) => i), x: drawn.map(p => p[0]), y: drawn.map(p => p[1]) } : null; $('#cap').textContent = 'Click on the plot to add points in order.'; }
    else if (kind !== 'pasted') { track = simulateTrack(rng(seed), { n: +$('#n').value, kind }); $('#cap').textContent = `Synthetic ${kind} random walk, seed ${seed} (not data).`; }
    draw();
  };

  function draw() {
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    if (!track || track.x.length < 3) { $('#vals').innerHTML = '<p class="mute">Need at least 3 points.</p>'; $('#c1').innerHTML = $('#c2').innerHTML = ''; return; }
    const pts = track.x.map((x, i) => [x, track.y[i]]), r = analyseTrack(pts);
    const fixed = kind === 'draw', pad = 30, xs = track.x, ys = track.y;
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 2);
    const sc = fixed ? 1 : Math.min((W - 2 * pad) / span, (H - 2 * pad) / span);
    const ox = fixed ? 0 : W / 2 - sc * (Math.min(...xs) + Math.max(...xs)) / 2, oy = fixed ? 0 : H / 2 - sc * (Math.min(...ys) + Math.max(...ys)) / 2;
    const P = (x, y) => [ox + sc * x, oy + sc * y], [cx, cy] = P(...r.center);
    g.strokeStyle = 'rgba(61,184,255,.4)'; g.beginPath(); g.arc(cx, cy, r.rg * sc, 0, 7); g.stroke();
    g.setLineDash([6, 5]); g.strokeStyle = '#ff6bd6'; g.beginPath();
    g.moveTo(cx - r.dominant[0] * 2 * r.rg * sc, cy - r.dominant[1] * 2 * r.rg * sc); g.lineTo(cx + r.dominant[0] * 2 * r.rg * sc, cy + r.dominant[1] * 2 * r.rg * sc); g.stroke(); g.setLineDash([]);
    g.strokeStyle = '#3dffb0'; g.lineWidth = 1.6; g.beginPath();
    pts.forEach(([x, y], i) => { const [a, b] = P(x, y); i ? g.lineTo(a, b) : g.moveTo(a, b); }); g.stroke(); g.lineWidth = 1;
    pts.forEach(([x, y], i) => { const [a, b] = P(x, y); g.fillStyle = i === 0 ? '#3dffb0' : i === pts.length - 1 ? '#ff5c7a' : '#bfffe6'; g.beginPath(); g.arc(a, b, i === 0 || i === pts.length - 1 ? 4.5 : 2, 0, 7); g.fill(); });
    $('#vals').innerHTML = vals([['radius of gyration', r.rg, 'px'], ['asymmetry', r.asymmetry], ['skewness', r.skewness], ['kurtosis', r.kurtosis],
      ['fractal dimension', r.fractal_dimension], ['net displacement', r.net_displacement, 'px'], ['efficiency', r.efficiency],
      ['straightness (mean cos)', r.cos_mean], ['mean sin', r.sin_mean], ['Rg (simple)', r.rg_simple, 'px'], ['mean step', r.mean_step, 'px'],
      ['scaled Rg', r.scaled_rg], ['directionality ratio', r.linear.directionality_ratio], ['persistence', r.persistence]])
      + `<p class="small">Linear-motion class: <b>${esc(r.linear.classification.replace(/_/g, ' '))}</b></p>`;
    const steps = r.directions.map((_, i) => i + 1);
    $('#c1').innerHTML = `<h3>Step direction</h3>${chart([{ x: steps, y: r.directions, color: '#3dffb0', dots: true, line: false }], { xlabel: 'step', ylabel: 'degrees (y up)' })}`;
    const ac = r.autocorrelation.map((v, i) => [i, v]).filter(([, v]) => Number.isFinite(v));
    $('#c2').innerHTML = `<h3>This track's directional autocorrelation</h3>${chart([{ x: ac.map(a => a[0]), y: ac.map(a => a[1]), color: '#3db8ff', dots: true }], { xlabel: 'lag (steps)', ylabel: 'mean cos(turn)' })}`;
  }

  function drawAc() {
    const minLen = Math.max(3, +$('#minlen').value || 10), n = Math.min(40, Math.max(2, +$('#nint').value || 15));
    const set = data.tracks.filter(t => t.frame.length >= minLen), r = autocorrelation(set, 1, n);
    $('#c3').innerHTML = chart([{ x: r.lags, y: r.avg, err: r.sem, color: '#ff6bd6', dots: true }], { w: 1040, h: 240, xlabel: 'lag (steps)', ylabel: 'mean cos (unit step vectors)' });
    $('#acnote').textContent = `${set.length} tracks; error bars are SEM; lag 1 averages ${r.count[1]} vector pairs.`;
  }

  const setKind = k => { kind = k; host.querySelectorAll('[data-k]').forEach(b => b.classList.toggle('on', b.dataset.k === k)); make(); };
  host.querySelectorAll('[data-k]').forEach(b => b.addEventListener('click', () => { if (b.dataset.k === 'draw') drawn = []; setKind(b.dataset.k); }));
  $('#tid').addEventListener('change', make);
  $('#new').addEventListener('click', () => { seed++; make(); });
  $('#n').addEventListener('input', () => { $('#nv').textContent = $('#n').value; make(); });
  cv.addEventListener('click', e => {
    if (kind !== 'draw') return;
    const b = cv.getBoundingClientRect();
    drawn.push([+((e.clientX - b.left) * W / b.width).toFixed(1), +((e.clientY - b.top) * H / b.height).toFixed(1)]);
    make();
  });
  $('#use').addEventListener('click', () => {
    try { track = parseTrack($('#txt').value); $('#err').textContent = ''; kind = 'pasted'; host.querySelectorAll('[data-k]').forEach(b => b.classList.remove('on')); $('#cap').textContent = 'Your pasted track.'; draw(); }
    catch (err) { $('#err').textContent = err.message; }
  });
  ['#minlen', '#nint'].forEach(s => $(s).addEventListener('change', drawAc));
  setKind('real'); drawAc();
}
