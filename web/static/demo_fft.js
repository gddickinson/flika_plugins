// FFT-in-chunks demo: fft_engine.js (the fft_Chunker port) on the plugin's own test signal or a pasted trace.
import { esc } from './core.js';
import { fakeSignal, fftChunks, groupAverage, parseTrace } from './fft_engine.js';
import { chart, fmt } from './ui.js';

const COLORS = ['#3dffb0', '#3db8ff', '#ff6bd6', '#ffd23d', '#ff8a3d'];

export function render(host) {
  host.innerHTML = `<div class="demo-grid">
    <div><div id="trace"></div><div id="spec"></div></div>
    <div class="ctl2">
      <label for="sig">Trace</label><select id="sig"><option value="fake">fakeData.py signal (50 Hz + 0.5 x 80 Hz)</option><option value="noisy">same, plus noise</option><option value="paste">pasted trace</option></select>
      <label for="chunk">Chunk size <b id="chv"></b> samples</label><input id="chunk" type="range" min="16" max="600" step="1" value="100">
      <label for="dt">Time step (s)</label><input id="dt" type="number" step="any" min="0" value="0.00125">
      <label for="show">Chunk to plot</label><select id="show"></select>
      <label class="chk"><input id="logy" type="checkbox" checked> log power axis (as the plugin's plots)</label>
      <details class="box"><summary>Paste a trace</summary><label for="txt">one value per line (or CSV: the last column is used)</label>
        <textarea id="txt"></textarea><button type="button" id="use">use pasted trace</button><p id="err" class="err small"></p></details>
    </div>
  </div>
  <h3>Group averages (makeAverage)</h3>
  <p class="small">Chunks are numbered from 1. As in the plugin, a group averages chunks <i>start</i> to <i>stop</i>
    (stop included; start = stop is one chunk), and is used when stop > 0 and stop >= start.</p>
  <div class="groups" id="groups"></div><div id="avg"></div><div class="tblwrap"><table class="tbl" id="peaks"></table></div>
  <p class="small mute">"Power" is |FFT| (the magnitude), as the plugin computes it; negative frequencies are dropped.
    The plugin exports log10 of these values. The noise added here is from a seeded generator (not data).</p>`;
  const $ = s => host.querySelector(s);
  const groups = [['baseline', 1, 1], ['puff1', 2, 3], ['puff2', 4, 5], ['puff3', 6, 6], ['puff4', 0, 0]];
  $('#groups').innerHTML = groups.map(([g, a, b]) => `<div><span>${g}</span><label>start <input type="number" min="1" value="${a}" data-g="${g}" data-e="a"></label>
    <label>stop <input type="number" min="0" value="${b}" data-g="${g}" data-e="b"></label></div>`).join('');
  let pasted = null;

  const trace = () => {
    const f = fakeSignal();
    if ($('#sig').value === 'paste' && pasted) return pasted;
    if ($('#sig').value === 'noisy') { let s = 7; const r = () => ((s = (s * 16807) % 2147483647) / 2147483647); return f.y.map(v => v + (r() + r() + r() - 1.5)); }
    return f.y;
  };

  function run() {
    const y = trace(), n = Math.min(+$('#chunk').value, y.length), dt = +$('#dt').value || 1;
    $('#chv').textContent = n;
    const r = fftChunks(y, n, dt), keep = $('#show').value;
    $('#show').innerHTML = r.chunks.map((c, i) => `<option value="${i}">chunk ${i + 1} (${c.n} samples)</option>`).join('');
    $('#show').value = keep && +keep < r.chunks.length ? keep : '0';
    const ci = +$('#show').value, c = r.chunks[ci], t = y.map((_, i) => i * dt);
    $('#trace').innerHTML = `<h3>Trace (${y.length} samples, ${r.chunks.length} chunks)</h3>` + chart([{ x: t, y, color: '#3db8ff' },
      { x: t.slice(ci * n, ci * n + c.n), y: c.chunk, color: '#ff6bd6' }], { xlabel: 'time (s)', ylabel: 'value' });
    $('#spec').innerHTML = `<h3>Chunk ${ci + 1}: |FFT|</h3>` + chart([{ x: c.frequency, y: c.power, color: '#3dffb0', dots: true, line: false }],
      { xlabel: 'frequency (Hz)', ylabel: 'power', logy: $('#logy').checked });
    const series = [], rows = [];
    groups.forEach(([g], k) => {
      const a = +host.querySelector(`[data-g="${g}"][data-e="a"]`).value, b = +host.querySelector(`[data-g="${g}"][data-e="b"]`).value;
      const av = groupAverage(r.chunks, a, b);
      if (!av) return;
      if (av.error) { rows.push(`<tr><td>${g}</td><td colspan="3" class="err">${esc(av.error)}</td></tr>`); return; }
      series.push({ x: av.frequency, y: av.power, color: COLORS[k], dots: true, line: false });
      let best = -1;
      av.power.forEach((p, i) => { if (i > 0 && Number.isFinite(p) && (best < 0 || p > av.power[best])) best = i; });
      rows.push(`<tr><td><span style="color:${COLORS[k]}">${g}</span></td><td>${av.chunks} chunk${av.chunks === 1 ? '' : 's'} (${a} to ${b})</td><td>${best > 0 ? fmt(av.frequency[best], 3) + ' Hz' : '-'}</td><td>${best > 0 ? fmt(av.power[best]) : '-'}</td></tr>`);
    });
    $('#avg').innerHTML = chart(series, { w: 1040, h: 240, xlabel: 'frequency (Hz)', ylabel: 'mean power', logy: $('#logy').checked });
    $('#peaks').innerHTML = '<tr><th>group</th><th>chunks</th><th>peak frequency (above 0 Hz)</th><th>its mean power</th></tr>' + rows.join('');
  }

  $('#sig').addEventListener('change', () => { if ($('#sig').value !== 'paste') $('#dt').value = 0.00125; run(); });
  ['#chunk', '#dt'].forEach(s => $(s).addEventListener('input', run));
  ['#show', '#logy'].forEach(s => $(s).addEventListener('change', run));
  host.querySelectorAll('[data-g]').forEach(i => i.addEventListener('input', run));
  $('#use').addEventListener('click', () => {
    try { pasted = parseTrace($('#txt').value); $('#err').textContent = ''; $('#sig').value = 'paste'; $('#chunk').max = pasted.length; run(); }
    catch (e) { $('#err').textContent = e.message; }
  });
  run();
}
