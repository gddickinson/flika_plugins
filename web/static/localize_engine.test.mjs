// Node test: localize_engine.js reproduces thunderstorm_python (wavelet filter, threshold, local maxima, LSQ/WLSQ
// fits, uncertainty) on the fixture frames. Run: node web/static/localize_engine.test.mjs
import assert from 'node:assert/strict';
import { checks, close, closeAll, fx } from './fixture.test.mjs';
import { analyzeFrame, bsplineKernel, precision, waveletFilter } from './localize_engine.js';
import { readFileSync } from 'node:fs';

const L = fx.localization;
closeAll(bsplineKernel(3, 2), L.kernel, 'kernel', 1e-15);
const frames = JSON.parse(readFileSync(new URL('../../tests/fixtures/localization_frames.json', import.meta.url), 'utf8'));
const frame = (sid, i) => { const f = frames.sources.find(s => s.id === sid).frames[i];
  return { width: f.width, height: f.height, data: Float64Array.from(f.image) }; };
const w = waveletFilter(frame(L.wavelet.source, L.wavelet.frame));
closeAll([...w.F], L.wavelet.F, 'wavelet F', 1e-9);
closeAll([...w.F1], L.wavelet.F1, 'wavelet F1', 1e-9);
for (const p of L.precision) close(precision(...p.args), p.value, 'precision', 1e-12);
let fits = 0;
for (const c of L.cases) {
  const src = frames.sources.find(s => s.id === c.source), cfg = c.config;
  const r = analyzeFrame(frame(c.source, c.frame), { connectivity: cfg.connectivity, fitter: cfg.fitter,
    initialSigma: cfg.initial_sigma, threshold: cfg.threshold, fitRadius: cfg.fit_radius, pixelSize: src.pixel_size,
    camera: src.camera });
  const what = `${c.source}[${c.frame}] ${cfg.fitter}/${cfg.connectivity}`;
  close(r.threshold, c.threshold, what + ' threshold', 1e-9);
  assert.deepEqual(r.detected, c.positions, what + ' positions');
  for (const k of Object.keys(c.result)) closeAll(r.result[k], c.result[k], `${what} ${k}`, 1e-6);
  fits += c.result.x.length;
}
console.log(`localize_engine: ${L.cases.length} frame/settings cases, ${fits} fits, ${checks} values match Python`);
