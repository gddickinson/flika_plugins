// Node test: track_engine.js reproduces spt_batch_analysis's track features and directional autocorrelation on the
// fixture tracks. Run: node web/static/track_engine.test.mjs
import assert from 'node:assert/strict';
import { checks, close, closeAll, fx } from './fixture.test.mjs';
import { analyseTrack, autocorrelation } from './track_engine.js';

const T = fx.tracks;
for (const c of T.single) {
  const pts = c.input.x.map((x, i) => [x, c.input.y[i]]), r = analyseTrack(pts), name = c.input.name;
  for (const k of ['rg', 'asymmetry', 'skewness', 'kurtosis', 'fractal_dimension', 'net_displacement', 'efficiency',
    'sin_mean', 'cos_mean', 'rg_simple', 'mean_step', 'scaled_rg', 'persistence']) close(r[k], c[k], `${name} ${k}`);
  for (const k of ['sin_vals', 'cos_vals', 'directions', 'autocorrelation']) closeAll(r[k], c[k], `${name} ${k}`);
  assert.equal(r.linear.classification, c.linear.classification, `${name} classification`);
  for (const k of ['directionality_ratio', 'mean_perpendicular_distance', 'normalized_perpendicular_distance'])
    close(r.linear[k], c.linear[k], `${name} ${k}`);
}
for (const s of T.autocorrelation) {
  const r = autocorrelation(s.inputs, s.time_interval, s.num_intervals);
  closeAll(r.lags, s.lags, `${s.name} lags`);
  closeAll(r.avg, s.avg, `${s.name} avg`);
  closeAll(r.sem, s.sem, `${s.name} sem`);
  assert.deepEqual(r.count.slice(1), s.count.slice(1), `${s.name} counts`);
  const flat = r.individual.flatMap(t => t.values.map(v => v[1]));
  closeAll(flat, s.individual.correlation, `${s.name} per-track`);
}
console.log(`track_engine: ${T.single.length} tracks + ${T.autocorrelation.length} autocorrelation sets, ${checks} values match Python`);
