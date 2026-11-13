// Node test: kymo_engine.js reproduces kymograph_analyzer's measure_velocities and detect_events on the fixture
// kymographs. Run: node web/static/kymo_engine.test.mjs
import assert from 'node:assert/strict';
import { checks, close, closeAll, fx } from './fixture.test.mjs';
import { detectEvents, measureVelocities } from './kymo_engine.js';

for (const c of fx.kymograph.cases) {
  const k = { rows: c.rows, cols: c.cols, data: Float64Array.from(c.data) }, what = `speed ${c.speed}`;
  const v = measureVelocities(k, { spacing: c.spacing, rowTime: c.row_time });
  for (const key of ['velocity', 'mean_velocity', 'median_velocity', 'std_velocity']) close(v[key], c[key], `${what} ${key}`);
  closeAll([...v.map], c.map, `${what} map`);
  for (const m of ['threshold', 'zscore']) {
    const ev = detectEvents(k, { method: m, threshold: 1, minDuration: 3 });
    assert.equal(ev.length, c.events[m].length, `${what} ${m} count`);
    ev.forEach((e, i) => { for (const key of Object.keys(c.events[m][i])) close(e[key], c.events[m][i][key], `${what} ${m} ${key}`); });
  }
}
console.log(`kymo_engine: ${fx.kymograph.cases.length} kymographs, ${checks} values match Python`);
