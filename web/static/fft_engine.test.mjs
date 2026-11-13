// Node test: fft_engine.js reproduces fft_Chunker's fft_chunks and group averages. Run: node web/static/fft_engine.test.mjs
import assert from 'node:assert/strict';
import { checks, close, closeAll, fx } from './fixture.test.mjs';
import { fakeSignal, fftChunks, groupAverage } from './fft_engine.js';

const F = fx.fft;
closeAll(fakeSignal().y, F.cases[0].trace, 'fakeData signal', 1e-12);
for (const c of F.cases) {
  const r = fftChunks(c.trace, c.chunk, c.time_step), what = `${c.name}/${c.chunk}`;
  assert.equal(r.chunks.length, c.chunks.length, what + ' chunk count');
  r.chunks.forEach((ch, i) => {
    closeAll(ch.power.filter(Number.isFinite), c.chunks[i].power, `${what} power ${i + 1}`, 1e-9);
    closeAll(ch.frequency.filter(Number.isFinite), c.chunks[i].frequency, `${what} frequency ${i + 1}`, 1e-12);
    closeAll(ch.power, c.power_raw[i], `${what} power column ${i + 1}`, 1e-9);
  });
  closeAll(r.ranges, c.ranges, what + ' ranges', 1e-9);
}
const A = F.average, chunks = fftChunks(F.cases[2].trace, A.chunk, A.time_step).chunks;
for (const [g, [a, b]] of Object.entries(A.groups)) {
  const r = groupAverage(chunks, a, b), key = g === 'baseline' ? 'baseline' : g;
  if (!(`${key}_PowerAverage` in A.result)) { assert.equal(r, null, g); continue; }
  closeAll(r.power, A.result[`${key}_PowerAverage`], g + ' power', 1e-9);
  closeAll(r.frequency, A.result[`${key}_FreqAverage`], g + ' frequency', 1e-12);
}
console.log(`fft_engine: ${F.cases.length} traces, ${checks} values match Python`);
