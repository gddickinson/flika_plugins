// FFT in chunks: a JavaScript port of fft_Chunker/fft_Chunker.py (fft_chunks and the group averages of
// FFT_Chunker.makeAverage). Checked against the Python by fft_engine.test.mjs (tests/fixtures/demo_cases.json).

// |DFT| of a real sequence (any length), twiddle angles reduced mod n for accuracy. scipy.fft gives the same
// magnitudes to rounding.
export function dftMagnitude(x) {
  const n = x.length, out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let re = 0, im = 0;
    for (let j = 0; j < n; j++) {
      const a = -2 * Math.PI * ((k * j) % n) / n;
      re += x[j] * Math.cos(a); im += x[j] * Math.sin(a);
    }
    out[k] = Math.hypot(re, im);
  }
  return out;
}

// numpy/scipy fftfreq(n, d)
export function fftfreq(n, d) {
  const val = 1 / (n * d);
  return Array.from({ length: n }, (_, k) => (k <= Math.floor((n - 1) / 2) ? k : k - n) * val);
}

// fft_chunks(l, n, time_step): chunks of n samples (the last may be shorter); for each, power = |FFT| and frequency,
// with negative frequencies set to NaN. Columns are padded with NaN to n rows, as in the plugin's DataFrame.
export function fftChunks(trace, n, timeStep) {
  const chunks = [];
  let fmin = Infinity, fmax = -Infinity, pmin = Infinity, pmax = -Infinity;
  for (let i = 0; i < trace.length; i += n) {
    const chunk = trace.slice(i, i + n), mag = dftMagnitude(chunk), freq = fftfreq(chunk.length, timeStep);
    const power = [], frequency = [];
    for (let k = 0; k < n; k++) {
      const ok = k < chunk.length && freq[k] >= 0;
      power.push(ok ? mag[k] : NaN); frequency.push(ok ? freq[k] : NaN);
    }
    for (let k = 0; k < chunk.length; k++) if (freq[k] >= 0) {
      fmin = Math.min(fmin, freq[k]); fmax = Math.max(fmax, freq[k]);
      pmin = Math.min(pmin, mag[k]); pmax = Math.max(pmax, mag[k]);
    }
    chunks.push({ chunk, power, frequency, n: chunk.length });
  }
  return { chunks, ranges: [fmin, fmax, pmin, pmax, Math.min(...trace), Math.max(...trace)] };
}

// makeAverage's getAverage(start, stop): element-wise mean of the power (and frequency) columns of chunks
// start .. stop inclusive (chunks are numbered from 1; start == stop is one chunk). A row is NaN when any averaged
// chunk is NaN there. A group is used when stop > 0 and stop >= start.
export function groupAverage(chunks, start, stop) {
  if (!(stop > 0 && stop >= start)) return null;
  const sel = chunks.slice(Math.max(start, 1) - 1, stop);
  if (start < 1 || stop > chunks.length) return { error: `chunks ${start} to ${stop}: there are only ${chunks.length} chunks (the plugin limits the boxes to that)` };
  const avg = key => sel[0][key].map((_, r) => sel.reduce((s, c) => s + c[key][r], 0) / sel.length);
  return { power: avg('power'), frequency: avg('frequency'), chunks: sel.length };
}

// fft_Chunker/fakeData.py's test signal: 50 Hz + 0.5 x 80 Hz, N = 600 samples spaced T = 1/800 s (np.linspace(0, N*T, N)).
export function fakeSignal(N = 600, T = 1 / 800) {
  const step = N * T / (N - 1);
  const x = Array.from({ length: N }, (_, i) => (i === N - 1 ? N * T : i * step));
  return { t: x, y: x.map(v => Math.sin(50 * 2 * Math.PI * v) + 0.5 * Math.sin(80 * 2 * Math.PI * v)), T };
}

// One numeric column from pasted text (the last number on each line; a header line is skipped).
export function parseTrace(text) {
  const v = text.trim().split(/\r?\n/).map(l => l.split(/[,\t; ]+/).filter(Boolean).map(Number).pop()).filter(Number.isFinite);
  if (v.length < 8) throw new Error('Need at least 8 numbers, one per line (the last number on each line is used).');
  return v;
}
