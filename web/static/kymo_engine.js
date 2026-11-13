// Kymograph velocity and event detection: a JavaScript port of kymograph_analyzer.py (measure_velocities,
// detect_events 'threshold' / 'zscore' with _event_bounds). Checked by kymo_engine.test.mjs against the Python.
// A kymograph is { rows, cols, data } (row = time, column = position along the line).

// np.gradient (edge_order=1) along rows (axis 0) or columns (axis 1).
function gradient(k, axis) {
  const { rows, cols, data } = k, out = new Float64Array(data.length), n = axis === 0 ? rows : cols;
  const at = (i, j) => data[i * cols + j];
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const p = axis === 0 ? i : j, get = q => (axis === 0 ? at(q, j) : at(i, q));
    out[i * cols + j] = p === 0 ? get(1) - get(0) : p === n - 1 ? get(n - 1) - get(n - 2) : (get(p + 1) - get(p - 1)) / 2;
  }
  return out;
}

// measure_velocities: least-squares v = -sum(It Ix)/sum(Ix^2), and the per-pixel map -It/Ix where |Ix| is at least
// minGradient x its maximum. Units: samples per row x spacing / rowTime. Positive = towards the end of the line.
export function measureVelocities(k, { spacing = 1, rowTime = 1, minGradient = 0.1 } = {}) {
  const It = gradient(k, 0), Ix = gradient(k, 1), scale = spacing / rowTime;
  let num = 0, den = 0, mx = 0;
  for (let i = 0; i < Ix.length; i++) { num += It[i] * Ix[i]; den += Ix[i] * Ix[i]; mx = Math.max(mx, Math.abs(Ix[i])); }
  const velocity = den > 0 ? -num / den * scale : NaN;
  const map = new Float64Array(Ix.length).fill(NaN), v = [];
  if (den > 0) for (let i = 0; i < Ix.length; i++) if (Math.abs(Ix[i]) > minGradient * mx) { map[i] = -It[i] / Ix[i] * scale; v.push(map[i]); }
  const mean = v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
  const s = [...v].sort((a, b) => a - b), median = !s.length ? NaN : s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
  const std = v.length ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length) : NaN;
  return { velocity, mean_velocity: mean, median_velocity: median, std_velocity: std, map };
}

// _event_bounds: complete runs of frames above threshold, [start, end) with start = first frame above.
function eventBounds(above) {
  const starts = [], ends = [];
  for (let i = 0; i + 1 < above.length; i++) {
    if (!above[i] && above[i + 1]) starts.push(i + 1);
    if (above[i] && !above[i + 1]) ends.push(i + 1);
  }
  if (ends.length && starts.length && ends[0] < starts[0]) ends.shift();
  return starts.slice(0, ends.length).map((s, i) => [s, ends[i]]);
}

// detect_events, 'threshold' (trace > threshold*std + mean) or 'zscore' (z > threshold; std with ddof 0).
export function detectEvents(k, { method = 'threshold', threshold = 2, minDuration = 3 } = {}) {
  const events = [];
  for (let j = 0; j < k.cols; j++) {
    const tr = Array.from({ length: k.rows }, (_, i) => k.data[i * k.cols + j]);
    const m = tr.reduce((a, b) => a + b, 0) / tr.length, sd = Math.sqrt(tr.reduce((a, b) => a + (b - m) ** 2, 0) / tr.length);
    const z = tr.map(x => (x - m) / sd);
    const above = method === 'zscore' ? z.map(x => x > threshold) : tr.map(x => x > threshold * sd + m);
    for (const [s, e] of eventBounds(above)) if (e - s >= minDuration) {
      const ev = { time: s, position: j, duration: e - s, amplitude: Math.max(...tr.slice(s, e)) };
      if (method === 'zscore') ev.z_score = Math.max(...z.slice(s, e));
      events.push(ev);
    }
  }
  return events;
}

// A synthetic kymograph for the demo (not data): Gaussian spots moving at constant speed plus seeded noise.
export function synthKymo(rand, { rows = 80, cols = 120, speed = 0.8, width = 4, noise = 3, spots = 2 } = {}) {
  const data = new Float64Array(rows * cols), x0 = Array.from({ length: spots }, (_, s) => (s + 0.5) * cols / spots - speed * rows / 2);
  const g = () => { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    let v = 10;
    for (const x of x0) v += 100 * Math.exp(-0.5 * ((j - x - speed * i) / width) ** 2);
    data[i * cols + j] = v + noise * g();
  }
  return { rows, cols, data };
}
