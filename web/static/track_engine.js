// Track features and directional autocorrelation: JavaScript ports of spt_batch_analysis (__init__.py) classes
// FeatureCalculator, GeometricAnalyzer, DirectionAnalyzer and AutocorrelationAnalyzer, line for line. Checked against
// the Python by track_engine.test.mjs (tests/fixtures/demo_cases.json). Points are [x, y] in pixels.

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const deg = r => r * 180 / Math.PI;

// np.linalg.eig on a symmetric 2x2 matrix as LAPACK dgeev computes it (dlanv2's standardisation), so the eigenvector
// sign, and so the sign of the skewness, matches the Python result. Same routine as the flika_scripts port.
export function eig2(a, b, d) {
  if (b === 0) return { w: [a, d], v: [[1, 0], [0, 1]] };
  const p = 0.5 * (a - d), bcmax = Math.abs(b), bcmis = Math.abs(b), scale = Math.max(Math.abs(p), bcmax);
  let z = p / scale * p + bcmax / scale * bcmis;
  z = p + (p >= 0 ? 1 : -1) * Math.sqrt(scale) * Math.sqrt(z);
  const w0 = d + z, w1 = d - bcmax / z * bcmis, tau = Math.hypot(b, z), cs = z / tau, sn = b / tau;
  return { w: [w0, w1], v: [[cs, -sn], [sn, cs]] };
}

// scipy.stats.skew / kurtosis (bias=True, Fisher): NaN when the values are all (numerically) equal.
function moments(a) {
  const m = mean(a), d = a.map(v => v - m);
  const m2 = mean(d.map(v => v * v)), m3 = mean(d.map(v => v * v * v)), m4 = mean(d.map(v => (v * v) * (v * v)));
  const zero = m2 <= (Number.EPSILON * m) ** 2;
  return { skew: zero ? NaN : m3 / m2 ** 1.5, kurt: zero ? NaN : m4 / m2 ** 2 - 3 };
}

// FeatureCalculator.radius_gyration_asymmetry
export function radiusGyration(pts) {
  const n = pts.length, cx = mean(pts.map(p => p[0])), cy = mean(pts.map(p => p[1]));
  let sxx = 0, sxy = 0, syy = 0;
  for (const [x, y] of pts) { const dx = x - cx, dy = y - cy; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
  const { w, v } = eig2(sxx / n, sxy / n, syy / n);
  const rg = Math.sqrt(w[0] + w[1]);
  const num = (w[0] - w[1]) ** 2, den = 2 * (w[0] + w[1]) ** 2;
  const asymmetry = den > 0 ? -Math.log(1 - num / den) : 0;
  const k = w[0] >= w[1] ? 0 : 1, ev = [v[0][k], v[1][k]], n2 = ev[0] ** 2 + ev[1] ** 2;
  const proj = [];
  for (let i = 0; i + 1 < n; i++) proj.push(((pts[i + 1][0] - pts[i][0]) * ev[0] + (pts[i + 1][1] - pts[i][1]) * ev[1]) / n2);
  const { skew, kurt } = proj.length ? moments(proj) : { skew: 0, kurt: 0 };
  return { rg, asymmetry, skewness: skew, kurtosis: kurt, eigenvalues: w, dominant: ev, center: [cx, cy] };
}

// FeatureCalculator.fractal_dimension: 1.0 for < 3 points, collinear points, or when the convex hull fails (Qhull
// rejects a degenerate set: all points on one line). The largest distance between hull vertices is the largest
// distance between any two points.
export function fractalDimension(pts) {
  if (pts.length < 3) return 1;
  const [x0, y0] = pts[0];
  const others = pts.filter(([x, y]) => x !== x0 || y !== y0);
  if (others.length < 2) return 1;
  const slopes = others.map(([x, y]) => (x !== x0 ? (y - y0) / (x - x0) : null)).filter(s => s !== null);
  const s0 = others[0][0] !== x0 ? (others[0][1] - y0) / (others[0][0] - x0) : null;
  if (slopes.every(s => s === s0)) return 1;
  const [ax, ay] = others[0];
  if (others.every(([x, y]) => (ax - x0) * (y - y0) - (ay - y0) * (x - x0) === 0)) return 1;
  let total = 0, largest = 0;
  for (let i = 1; i < pts.length; i++) total += Math.sqrt((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2);
  for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++)
    largest = Math.max(largest, Math.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2));
  if (!(total > 0 && largest > 0)) return 1;
  return Math.log(pts.length) / Math.log(pts.length * largest / total);
}

// FeatureCalculator.net_displacement_efficiency
export function netDisplacement(pts) {
  const n = pts.length;
  if (n < 2) return { net: 0, efficiency: 0 };
  const net = Math.hypot(pts[0][0] - pts[n - 1][0], pts[0][1] - pts[n - 1][1]);
  if (n < 3) return { net, efficiency: 1 };
  let ss = 0;
  for (let i = 1; i < n; i++) ss += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) ** 2;
  return { net, efficiency: ss > 0 ? net ** 2 / ((n - 1) * ss) : 0 };
}

// FeatureCalculator.summed_sines_cosines: consecutive repeated positions dropped; zero-length steps give 0.
export function sinesCosines(pts) {
  if (pts.length < 3) return { sin_mean: 0, sin_vals: [], cos_mean: 0, cos_vals: [] };
  const keep = [pts[0]];
  for (let i = 1; i < pts.length; i++) if (!(pts[i][0] === pts[i - 1][0] && pts[i][1] === pts[i - 1][1])) keep.push(pts[i]);
  if (keep.length < 3) return { sin_mean: 0, sin_vals: [], cos_mean: 0, cos_vals: [] };
  const sin = [], cos = [];
  for (let i = 0; i + 2 < keep.length; i++) {
    const ab = [keep[i + 1][0] - keep[i][0], keep[i + 1][1] - keep[i][1]];
    const bc = [keep[i + 2][0] - keep[i + 1][0], keep[i + 2][1] - keep[i + 1][1]];
    const mag = Math.hypot(ab[0], ab[1]) * Math.hypot(bc[0], bc[1]);
    cos.push(mag > 0 ? (ab[0] * bc[0] + ab[1] * bc[1]) / mag : 0);
    sin.push(mag > 0 ? (ab[0] * bc[1] - ab[1] * bc[0]) / mag : 0);
  }
  return { sin_mean: mean(sin), sin_vals: sin, cos_mean: mean(cos), cos_vals: cos };
}

// GeometricAnalyzer: simple Rg, mean step, scaled Rg (Golan & Sherman 2017), linear-motion classification.
export function rgSimple(pts) {
  if (pts.length < 2) return NaN;
  const mx = mean(pts.map(p => p[0])), my = mean(pts.map(p => p[1]));
  const m2x = mean(pts.map(p => p[0] ** 2)), m2y = mean(pts.map(p => p[1] ** 2));
  return Math.sqrt((m2x - mx ** 2) + (m2y - my ** 2));
}
export function meanStep(pts) {
  if (pts.length < 2) return NaN;
  const s = [];
  for (let i = 1; i < pts.length; i++) s.push(Math.sqrt((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2));
  return mean(s);
}
export const scaledRg = (rg, step) => (Number.isNaN(rg) || Number.isNaN(step) || step === 0 ? NaN : Math.sqrt(Math.PI / 2) * rg / step);
export function classifyLinear(pts, dirThr = 0.8, perpThr = 0.15) {
  const none = { classification: 'unclassified', directionality_ratio: NaN, mean_perpendicular_distance: NaN,
    normalized_perpendicular_distance: NaN };
  if (pts.length < 3) return none;
  const s = pts[0], e = pts[pts.length - 1], net = Math.hypot(e[0] - s[0], e[1] - s[1]);
  let total = 0;
  for (let i = 1; i < pts.length; i++) total += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  if (total === 0) return none;
  const ratio = net / total;
  let mpd = NaN, npd = NaN;
  if (net > 0) {
    const d = [(e[0] - s[0]) / net, (e[1] - s[1]) / net];
    mpd = mean(pts.map(([x, y]) => { const v = [x - s[0], y - s[1]], l = v[0] * d[0] + v[1] * d[1];
      return Math.hypot(v[0] - l * d[0], v[1] - l * d[1]); }));
    npd = mpd / net;
  }
  const directional = ratio >= dirThr, straight = Number.isNaN(npd) ? false : npd <= perpThr;
  return { classification: directional && straight ? 'linear_unidirectional' : !directional && straight ? 'linear_bidirectional' : 'non_linear',
    directionality_ratio: ratio, mean_perpendicular_distance: mpd, normalized_perpendicular_distance: npd };
}

// DirectionAnalyzer: step direction in degrees with the y axis inverted, persistence 1 - mean|turn|/180, and the
// per-track directional autocorrelation (mean cos of the angle change at each lag, padded with NaN).
export function direction(x1, y1, x2, y2) {
  const dx = x2 - x1, dy = -(y2 - y1);
  if (dx === 0 && dy === 0) return NaN;
  const a = deg(Math.atan2(dy, dx));
  return a < 0 ? a + 360 : a;
}
export function persistence(angles) {
  const v = angles.filter(a => !Number.isNaN(a));
  if (angles.length < 2 || v.length < 2) return NaN;
  const d = [];
  for (let i = 1; i < v.length; i++) { let x = Math.abs(v[i] - v[i - 1]); if (x > 180) x = 360 - x; d.push(x); }
  return 1 - mean(d) / 180;
}
export function trackAutocorrelation(pts) {
  const ang = [];
  for (let i = 1; i < pts.length; i++) ang.push(Math.atan2(-(pts[i][1] - pts[i - 1][1]), pts[i][0] - pts[i - 1][0]));
  const N = ang.length;
  if (N < 2) return pts.map(() => NaN);
  const out = [];
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let j = 0; j < N - i; j++) s += Math.cos(ang[j + i] - ang[j]);
    out.push(s / (N - i));
  }
  while (out.length < pts.length) out.push(NaN);
  return out;
}

// Everything the demo shows for one track (points [x, y]).
export function analyseTrack(pts) {
  const rg = radiusGyration(pts), nd = netDisplacement(pts), rgs = rgSimple(pts), step = meanStep(pts);
  const directions = [];
  for (let i = 0; i + 1 < pts.length; i++) directions.push(direction(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]));
  return { ...rg, fractal_dimension: fractalDimension(pts), net_displacement: nd.net, efficiency: nd.efficiency,
    ...sinesCosines(pts), rg_simple: rgs, mean_step: step, scaled_rg: scaledRg(rgs, step), linear: classifyLinear(pts),
    directions, persistence: persistence(directions), autocorrelation: trackAutocorrelation(pts) };
}

// AutocorrelationAnalyzer.process_track_data (Gorelik & Gautreau 2014) on tracks [{frame, x, y}] (kept if >= 3 points,
// sorted by frame): unit step vectors, their dot products at lags 1..numIntervals within each track, then the mean and
// SEM (sample SD / sqrt(n)) per lag; lag 0 is 1 by definition.
export function autocorrelation(tracks, timeInterval = 1, numIntervals = 10) {
  const lagVals = Array.from({ length: numIntervals }, () => []), individual = [];
  tracks.filter(t => t.frame.length >= 3).forEach((t, k) => {
    const idx = t.frame.map((_, i) => i).sort((a, b) => t.frame[a] - t.frame[b]);
    const x = idx.map(i => t.x[i]), y = idx.map(i => t.y[i]), L = x.length;
    const vx = [NaN], vy = [NaN];
    for (let i = 0; i + 1 < L; i++) {
      const dx = x[i] - x[i + 1], dy = y[i] - y[i + 1], m = Math.sqrt(dx * dx + dy * dy);
      vx.push(m > 0 ? dx / m : NaN); vy.push(m > 0 ? dy / m : NaN);
    }
    const per = [];  // [lag, mean dot product] in lag order
    for (let step = 1; step <= Math.min(numIntervals, L - 2); step++) {
      const dots = [];
      for (let i = 0; i + step < L; i++) { const d = vx[i] * vx[i + step] + vy[i] * vy[i + step]; if (!Number.isNaN(d)) dots.push(d); }
      lagVals[step - 1].push(...dots);
      if (dots.length) per.push([step * timeInterval, mean(dots)]);
    }
    individual.push({ track: t.id ?? k, values: per });
  });
  const lags = [0], avg = [1], sem = [0], count = [null];
  lagVals.forEach((v, i) => {
    lags.push((i + 1) * timeInterval); count.push(v.length);
    const m = v.length ? mean(v) : NaN;
    avg.push(m);
    sem.push(v.length ? Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)) / Math.sqrt(v.length) : 0);
  });
  return { lags, avg, sem, count, individual };
}

// Seeded random walks for the demo (not data): 'free' | 'confined' | 'directed'.
export function simulateTrack(rand, { n = 60, kind = 'free', drift = 0.8, confine = 0.4 } = {}) {
  const gauss = () => { let u = 0, v = 0; while (!u) u = rand(); while (!v) v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const x = [0], y = [0], ang = rand() * 2 * Math.PI;
  for (let i = 1; i < n; i++) {
    let dx = gauss(), dy = gauss();
    if (kind === 'directed') { dx += Math.cos(ang) * drift; dy += Math.sin(ang) * drift; }
    if (kind === 'confined') { dx -= confine * x[i - 1]; dy -= confine * y[i - 1]; }
    x.push(x[i - 1] + dx); y.push(y[i - 1] + dy);
  }
  return { frame: x.map((_, i) => i), x: x.map(v => v + 30), y: y.map(v => v + 30) };
}

// "frame,x,y" (or "x,y") text, one point per line; a header line is skipped.
export function parseTrack(text) {
  const rows = text.trim().split(/\r?\n/).map(l => l.split(/[,\t; ]+/).filter(Boolean).map(Number));
  const body = rows.filter(r => r.length >= 2 && r.every(Number.isFinite));
  if (body.length < 3) throw new Error('Need at least 3 rows of numbers: frame,x,y (or x,y).');
  const three = body.every(r => r.length >= 3);
  return { frame: body.map((r, i) => (three ? r[0] : i)), x: body.map(r => (three ? r[1] : r[0])), y: body.map(r => (three ? r[2] : r[1])) };
}
