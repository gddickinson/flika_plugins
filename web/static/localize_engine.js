// Spot detection and localization: a JavaScript port of spt_batch_analysis/thunderstorm_python, line for line, for
// one frame (ThunderSTORM.analyze_frame with the wavelet filter, the local-maximum detector and the integrated-Gaussian
// LSQ / WLSQ fitters). Checked against the Python by localize_engine.test.mjs (tests/fixtures/demo_cases.json).
// Images are { width, height, data: Float64Array (row-major) }; positions are [row, col].
import { erf } from './erf.js';

// filters.py WaveletFilter._bspline_N / _create_bspline_kernel
function bsplineN(k, t) {
  if (k <= 1) return t >= 0 && t < 1 ? 1 : 0;
  return t / (k - 1) * bsplineN(k - 1, t) + (k - t) / (k - 1) * bsplineN(k - 1, t - 1);
}
export function bsplineKernel(order = 3, scale = 2) {
  const n = Math.trunc(order * scale - 1), half = Math.floor(n / 2);
  const k = Array.from({ length: n }, (_, i) => bsplineN(order, (-half + i) / scale + order / 2));
  const s = k.reduce((a, b) => a + b, 0);
  return s > 0 ? k.map(v => v / s) : k;
}

// scipy.ndimage.convolve1d(..., mode='nearest') along rows (axis=1) or columns (axis=0), odd-length weights, with
// scipy's summation order for symmetric weights (centre first, then the pairs from the outside in).
function convolve1d(src, w, h, wd, axis) {
  const L = w.length, c = (L - 1) / 2, out = new Float64Array(src.length);
  const rev = w.slice().reverse(), sym = rev.every((v, i) => v === rev[L - 1 - i]);
  const n = axis === 1 ? wd : h;
  const at = (line, i) => line[Math.min(Math.max(i, 0), n - 1)];
  const line = new Float64Array(n);
  for (let o = 0; o < (axis === 1 ? h : wd); o++) {
    for (let i = 0; i < n; i++) line[i] = axis === 1 ? src[o * wd + i] : src[i * wd + o];
    for (let i = 0; i < n; i++) {
      let t;
      if (sym) {
        t = at(line, i) * rev[c];
        for (let j = -c; j < 0; j++) t += (at(line, i + j) + at(line, i - j)) * rev[c + j];
      } else {
        t = 0;
        for (let j = 0; j < L; j++) t += rev[j] * at(line, i + j - c);
      }
      if (axis === 1) out[o * wd + i] = t; else out[i * wd + o] = t;
    }
  }
  return out;
}

// WaveletFilter.apply: two a-trous planes on an edge-padded image. Returns F = V1 - V2 and F1 = I - V1.
export function waveletFilter(img, { order = 3, scale = 2 } = {}) {
  const h = bsplineKernel(order, scale), n = h.length, h2 = new Array(2 * (n - 1) + 1).fill(0);
  h.forEach((v, i) => { h2[2 * i] = v; });
  const m = Math.floor(h2.length / 2), W = img.width + 2 * m, H = img.height + 2 * m;
  const pad = new Float64Array(W * H);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const rr = Math.min(Math.max(r - m, 0), img.height - 1), cc = Math.min(Math.max(c - m, 0), img.width - 1);
    pad[r * W + c] = img.data[rr * img.width + cc];
  }
  const v1 = convolve1d(convolve1d(pad, h, H, W, 1), h, H, W, 0);
  const v2 = convolve1d(convolve1d(v1, h2, H, W, 1), h2, H, W, 0);
  const F = new Float64Array(img.data.length), F1 = new Float64Array(img.data.length);
  for (let r = 0; r < img.height; r++) for (let c = 0; c < img.width; c++) {
    const p = (r + m) * W + c + m, q = r * img.width + c;
    F[q] = v1[p] - v2[p];
    F1[q] = img.data[q] - v1[p];
  }
  return { F, F1, kernel: h };
}

const mean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };
export function std(a) { const m = mean(a); let s = 0; for (const v of a) s += (v - m) * (v - m); return Math.sqrt(s / a.length); }

// LocalMaximumDetector.detect. 8-neighbourhood: skimage peak_local_max (min_distance=1, no border exclusion), i.e.
// pixels equal to the 3x3 maximum (edges repeated) and above threshold, strongest first. 4-neighbourhood: pixels
// >= their 4 neighbours (grey_dilation with a cross) and above threshold, in row order.
export function localMaxima(F, width, height, threshold, connectivity = 8) {
  const at = (r, c) => F[Math.min(Math.max(r, 0), height - 1) * width + Math.min(Math.max(c, 0), width - 1)];
  const nb = connectivity === 4 ? [[-1, 0], [1, 0], [0, -1], [0, 1]]
    : [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
  const out = [];
  let trivial = true;
  for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) {
    const v = F[r * width + c];
    let isMax = true;
    for (const [dr, dc] of nb) if (at(r + dr, c + dc) > v) { isMax = false; break; }
    if (!isMax) trivial = false;
    if (isMax && v > threshold) out.push([r, c]);
  }
  if (connectivity === 4) return out;
  if (trivial) return [];
  return out.map((p, i) => [p, i]).sort((a, b) => F[b[0][0] * width + b[0][1]] - F[a[0][0] * width + a[0][1]] || a[1] - b[1])
    .map(x => x[0]);
}

// detection.remove_border_detections
export const removeBorder = (pos, width, height, b) => pos.filter(([r, c]) => r >= b && r < height - b && c >= b && c < width - b);

// fitting.integrated_gaussian_value / integrated_gaussian_jacobian
const SQRT2 = Math.SQRT2, TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);
function model(jj, ii, x0, y0, s, I, b) {
  const s2 = SQRT2 * s;
  const ex = 0.5 * (erf((jj - x0 + 0.5) / s2) - erf((jj - x0 - 0.5) / s2));
  const ey = 0.5 * (erf((ii - y0 + 0.5) / s2) - erf((ii - y0 - 0.5) / s2));
  return b + I * ex * ey;
}
function jacobian(jj, ii, x0, y0, s, I) {
  const s2 = SQRT2 * s, inv = 1 / s2;
  const ex = 0.5 * (erf((jj - x0 + 0.5) / s2) - erf((jj - x0 - 0.5) / s2));
  const ey = 0.5 * (erf((ii - y0 + 0.5) / s2) - erf((ii - y0 - 0.5) / s2));
  const uxp = (jj - x0 + 0.5) * inv, uxm = (jj - x0 - 0.5) * inv, uyp = (ii - y0 + 0.5) * inv, uym = (ii - y0 - 0.5) * inv;
  const gxp = TWO_OVER_SQRTPI * Math.exp(-uxp * uxp), gxm = TWO_OVER_SQRTPI * Math.exp(-uxm * uxm);
  const gyp = TWO_OVER_SQRTPI * Math.exp(-uyp * uyp), gym = TWO_OVER_SQRTPI * Math.exp(-uym * uym);
  const dex = 0.5 * inv * (gxm - gxp), dey = 0.5 * inv * (gym - gyp);
  const dexs = 0.5 * (-uxp * gxp + uxm * gxm) / s, deys = 0.5 * (-uyp * gyp + uym * gym) / s;
  return [I * dex * ey, I * ex * dey, I * (dexs * ey + ex * deys), ex * ey, 1];
}

// fitting.solve_5x5: Gaussian elimination with partial pivoting.
function solve5(A, b) {
  const n = 5, x = new Array(n).fill(0);
  for (let k = 0; k < n; k++) {
    let mx = Math.abs(A[k][k]), mr = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > mx) { mx = Math.abs(A[i][k]); mr = i; }
    if (mx < 1e-30) return x;
    if (mr !== k) { [A[k], A[mr]] = [A[mr], A[k]]; [b[k], b[mr]] = [b[mr], b[k]]; }
    const piv = A[k][k];
    for (let i = k + 1; i < n; i++) {
      const f = A[i][k] / piv;
      for (let j = k + 1; j < n; j++) A[i][j] -= f * A[k][j];
      A[i][k] = 0; b[i] -= f * b[k];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs(A[i][i]) < 1e-30) { x[i] = 0; continue; }
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}

// fit_gaussian_lsq_batch_numba / fit_gaussian_wlsq_batch_numba for one position (weighted = WLSQ, w = 1/max(data,1)).
// Levenberg-Marquardt on [x0, y0, sqrt(sigma), sqrt(intensity), sqrt(offset)]. Returns null when the ROI is < 3 px.
export function fitSpot(img, row, col, { fitRadius = 3, initialSigma = 1.3, weighted = false, maxIterations = 1001 } = {}) {
  const r0 = Math.max(0, row - fitRadius), r1 = Math.min(img.height, row + fitRadius + 1);
  const c0 = Math.max(0, col - fitRadius), c1 = Math.min(img.width, col + fitRadius + 1);
  if (r1 - r0 < 3 || c1 - c0 < 3) return null;
  const ny = r1 - r0, nx = c1 - c0, roi = [];
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) roi.push(img.data[(r0 + i) * img.width + c0 + j]);
  let offset = Math.max(Math.min(...roi), 0.01);
  const peak = Math.max(...roi) - offset;
  let x0 = col - c0, y0 = row - r0, sigma = initialSigma;
  let intensity = Math.max(peak * 2 * Math.PI * sigma * sigma, 1);
  let p2 = Math.sqrt(sigma), p3 = Math.sqrt(intensity), p4 = Math.sqrt(offset), lambda = 1;
  const wt = d => (weighted ? 1 / Math.max(d, 1) : 1);
  const chi = (x, y, s, I, b) => {
    let c = 0;
    for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
      const d = roi[i * nx + j], r = d - model(j, i, x, y, s, I, b);
      c += weighted ? wt(d) * r * r : r * r;
    }
    return c;
  };
  let chiSq = chi(x0, y0, sigma, intensity, offset), iterations = 0;
  for (let it = 0; it < maxIterations; it++) {
    iterations = it + 1;
    const H = Array.from({ length: 5 }, () => new Array(5).fill(0)), g = new Array(5).fill(0);
    for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
      const d = roi[i * nx + j], res = d - model(j, i, x0, y0, sigma, intensity, offset), w = wt(d);
      const Jo = jacobian(j, i, x0, y0, sigma, intensity);
      const J = [Jo[0], Jo[1], Jo[2] * 2 * p2, Jo[3] * 2 * p3, Jo[4] * 2 * p4];
      for (let p = 0; p < 5; p++) {
        g[p] += weighted ? w * J[p] * res : J[p] * res;
        for (let q = 0; q < 5; q++) H[p][q] += weighted ? w * J[p] * J[q] : J[p] * J[q];
      }
    }
    for (let p = 0; p < 5; p++) H[p][p] *= 1 + lambda;
    const dp = solve5(H, g.slice());
    const xn = x0 + dp[0], yn = y0 + dp[1];
    let p2n = p2 + dp[2];
    const p3n = p3 + dp[3], p4n = p4 + dp[4];
    let sn = Math.min(p2n * p2n, 10);
    if (sn < 0.1) { sn = 0.1; p2n = Math.sqrt(sn); }
    const In = p3n * p3n, bn = p4n * p4n, chiNew = chi(xn, yn, sn, In, bn);
    if (chiNew < chiSq) {
      const rel = Math.abs(chiSq - chiNew) / Math.max(chiSq, 1e-30);
      [x0, y0, p2, sigma, intensity, offset, chiSq] = [xn, yn, p2n, sn, In, bn, chiNew];
      p3 = p3n; p4 = p4n;
      lambda = Math.max(lambda / 10, 1e-7);
      if (rel < 1e-8) break;
    } else {
      lambda *= 10;
      if (lambda > 1e10) break;
    }
  }
  return { x: x0 + c0, y: y0 + r0, intensity, sigma, background: offset, chi_squared: chiSq,
           success: sigma < 9.9, iterations, roi: [r0, r1, c0, c1] };
}

// compute_localization_precision (Thompson; F = 2 for EMCCD) and BaseFitter._photons_and_background_sd.
export function precision(N, b, s, a, isEmccd) {
  if (N <= 0) return NaN;
  const F = isEmccd ? 2 : 1, sa2 = s * s + a * a / 12, tau = 2 * Math.PI * sa2 * b * b / (N * a * a);
  return Math.sqrt(sa2 / N * (F + 4 * tau));
}
export function photonsAndBackgroundSd(I, bg, { photons_per_adu = 1, baseline = 0 } = {}) {
  const ppa = photons_per_adu > 0 ? photons_per_adu : 1;
  return [Math.max(I, 0) * ppa, Math.sqrt(Math.max(bg - baseline, 0) * ppa)];
}

// ThunderSTORM.analyze_frame: filter, threshold k*std(Wave.F1), detect, drop border detections (border = fit radius),
// fit, keep successful fits inside the image, convert to nm with the (x + 0.5) * pixel_size corner-origin convention.
export function analyzeFrame(img, cfg) {
  const { connectivity = 8, fitter = 'lsq', initialSigma = 1.3, threshold = 1, fitRadius = 3, pixelSize = 100,
    camera = {} } = cfg;
  const { F, F1 } = waveletFilter(img);
  const thr = threshold * std(F1);
  const detected = localMaxima(F, img.width, img.height, thr, connectivity);
  const positions = removeBorder(detected, img.width, img.height, fitRadius);
  const fits = [];
  for (const [r, c] of positions) {
    const f = fitSpot(img, r, c, { fitRadius, initialSigma, weighted: fitter === 'wlsq' });
    if (!f || !f.success) continue;
    const [N, b] = photonsAndBackgroundSd(f.intensity, f.background, camera);
    const unc = precision(N, b, f.sigma, 1, !!camera.is_emccd);
    if (f.x >= 0 && f.x < img.width && f.y >= 0 && f.y < img.height) fits.push({ ...f, uncertainty: unc, row: r, col: c });
  }
  const nm = v => (v + 0.5) * pixelSize;
  return { F, F1, threshold: thr, detected, positions, fits,
           result: { x: fits.map(f => nm(f.x)), y: fits.map(f => nm(f.y)), intensity: fits.map(f => f.intensity),
                     background: fits.map(f => f.background), sigma_x: fits.map(f => f.sigma * pixelSize),
                     chi_squared: fits.map(f => f.chi_squared), uncertainty: fits.map(f => f.uncertainty * pixelSize) } };
}

// Nearest-neighbour matching of localizations to reference positions within `radius` px (greedy, closest first).
export function matchToReference(locs, ref, radius = 1.5) {
  const pairs = [];
  locs.forEach((p, i) => ref.forEach((q, j) => { const d = Math.hypot(p[0] - q[0], p[1] - q[1]); if (d <= radius) pairs.push([d, i, j]); }));
  pairs.sort((a, b) => a[0] - b[0]);
  const ui = new Set(), uj = new Set(), matched = [];
  for (const [d, i, j] of pairs) if (!ui.has(i) && !uj.has(j)) { ui.add(i); uj.add(j); matched.push({ i, j, d, dx: locs[i][0] - ref[j][0], dy: locs[i][1] - ref[j][1] }); }
  return { matched, missed: ref.length - matched.length, extra: locs.length - matched.length };
}
