// Core helpers and the hash router (shared web kit; Microscope Atlas pattern).
// Pages register with route(name, fn); fn(args, query) renders into #app. Data comes from api(path): the Python server
// at api/<path>, or, on the static site (window.APP_STATIC), static_api.js answering from exported JSON or in-browser code.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const isStatic = () => !!window.APP_STATIC;

export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
export const list = (items, fn, sep = '') => (items || []).map(fn).join(sep);
export const app = () => $('#app');

const cache = new Map();
// GET (or POST with body) an API path such as '/items' or '/items?q=x'. GETs are cached unless fresh.
export async function api(path, { fresh = false, body = null } = {}) {
  const key = body ? null : path;
  if (key && !fresh && cache.has(key)) return cache.get(key);
  let v;
  if (isStatic()) {
    v = await (await import('./static_api.js')).staticApi(path, body);
  } else {
    const opt = body instanceof FormData ? { method: 'POST', body }
      : body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {};
    const r = await fetch('api' + path, opt);
    if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
    v = await r.json();
  }
  if (key) cache.set(key, v);
  return v;
}

// --- router ---------------------------------------------------------------
const routes = {};
export function route(name, fn) { routes[name] = fn; }
const loaders = {};
// lazy({ name: () => import('./page.js') }): the module is loaded on first visit and registers its routes.
export function lazy(map) { Object.assign(loaders, map); }
export function go(hash) { location.hash = hash; }

function parse() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [path, qs] = raw.split('?');
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  return { name: parts[0] || 'home', args: parts.slice(1), query: Object.fromEntries(new URLSearchParams(qs || '')) };
}

let stopCurrent = null;
// A page with an animation loop registers its stop function here; it runs when the route changes.
export function onLeave(fn) { stopCurrent = fn; }

async function render() {
  if (stopCurrent) { try { stopCurrent(); } catch (e) { console.warn(e); } stopCurrent = null; }
  const { name, args, query } = parse();
  $$('[data-nav]').forEach(a => a.classList.toggle('on', a.dataset.nav === name));
  $$('nav details').forEach(d => d.removeAttribute('open'));
  const root = app();
  root.innerHTML = '<p class="mute">Loading...</p>';
  try {
    if (!routes[name] && loaders[name]) await loaders[name]();
    await (routes[name] || routes.notfound || (() => { root.innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; }))(args, query);
  } catch (e) {
    console.error(e);
    root.innerHTML = `<h1>Something went wrong</h1><p class="err">${esc(e.message)}</p>`;
  }
  if (!query.keep) window.scrollTo(0, 0);
  root.focus({ preventScroll: true });
}

export function start() {
  window.addEventListener('hashchange', render);
  document.addEventListener('click', e => {
    $$('nav details[open]').forEach(d => { if (!d.contains(e.target)) d.removeAttribute('open'); });
  });
  render();
}

// --- small UI helpers -------------------------------------------------------
export const tags = (items, cls = '') => list(items, t => `<span class="tag ${cls}">${esc(t)}</span>`);

// Slider with a live value label; returns the input element. onInput(value) fires on every change.
export function slider(host, { id, label, min, max, step = 1, value, unit = '', fmt = v => v }, onInput) {
  const wrap = document.createElement('div');
  wrap.className = 'ctl';
  wrap.innerHTML = `<label for="sl-${id}">${esc(label)} <b>${fmt(value)}${unit}</b></label>
    <input type="range" id="sl-${id}" min="${min}" max="${max}" step="${step}" value="${value}">`;
  host.appendChild(wrap);
  const input = wrap.querySelector('input'), out = wrap.querySelector('b');
  input.addEventListener('input', () => { out.textContent = fmt(+input.value) + unit; onInput(+input.value); });
  return input;
}

// Notice shown on the static site where a feature needs the local Python server.
export function needsServer(what = 'This feature') {
  return `<p class="static-note">${esc(what)} needs the full app running locally (Python server): see the README, "Run the web app".</p>`;
}

// requestAnimationFrame loop that stops when the page is left (and respects reduced motion if asked).
export function animate(step) {
  let id = 0, alive = true;
  const tick = t => { if (!alive) return; step(t); id = requestAnimationFrame(tick); };
  id = requestAnimationFrame(tick);
  const stop = () => { alive = false; cancelAnimationFrame(id); };
  onLeave(stop);
  return stop;
}

// Download a Blob or string as a file.
export function download(name, data, type = 'text/plain') {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// Seeded RNG (mulberry32) so simulations and tests are reproducible.
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
