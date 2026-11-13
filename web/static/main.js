// Entry point: lazy page modules, the mode label, a faint drifting-particles background, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = { 'home.js': ['home'], 'browse.js': ['plugins'], 'plugin.js': ['plugin'], 'demos.js': ['demos', 'demo'],
  'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: catalogue read from the repository at build time; demos run in your browser'
  : 'local app: catalogue read from this checkout; demos run in your browser';

// Background: a few diffusing spots leaving short trails, like single-particle tracks. Off under reduced motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d'), cols = ['#3dffb0', '#3db8ff', '#ff6bd6'];
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const ps = Array.from({ length: 14 }, (_, i) => ({ c: cols[i % 3], t: [[Math.random() * innerWidth, Math.random() * innerHeight]] }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 80) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of ps) {
      const [x, y] = p.t[p.t.length - 1];
      p.t.push([(x + (Math.random() - 0.5) * 9 + canvas.width) % canvas.width, (y + (Math.random() - 0.5) * 9 + canvas.height) % canvas.height]);
      if (p.t.length > 40) p.t.shift();
      g.strokeStyle = p.c; g.globalAlpha = 0.5; g.beginPath();
      p.t.forEach(([a, b], i) => (i && Math.abs(a - p.t[i - 1][0]) < 20 && Math.abs(b - p.t[i - 1][1]) < 20 ? g.lineTo(a, b) : g.moveTo(a, b)));
      g.stroke(); g.globalAlpha = 1; g.fillStyle = p.c;
      const [a, b] = p.t[p.t.length - 1]; g.beginPath(); g.arc(a, b, 2.2, 0, 7); g.fill();
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
