// Small shared UI pieces: status labels, number formatting, an SVG line/scatter chart, a safe Markdown renderer.
import { esc } from './core.js';

export const GROUP_SHORT = { main: 'plugin', experimental: 'experimental', py311: 'not on Py 3.11', broken: 'not working',
  old: 'old', backup: 'backup' };
export const groupBadge = g => `<span class="st ${esc(g)}" title="status, from the folder it is in">${esc(GROUP_SHORT[g] || g)}</span>`;

export function fmt(v, d = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return 'NaN';
  if (typeof v !== 'number') return esc(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  return a >= 1e5 || a < 1e-3 ? v.toExponential(2) : String(+v.toFixed(d));
}

// series: [{ x: [...], y: [...], color, dots?, line?, err? }]. Returns an SVG string.
export function chart(series, { w = 520, h = 220, xlabel = '', ylabel = '', title = '', logy = false } = {}) {
  const tf = v => (logy ? (v > 0 ? Math.log10(v) : NaN) : v);
  const pts = series.flatMap(s => s.x.map((x, i) => [x, tf(s.y[i])])).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!pts.length) return '<p class="mute small">nothing to plot</p>';
  let [x0, x1] = [Math.min(...pts.map(p => p[0])), Math.max(...pts.map(p => p[0]))];
  let [y0, y1] = [Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))];
  if (x0 === x1) { x0 -= 1; x1 += 1; } if (y0 === y1) { y0 -= 1; y1 += 1; }
  const L = 56, R = 10, T = 12, B = 34, X = x => L + (x - x0) / (x1 - x0) * (w - L - R), Y = y => h - B - (y - y0) / (y1 - y0) * (h - T - B);
  const yl = v => (logy ? '1e' + fmt(v, 1) : fmt(v, 2));
  let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title || ylabel)}">`;
  s += `<line class="axis" x1="${L}" y1="${h - B}" x2="${w - R}" y2="${h - B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${h - B}"/>`;
  s += `<text x="${L}" y="${h - B + 14}">${fmt(x0, 2)}</text><text x="${w - R}" y="${h - B + 14}" text-anchor="end">${fmt(x1, 2)}</text>`;
  s += `<text x="${L - 4}" y="${h - B}" text-anchor="end">${yl(y0)}</text><text x="${L - 4}" y="${T + 8}" text-anchor="end">${yl(y1)}</text>`;
  s += `<text x="${(L + w - R) / 2}" y="${h - 4}" text-anchor="middle">${esc(xlabel)}</text>`;
  s += `<text x="12" y="${(T + h - B) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(T + h - B) / 2})">${esc(ylabel)}</text>`;
  for (const se of series) {
    const p = se.x.map((x, i) => [x, tf(se.y[i]), se.err ? se.err[i] : 0]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
    if (se.err) s += p.filter(q => Number.isFinite(q[2]) && q[2] > 0).map(([x, y, e]) =>
      `<line x1="${X(x).toFixed(1)}" x2="${X(x).toFixed(1)}" y1="${Y(y - e).toFixed(1)}" y2="${Y(y + e).toFixed(1)}" stroke="${se.color}" opacity=".6"/>`).join('');
    if (se.line !== false && p.length > 1) s += `<polyline fill="none" stroke="${se.color}" stroke-width="1.6" points="${p.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(' ')}"/>`;
    if (se.dots) s += p.map(([x, y]) => `<circle cx="${X(x).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="${se.r || 2.4}" fill="${se.color}" opacity="${se.opacity || 0.9}"/>`).join('');
  }
  return s + '</svg>';
}

export const vals = items => `<div class="vals">${items.map(([k, v, unit]) => `<div><b>${fmt(v)}${unit ? ' <small>' + esc(unit) + '</small>' : ''}</b><span>${esc(k)}</span></div>`).join('')}</div>`;

// Markdown subset for READMEs: headings, fenced code, lists, tables (shown as text), quotes, paragraphs, inline code,
// bold/italic and http(s) links. Everything is escaped first; images are shown as their alt text.
function inline(s) {
  let t = esc(s);
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  t = t.replace(/!\[([^\]]*)\]\([^)]*\)/g, (_, a) => (a ? `<span class="mute">[image: ${a}]</span>` : ''));
  t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  t = t.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');
  t = t.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/(^|\W)\*([^*\s][^*]*)\*(?=\W|$)/g, '$1<i>$2</i>');
  return t.replace(/\\([_*`#\\])/g, '$1');
}
export function markdown(md) {
  const lines = md.replace(/\r/g, '').split('\n'), out = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^\s*```/.test(l)) {
      const buf = [];
      for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) buf.push(lines[i]);
      out.push(`<pre>${esc(buf.join('\n'))}</pre>`); i++; continue;
    }
    const hm = l.match(/^(#{1,6})\s+(.*)/);
    if (hm) { const n = Math.min(hm[1].length + 2, 6); out.push(`<h${n}>${inline(hm[2].replace(/#+\s*$/, ''))}</h${n}>`); i++; continue; }
    if (/^\s*\|/.test(l)) {
      const buf = [];
      for (; i < lines.length && /^\s*\|/.test(lines[i]); i++) if (!/^\s*\|[\s:|-]+\|\s*$/.test(lines[i])) buf.push(lines[i]);
      const rows = buf.map(r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => inline(c.trim())));
      out.push(`<div class="tblwrap"><table class="tbl"><tr>${rows[0].map(c => `<th>${c}</th>`).join('')}</tr>${rows.slice(1)
        .map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</table></div>`);
      continue;
    }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(l)) {
      const ord = /^\s*\d/.test(l), buf = [];
      for (; i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i]); i++) buf.push(lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, ''));
      out.push(`<${ord ? 'ol' : 'ul'}>${buf.map(b => `<li>${inline(b)}</li>`).join('')}</${ord ? 'ol' : 'ul'}>`); continue;
    }
    if (/^\s*>/.test(l)) {
      const buf = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) buf.push(lines[i].replace(/^\s*>\s?/, ''));
      out.push(`<blockquote>${inline(buf.join(' '))}</blockquote>`); continue;
    }
    if (!l.trim() || /^\s*([-*_])\1{2,}\s*$/.test(l) || /^\s*<\/?[a-z]/i.test(l)) { i++; continue; }
    const buf = [];
    for (; i < lines.length && lines[i].trim() && !/^\s*(#|```|\||[-*+]\s|\d+[.)]\s|>)/.test(lines[i]); i++) buf.push(lines[i]);
    if (!buf.length) { out.push(`<p>${inline(l)}</p>`); i++; continue; }
    out.push(`<p>${inline(buf.join(' '))}</p>`);
  }
  return out.join('\n');
}
