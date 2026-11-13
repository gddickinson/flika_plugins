// One plugin: info.xml fields, status, README, Python files (classes, functions, docstrings), links to demos and copies.
import { api, app, esc, list, route } from './core.js';
import { groupBadge, markdown } from './ui.js';

const GROUP_NOTE = {
  experimental: 'In EXPERIMENTAL/: still being developed.', py311: 'In NOT_WORKING_WITH_PY311/: does not run on Python 3.11 or later.',
  broken: 'In NOT_WORKING/: marked as not working.', old: 'In OLD/: a superseded version kept for reference.',
  backup: 'In backups/: a snapshot, not the current plugin.',
};

route('plugin', async ([id]) => {
  let p;
  try { p = await api('/plugin/' + id); } catch { app().innerHTML = '<h1>No such plugin</h1><p><a href="#/plugins">All plugins</a></p>'; return; }
  const cat = await api('/catalogue'), topic = cat.topics.find(t => t.id === p.topic);
  const kv = [['Folder', `<a href="${esc(p.github)}" target="_blank" rel="noopener">${esc(p.path)}/</a>`],
    ['Author', esc(p.author || 'not given')], ['Version', esc(p.version || 'not given')],
    ['Topic', `<a href="#/plugins?topic=${p.topic}">${esc(topic.title)}</a>`],
    ['Menu entries', p.menu.length ? p.menu.map(esc).join(' · ') : 'none in info.xml'],
    ['Dependencies (info.xml)', p.dependencies.length ? p.dependencies.map(d => `<code>${esc(d)}</code>`).join(' ') : 'none listed'],
    ['Code', `${p.files.length} Python files, ${p.lines.toLocaleString()} lines, ${p.n_classes} classes, ${p.n_functions} top-level functions${p.uses_flika ? '; imports flika' : ''}${p.uses_qt ? ' and Qt' : ''}`]];
  if (p.url) kv.push(['Download URL', `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.url)}</a>`]);
  if (p.docs.length) kv.push(['Other documents', p.docs.map(esc).join(', ')]);
  const files = p.files.map(f => `<div class="fn"><code class="sig">${esc(f.path)}</code> <span class="mute small">${f.lines} lines</span>
    ${f.error ? `<p class="err small">Does not parse with Python 3.12: ${esc(f.error)}</p>` : ''}
    ${f.doc ? `<p class="small">${esc(f.doc)}</p>` : ''}
    ${f.classes && f.classes.length ? `<p class="small"><span class="mute">classes</span> ${f.classes.map(c => `<code>${esc(c)}</code>`).join(' ')}</p>` : ''}
    ${f.functions && f.functions.length ? `<p class="small"><span class="mute">functions</span> ${f.functions.map(c => `<code>${esc(c)}</code>`).join(' ')}</p>` : ''}</div>`).join('');
  app().innerHTML = `<p class="kicker"><a href="#/plugins">Plugins</a></p>
    <h1>${esc(p.name)} ${groupBadge(p.group)}</h1>
    ${p.description ? `<p class="lead">${esc(p.description)}</p>` : '<p class="mute">No description in info.xml or README.</p>'}
    ${GROUP_NOTE[p.group] ? `<p class="static-note">${GROUP_NOTE[p.group]}</p>` : ''}
    ${p.nested_in ? `<p class="small">Inside the folder of <a href="#/plugin/${esc(p.nested_in)}">${esc(p.nested_in)}</a>.</p>` : ''}
    ${p.demos.length ? `<div class="panel"><b>Run it here:</b> ${list(p.demos, d => `<a href="#/demo/${d.id}">${esc(d.title)}</a>`, ' · ')}</div>` : ''}
    <dl class="kv">${kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>
    ${p.same_folder.length ? `<p class="small">Same folder name elsewhere in the repository: ${list(p.same_folder, o => `<a href="#/plugin/${esc(o.id)}">${esc(o.path)}</a>`, ', ')}</p>` : ''}
    <details class="box" ${p.readme ? 'open' : ''}><summary>${esc(p.readme_name || 'README')}</summary>
      ${p.readme ? `<div class="md">${markdown(p.readme)}</div>${p.readme_cut ? '<p class="mute small">(truncated: see the file on GitHub)</p>' : ''}` : '<p class="mute">No README in this folder.</p>'}</details>
    <details class="box"><summary>Python files (${p.files.length})</summary>${files || '<p class="mute">none</p>'}</details>
    <div class="pager">${p.prev ? `<a href="#/plugin/${esc(p.prev)}">&larr; ${esc(p.prev)}</a>` : '<span></span>'}${p.next ? `<a href="#/plugin/${esc(p.next)}">${esc(p.next)} &rarr;</a>` : ''}</div>`;
});
