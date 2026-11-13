// Browse: search and filter the plugins (name, description, folder, menu entries, classes/functions, dependencies).
import { api, app, esc, route } from './core.js';
import { groupBadge } from './ui.js';

route('plugins', async (_args, query) => {
  const cat = await api('/catalogue');
  const topics = Object.fromEntries(cat.topics.map(t => [t.id, t.title]));
  app().innerHTML = `<h1>Plugins</h1>
    <div class="filters">
      <div><label for="q">Search</label><input id="q" type="search" placeholder="name, menu entry, function, dependency..." value="${esc(query.q || '')}"></div>
      <div><label for="grp">Status</label><select id="grp"><option value="">all</option>${cat.groups.map(g => `<option value="${g.id}">${esc(g.title)}</option>`).join('')}</select></div>
      <div><label for="top">Topic</label><select id="top"><option value="">all</option>${cat.topics.map(t => `<option value="${t.id}">${esc(t.title)}</option>`).join('')}</select></div>
    </div>
    <p class="small mute" id="count"></p><ul class="slist" id="list"></ul>`;
  const $ = s => app().querySelector(s);
  $('#grp').value = query.group || ''; $('#top').value = query.topic || '';
  const hay = p => [p.name, p.folder, p.path, p.description, p.author, ...p.menu, ...p.names, ...p.dependencies].join(' ').toLowerCase();
  const draw = () => {
    const q = $('#q').value.trim().toLowerCase().split(/\s+/).filter(Boolean), g = $('#grp').value, t = $('#top').value;
    const hits = cat.plugins.filter(p => (!g || p.group === g) && (!t || p.topic === t) && q.every(w => hay(p).includes(w)));
    $('#count').textContent = `${hits.length} of ${cat.plugins.length} plugins`;
    $('#list').innerHTML = hits.map(p => `<li><a class="nm" href="#/plugin/${esc(p.id)}">${esc(p.name)}</a> ${groupBadge(p.group)}
      <span class="sum">${esc(p.description || 'No description in info.xml or README.')}</span>
      <span class="meta">${esc(p.path)} · ${esc(topics[p.topic])} · ${p.lines.toLocaleString()} lines${p.version ? ' · v' + esc(p.version) : ''}${p.menu.length ? ' · menu: ' + esc(p.menu.join(', ')) : ''}</span></li>`).join('');
  };
  ['#q', '#grp', '#top'].forEach(s => $(s).addEventListener('input', draw));
  draw();
});
