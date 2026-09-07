/* ============================================================================
   Hybrid RAG — client
   No framework, no build step. One file, one job: consume the SSE stream and
   render it as it arrives.
   ========================================================================== */

const $ = (id) => document.getElementById(id);

const el = {
  app:        document.querySelector('.app'),
  sidebar:    $('sidebar'),
  messages:   $('messages'),
  input:      $('input'),
  send:       $('send'),
  status:     $('status'),
  statusText: $('status-text'),
  debug:      $('debug-toggle'),
  corpus:     $('corpus-list'),
  docCount:   $('doc-count'),
  stats:      $('index-stats'),
  fileInput:  $('file-input'),
  uploadLabel:$('upload-label'),
  uploadText: $('upload-text'),
  uploaded:   $('uploaded-list'),
  sourcesBody:$('sources-body'),
  eyebrow:    $('eyebrow'),
  closeSources: $('close-sources'),
  menuBtn:    $('menu-btn'),
};

const state = {
  history: [],        // [{role, content}] sent back for follow-up condensing
  sessionId: null,    // set on first upload
  streaming: false,
  lastSources: [],
};

/* ── Boot ─────────────────────────────────────────────────────────────── */

async function boot() {
  try {
    const health = await (await fetch('/api/health')).json();
    const ok = health.index_ready;
    el.status.className = 'status ' + (ok ? 'ok' : 'bad');
    el.statusText.textContent = ok
      ? `${health.stats.chunks} chunks · ${health.llm_model}`
      : 'index unavailable';

    // Drive the headline counts from the live index rather than hardcoding
    // them in the markup, where they would silently go stale on re-ingest.
    if (el.eyebrow && health.stats && health.stats.documents) {
      el.eyebrow.textContent =
        `${health.stats.documents} papers · ` +
        `${health.stats.chunks.toLocaleString()} passages · exact search`;
    }

    if (health.stats && health.stats.chunks) {
      el.stats.innerHTML = [
        `${health.stats.documents} docs · ${health.stats.chunks} chunks`,
        `${health.stats.vector_backend} · ${health.stats.vector_dimension}d`,
        `bm25 vocab ${health.stats.bm25_vocabulary}`,
      ].join('<br>');
    }
  } catch {
    el.status.className = 'status bad';
    el.statusText.textContent = 'offline';
  }

  try {
    const { documents, count } = await (await fetch('/api/documents')).json();
    el.docCount.textContent = `(${count})`;
    el.corpus.innerHTML = documents
      .map((d) => `<li title="${escapeAttr(d.title)}">${escapeHtml(truncate(d.title, 52))}</li>`)
      .join('');
  } catch { /* corpus list is decorative; ignore */ }
}

/* ── Sending ──────────────────────────────────────────────────────────── */

async function ask(question) {
  if (state.streaming || !question.trim()) return;
  state.streaming = true;
  el.send.disabled = true;

  document.querySelector('.welcome')?.remove();
  addMessage('user', escapeHtml(question));

  const bot = addMessage('bot', '');
  const bubble = bot.querySelector('.bubble');
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  bubble.appendChild(cursor);

  let raw = '';
  let sources = [];
  let done = null;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        history: state.history.slice(-8),
        session_id: state.sessionId,
      }),
    });
    if (!resp.ok) throw new Error(`server returned ${resp.status}`);

    for await (const { event, data } of readSSE(resp)) {
      if (event === 'sources') {
        sources = data;
        state.lastSources = data;
        renderSources(data);
      } else if (event === 'token') {
        raw += data;
        // Re-render the whole answer each tick. At the token rate of a
        // streamed LLM this is far cheaper than it sounds, and it keeps
        // markdown and citation chips correct mid-stream rather than
        // flickering as a partial marker like "[1" is completed.
        bubble.innerHTML = renderMarkdown(raw);
        bubble.appendChild(cursor);
        scrollToBottom();
      } else if (event === 'done') {
        done = data;
      } else if (event === 'error') {
        bubble.innerHTML += `<div class="notice">${escapeHtml(String(data))}</div>`;
      }
    }
  } catch (err) {
    bubble.innerHTML += `<div class="notice">Connection failed: ${escapeHtml(err.message)}</div>`;
  }

  cursor.remove();

  if (done) {
    // The server's validated answer replaces the streamed text: invalid
    // citation markers have been stripped and the rest renumbered.
    bubble.innerHTML = renderMarkdown(done.answer || raw);
    wireCitations(bubble);
    if (el.debug.checked) bubble.appendChild(buildTrace(done, sources));
    state.history.push({ role: 'user', content: question });
    state.history.push({ role: 'assistant', content: done.answer || raw });
  } else if (raw) {
    bubble.innerHTML = renderMarkdown(raw);
    wireCitations(bubble);
  }

  state.streaming = false;
  el.send.disabled = false;
  el.input.focus();
  scrollToBottom();
}

/* ── SSE reader ───────────────────────────────────────────────────────── */

async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line. Anything after the last blank
    // line is a partial frame and stays in the buffer.
    const frames = buffer.split('\n\n');
    buffer = frames.pop();

    for (const frame of frames) {
      let event = 'message';
      const dataLines = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7).trim();
        else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
      }
      if (!dataLines.length) continue;
      try {
        yield { event, data: JSON.parse(dataLines.join('\n')) };
      } catch { /* skip malformed frame */ }
    }
  }
}

/* ── Rendering ────────────────────────────────────────────────────────── */

function addMessage(role, html) {
  const node = document.createElement('div');
  node.className = `msg ${role}`;
  node.innerHTML =
    `<div class="msg-role">${role === 'user' ? 'You' : 'Assistant'}</div>` +
    `<div class="bubble">${html}</div>`;
  el.messages.appendChild(node);
  scrollToBottom();
  return node;
}

/* A deliberately small markdown subset: bold, italic, inline code, lists and
   paragraphs. A full markdown library would be ~40KB for features the model
   is instructed not to use, and every HTML-producing dependency is an XSS
   surface. Input is escaped first, so nothing here can inject markup. */
function renderMarkdown(text) {
  let out = escapeHtml(text);

  out = out
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/\[(\d{1,2})\]/g, '<button class="cite" data-n="$1">$1</button>');

  const blocks = out.split(/\n{2,}/).map((block) => {
    const lines = block.split('\n');
    if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
      return '<ul>' + lines.map((l) => `<li>${l.replace(/^\s*[-*]\s+/, '')}</li>`).join('') + '</ul>';
    }
    if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
      return '<ol>' + lines.map((l) => `<li>${l.replace(/^\s*\d+[.)]\s+/, '')}</li>`).join('') + '</ol>';
    }
    return `<p>${lines.join('<br>')}</p>`;
  });

  return blocks.join('');
}

function wireCitations(bubble) {
  bubble.querySelectorAll('.cite').forEach((chip) => {
    chip.addEventListener('click', () => {
      el.app.classList.add('sources-open');
      const card = $(`src-${chip.dataset.n}`);
      if (card) {
        card.scrollIntoView({ block: 'center', behavior: 'smooth' });
        card.classList.add('flash');
        setTimeout(() => card.classList.remove('flash'), 1400);
      }
    });
  });
}

function renderSources(sources) {
  if (!sources.length) {
    el.sourcesBody.innerHTML = '<p class="empty">Nothing cleared the relevance threshold.</p>';
    return;
  }
  el.sourcesBody.innerHTML = sources.map((s) => {
    const meta = [s.section, s.page ? `p. ${s.page}` : '', s.source]
      .filter(Boolean).map(escapeHtml).join(' · ');
    const arxiv = /^(\d{4}\.\d{4,5})/.exec(s.source || '');
    const link = arxiv
      ? `<a class="sc-link" href="https://arxiv.org/abs/${arxiv[1]}" target="_blank" rel="noopener">arXiv ↗</a>`
      : '';
    const score = s.rerank_score != null
      ? `<span class="sc-meta" style="margin:0">${s.rerank_score.toFixed(2)}</span>` : '';
    return `
      <div class="source-card" id="src-${s.n}">
        <div class="sc-head">
          <span class="sc-n">${s.n}</span>
          <span class="sc-title">${escapeHtml(truncate(s.title, 62))}</span>
        </div>
        <div class="sc-meta">${meta}</div>
        <div class="sc-snippet">${escapeHtml(s.snippet)}…</div>
        <div class="sc-foot">
          <span class="tag ${s.retriever}">${s.retriever}</span>${score}${link}
        </div>
      </div>`;
  }).join('');
  el.app.classList.add('sources-open');
}

function buildTrace(done, sources) {
  const t = done.timings_ms || {};
  const details = document.createElement('details');
  details.className = 'trace';
  const rewritten = done.search_query && done.search_query !== ''
    ? `<div class="timings">search query: ${escapeHtml(done.search_query)}</div>` : '';
  details.innerHTML = `
    <summary>retrieval trace · ${sources.length} passages · ${(t.total_ms || 0).toFixed(0)}ms</summary>
    <div class="trace-body">
      ${rewritten}
      ${sources.map((s) => `
        <div class="trace-row">
          <span class="t-n">${s.n}</span>
          <span class="tag ${s.retriever}">${s.retriever}</span>
          <span class="t-title">${escapeHtml(s.title)}</span>
          <span class="t-score">${s.rerank_score != null ? s.rerank_score.toFixed(2) : '–'}</span>
        </div>`).join('')}
      <div class="timings">${Object.entries(t).map(([k, v]) => `${k.replace('_ms', '')}=${v}ms`).join('  ')}</div>
    </div>`;
  return details;
}

/* ── Upload ───────────────────────────────────────────────────────────── */

el.fileInput.addEventListener('change', async () => {
  const file = el.fileInput.files[0];
  if (!file) return;

  el.uploadLabel.className = 'upload busy';
  el.uploadText.textContent = `Indexing ${truncate(file.name, 24)}…`;

  const form = new FormData();
  form.append('file', file);
  if (state.sessionId) form.append('session_id', state.sessionId);

  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'upload failed');

    state.sessionId = data.session_id;
    el.uploaded.innerHTML = data.documents
      .map((d) => `<li>${escapeHtml(truncate(d, 26))}</li>`).join('');
    const plural = data.chunks === 1 ? 'chunk' : 'chunks';
    el.uploaded.lastElementChild.innerHTML += `<span>${data.chunks} ${plural}</span>`;
    el.uploadLabel.className = 'upload';
    el.uploadText.textContent = 'Upload another document';
  } catch (err) {
    el.uploadLabel.className = 'upload error';
    el.uploadText.textContent = err.message;
    setTimeout(() => {
      el.uploadLabel.className = 'upload';
      el.uploadText.textContent = 'Upload PDF, MD, TXT or DOCX';
    }, 4000);
  }
  el.fileInput.value = '';
});

/* ── Input handling ───────────────────────────────────────────────────── */

function submit() {
  const q = el.input.value.trim();
  if (!q) return;
  el.input.value = '';
  autosize();
  ask(q);
}

function autosize() {
  el.input.style.height = 'auto';
  el.input.style.height = Math.min(el.input.scrollHeight, 168) + 'px';
}

el.send.addEventListener('click', submit);
el.input.addEventListener('input', autosize);
el.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});

document.querySelectorAll('#suggestions button').forEach((b) =>
  b.addEventListener('click', () => ask(b.textContent))
);

el.closeSources.addEventListener('click', () => el.app.classList.remove('sources-open'));
el.menuBtn?.addEventListener('click', () => el.sidebar.classList.toggle('open'));
el.debug.addEventListener('change', () => {
  localStorage.setItem('rag-debug', el.debug.checked ? '1' : '0');
});

try {
  el.debug.checked = localStorage.getItem('rag-debug') === '1';
} catch { /* storage can throw in private mode; the default is fine */ }

/* ── Helpers ──────────────────────────────────────────────────────────── */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const escapeAttr = escapeHtml;
const truncate = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + '…' : s || '');
const scrollToBottom = () => { el.messages.scrollTop = el.messages.scrollHeight; };

boot();
