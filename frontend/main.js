(() => {
  const API_BASE = `http://localhost:8000`; // assumes served from same host; adjust if needed

  // --- Simple router
  const pages = {
    home: document.getElementById('page-home'),
    jobs: document.getElementById('page-jobs'),
    candidates: document.getElementById('page-candidates'),
    search: document.getElementById('page-search'),
  };
  const navBtns = Array.from(document.querySelectorAll('#top-nav .nav-btn'));
  function showPage(name) {
    Object.values(pages).forEach(p => p.classList.remove('visible'));
    pages[name]?.classList.add('visible');
    navBtns.forEach(b => b.classList.toggle('active', b.dataset.page === name));
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  navBtns.forEach(btn => btn.addEventListener('click', () => showPage(btn.dataset.page)));

  // --- Generic list renderer
  function renderList(container, items, buildFn) {
    container.innerHTML = '';
    items.forEach(it => container.appendChild(buildFn(it)));
  }

  // --- Modal helpers
  const backdrop = document.getElementById('modal-backdrop');
  const addJobModal = document.getElementById('modal-add-job');
  const addCandidateModal = document.getElementById('modal-add-candidate');
  function openModal(modal) {
    backdrop.classList.remove('hidden');
    modal.classList.remove('hidden');
  }
  function closeModal(modal) {
    modal.classList.add('hidden');
    backdrop.classList.add('hidden');
  }
  document.querySelectorAll('[data-close]').forEach(el => {
    el.addEventListener('click', () => closeModal(el.closest('.modal')));
  });
  backdrop.addEventListener('click', () => {
    [addJobModal, addCandidateModal].forEach(m => m.classList.add('hidden'));
    backdrop.classList.add('hidden');
  });

  // --- Jobs page
  const jobsListEl = document.getElementById('jobs-list');
  const btnAddJob = document.getElementById('btn-add-job');
  const formAddJob = document.getElementById('form-add-job');

  async function fetchJobs() {
    const res = await fetch(`${API_BASE}/jobs`);
    const data = await res.json();
    return data.jobs || [];
  }
  
  function buildJobCard(job) {
    return makeCard({
      id: job.id,
      onInit: (_id, root) => {
        if(!root) {console.log("buildJob root not found; skipping."); return }; 
        const header = root.querySelector('.card-header'); 
        const body = root.querySelector('.card-body'); 
        const footer = root.querySelector('.card-footer');
        if(!header || !body || !footer) return;
        header.innerHTML = `
          <div>
            <div class="card-title">${escapeHtml(job.job_title)}</div>
            <div class="card-subtitle">${escapeHtml(job.company_name || '')}</div>
          </div>
          <button class="link" aria-expanded="false">Details</button>`;
        body.innerHTML = `
          <div class="field">
            <label>Job Title
              <input data-f-job-title type="text" value="${escapeHtml(job.job_title)}" disabled />
            </label>
          </div>
            <div class="field">
            <label>Company Name
              <input data-f-company-name type="text" value="${escapeHtml(job.company_name || '')}" disabled />
            </label>
          </div>
          <div class="field">
            <label>Descriptionaccording to my google chrome.
              <textarea data-f-job-desc rows="6" disabled>${escapeHtml(job.job_description || '')}</textarea>
            </label>
          </div>`;
        footer.innerHTML = `
          <button class="secondary" data-act-edit>Edit</button>
          <button class="primary" data-act-search>Search</button>
          <button class="danger" data-act-delete>Delete</button>`;
        const btnEdit = footer.querySelector('[data-act-edit]');
        const btnSearch = footer.querySelector('[data-act-search]');
        const btnDelete = footer.querySelector('[data-act-delete]');
        btnEdit.addEventListener('click', async () => {
          const titleInput = body.querySelector('[data-f-job-title]');
          const companyInput = body.querySelector('[data-f-company-name]');
          const desc = body.querySelector('[data-f-job-desc]');
          const entering = titleInput.disabled;
          [titleInput, companyInput, desc].forEach(i => i.disabled = !entering ? true : false);
          btnEdit.textContent = entering ? 'Save' : 'Edit';
          if (!entering) {
            try {
              const res = await fetch(`${API_BASE}/jobs/${job.id}`, { method:'PATCH', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ job_title: titleInput.value, company_name: companyInput.value, job_description: desc.value }) });
              if(!res.ok) throw new Error();
              job.job_title = titleInput.value; job.company_name = companyInput.value; job.job_description = desc.value;
              header.querySelector('.card-title').textContent = job.job_title;
              header.querySelector('.card-subtitle').textContent = job.company_name || '';
            } catch { alert('Failed to save job'); }
          }
        });
        btnSearch.addEventListener('click', async () => {
          await ensureSearchJobs(); searchSelect.value = job.id; updateSearchInfo(job); showPage('search');
        });
        btnDelete.addEventListener('click', async () => {
          if(!confirm('Delete this job?')) return;
          try {
            const res = await fetch(`${API_BASE}/jobs/${job.id}`, { method:'DELETE' });
            if(!res.ok) throw new Error();
            await loadJobs();
          } catch { alert('Failed to delete job'); }
        });
      },
      onExpand: () => {},
      onCollapse: () => {},
    });
  }
  async function loadJobs() {
    const jobs = await fetchJobs();
    renderList(jobsListEl, jobs, buildJobCard);
    await ensureSearchJobs();
  }
  btnAddJob.addEventListener('click', () => openModal(addJobModal));
  formAddJob.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(formAddJob);
    const payload = Object.fromEntries(fd.entries());
    const res = await fetch(`${API_BASE}/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) return alert('Failed to add job');
    const { job_id } = await res.json();
    closeModal(addJobModal);
    formAddJob.reset();
    await loadJobs();
    // If opened from Search page button, preselect
    if (pages.search.classList.contains('visible')) {
      await ensureSearchJobs();
      searchSelect.value = job_id;
      const job = (await fetchJobs()).find(j => j.id === job_id);
      updateSearchInfo(job);
    }
  });

  // --- Candidates page
  const candidatesListEl = document.getElementById('candidates-list');
  const btnAddCandidate = document.getElementById('btn-add-candidate');
  const formAddCandidate = document.getElementById('form-add-candidate');

  async function fetchCandidates() {
    const res = await fetch(`${API_BASE}/candidates`);
    const data = await res.json();
    return data.candidates || [];
  }
  async function fetchDocument(id) {
    const res = await fetch(`${API_BASE}/documents/${id}`);
    if (!res.ok) return null;
    return await res.json();
  }
  function buildCandidateCard(c) {
    return makeCard({
      id: c.id,
      onInit: (_id, root) => {
        if(!root) return;
        const header = root.querySelector('.card-header');
        const body = root.querySelector('.card-body');
        const footer = root.querySelector('.card-footer');
        if(!header || !body || !footer) return;
        header.innerHTML = `
          <div class="card-title">${escapeHtml(c.candidate_name || '(Unnamed)')}</div>
          <button class="link" aria-expanded="false">Details</button>`;
        body.innerHTML = `
          <div class="field">
            <label>Name
              <input data-f-cand-name type="text" value="${escapeHtml(c.candidate_name || '')}" disabled />
            </label>
          </div>
          <div data-nested-cv></div>
          `;
        footer.innerHTML = `<button class="secondary" data-act-edit>Edit</button><button class="danger" data-act-delete>Delete</button>`;
        const btnEdit = footer.querySelector('[data-act-edit]');
        const btnDelete = footer.querySelector('[data-act-delete]');
        const nameInput = body.querySelector('[data-f-cand-name]');
        btnEdit.addEventListener('click', async () => {
          const entering = nameInput.disabled;
          nameInput.disabled = !entering ? true : false;
          btnEdit.textContent = entering ? 'Save' : 'Edit';
          if (!entering) {
            try { 
              const r = await fetch(`${API_BASE}/candidates/${c.id}`, { method:'PATCH', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ candidate_name: nameInput.value })});
              if(!r.ok) throw new Error();
              c.candidate_name = nameInput.value; header.querySelector('.card-title').textContent = c.candidate_name || '(Unnamed)'; 
            }
            catch { alert('Failed to save candidate'); }
          }
        });
        btnDelete.addEventListener('click', async () => {
          if(!confirm('Delete this candidate?')) return;
          try {
            const res = await fetch(`${API_BASE}/candidates/${c.id}`, { method:'DELETE' });
            if(!res.ok) throw new Error();
            await loadCandidates();
          } catch { alert('Failed to delete candidate'); }
        });
        // Nested CV card
        const nestedMount = body.querySelector('[data-nested-cv]');
        const cvCardId = `${c.id}-cv`;
        const cvCard = makeCard({
          id: cvCardId,
          onInit: (_nid, root2) => {
            if(!root2) return;
            const h2 = root2.querySelector('.card-header');
            const b2 = root2.querySelector('.card-body');
            const f2 = root2.querySelector('.card-footer');
            if(h2) h2.innerHTML = `<div class="card-title">CV</div><button class="link" aria-expanded="false">Details</button>`;
            if(b2) b2.innerHTML = `<pre data-f-cv class="muted" style="white-space:pre-wrap; margin:0;"></pre>`;
            if(f2) f2.innerHTML = `
              <label class="cv-upload-label">
                <input data-f-cv-file type="file" accept=".pdf,.doc,.docx,.txt,.md"/>
                <span>Click to browse or drop files here to upload</span>
              </label>
            `;
            const label = root2.querySelector('.cv-upload-label');
            const input = root2.querySelector('[data-f-cv-file]');
            ['dragover','dragenter'].forEach(evt => label.addEventListener(evt, e => { e.preventDefault(); label.classList.add('dragover'); }));
            ['dragleave','dragend','drop'].forEach(evt => label.addEventListener(evt, e => { e.preventDefault(); label.classList.remove('dragover'); }));
            label.addEventListener('drop', e => {
              if (e.dataTransfer.files && e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                input.dispatchEvent(new Event('change', { bubbles: true }));
              }
            });
          },
          onExpand: async () => {
            const pre = nestedMount.querySelector('[data-f-cv]');
            if(!pre || pre.textContent) return;
            if (c.candidate_cv_id) {
              try { const res = await fetch(`${API_BASE}/documents/${c.candidate_cv_id}`); if(res.ok){ const doc = await res.json(); pre.textContent = doc ? doc.content : '(No CV)'; } else pre.textContent='(No CV)'; }
              catch { pre.textContent='(No CV)'; }
            } else {
              pre.textContent='(No CV)';
            }
          },
          onCollapse: () => {}
        });
        nestedMount.appendChild(cvCard);

        const fileInput = body.querySelector('[data-f-cv-file]');
        const pre = () => nestedMount.querySelector('[data-f-cv]');
        fileInput.addEventListener('change', async () => {
          if(!fileInput.files?.length) return;
          const file = fileInput.files[0];
          const fd = new FormData(); fd.append('file', file);
          try {
            const upRes = await fetch(`${API_BASE}/documents`, { method:'POST', body: fd });
            if(!upRes.ok) throw new Error();
            const { document_id } = await upRes.json();
            const patchRes = await fetch(`${API_BASE}/candidates/${c.id}`, { method:'PATCH', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ candidate_cv_id: document_id })});
            if(!patchRes.ok) throw new Error();
            c.candidate_cv_id = document_id;
            const p = pre();
            if(p) {
              try { const docRes = await fetch(`${API_BASE}/documents/${document_id}`); if(docRes.ok){ const doc = await docRes.json(); p.textContent = doc ? doc.content : ''; } }
              catch { /* ignore */ }
            }
          } catch { alert('Failed to upload/link CV'); }
        });
  },
  onExpand: async (_id, root) => {
        if (c.candidate_cv_id) {
          const nested = root?.querySelector(`[data-card-id="${c.id}-cv"]`) || root;
          const preEl = nested?.querySelector('[data-f-cv]');
          if(preEl && !preEl.textContent) {
            try { const r = await fetch(`${API_BASE}/documents/${c.candidate_cv_id}`); if(r.ok){ const doc = await r.json(); preEl.textContent = doc ? doc.content : '(No CV)'; } else preEl.textContent='(No CV)'; }
            catch { preEl.textContent='(No CV)'; }
          }
        }
      },
      onCollapse: () => {},
    });
  }
  async function loadCandidates() {
    const candidates = await fetchCandidates();
    renderList(candidatesListEl, candidates, buildCandidateCard);
  }
  btnAddCandidate.addEventListener('click', () => openModal(addCandidateModal));
  formAddCandidate.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(formAddCandidate);
    const name = fd.get('candidate_name');
    const file = fd.get('cv_file');
    if (!file || !(file instanceof File)) return alert('Please choose a CV file');
    // 1) upload doc
    const up = new FormData();
    up.append('file', file);
    up.append('document_name', file.name);
    const r1 = await fetch(`${API_BASE}/documents`, { method: 'POST', body: up });
    if (!r1.ok) return alert('Failed to upload document');
    const { document_id } = await r1.json();
    // 2) create candidate with candidate_cv_id
    const r2 = await fetch(`${API_BASE}/candidates`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_name: name, candidate_cv_id: document_id }),
    });
    if (!r2.ok) return alert('Failed to create candidate');
    closeModal(addCandidateModal);
    formAddCandidate.reset();
    await loadCandidates();
  });

  // --- Search page
  const searchSelect = document.getElementById('search-job-select');
  const btnSearchAddJob = document.getElementById('btn-search-add-job');
  const searchInfo = document.getElementById('search-info');
  async function ensureSearchJobs() {
    const jobs = await fetchJobs();
    const curr = searchSelect.value;
    searchSelect.innerHTML = '';
    jobs.forEach(j => {
      const opt = document.createElement('option');
      opt.value = j.id; opt.textContent = `${j.job_title}${j.company_name ? ' — ' + j.company_name : ''}`;
      searchSelect.appendChild(opt);
    });
    if (jobs.length && !searchSelect.value) searchSelect.value = jobs[0].id;
    if (curr && [...searchSelect.options].some(o => o.value === curr)) searchSelect.value = curr;
    const sel = jobs.find(j => j.id === searchSelect.value);
    updateSearchInfo(sel);
  }
  function updateSearchInfo(job) {
    if (!job) { searchInfo.textContent = 'No job selected.'; return; }
    searchInfo.textContent = `Selected: ${job.job_title}${job.company_name ? ' @ ' + job.company_name : ''}`;
  }
  searchSelect.addEventListener('change', async () => {
    const jobs = await fetchJobs();
    const job = jobs.find(j => j.id === searchSelect.value);
    updateSearchInfo(job);
  });
  btnSearchAddJob.addEventListener('click', () => {
    openModal(addJobModal);
  });

  // --- Utils
  function escapeHtml(s) {
    return String(s || '').replace(/[&<>"]+/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  }

  // --- Init
  (async function init() {
    await loadJobs();
    await loadCandidates();
    await ensureSearchJobs();
    showPage('home');
  })();
})();
