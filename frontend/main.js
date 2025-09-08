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
  function jobCard(job) {
    const el = document.createElement('div');
    el.className = 'card';
    el.innerHTML = `
      <div class="card-header">
        <div>
          <div class="card-title">${escapeHtml(job.job_title)}</div>
          <div class="card-subtitle">${escapeHtml(job.company_name || '')}</div>
        </div>
        <button class="link" aria-expanded="false">Details</button>
      </div>
      <div class="card-body">
        <div class="field">
          <label>Job Title
            <input type="text" value="${escapeHtml(job.job_title)}" disabled />
          </label>
        </div>
        <div class="field">
          <label>Company Name
            <input type="text" value="${escapeHtml(job.company_name || '')}" disabled />
          </label>
        </div>
        <div class="field">
          <label>Description
            <textarea rows="6" disabled>${escapeHtml(job.job_description || '')}</textarea>
          </label>
        </div>
        <div class="card-footer">
          <button class="secondary btn-edit">Edit</button>
          <button class="primary btn-go-search">Search</button>
        </div>
      </div>`;

    const headerBtn = el.querySelector('.card-header .link');
    const body = el.querySelector('.card-body');
    headerBtn.addEventListener('click', () => {
      const expanded = el.classList.toggle('expanded');
      headerBtn.setAttribute('aria-expanded', String(expanded));
    });

    // Edit mode toggle
    const btnEdit = el.querySelector('.btn-edit');
    btnEdit.addEventListener('click', () => {
      const inputs = body.querySelectorAll('input, textarea');
      const currentlyDisabled = inputs[0]?.disabled ?? true;
      inputs.forEach(i => (i.disabled = !currentlyDisabled));
      btnEdit.textContent = currentlyDisabled ? 'Save' : 'Edit';
      if (!currentlyDisabled) {
        // TODO: call API to persist edit when implemented
      }
    });

    // Go to Search and pre-select this job
    el.querySelector('.btn-go-search').addEventListener('click', async () => {
      await ensureSearchJobs();
      searchSelect.value = job.id;
      updateSearchInfo(job);
      showPage('search');
    });
    return el;
  }
  async function loadJobs() {
    jobsListEl.innerHTML = '';
    const jobs = await fetchJobs();
    jobs.forEach(j => jobsListEl.appendChild(jobCard(j)));
    // also refresh search dropdown
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
  function candidateCard(c) {
    const el = document.createElement('div');
    el.className = 'card';
    el.innerHTML = `
      <div class="card-header">
        <div class="card-title">${escapeHtml(c.candidate_name || '(Unnamed)')}</div>
        <button class="link" aria-expanded="false">Details</button>
      </div>
      <div class="card-body">
        <div class="field">
          <label>Name
            <input type="text" value="${escapeHtml(c.candidate_name || '')}" disabled />
          </label>
        </div>
        <div class="field">
          <label>CV</label>
          <pre class="muted" style="white-space:pre-wrap"></pre>
        </div>
        <div class="card-footer">
          <button class="secondary btn-edit">Edit</button>
          <label class="secondary" style="display:inline-flex; align-items:center; gap:8px;">
            <input type="file" accept=".pdf,.doc,.docx,.txt,.md" style="display:none;" />
            <span>Upload new CV</span>
          </label>
        </div>
      </div>`;

    const headerBtn = el.querySelector('.card-header .link');
    const body = el.querySelector('.card-body');
    const pre = body.querySelector('pre');
    headerBtn.addEventListener('click', async () => {
      const expanded = el.classList.toggle('expanded');
      headerBtn.setAttribute('aria-expanded', String(expanded));
      if (expanded && c.candidate_cv_id) {
        const doc = await fetchDocument(c.candidate_cv_id);
        pre.textContent = doc ? doc.contents : '(No CV)';
      }
    });

    // Edit toggle
    const btnEdit = el.querySelector('.btn-edit');
    const nameInput = body.querySelector('input[type="text"]');
    btnEdit.addEventListener('click', () => {
      const editable = nameInput.disabled;
      nameInput.disabled = !editable;
      btnEdit.textContent = editable ? 'Save' : 'Edit';
      if (!editable) {
        // TODO: persist edit when API exists
      }
    });

    // Upload new CV
    const fileInput = body.querySelector('input[type="file"]');
    fileInput.addEventListener('change', async () => {
      if (!fileInput.files?.length) return;
      const file = fileInput.files[0];
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_BASE}/documents`, { method: 'POST', body: fd });
      if (!res.ok) return alert('Failed to upload CV');
      const { document_id } = await res.json();
      // TODO: call API to update candidate's candidate_cv_id once endpoint exists
      c.candidate_cv_id = document_id;
      const doc = await fetchDocument(document_id);
      pre.textContent = doc ? doc.contents : '';
    });

    return el;
  }
  async function loadCandidates() {
    candidatesListEl.innerHTML = '';
    const candidates = await fetchCandidates();
    candidates.forEach(c => candidatesListEl.appendChild(candidateCard(c)));
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
