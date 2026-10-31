/* ── CiteGuard – Dashboard App ───────────────────────────────── */

// ── Document Store (session memory) ─────────────────────────────
const _docStore = []; // { job_id, filename, stem, total_citations, verified_count, overall_score }

// ── Dashboard stats ───────────────────────────────────────────────
function refreshDashboardStats() {
  const total   = _docStore.length;
  const claims  = _docStore.reduce((s, d) => s + d.total_citations, 0);
  const sup     = _docStore.reduce((s, d) => s + d.verified_count, 0);
  const unsup   = claims - sup;
  const el = id => document.getElementById(id);
  if (el("stat-docs"))        el("stat-docs").textContent        = total;
  if (el("stat-claims"))      el("stat-claims").textContent      = claims;
  if (el("stat-supported"))   el("stat-supported").textContent   = sup;
  if (el("stat-unsupported")) el("stat-unsupported").textContent = Math.max(0, unsup);
}

// ── My Documents table ────────────────────────────────────────────
function refreshDocTable() {
  const tbody   = document.getElementById("doc-table-body");
  const emptyRow= document.getElementById("doc-empty-row");
  if (!tbody) return;
  // Remove all non-empty rows
  tbody.querySelectorAll("tr.doc-data-row").forEach(r => r.remove());

  if (_docStore.length === 0) {
    if (emptyRow) emptyRow.style.display = "";
    return;
  }
  if (emptyRow) emptyRow.style.display = "none";

  _docStore.slice().reverse().forEach(doc => {
    const score = Math.round((doc.overall_score || 0) * 100);
    const tr = document.createElement("tr");
    tr.className = "doc-data-row";
    tr.innerHTML = `
      <td><div class="doc-name">${esc(doc.filename)}</div></td>
      <td>${doc.total_citations}</td>
      <td>${doc.verified_count}</td>
      <td>${score}%</td>
      <td><span class="badge-done">Complete</span></td>
      <td style="display:flex;gap:6px;flex-wrap:wrap">
        <button class="btn-sm" onclick="viewDoc('${doc.stem}')">View Results</button>
        <a class="btn-sm" href="/report/${doc.stem}" target="_blank">⬇ HTML Report</a>
        <a class="btn-sm" href="/download/${doc.stem}">⬇ JSON Report</a>
      </td>`;
    tbody.appendChild(tr);
  });
}

function viewDoc(stem) {
  // Find matching doc and re-load its result
  const doc = _docStore.find(d => d.stem === stem);
  if (doc && doc._result) {
    renderResults(doc._result);
  }
}

// Initialize stats to zero on load
document.addEventListener("DOMContentLoaded", () => {
  refreshDashboardStats();
  refreshDocTable();
});

const STAGES = ["Uploading PDF","Extracting text","Parsing citations",
  "Resolving metadata","Retrieving evidence","Verifying claims","Generating report","Done"];

const WHY = {
  SUPPORTS:    "The cited paper's content directly supports this claim. The evidence passage semantically aligns with what the author states.",
  NEUTRAL:     "The cited paper exists and is related, but its abstract doesn't directly prove this specific claim. The abstract may be too general or use different terminology.",
  CONTRADICTS: "The cited paper's content appears to oppose this claim. This often happens when a paper cites a prior work as a baseline it's surpassing.",
  NO_EVIDENCE: "This paper could not be found in Semantic Scholar, CrossRef, or arXiv. It may be a book chapter, very old paper, or an incorrectly cited reference.",
};

const VICON = { SUPPORTS:"✅", NEUTRAL:"⚠️", CONTRADICTS:"❌", NO_EVIDENCE:"🔍" };

// ── State ─────────────────────────────────────────────────────────
let selectedFile = null;
let pollInterval  = null;
let allRecords    = [];
let activeFilter  = "ALL";

// ── Page Navigation ───────────────────────────────────────────────
function navigate(page) {
  // Hide all pages
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  // Show target page
  const target = document.getElementById("page-" + page);
  if (target) target.classList.add("active");

  // Update sidebar active state
  document.querySelectorAll(".sb-item").forEach(item => {
    item.classList.toggle("active", item.dataset.page === page);
  });
}

// Sidebar click handlers
document.querySelectorAll(".sb-item").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    navigate(item.dataset.page);
  });
});

// Global navigate function is also available for onclick= attributes

// ── Stage pills ───────────────────────────────────────────────────
STAGES.slice(1).forEach((n, i) => {
  const p = document.createElement("span");
  p.className = "spill"; p.id = `sp-${i+1}`; p.textContent = n;
  document.getElementById("stage-pills").appendChild(p);
});

// ── Drag & Drop Upload ────────────────────────────────────────────
const dz   = document.getElementById("drop-zone");
const fi   = document.getElementById("file-input");
const dzf  = document.getElementById("dz-file");
const vbtn = document.getElementById("verify-btn");

if (dz) {
  dz.addEventListener("dragover",  e => { e.preventDefault(); dz.classList.add("drag-over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
  dz.addEventListener("drop",      e => { e.preventDefault(); dz.classList.remove("drag-over"); pick(e.dataTransfer.files[0]); });
}
if (fi)   fi.addEventListener("change", () => pick(fi.files[0]));
if (vbtn) vbtn.addEventListener("click", () => { if (selectedFile) startVerify(); });

function pick(f) {
  if (!f || !f.name.toLowerCase().endsWith(".pdf")) { alert("PDF files only."); return; }
  selectedFile = f;
  dzf.textContent = `📄 ${f.name}  (${(f.size/1024).toFixed(0)} KB)`;
  dzf.style.display = "block";
  vbtn.disabled = false;
}

// ── Upload & poll ─────────────────────────────────────────────────
async function startVerify() {
  navigate("progress");
  resetPills(); setProg(0);
  document.getElementById("prog-stage").textContent = "Uploading…";

  const form = new FormData();
  form.append("file", selectedFile);
  let jobId;
  try {
    const r = await fetch("/verify", { method:"POST", body:form });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Upload failed.");
    jobId = d.job_id;
  } catch(e) { showErr(e.message); return; }

  pollInterval = setInterval(() => poll(jobId), 1500);
}

async function poll(jobId) {
  try {
    const d = await fetch(`/status/${jobId}`).then(r => r.json());
    if (d.status === "running") {
      const i = d.stage_index || 1;
      setProg(Math.round(i / (STAGES.length - 1) * 100));
      document.getElementById("prog-stage").textContent = d.stage || "";
      updatePills(i);
    }
    if (d.status === "done") {
      clearInterval(pollInterval);
      setProg(100);
      document.getElementById("prog-stage").textContent = "Done ✓";
      updatePills(STAGES.length - 1);
      setTimeout(() => renderResults(d.result), 400);
    }
    if (d.status === "error") { clearInterval(pollInterval); showErr(d.error || "Unknown error."); }
  } catch { clearInterval(pollInterval); showErr("Lost connection to server."); }
}

function setProg(p)   { const b = document.getElementById("prog-bar"); if(b) b.style.width = `${p}%`; }
function resetPills() { document.querySelectorAll(".spill").forEach(p => p.classList.remove("active","done")); }
function updatePills(ai) {
  STAGES.slice(1).forEach((_, i) => {
    const p = document.getElementById(`sp-${i+1}`); if (!p) return;
    p.classList.toggle("done",   i+1 < ai);
    p.classList.toggle("active", i+1 === ai);
  });
}

// ── Render results ────────────────────────────────────────────────
function renderResults(result) {
  allRecords   = result.records || [];
  activeFilter = "ALL";

  // Summary numbers
  S("s-total").textContent = result.total_citations;
  S("s-sup").textContent   = result.verified_count;
  S("s-neu").textContent   = result.neutral_count;
  S("s-con").textContent   = result.contradicts_count;
  S("s-nf").textContent    = result.not_found_count;

  const pct = Math.round((result.overall_score || 0) * 100);
  S("sb-fill").style.width = `${pct}%`;
  S("sb-pct").textContent  = `${pct}%`;

  // Subtitle
  if (S("results-subtitle")) {
    S("results-subtitle").textContent =
      `${result.total_citations} citations verified in ${result.stem || "your paper"}.`;
  }

  // Header links
  if (S("hbtn-report"))   S("hbtn-report").href   = `/report/${result.stem}`;
  if (S("hbtn-download")) S("hbtn-download").href = `/download/${result.stem}`;

  // Register in doc store (only if not already there)
  if (result.stem && !_docStore.find(d => d.stem === result.stem)) {
    _docStore.push({
      job_id:          result.stem.split("_")[0],
      filename:        selectedFile ? selectedFile.name : result.stem,
      stem:            result.stem,
      total_citations: result.total_citations,
      verified_count:  result.verified_count,
      overall_score:   result.overall_score,
      _result:         result,           // keep full result for View button
    });
    refreshDashboardStats();
    refreshDocTable();
  }

  buildList(allRecords);
  navigate("analysis");

  // Reset filter pills
  document.querySelectorAll(".fpill").forEach(b => b.classList.toggle("active", b.dataset.f === "ALL"));

  if (allRecords.length > 0) selectRecord(0);
}

// ── Build citation list ───────────────────────────────────────────
function buildList(records) {
  const list = S("cite-list"); list.innerHTML = "";
  S("list-count").textContent = `${records.length} citations`;

  records.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "cite-row";
    row.dataset.idx     = i;
    row.dataset.verdict = r.verdict;

    const title = r.paper_title || "Unknown paper";
    row.innerHTML = `
      <div class="vdot vdot-${r.verdict}"></div>
      <div class="cr-body">
        <div class="cr-ref">${esc(r.raw_citation)} · ${esc(r.paper_year || "—")}</div>
        <div class="cr-title">${esc(title)}</div>
        <div class="cr-claim">${esc(r.claim)}</div>
      </div>
      <div class="cr-pct">${r.alignment_score || 0}%</div>`;

    row.addEventListener("click", () => selectRecord(i));
    list.appendChild(row);
  });
}

// ── Select record → fill detail panel ────────────────────────────
function selectRecord(idx) {
  const r = allRecords[idx];

  document.querySelectorAll(".cite-row").forEach((row, i) =>
    row.classList.toggle("active", i === idx));

  const icon   = VICON[r.verdict] || "";
  const vlabel = r.verdict.replace("_", " ");

  const titleHtml = r.doi
    ? `<a href="https://doi.org/${r.doi}" target="_blank" rel="noopener">${esc(r.paper_title || "Unknown paper")}</a>`
    : esc(r.paper_title || "Unknown paper");

  const authors = (r.paper_authors || []).slice(0, 4).join(", ")
    + ((r.paper_authors || []).length > 4 ? ` +${r.paper_authors.length - 4} more` : "");

  S("dp-ref").textContent = `Citation ${r.raw_citation}`;
  S("dp-verdict").innerHTML =
    `<span class="dp-verdict-badge vb-${r.verdict}">${icon} ${esc(vlabel)}</span>`;

  S("dp-title").innerHTML = titleHtml;
  S("dp-meta").innerHTML  =
    `Year: <strong>${r.paper_year || "—"}</strong>`
    + (r.source ? ` &nbsp;·&nbsp; Source: <strong>${esc(r.source)}</strong>` : "")
    + (r.doi ? ` &nbsp;·&nbsp; <a href="https://doi.org/${r.doi}" target="_blank">DOI ↗</a>` : "");
  S("dp-authors").textContent = authors ? `Authors: ${authors}` : "";
  S("dp-found").innerHTML = r.paper_found
    ? `<span style="color:#15803d;font-size:.8rem">✅ Found in database</span>`
    : `<span style="color:#b91c1c;font-size:.8rem">❌ Not found in any database</span>`
      + (r.warning ? ` <span style="color:#b91c1c;font-size:.76rem">· ${esc(r.warning)}</span>` : "");

  S("dp-claim").textContent = r.claim || "—";

  if (r.best_evidence) {
    S("dp-evidence-section").style.display = "block";
    S("dp-evidence").textContent = `"${r.best_evidence}"`;
  } else {
    S("dp-evidence-section").style.display = "none";
  }

  const ap = r.alignment_score || 0;
  S("dp-align-fill").style.width = `${ap}%`;
  S("dp-align-pct").textContent  = `${ap}%`;
  S("dp-conf").textContent       = `NLI confidence: ${r.confidence || 0}%`;

  S("dp-why").className = `dp-why2 ${r.verdict}`;
  S("dp-why").innerHTML = `<strong>Why ${esc(vlabel)}?</strong> ${WHY[r.verdict] || ""}`;

  S("detail-empty").style.display  = "none";
  S("detail-panel").style.display  = "flex";
}

// ── Filter pills ──────────────────────────────────────────────────
document.querySelectorAll(".fpill").forEach(btn => {
  btn.addEventListener("click", () => {
    activeFilter = btn.dataset.f;
    document.querySelectorAll(".fpill").forEach(b =>
      b.classList.toggle("active", b.dataset.f === activeFilter));
    applyFilter();
  });
});

// Summary card clicks
document.querySelectorAll(".rs-card.clickable").forEach(card => {
  card.addEventListener("click", () => {
    activeFilter = card.dataset.f;
    document.querySelectorAll(".fpill").forEach(b =>
      b.classList.toggle("active", b.dataset.f === activeFilter));
    document.querySelectorAll(".rs-card.clickable").forEach(c =>
      c.classList.toggle("active", c.dataset.f === activeFilter));
    applyFilter();
  });
});

// Search
const searchBox = document.getElementById("search-box");
if (searchBox) {
  searchBox.addEventListener("input", e => {
    applyFilter(e.target.value.toLowerCase());
  });
}

function applyFilter(q = "") {
  if (!q && searchBox) q = searchBox.value.toLowerCase();
  let visible = 0;
  document.querySelectorAll(".cite-row").forEach(row => {
    const i   = +row.dataset.idx;
    const r   = allRecords[i];
    const txt = ((r.paper_title || "") + (r.raw_citation || "") + (r.claim || "")).toLowerCase();
    const show = (activeFilter === "ALL" || row.dataset.verdict === activeFilter)
              && (!q || txt.includes(q));
    row.classList.toggle("hidden", !show);
    if (show) visible++;
  });
  if (S("list-count")) S("list-count").textContent = `${visible} of ${allRecords.length} citations`;
}

// ── Reset / New Analysis ──────────────────────────────────────────
document.querySelectorAll(".reset-btn").forEach(b => b.addEventListener("click", resetApp));

function resetApp() {
  selectedFile = null; allRecords = []; activeFilter = "ALL";
  if (pollInterval) clearInterval(pollInterval);
  if (fi)  { fi.value = ""; }
  if (dzf) { dzf.style.display = "none"; }
  if (vbtn){ vbtn.disabled = true; }
  if (searchBox) searchBox.value = "";
  document.querySelectorAll(".fpill").forEach(b =>
    b.classList.toggle("active", b.dataset.f === "ALL"));
  resetPills(); setProg(0);
  navigate("upload");
}

// ── Error ─────────────────────────────────────────────────────────
function showErr(msg) {
  const el = document.getElementById("error-msg");
  if (el) el.textContent = msg;
  navigate("error");
}

// ── Util ──────────────────────────────────────────────────────────
const S = id => document.getElementById(id);

function esc(s) {
  if (!s) return "";
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
