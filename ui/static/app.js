/* ═══════════════════════════════════════════════════════════
   CosmicDIP  |  app.js  — UI logic + API calls
   ═══════════════════════════════════════════════════════════ */

// ── State ──────────────────────────────────────────────────
let currentTab      = "stage8";
let currentImagePath = null;   // null = use server default

// ── Starfield ──────────────────────────────────────────────
(function initStarfield() {
  const canvas = document.getElementById("starfield");
  const ctx    = canvas.getContext("2d");
  let stars    = [];

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    stars = Array.from({ length: 260 }, () => ({
      x:    Math.random() * canvas.width,
      y:    Math.random() * canvas.height,
      r:    Math.random() * 1.4 + 0.2,
      a:    Math.random() * 0.7 + 0.15,
      da:   (Math.random() - 0.5) * 0.003,
      dx:   (Math.random() - 0.5) * 0.04,
      dy:   (Math.random() - 0.5) * 0.04,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const s of stars) {
      s.a  = Math.max(0.05, Math.min(0.9, s.a + s.da));
      s.x  = (s.x + s.dx + canvas.width)  % canvas.width;
      s.y  = (s.y + s.dy + canvas.height) % canvas.height;
      if (Math.random() < 0.001) s.da = -s.da;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(160,200,255,${s.a.toFixed(2)})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  draw();
})();

// ── Tab switching ────────────────────────────────────────────
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-page").forEach(p => p.classList.remove("active"));
  document.getElementById("tab-"  + tab).classList.add("active");
  document.getElementById("page-" + tab).classList.add("active");
}

// ── Threshold param visibility ───────────────────────────────
function toggleThreshParams() {
  const method = document.getElementById("thresh-method").value;
  document.getElementById("params-global").classList.toggle("hidden", method !== "global");
  document.getElementById("params-adaptive").classList.toggle("hidden", method !== "adaptive");
}

// ── Read all slider/select params ───────────────────────────
function getParams() {
  return {
    p_low:        parseFloat(document.getElementById("p-low").value),
    p_high:       parseFloat(document.getElementById("p-high").value),
    filter_type:  document.getElementById("filter-type").value,
    kernel_size:  parseInt(document.getElementById("kernel-size").value),
    thresh_method: document.getElementById("thresh-method").value,
    global_t:     parseInt(document.getElementById("global-t").value),
    block_size:   parseInt(document.getElementById("block-size").value),
    C_val:        parseInt(document.getElementById("c-val").value),
    min_area:     parseInt(document.getElementById("min-area").value),
    max_area:     parseInt(document.getElementById("max-area").value) || null,
  };
}

// ── File upload ──────────────────────────────────────────────
document.getElementById("file-input").addEventListener("change", async function () {
  const file = this.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("image", file);
  showToast("Uploading image…");
  const res  = await fetch("/api/upload", { method: "POST", body: fd });
  const data = await res.json();
  if (data.error) { showToast("Upload failed: " + data.error, true); return; }
  currentImagePath = data.path;
  document.getElementById("upload-hint").textContent = file.name;
  showToast("Image uploaded ✓");
});

// Drag-and-drop on upload zone
const zone = document.getElementById("upload-zone");
zone.addEventListener("dragover",  e => { e.preventDefault(); zone.style.borderColor = "#7b61ff"; });
zone.addEventListener("dragleave", () => { zone.style.borderColor = ""; });
zone.addEventListener("drop", async e => {
  e.preventDefault();
  zone.style.borderColor = "";
  const file = e.dataTransfer.files[0];
  if (!file) return;
  document.getElementById("file-input").files = e.dataTransfer.files;
  document.getElementById("file-input").dispatchEvent(new Event("change"));
});

// ── Run dispatcher ───────────────────────────────────────────
function runCurrentTab() {
  if (currentTab === "stage8") runStage8();
  else                         runStage9();
}

// ── Stage 8 ─────────────────────────────────────────────────
async function runStage8() {
  setLoading("s8", true);
  clearError("s8");
  document.getElementById("s8-stats").classList.add("hidden");

  try {
    const body = {
      image_path: currentImagePath,
      params:     getParams(),
    };
    const res  = await fetch("/api/run_stage8", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) { showError("s8", data.error); return; }

    // Populate 6 panels
    const map = {
      original: "img-original",
      gray:     "img-gray",
      enhanced: "img-enhanced",
      filtered: "img-filtered",
      mask:     "img-mask",
      overlay:  "img-overlay",
    };
    for (const [key, id] of Object.entries(map)) {
      const el = document.getElementById(id);
      if (data.panels[key]) el.src = data.panels[key];
    }

    // Stats bar
    const [H, W] = data.image_shape;
    document.getElementById("s8-n-objects").textContent = `${data.n_objects} candidates`;
    document.getElementById("s8-otsu").textContent      = `Otsu T = ${data.otsu_val}`;
    document.getElementById("s8-shape").textContent     = `${W} × ${H} px`;
    document.getElementById("s8-stats").classList.remove("hidden");

    showToast(`Stage 8 complete — ${data.n_objects} objects detected`);
  } catch (err) {
    showError("s8", "Network error: " + err.message);
  } finally {
    setLoading("s8", false);
  }
}

// ── Stage 9 ─────────────────────────────────────────────────
async function runStage9() {
  setLoading("s9", true);
  clearError("s9");

  try {
    const body = {
      image_path:      currentImagePath,
      params:          getParams(),
      min_intensity:   parseFloat(document.getElementById("min-intensity").value),
      max_intensity:   parseFloat(document.getElementById("max-intensity").value),
      min_area_filter: parseInt(document.getElementById("min-area-f").value),
      max_area_filter: parseInt(document.getElementById("max-area-f").value) || 999999,
    };
    const res  = await fetch("/api/run_stage9", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) { showError("s9", data.error); return; }

    // Overlay & scatter
    if (data.overlay) document.getElementById("s9-overlay").src = data.overlay;
    if (data.scatter) document.getElementById("s9-scatter").src = data.scatter;

    // Count
    document.getElementById("s9-count").textContent =
      `${data.n_filtered} / ${data.n_total} objects`;

    // Table
    buildCatalogueTable(data.catalogue);

    showToast(`Stage 9 — ${data.n_filtered} objects in catalogue`);
  } catch (err) {
    showError("s9", "Network error: " + err.message);
  } finally {
    setLoading("s9", false);
  }
}

// ── Catalogue table builder ───────────────────────────────────
function buildCatalogueTable(records) {
  const tbody = document.getElementById("cat-tbody");
  tbody.innerHTML = "";
  if (!records || records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--text3);padding:20px">No objects match filters</td></tr>`;
    return;
  }
  const frag = document.createDocumentFragment();
  for (const r of records) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>#${r.object_id}</td>
      <td>${r.centroid_x !== undefined ? r.centroid_x.toFixed(1) : "—"}</td>
      <td>${r.centroid_y !== undefined ? r.centroid_y.toFixed(1) : "—"}</td>
      <td>${r.area_px}</td>
      <td>${r.mean_intensity !== undefined ? r.mean_intensity.toFixed(1) : "—"}</td>
      <td>${r.aspect_ratio  !== undefined ? r.aspect_ratio.toFixed(3)  : "—"}</td>
      <td>${r.extent        !== undefined ? r.extent.toFixed(3)        : "—"}</td>
    `;
    tr.addEventListener("click", () => showObjectDetail(r));
    frag.appendChild(tr);
  }
  tbody.appendChild(frag);
}

// ── Object detail card ────────────────────────────────────────
function showObjectDetail(obj) {
  document.getElementById("detail-id").textContent = `#${obj.object_id}`;
  const grid = document.getElementById("detail-grid");
  grid.innerHTML = "";
  const fields = [
    ["Centroid X",     obj.centroid_x?.toFixed(1)],
    ["Centroid Y",     obj.centroid_y?.toFixed(1)],
    ["BBox X",         obj.bbox_x],
    ["BBox Y",         obj.bbox_y],
    ["BBox W",         obj.bbox_width],
    ["BBox H",         obj.bbox_height],
    ["Area (px)",      obj.area_px],
    ["Mean Intensity", obj.mean_intensity?.toFixed(2)],
    ["Aspect Ratio",   obj.aspect_ratio?.toFixed(3)],
    ["Extent",         obj.extent?.toFixed(3)],
  ];
  for (const [k, v] of fields) {
    grid.insertAdjacentHTML("beforeend", `
      <div class="detail-item">
        <div class="detail-key">${k}</div>
        <div class="detail-val">${v ?? "—"}</div>
      </div>`);
  }
  document.getElementById("obj-detail").classList.remove("hidden");
}

// ── CSV Export ────────────────────────────────────────────────
async function exportCSV() {
  showToast("Generating CSV…");
  const body = {
    image_path: currentImagePath,
    params:     getParams(),
  };
  const res  = await fetch("/api/export_csv", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  if (!res.ok) { showToast("Export failed", true); return; }
  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = "object_catalogue.csv";
  a.click();
  URL.revokeObjectURL(url);
  showToast("CSV downloaded ✓");
}

// ── Helpers ──────────────────────────────────────────────────
function setLoading(stage, state) {
  const spinner = document.getElementById(stage + "-spinner");
  const runBtn  = document.getElementById("run-btn");
  if (spinner) spinner.style.display = state ? "flex" : "none";
  if (runBtn)  runBtn.disabled = state;
}

function showError(stage, msg) {
  const el = document.getElementById(stage + "-error");
  if (!el) return;
  el.textContent = "Error: " + msg;
  el.classList.remove("hidden");
}
function clearError(stage) {
  const el = document.getElementById(stage + "-error");
  if (el) el.classList.add("hidden");
}

let toastTimeout;
function showToast(msg, isError = false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.style.borderColor = isError ? "var(--accent3)" : "var(--border2)";
  t.classList.remove("hidden");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => t.classList.add("hidden"), 3200);
}

// ── Auto-run Stage 8 on load with default image ───────────────
window.addEventListener("DOMContentLoaded", () => {
  setTimeout(runStage8, 500);
});
