// ════════════════════════════════════════════════════════
// SIDEBAR COLLAPSE TOGGLE — works on all pages
// ════════════════════════════════════════════════════════
(function () {
  var sidebar     = document.querySelector('.sidebar');
  var mainContent = document.querySelector('.main-content');
  var toggleBtn   = document.getElementById('sidebar-toggle');
  var STORAGE_KEY = 'dg_sidebar_collapsed';

  if (!sidebar || !toggleBtn) return;

  // Restore saved state on load
  var isCollapsed = localStorage.getItem(STORAGE_KEY) === '1';
  if (isCollapsed) {
    sidebar.classList.add('collapsed');
    if (mainContent) mainContent.classList.add('sidebar-collapsed');
    toggleBtn.innerHTML = '&#10095;';
    toggleBtn.title = 'Expand sidebar';
  }

  toggleBtn.addEventListener('click', function () {
    isCollapsed = !sidebar.classList.contains('collapsed');
    sidebar.classList.toggle('collapsed', isCollapsed);
    if (mainContent) mainContent.classList.toggle('sidebar-collapsed', isCollapsed);
    toggleBtn.innerHTML = isCollapsed ? '&#10095;' : '&#10094;';
    toggleBtn.title = isCollapsed ? 'Expand sidebar' : 'Collapse sidebar';
    localStorage.setItem(STORAGE_KEY, isCollapsed ? '1' : '0');
  });
})();

// ════════════════════════════════════════════════════════
// UPLOAD PAGE — Element references
// ════════════════════════════════════════════════════════

(function () {
  const input      = document.getElementById('image');
  const uploadArea = document.getElementById('upload-area');
const preview    = document.getElementById('preview');
const fileName   = document.getElementById('file-name');
const fileError  = document.getElementById('file-error');
const form       = document.getElementById('detect-form');
const btn        = document.getElementById('submit-btn');
const spinner    = document.getElementById('spinner');
const btnLabel   = document.getElementById('btn-label');
const resultBox  = document.getElementById('result-box');

const MAX_SIZE_MB    = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

// ── Show result box if server returned a result ───────────
if (resultBox) {
  resultBox.style.display = 'block';
  setTimeout(function () {
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, 300);
}

// ── Guard: exit early if not on the upload page ───────────
if (!input) return; // elements below only exist on index.html

// ── Handle file selection ─────────────────────────────────
input.addEventListener('change', function () {
  handleFile(input.files[0]);
});

// ── Handle file (shared by input + drag-and-drop) ─────────
function handleFile(file) {
  if (fileError) { fileError.style.display = 'none'; fileError.textContent = ''; }
  if (fileName)  { fileName.textContent = ''; }
  if (preview)   { preview.style.display = 'none'; preview.src = ''; }

  if (!file) return;

  // Validate size
  if (file.size > MAX_SIZE_BYTES) {
    showError('File too large (' + (file.size / 1024 / 1024).toFixed(1) + ' MB). Max allowed: ' + MAX_SIZE_MB + ' MB.');
    input.value = '';
    return;
  }

  // Validate type
  var allowed = ['image/png', 'image/jpeg', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showError('Unsupported file type. Please upload PNG, JPG, or WEBP.');
    input.value = '';
    return;
  }

  // Show file name
  if (fileName) fileName.textContent = '\u2713 ' + file.name;

  // Show preview
  var reader = new FileReader();
  reader.onload = function (e) {
    if (preview) {
      preview.src = e.target.result;
      preview.style.display = 'block';
    }
  };
  reader.readAsDataURL(file);
}

function showError(msg) {
  if (!fileError) return;
  fileError.textContent = msg;
  fileError.style.display = 'block';
}

// ── Drag-and-drop support ─────────────────────────────────
if (uploadArea) {
  uploadArea.addEventListener('dragover', function (e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });

  uploadArea.addEventListener('dragleave', function () {
    uploadArea.classList.remove('dragover');
  });

  uploadArea.addEventListener('drop', function (e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    var file = e.dataTransfer.files[0];
    if (!file) return;
    var dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    handleFile(file);
  });
}

  // ── Loading state on submit ───────────────────────────────
  if (form) {
    form.addEventListener('submit', function (e) {
      if (fileError && fileError.style.display === 'block') {
        e.preventDefault();
        return;
      }
      if (btn)      btn.disabled = true;
      if (spinner)  spinner.style.display = 'inline-block';
      if (btnLabel) btnLabel.textContent = 'Analysing\u2026';
    });
  }
})();