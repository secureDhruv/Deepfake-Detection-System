// ── Element references ──────────────────────────────
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

const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

// ── Show result box on page load if result exists ───
if (resultBox) {
  resultBox.style.display = 'block';
  // Smoothly scroll to result after a short delay
  setTimeout(() => {
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, 300);
}

// ── Handle file selection ───────────────────────────
input.addEventListener('change', () => {
  const file = input.files[0];
  handleFile(file);
});

// ── Handle file (shared by input + drag-and-drop) ───
function handleFile(file) {
  fileError.style.display = 'none';
  fileName.textContent = '';
  preview.style.display = 'none';

  if (!file) return;

  // Validate file size
  if (file.size > MAX_SIZE_BYTES) {
    fileError.textContent = `File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum allowed size is ${MAX_SIZE_MB} MB.`;
    fileError.style.display = 'block';
    input.value = '';
    return;
  }

  // Validate file type
  const allowedTypes = ['image/png', 'image/jpeg', 'image/webp'];
  if (!allowedTypes.includes(file.type)) {
    fileError.textContent = 'Unsupported file type. Please upload a PNG, JPG, or WEBP image.';
    fileError.style.display = 'block';
    input.value = '';
    return;
  }

  // Show file name
  fileName.textContent = '✓ ' + file.name;

  // Show image preview
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

// ── Drag-and-drop support ───────────────────────────
uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');

  const file = e.dataTransfer.files[0];
  if (!file) return;

  // Assign dropped file to the input
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  input.files = dataTransfer.files;

  handleFile(file);
});

// ── Loading state on submit ─────────────────────────
form.addEventListener('submit', (e) => {
  // Block submit if there's a validation error
  if (fileError.style.display === 'block') {
    e.preventDefault();
    return;
  }

  btn.disabled = true;
  spinner.style.display = 'inline-block';
  btnLabel.textContent = 'Analysing…';
});