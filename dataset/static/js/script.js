/* ════════════════════════════════════════════════════════
   DeepGuard AI — Global Script
   Covers: Sidebar, Upload, Dashboard, Analytics, Analysis
   ════════════════════════════════════════════════════════ */

'use strict';

/* ════════════════════════════════════════════════════════
   1. SIDEBAR — Desktop collapse + Mobile drawer
   ════════════════════════════════════════════════════════ */
(function () {
  var sidebar      = document.querySelector('.sidebar');
  var mainContent  = document.querySelector('.main-content');
  var toggleBtn    = document.getElementById('sidebar-toggle');
  var hamburgerBtn = document.getElementById('hamburger-btn');
  var overlay      = document.getElementById('mobile-overlay');
  var STORAGE_KEY  = 'dg_sidebar_collapsed';

  var EXPANDED_WIDTH  = 260;
  var COLLAPSED_WIDTH = 64;
  var HALF_BTN        = 13;

  if (!sidebar) return;

  function isMobile() { return window.innerWidth <= 900; }

  /* ── Desktop: apply collapsed/expanded state ── */
  function applyDesktopState(collapsed, animate) {
    sidebar.classList.toggle('collapsed', collapsed);
    if (mainContent) mainContent.classList.toggle('sidebar-collapsed', collapsed);

    if (toggleBtn) {
      toggleBtn.innerHTML = collapsed ? '&#10095;' : '&#10094;';
      toggleBtn.title     = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
      var targetLeft = (collapsed ? COLLAPSED_WIDTH : EXPANDED_WIDTH) - HALF_BTN;

      toggleBtn.style.transition = animate
        ? 'left 0.28s cubic-bezier(0.4,0,0.2,1), background 0.2s, box-shadow 0.2s, transform 0.2s'
        : 'none';
      toggleBtn.style.left = targetLeft + 'px';
    }
  }

  /* ── Mobile: open drawer ── */
  function openMobileSidebar() {
    sidebar.classList.add('mobile-open');
    if (overlay) overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  /* ── Mobile: close drawer ── */
  function closeMobileSidebar() {
    sidebar.classList.remove('mobile-open');
    if (overlay) overlay.classList.remove('active');
    document.body.style.overflow = '';
  }

  /* ── Init desktop state from localStorage ── */
  if (!isMobile()) {
    var savedCollapsed = localStorage.getItem(STORAGE_KEY) === '1';
    applyDesktopState(savedCollapsed, false);
  }

  /* ── Desktop toggle ── */
  if (toggleBtn) {
    toggleBtn.addEventListener('click', function () {
      var nowCollapsed = !sidebar.classList.contains('collapsed');
      applyDesktopState(nowCollapsed, true);
      localStorage.setItem(STORAGE_KEY, nowCollapsed ? '1' : '0');
    });
  }

  /* ── Hamburger (mobile) ── */
  if (hamburgerBtn) {
    hamburgerBtn.addEventListener('click', function () {
      if (sidebar.classList.contains('mobile-open')) {
        closeMobileSidebar();
      } else {
        openMobileSidebar();
      }
    });
  }

  /* ── Overlay click closes drawer ── */
  if (overlay) {
    overlay.addEventListener('click', closeMobileSidebar);
  }

  /* ── ESC closes mobile drawer ── */
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && isMobile()) closeMobileSidebar();
  });

  /* ── Debounced resize handler ── */
  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (!isMobile()) {
        closeMobileSidebar();
        var isCollapsed = localStorage.getItem(STORAGE_KEY) === '1';
        applyDesktopState(isCollapsed, false);
      }
    }, 150);
  });
})();


/* ════════════════════════════════════════════════════════
   2. UPLOAD PAGE — File handling, preview, validation
   ════════════════════════════════════════════════════════ */
(function () {
  var input         = document.getElementById('image');
  var uploadArea    = document.getElementById('upload-area');
  var previewWrap   = document.getElementById('preview-wrap');
  var preview       = document.getElementById('preview');
  var fileChip      = document.getElementById('file-chip');
  var chipName      = document.getElementById('file-chip-name');
  var chipType      = document.getElementById('file-chip-type');
  var chipSize      = document.getElementById('file-chip-size');
  var chipRemove    = document.getElementById('file-chip-remove');
  var fileError     = document.getElementById('file-error');
  var uploadLabel   = document.getElementById('upload-label');
  var form          = document.getElementById('detect-form');
  var submitBtn     = document.getElementById('submit-btn');
  var scanOverlay   = document.getElementById('preview-scan-overlay');
  var resultBox     = document.getElementById('result-box');

  var MAX_MB    = 10;
  var ALLOWED   = ['image/png', 'image/jpeg', 'image/webp'];

  /* ── Auto-scroll to result if present ── */
  if (resultBox && resultBox.textContent.trim().length > 0) {
    resultBox.style.display = 'block';
    setTimeout(function () {
      resultBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 350);
  }

  /* Exit early if not on upload page */
  if (!input) return;

  /* ── Prevent browser navigation on missed drag ── */
  document.addEventListener('dragover', function (e) { e.preventDefault(); });
  document.addEventListener('drop',     function (e) { e.preventDefault(); });

  /* ── Show error message ── */
  function showError(msg) {
    if (!fileError) return;
    fileError.textContent  = msg;
    fileError.style.display = 'block';
  }

  /* ── Clear error ── */
  function clearError() {
    if (!fileError) return;
    fileError.textContent  = '';
    fileError.style.display = 'none';
  }

  /* ── Reset all file UI ── */
  function resetFile(opts) {
    opts = opts || {};
    input.value = '';

    if (previewWrap) previewWrap.classList.remove('visible');
    if (fileChip)    fileChip.classList.remove('visible');
    if (preview)     preview.src = '';
    if (uploadLabel) uploadLabel.textContent = 'Drop image here or click to browse';

    if (!opts.keepError) clearError();
  }

  /* ── Load and validate a File object ── */
  function loadFile(file) {
    if (!file) return;

    /* Type check */
    if (!ALLOWED.includes(file.type)) {
      resetFile({ keepError: true });
      showError('Unsupported file type. Please upload PNG, JPG, or WEBP.');
      return;
    }

    /* Size check */
    if (file.size > MAX_MB * 1024 * 1024) {
      resetFile({ keepError: true });
      showError(
        'File too large (' + (file.size / 1024 / 1024).toFixed(1) +
        ' MB). Max: ' + MAX_MB + ' MB.'
      );
      return;
    }

    clearError();

    /* File info chip */
    if (chipName && chipType && chipSize && fileChip) {
      var ext = file.name.split('.').pop().toUpperCase();
      chipName.textContent = file.name;
      chipType.textContent = ext;
      chipSize.textContent = file.size > 1024 * 1024
        ? (file.size / 1024 / 1024).toFixed(1) + ' MB'
        : (file.size / 1024).toFixed(0) + ' KB';
      fileChip.classList.add('visible');
    }

    /* Image preview */
    var reader = new FileReader();
    reader.onload = function (e) {
      if (preview) preview.src = e.target.result;
      if (previewWrap) previewWrap.classList.add('visible');
    };
    reader.readAsDataURL(file);

    if (uploadLabel) uploadLabel.textContent = '✓ Ready to scan';
  }

  /* ── Input change ── */
  input.addEventListener('change', function () {
    loadFile(this.files[0]);
  });

  /* ── Chip remove button ── */
  if (chipRemove) {
    chipRemove.addEventListener('click', function () { resetFile(); });
  }

  /* ── Drag and drop ── */
  if (uploadArea) {
    uploadArea.addEventListener('dragover', function (e) {
      e.preventDefault();
      uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function (e) {
      if (!uploadArea.contains(e.relatedTarget)) {
        uploadArea.classList.remove('drag-over');
      }
    });

    uploadArea.addEventListener('drop', function (e) {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
      var file = e.dataTransfer.files[0];
      if (!file) return;

      /* Try to assign to file input for form submit */
      try {
        var dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
      } catch (_) {}

      loadFile(file);
    });
  }

  /* ── Form submit ── */
  if (form) {
    form.addEventListener('submit', function (e) {
      if (fileError && fileError.style.display === 'block') {
        e.preventDefault();
        return;
      }

      if (!input.files || input.files.length === 0) {
        e.preventDefault();
        showError('Please select an image before scanning.');
        return;
      }

      if (scanOverlay) scanOverlay.classList.add('active');
      if (submitBtn)   submitBtn.classList.add('loading');
    });
  }

  /* ── Re-enable on bfcache navigation ── */
  window.addEventListener('pageshow', function (e) {
    if (e.persisted) {
      if (submitBtn)  submitBtn.classList.remove('loading');
      if (scanOverlay) scanOverlay.classList.remove('active');
    }
  });
})();


/* ════════════════════════════════════════════════════════
   3. DASHBOARD — Search, delete modal, timestamps
   ════════════════════════════════════════════════════════ */
(function () {
  /* ── Live search ── */
  var searchInput = document.getElementById('search-input');
  var tableRows   = document.querySelectorAll('#table-body tr');

  if (searchInput && tableRows.length) {
    /* Pre-fill from URL ?q= parameter */
    var urlQ = new URLSearchParams(window.location.search).get('q');
    if (urlQ) {
      searchInput.value = urlQ;
      filterRows(urlQ);
    }

    searchInput.addEventListener('input', function () {
      filterRows(this.value);
    });
  }

  function filterRows(query) {
    var q = query.toLowerCase().trim();
    tableRows.forEach(function (row) {
      var fname = (row.getAttribute('data-filename') || '').toLowerCase();
      row.style.display = fname.includes(q) ? '' : 'none';
    });
  }

  /* ── Delete / clear modal ── */
  var pendingId = null;
  var deleteModal = document.getElementById('delete-modal');

  window.openDeleteModal = function (id) {
    pendingId = id;
    var isAll = (id === 'all');

    var title = document.getElementById('modal-title');
    var text  = document.getElementById('modal-text');
    var btn   = document.getElementById('confirm-delete-btn');

    if (title) title.textContent = isAll ? 'Clear all history?' : 'Delete this record?';
    if (text)  text.textContent  = isAll
      ? 'This will permanently delete every scan record. This cannot be undone.'
      : 'This action cannot be undone. The scan record will be permanently removed.';
    if (btn)   btn.textContent   = isAll ? 'Clear All' : 'Delete';

    if (deleteModal) deleteModal.classList.add('show');
  };

  window.closeDeleteModal = function () {
    pendingId = null;
    if (deleteModal) deleteModal.classList.remove('show');
  };

  var confirmBtn = document.getElementById('confirm-delete-btn');
  if (confirmBtn) {
    confirmBtn.addEventListener('click', function () {
      if (pendingId === null) return;

      if (pendingId === 'all') {
        fetch('/clear-history', { method: 'POST' })
          .then(function () { window.location.reload(); })
          .catch(function () { window.location.reload(); });
        window.closeDeleteModal();
        return;
      }

      var row = document.getElementById('row-' + pendingId);
      fetch('/delete/' + pendingId, { method: 'POST' })
        .then(function (r) {
          if (!r.ok) throw new Error('Delete failed');
          if (row) {
            row.classList.add('removing');
            setTimeout(function () { row.remove(); }, 360);
          }
          window.closeDeleteModal();
        })
        .catch(function () {
          window.closeDeleteModal();
          window.location.reload();
        });
    });
  }

  if (deleteModal) {
    deleteModal.addEventListener('click', function (e) {
      if (e.target === this) window.closeDeleteModal();
    });
  }

  /* ── Sidebar quick-action clear ── */
  var dashClearBtn = document.getElementById('dash-clear-btn');
  if (dashClearBtn) {
    dashClearBtn.addEventListener('click', function () {
      window.openDeleteModal('all');
    });
  }

  /* ── Format timestamps ── */
  formatTimestamps();
})();


/* ════════════════════════════════════════════════════════
   4. ANALYTICS — KPI counters, charts, donut hover
   ════════════════════════════════════════════════════════ */
(function () {
  /* Only run if analytics elements are present */
  if (!document.querySelector('.kpi-value[data-count]')) return;

  /* ── KPI count-up animation ── */
  document.querySelectorAll('.kpi-value[data-count]').forEach(function (el) {
    var target   = parseFloat(el.dataset.count) || 0;
    var decimals = parseInt(el.dataset.decimals, 10) || 0;
    var suffix   = el.dataset.suffix || '';
    var duration = 1000;
    var startTs  = null;

    function step(ts) {
      if (!startTs) startTs = ts;
      var progress = Math.min((ts - startTs) / duration, 1);
      var ease     = 1 - Math.pow(1 - progress, 3);
      el.textContent = (target * ease).toFixed(decimals) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  });

  /* ── Animate bars + donut on load ── */
  window.addEventListener('load', function () {
    setTimeout(function () {

      /* Distribution bars */
      document.querySelectorAll('.dist-bar-fill[data-target]').forEach(function (bar) {
        bar.style.width = bar.dataset.target + '%';
      });

      /* High-confidence bar */
      var hfill = document.getElementById('hconf-fill');
      if (hfill) hfill.style.width = (hfill.dataset.target || 0) + '%';

      /* Donut arcs */
      ['donut-real', 'donut-fake'].forEach(function (id) {
        var arc = document.getElementById(id);
        if (arc && arc.dataset.final !== undefined) {
          arc.style.strokeDashoffset = arc.dataset.final;
        }
      });

    }, 180);
  });

  /* ── Donut hover interactions ── */
  var centerVal   = document.getElementById('donut-center-val');
  var centerLabel = document.getElementById('donut-center-label');
  var totalVal    = centerVal   ? centerVal.textContent   : '';
  var totalLabel  = centerLabel ? centerLabel.textContent : '';

  window.highlightArc = function (type) {
    var arcId   = type === 'real' ? 'donut-real' : 'donut-fake';
    var otherId = type === 'real' ? 'donut-fake' : 'donut-real';
    var arc     = document.getElementById(arcId);
    var other   = document.getElementById(otherId);

    if (!arc) return;
    if (other) other.style.opacity = '0.22';

    if (centerVal)   {
      centerVal.textContent = arc.dataset.value + ' (' + arc.dataset.pct + '%)';
      centerVal.style.color = arc.dataset.color;
    }
    if (centerLabel) centerLabel.textContent = arc.dataset.label;

    ['real', 'fake'].forEach(function (t) {
      var li = document.getElementById('legend-' + t);
      if (li) li.classList.remove('highlight-real', 'highlight-fake');
    });

    var legend = document.getElementById('legend-' + type);
    if (legend) legend.classList.add('highlight-' + type);
  };

  window.resetArc = function () {
    ['donut-real', 'donut-fake'].forEach(function (id) {
      var arc = document.getElementById(id);
      if (arc) arc.style.opacity = '1';
    });

    if (centerVal)   { centerVal.textContent = totalVal;   centerVal.style.color = ''; }
    if (centerLabel)   centerLabel.textContent = totalLabel;

    ['real', 'fake'].forEach(function (t) {
      var li = document.getElementById('legend-' + t);
      if (li) li.classList.remove('highlight-real', 'highlight-fake');
    });
  };

  /* Wire SVG arc hover events */
  ['donut-real', 'donut-fake'].forEach(function (id) {
    var arc  = document.getElementById(id);
    if (!arc) return;
    var type = id === 'donut-real' ? 'real' : 'fake';
    arc.addEventListener('mouseenter', function () { window.highlightArc(type); });
    arc.addEventListener('mouseleave', window.resetArc);
  });

  /* ── Format timestamps ── */
  formatTimestamps();
})();


/* ════════════════════════════════════════════════════════
   5. ANALYSIS PAGE — Gauge, vote bars, heatmap, copy
   ════════════════════════════════════════════════════════ */
(function () {
  /* Gauge arc animation */
  var gaugeArc = document.getElementById('gauge-arc');
  if (gaugeArc) {
    var finalOffset = parseFloat(gaugeArc.dataset.final);
    if (!isNaN(finalOffset)) {
      requestAnimationFrame(function () {
        setTimeout(function () {
          gaugeArc.style.transition = 'stroke-dashoffset 1.3s cubic-bezier(0.25,1,0.5,1)';
          gaugeArc.setAttribute('stroke-dashoffset', finalOffset);
        }, 320);
      });
    }
  }

  /* Model vote bar animations */
  window.addEventListener('load', function () {
    setTimeout(function () {
      document.querySelectorAll('.model-vote-bar-fill[data-target]').forEach(function (bar) {
        bar.style.width = bar.dataset.target + '%';
      });
    }, 450);
  });

  /* Heatmap toggle */
  var heatmapToggle  = document.getElementById('heatmap-toggle');
  var heatmapOverlay = document.getElementById('heatmap-overlay');
  var overlayBadge   = document.getElementById('overlay-badge');
  var visGrid        = document.getElementById('vis-grid');

  if (heatmapToggle && heatmapOverlay) {
    heatmapToggle.addEventListener('change', function () {
      var on = this.checked;
      heatmapOverlay.classList.toggle('hidden', !on);
      if (visGrid)      visGrid.classList.toggle('active', on);
      if (overlayBadge) {
        overlayBadge.textContent = on ? 'Overlay ON' : 'Overlay OFF';
        overlayBadge.className   = 'overlay-badge ' + (on ? 'on' : 'off');
      }
    });

    /* Sync grid to initial toggle state */
    if (heatmapToggle.checked && visGrid) visGrid.classList.add('active');
  }

  /* Copy link with toast */
  window.copyLink = function () {
    function showToast() {
      var toast = document.getElementById('toast');
      if (!toast) return;
      toast.classList.add('show');
      setTimeout(function () { toast.classList.remove('show'); }, 2400);
    }

    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(window.location.href)
        .then(showToast)
        .catch(fallbackCopy);
    } else {
      fallbackCopy();
    }

    function fallbackCopy() {
      var input = document.createElement('input');
      input.value = window.location.href;
      input.style.position = 'fixed';
      input.style.opacity = '0';
      document.body.appendChild(input);
      input.select();
      try { document.execCommand('copy'); } catch (_) {}
      document.body.removeChild(input);
      showToast();
    }
  };

  /* Filename truncation in hero */
  document.querySelectorAll('.fmt-fname').forEach(function (el) {
    var name = el.textContent.trim();
    if (name.length > 24) {
      var ext = name.lastIndexOf('.');
      el.textContent = ext > 0
        ? name.slice(0, 14) + '…' + name.slice(ext)
        : name.slice(0, 20) + '…';
      el.title = name;
    }
  });

  /* Format timestamps */
  formatTimestamps();
})();


/* ════════════════════════════════════════════════════════
   6. SHARED UTILITIES
   ════════════════════════════════════════════════════════ */

/* ── Global timestamp formatter ── */
function formatTimestamps() {
  document.querySelectorAll('.fmt-ts').forEach(function (el) {
    var raw = el.dataset.ts || el.textContent.trim();
    if (!raw || raw === '—') return;
    var d = new Date(raw.replace(' ', 'T'));
    if (isNaN(d)) return;
    el.textContent = d.toLocaleDateString('en-IN', {
      day: 'numeric', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  });
}

/* ── Global toast helper ── */
window.showGlobalToast = function (msg, isError) {
  var toastIds = ['save-toast', 'toast'];
  var toast;
  for (var i = 0; i < toastIds.length; i++) {
    toast = document.getElementById(toastIds[i]);
    if (toast) break;
  }
  if (!toast) return;

  var msgEl = toast.querySelector('span') || toast;
  if (msgEl !== toast) msgEl.textContent = msg || 'Done';
  else toast.textContent = msg || 'Done';

  toast.style.borderColor = isError
    ? 'rgba(255,56,96,0.35)'
    : 'rgba(0,242,255,0.28)';
  toast.style.color = isError ? 'var(--red)' : 'var(--cyan)';
  toast.classList.add('show');
  setTimeout(function () { toast.classList.remove('show'); }, 2600);
};

/* ── Topbar search redirect (index.html) ── */
(function () {
  var el = document.getElementById('topbar-search');
  if (!el) return;
  el.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && this.value.trim()) {
      window.location.href = '/dashboard?q=' + encodeURIComponent(this.value.trim());
    }
  });
})();

/* ── Reduced motion: disable CSS animations if preferred ── */
(function () {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.documentElement.style.setProperty('--transition', '0.001ms');
  }
})();