'use strict';

// ── Stage ordering for progress stepper ──────────────────
const STAGES = [
  'extracting_frames',
  'pose_estimation',
  'gemini_analysis',
  'tts_generation',
  'assembling_output',
];

// ── State ─────────────────────────────────────────────────
let currentJobId = null;
let pollInterval = null;
let selectedFile = null;

// ── DOM references ─────────────────────────────────────────
const views = {
  welcome: document.getElementById('view-welcome'),
  upload: document.getElementById('view-upload'),
  processing: document.getElementById('view-processing'),
  results: document.getElementById('view-results'),
};

// ── Welcome screen ─────────────────────────────────────────
const btnReady = document.getElementById('btn-ready');
setTimeout(() => { btnReady.classList.add('visible'); }, 2800);
btnReady.addEventListener('click', () => {
  const welcome = views.welcome;
  welcome.classList.add('fade-out');
  welcome.addEventListener('animationend', () => {
    welcome.classList.remove('active');
    welcome.style.display = 'none';
    views.upload.classList.add('active');
  }, { once: true });
});

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileSelected = document.getElementById('file-selected');
const fileNameEl = document.getElementById('file-name');
const btnAnalyze = document.getElementById('btn-analyze');
const uploadError = document.getElementById('upload-error');
const progressBar = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const processingError = document.getElementById('processing-error');
const btnNew = document.getElementById('btn-new');

// ── View switching ─────────────────────────────────────────
function showView(name) {
  Object.values(views).forEach(v => {
    v.classList.remove('active');
    if (v.id === 'view-welcome') v.style.display = 'none';
  });
  views[name].classList.add('active');
  if (name !== 'results') stopVisualizer();
}

// ── Upload flow ────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileSelected.classList.remove('hidden');
  uploadError.classList.add('hidden');
}

btnAnalyze.addEventListener('click', async () => {
  if (!selectedFile) return;
  uploadError.classList.add('hidden');
  btnAnalyze.disabled = true;
  btnAnalyze.textContent = 'Uploading…';

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }

    const { job_id } = await resp.json();
    currentJobId = job_id;
    startPolling();
    showView('processing');

  } catch (err) {
    showError(uploadError, err.message);
  } finally {
    btnAnalyze.disabled = false;
    btnAnalyze.textContent = 'Analyze My Climb';
  }
});

// ── Elapsed timer ──────────────────────────────────────────
let elapsedInterval = null;
let startTime = null;
const elapsedTimer = document.getElementById('elapsed-timer');

function startElapsedTimer() {
  startTime = Date.now();
  elapsedTimer.textContent = '0:00';
  elapsedInterval = setInterval(() => {
    const secs = Math.floor((Date.now() - startTime) / 1000);
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    elapsedTimer.textContent = `${m}:${s.toString().padStart(2, '0')}`;
  }, 1000);
}

function stopElapsedTimer() {
  clearInterval(elapsedInterval);
  elapsedInterval = null;
}

// ── Polling ────────────────────────────────────────────────
function startPolling() {
  resetStepper();
  progressBar.style.width = '0%';
  progressLabel.textContent = 'Starting…';
  processingError.classList.add('hidden');
  startElapsedTimer();

  pollInterval = setInterval(pollStatus, 2000);
}

function stopPolling() {
  clearInterval(pollInterval);
  pollInterval = null;
  stopElapsedTimer();
}

async function pollStatus() {
  if (!currentJobId) return;
  try {
    const resp = await fetch(`/api/jobs/${currentJobId}/status`);
    if (!resp.ok) throw new Error(`Status check failed: ${resp.status}`);

    const data = await resp.json();
    updateProgress(data);

    if (data.status === 'completed') {
      stopPolling();
      await fetchAndRenderResult();
    } else if (data.status === 'failed') {
      stopPolling();
      showError(processingError, data.error || 'Processing failed. Please try again.');
    }
  } catch (err) {
    console.error('Poll error:', err);
  }
}

function updateProgress(data) {
  const pct = data.progress_pct || 0;
  progressBar.style.width = `${pct}%`;
  progressLabel.textContent = `${pct}%`;
  if (data.stage) updateStepper(data.stage);
}

// ── Stepper ────────────────────────────────────────────────
function resetStepper() {
  document.querySelectorAll('.stage').forEach(el => {
    el.classList.remove('active', 'done');
  });
}

function updateStepper(currentStage) {
  const currentIdx = STAGES.indexOf(currentStage);
  document.querySelectorAll('.stage').forEach(el => {
    const stage = el.dataset.stage;
    const idx = STAGES.indexOf(stage);
    el.classList.remove('active', 'done');
    if (idx < currentIdx) el.classList.add('done');
    else if (idx === currentIdx) el.classList.add('active');
  });
}

// ── Audio Visualizer (ChatGPT-style orb) ──────────────────
let audioCtx = null;
let analyser = null;
let animFrameId = null;
let isPlaying = false;

const canvas = document.getElementById('coach-visualizer');
const ctx = canvas.getContext('2d');
const coachAudio = document.getElementById('coach-audio');
const btnPlay = document.getElementById('btn-play-coach');

function initAudioContext() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.8;
  const source = audioCtx.createMediaElementSource(coachAudio);
  source.connect(analyser);
  analyser.connect(audioCtx.destination);
}

function startVisualizer() {
  if (animFrameId) return;
  drawOrb();
}

function stopVisualizer() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
}

function drawOrb() {
  animFrameId = requestAnimationFrame(drawOrb);

  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const baseRadius = Math.min(w, h) * 0.25;

  ctx.clearRect(0, 0, w, h);

  let amplitude = 0;
  let freqData = null;

  if (analyser) {
    const bufferLength = analyser.frequencyBinCount;
    freqData = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(freqData);
    // Average amplitude
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) sum += freqData[i];
    amplitude = sum / bufferLength / 255;
  }

  const time = performance.now() / 1000;

  // Outer glow
  const glowRadius = baseRadius + baseRadius * amplitude * 0.6;
  const gradient = ctx.createRadialGradient(cx, cy, glowRadius * 0.3, cx, cy, glowRadius * 1.8);
  gradient.addColorStop(0, `rgba(193, 95, 60, ${0.12 + amplitude * 0.15})`);
  gradient.addColorStop(0.5, `rgba(193, 95, 60, ${0.04 + amplitude * 0.06})`);
  gradient.addColorStop(1, 'rgba(193, 95, 60, 0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);

  // Draw morphing orb with multiple layered blobs
  const layers = [
    { color: 'rgba(193, 95, 60, 0.08)', radiusMult: 1.35, blobCount: 5, blobAmp: 0.15, speed: 0.4 },
    { color: 'rgba(193, 95, 60, 0.15)', radiusMult: 1.15, blobCount: 6, blobAmp: 0.12, speed: 0.6 },
    { color: 'rgba(193, 95, 60, 0.35)', radiusMult: 1.0, blobCount: 7, blobAmp: 0.08, speed: 0.8 },
    { color: 'rgba(210, 130, 90, 0.6)', radiusMult: 0.85, blobCount: 8, blobAmp: 0.06, speed: 1.0 },
    { color: 'rgba(230, 160, 120, 0.5)', radiusMult: 0.6, blobCount: 5, blobAmp: 0.04, speed: 1.2 },
  ];

  for (const layer of layers) {
    const r = baseRadius * layer.radiusMult + baseRadius * amplitude * 0.4 * layer.radiusMult;
    ctx.beginPath();
    const steps = 120;
    for (let i = 0; i <= steps; i++) {
      const angle = (i / steps) * Math.PI * 2;

      // Multiple sine waves for organic blob shape
      let blobOffset = 0;
      for (let b = 1; b <= layer.blobCount; b++) {
        const freqInfluence = freqData ? (freqData[b * 4] || 0) / 255 : 0;
        blobOffset += Math.sin(angle * b + time * layer.speed * b * 0.3) *
          (layer.blobAmp + freqInfluence * 0.08) * r;
      }

      const dr = r + blobOffset;
      const x = cx + Math.cos(angle) * dr;
      const y = cy + Math.sin(angle) * dr;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = layer.color;
    ctx.fill();
  }

  // Center bright spot
  const spotGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, baseRadius * 0.4);
  spotGrad.addColorStop(0, `rgba(244, 243, 238, ${0.15 + amplitude * 0.2})`);
  spotGrad.addColorStop(1, 'rgba(244, 243, 238, 0)');
  ctx.fillStyle = spotGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius * 0.4, 0, Math.PI * 2);
  ctx.fill();
}

btnPlay.addEventListener('click', () => {
  initAudioContext();
  if (audioCtx.state === 'suspended') audioCtx.resume();

  if (isPlaying) {
    coachAudio.pause();
    setPlayState(false);
  } else {
    coachAudio.play();
    setPlayState(true);
  }
});

coachAudio.addEventListener('ended', () => setPlayState(false));

function setPlayState(playing) {
  isPlaying = playing;
  btnPlay.querySelector('.play-icon').style.display = playing ? 'none' : '';
  btnPlay.querySelector('.pause-icon').style.display = playing ? '' : 'none';
}

// ── Results rendering ──────────────────────────────────────
async function fetchAndRenderResult() {
  try {
    const resp = await fetch(`/api/jobs/${currentJobId}/result`);
    if (!resp.ok) throw new Error('Failed to fetch result');
    const data = await resp.json();
    renderResult(data);
    showView('results');
  } catch (err) {
    showError(processingError, err.message);
  }
}

function renderResult(data) {
  const { feedback, output_urls } = data;

  // Annotated video
  if (output_urls?.annotated_video) {
    document.getElementById('annotated-src').src = output_urls.annotated_video;
    document.getElementById('annotated-video').load();
  }

  // Coach audio + visualizer
  if (output_urls?.coaching_audio) {
    coachAudio.src = output_urls.coaching_audio;
    coachAudio.load();
    startVisualizer();
  }

  if (!feedback) return;

  setText('fb-overall', feedback.overall_summary);
  setList('fb-form', feedback.form);
  setList('fb-movement', feedback.movement);
  setList('fb-route', feedback.route_reading);
  setText('fb-encouragement', feedback.encouragement);
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text || '';
}

function setList(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = '';
  (items || []).forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    el.appendChild(li);
  });
}

// ── Reset / new analysis ───────────────────────────────────
btnNew.addEventListener('click', () => {
  currentJobId = null;
  selectedFile = null;
  fileInput.value = '';
  fileSelected.classList.add('hidden');
  uploadError.classList.add('hidden');
  stopVisualizer();
  coachAudio.pause();
  coachAudio.src = '';
  setPlayState(false);
  showView('upload');
});

// ── Helpers ────────────────────────────────────────────────
function showError(el, message) {
  el.textContent = message;
  el.classList.remove('hidden');
}
