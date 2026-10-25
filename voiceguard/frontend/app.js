/**
 * app.js — VoiceGuard v2 Main Frontend Application
 * Handles: routing, enrollment, incremental voice notes, calibration,
 * live WebSocket monitoring with ECAPA-TDNN speaker verification, and settings.
 */

import { AudioCapture } from './audio_capture.js';
import { Visualizer }   from './visualizer.js';

const API_HOST = window.location.hostname || 'localhost';
const API_BASE = `http://${API_HOST}:8000`;
const WS_BASE  = `ws://${API_HOST}:8000`;

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  page:            'enroll',
  speakers:        [],
  recording:       false,
  recordDuration:  5,
  recordedBlob:    null,
  vnRecording:     false,
  vnDuration:      5,
  vnRecordedBlob:  null,
  listening:       false,
  ws:              null,
  audioCapture:    new AudioCapture(),
  visualizer:      new Visualizer(),
  speechRecognizer: null,   // Web Speech API instance for the door page
  enrollRecognizer: null,   // Web Speech API instance for enroll recording
  requireChallenge: true,   // from backend config
  phraseHeard:     false,   // backend confirmed the current challenge phrase
  lastTranscript:  '',
  doorUnlocked:    false,
  relockTimer:     null,    // countdown display interval
  spectrumSpeaker: null,    // speaker whose voiceprint spectrum is overlaid
};

// ─── DOM Helpers ──────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);

// ─── Navigation ───────────────────────────────────────────────────────────────
function navigate(page) {
  state.page = page;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  $(`page-${page}`)?.classList.add('active');
  $(`nav-${page}`)?.classList.add('active');

  if (page === 'voice-notes') initVoiceNotesPage();
  if (page === 'monitor')     initMonitorPage();
  if (page === 'settings')    initSettingsPage();
}

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => navigate(btn.dataset.page));
});

// ─── Backend Health ───────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    $('backend-status-dot').className  = 'status-dot online';
    $('backend-status-text').textContent = 'Backend online';
    return data;
  } catch {
    $('backend-status-dot').className  = 'status-dot offline';
    $('backend-status-text').textContent = 'Backend offline';
    return null;
  }
}

// ─── Speakers ─────────────────────────────────────────────────────────────────
async function fetchSpeakers() {
  try {
    const res  = await fetch(`${API_BASE}/speakers`);
    const data = await res.json();
    state.speakers = data.speakers || [];
    renderSpeakers();
    updateSpeakerSelects();
  } catch (e) {
    console.error('Failed to fetch speakers:', e);
  }
}

function renderSpeakers() {
  const list = $('speakers-list');
  if (!list) return;
  if (state.speakers.length === 0) {
    list.innerHTML = '<div class="empty-state">No speakers enrolled yet.</div>';
    return;
  }
  list.innerHTML = state.speakers.map(s => `
    <div class="speaker-item" id="speaker-${s.id}">
      <div class="speaker-info">
        <div class="speaker-name">${escHtml(s.name)}</div>
        <div class="speaker-meta">${s.enrolled_at ? new Date(s.enrolled_at).toLocaleString() : ''}</div>
        <div class="speaker-badges">
          <span class="badge badge-dl" title="ECAPA-TDNN 192-dim Embedding">ECAPA-TDNN</span>
          ${s.calibrated ? '<span class="badge badge-classifier" title="Personal threshold learned from voice notes">Calibrated</span>' : ''}
          <span class="badge badge-dtw" title="Voice Notes">${s.voice_note_count || 0} Notes</span>
        </div>
      </div>
      <button class="btn-delete" data-id="${s.id}">Delete</button>
    </div>
  `).join('');

  list.querySelectorAll('.btn-delete').forEach(btn => {
    btn.addEventListener('click', () => deleteSpeaker(btn.dataset.id));
  });
}

function updateSpeakerSelects() {
  const opts = state.speakers.map(s =>
    `<option value="${s.id}">${escHtml(s.name)}</option>`
  ).join('');

  ['vn-speaker-select', 'settings-speaker-select'].forEach(id => {
    const sel = $(id);
    if (!sel) return;
    const cur = sel.value;
    sel.innerHTML = `<option value="">— Select Speaker —</option>${opts}`;
    if (cur && state.speakers.some(s => s.id === cur)) sel.value = cur;
  });
}

async function deleteSpeaker(id) {
  if (!confirm('Delete this speaker and all associated voice notes and model checkpoints?')) return;
  try {
    const res = await fetch(`${API_BASE}/speakers/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    await fetchSpeakers();
    if (state.page === 'voice-notes') {
      loadVoiceNotesSection();
    }
  } catch (e) {
    alert(`Failed to delete speaker: ${e.message}`);
  }
}

// ─── Enrollment Page ──────────────────────────────────────────────────────────
let enrollRafId   = null;
let timerInterval = null;

function initEnrollPage() {
  document.querySelectorAll('#page-enroll .duration-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('#page-enroll .duration-chips .chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.recordDuration = parseInt(chip.dataset.duration);
    });
  });

  $('record-btn')?.addEventListener('click', toggleRecord);
  $('save-btn')?.addEventListener('click', saveEnrollment);
  $('refresh-speakers-btn')?.addEventListener('click', fetchSpeakers);

  const canvas = $('enroll-waveform');
  resizeCanvas(canvas);
}

function toggleRecord() {
  if (state.recording) return;
  startRecording();
}

async function startRecording() {
  const nameInput = $('speaker-name');
  if (!nameInput.value.trim()) {
    showEnrollStatus('Please enter a speaker name.', 'error');
    nameInput.focus();
    return;
  }

  state.recording = true;
  state.recordedBlob = null;

  const btn = $('record-btn');
  btn.classList.add('recording');
  $('record-btn-text').textContent = 'Recording…';
  $('record-timer').classList.remove('hidden');
  $('save-btn').classList.add('hidden');
  $('waveform-placeholder').style.display = 'none';
  hideEnrollStatus();

  const transcriptWrap = $('enroll-transcript-wrap');
  const transcriptEl   = $('enroll-transcript');
  if (transcriptWrap && transcriptEl) {
    transcriptEl.textContent = '…';
    transcriptEl.classList.remove('has-content');
    transcriptWrap.classList.add('hidden');
  }

  startEnrollSpeechRecognition((transcript) => {
    if (!transcriptWrap || !transcriptEl) return;
    transcriptWrap.classList.remove('hidden');
    transcriptEl.textContent = transcript || '(nothing recognised yet)';
    transcriptEl.classList.toggle('has-content', !!transcript);
  });

  const startTime = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    $('timer-value').textContent = `${elapsed}s`;
    if (parseFloat(elapsed) >= state.recordDuration) {
      clearInterval(timerInterval);
    }
  }, 50);

  try {
    await state.audioCapture.init();

    const canvas = $('enroll-waveform');
    function drawLive() {
      if (!state.recording) return;
      const data = state.audioCapture.getWaveformData();
      if (data) state.visualizer.drawStaticWaveform(canvas, data);
      enrollRafId = requestAnimationFrame(drawLive);
    }
    drawLive();

    const blob = await state.audioCapture.record(state.recordDuration);
    state.recordedBlob = blob;

    state.recording = false;
    if (enrollRafId) cancelAnimationFrame(enrollRafId);
    stopEnrollSpeechRecognition();

    blob.arrayBuffer().then(buf => {
      const audioCtx = new AudioContext();
      audioCtx.decodeAudioData(buf, decoded => {
        const samples = decoded.getChannelData(0);
        state.visualizer.drawStaticWaveform(canvas, samples);
        audioCtx.close();
      });
    });

    btn.classList.remove('recording');
    $('record-btn-text').textContent = 'Re-record';
    $('record-timer').classList.add('hidden');
    $('save-btn').classList.remove('hidden');
    showEnrollStatus('✅ Recording complete! Click "Save Voiceprint" to enroll.', 'success');

  } catch (err) {
    state.recording = false;
    if (enrollRafId) cancelAnimationFrame(enrollRafId);
    clearInterval(timerInterval);
    stopEnrollSpeechRecognition();
    btn.classList.remove('recording');
    $('record-btn-text').textContent = 'Record';
    $('record-timer').classList.add('hidden');
    showEnrollStatus(`❌ ${err.message}`, 'error');
  }
}

async function saveEnrollment() {
  if (!state.recordedBlob) return;
  const name = $('speaker-name').value.trim();
  if (!name) { showEnrollStatus('Please enter a speaker name.', 'error'); return; }

  showEnrollStatus('⏳ Computing ECAPA-TDNN embedding & saving voice note…', 'info');
  $('save-btn').disabled = true;

  try {
    const wavBlob = await blobToWav(state.recordedBlob, 16000);

    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', wavBlob, `${name}.wav`);

    const res  = await fetch(`${API_BASE}/enroll`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Enrollment failed');

    showEnrollStatus(
      `✅ "${name}" enrolled! Now add 2 more recordings on the Voice Notes page and press Calibrate.`,
      'success',
    );
    $('record-btn-text').textContent = 'Record';
    $('save-btn').classList.add('hidden');
    state.recordedBlob = null;
    $('speaker-name').value = '';
    $('enroll-transcript-wrap')?.classList.add('hidden');

    await fetchSpeakers();
  } catch (err) {
    showEnrollStatus(`❌ ${err.message}`, 'error');
  } finally {
    $('save-btn').disabled = false;
  }
}

function showEnrollStatus(msg, type) {
  const el = $('enroll-status');
  if (!el) return;
  el.textContent = msg;
  el.className = `enroll-status ${type}`;
  el.classList.remove('hidden');
}
function hideEnrollStatus() {
  $('enroll-status')?.classList.add('hidden');
}

// ─── Voice Notes Page ─────────────────────────────────────────────────────────
let vnPageInited = false;
let vnRafId     = null;
let vnTimerInt   = null;

function initVoiceNotesPage() {
  if (vnPageInited) {
    loadVoiceNotesSection();
    return;
  }
  vnPageInited = true;

  const sel = $('vn-speaker-select');
  sel?.addEventListener('change', loadVoiceNotesSection);

  document.querySelectorAll('.vn-dur-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.vn-dur-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.vnDuration = parseInt(chip.dataset.duration);
    });
  });

  $('vn-record-btn')?.addEventListener('click', toggleVnRecord);
  $('vn-save-btn')?.addEventListener('click', saveVoiceNote);

  loadVoiceNotesSection();
}

async function loadVoiceNotesSection() {
  const speakerId = $('vn-speaker-select')?.value;
  const recordSec = $('vn-record-section');

  if (!speakerId) {
    if (recordSec) recordSec.classList.add('hidden');
    $('vn-model-status').innerHTML = '<div class="empty-state-sm">Select a speaker to view model status.</div>';
    $('vn-notes-list').innerHTML   = '<div class="empty-state-sm">Select a speaker to view voice notes.</div>';
    $('vn-count-badge').textContent = '0';
    return;
  }

  if (recordSec) recordSec.classList.remove('hidden');
  await Promise.all([
    fetchVoiceNotes(speakerId),
    fetchModelStatus(speakerId),
  ]);
}

async function fetchVoiceNotes(speakerId) {
  const listEl  = $('vn-notes-list');
  const badgeEl = $('vn-count-badge');

  try {
    const res  = await fetch(`${API_BASE}/speakers/${speakerId}/voice-notes`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    const notes = data.voice_notes || [];

    if (badgeEl) badgeEl.textContent = notes.length;

    if (notes.length === 0) {
      listEl.innerHTML = '<div class="empty-state-sm">No voice notes recorded yet. Add one on the left!</div>';
      return;
    }

    listEl.innerHTML = notes.map((n, idx) => `
      <div class="note-item">
        <div class="note-info">
          <span class="note-icon">🎵</span>
          <div>
            <div class="note-title">Voice Note #${idx + 1} (${n.duration_s}s)</div>
            <div class="note-meta">${new Date(n.recorded_at).toLocaleString()}</div>
          </div>
        </div>
        <button class="btn-delete" data-note-id="${n.note_id}">Delete</button>
      </div>
    `).join('');

    listEl.querySelectorAll('.btn-delete').forEach(btn => {
      btn.addEventListener('click', () => deleteVoiceNote(speakerId, btn.dataset.noteId));
    });
  } catch (e) {
    listEl.innerHTML = `<div class="empty-state-sm">Failed to load voice notes: ${escHtml(e.message)}</div>`;
  }
}

async function fetchModelStatus(speakerId) {
  const statusEl = $('vn-model-status');
  try {
    const res  = await fetch(`${API_BASE}/speakers/${speakerId}/model-status`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    const minNotes     = data.min_notes_for_calibration || 3;
    const count        = data.voice_note_count || 0;
    const progress     = Math.min(100, Math.round((count / minNotes) * 100));
    const isCalibrated = data.calibrated;
    const isReady      = data.can_calibrate;
    const threshold    = data.similarity_threshold;

    let hint;
    if (!isReady) {
      hint = `Add ${minNotes - count} more voice note(s) to unlock calibration.`;
    } else if (data.calibration_stale) {
      hint = 'Voice notes changed since the last calibration — recalibrate to update your threshold.';
    } else if (isCalibrated) {
      hint = calibrationSummary(data.calibration);
    } else {
      hint = `Ready to calibrate with ${count} voice notes.`;
    }

    statusEl.innerHTML = `
      <div class="model-status-header">
        <span class="model-status-badge ${isCalibrated ? 'trained' : 'untrained'}">
          ${isCalibrated ? '✅ Calibrated' : '⚠️ Default Threshold'}
        </span>
        <span class="note-meta">${count} / ${minNotes} notes · threshold ${threshold.toFixed(2)}</span>
      </div>

      <div class="model-progress-bar" title="${progress}% of minimum required voice notes">
        <div class="model-progress-fill" style="width: ${progress}%"></div>
      </div>

      <div class="fine-tune-box">
        <button id="trigger-fine-tune-btn" class="btn-fine-tune" ${isReady ? '' : 'disabled'}>
          <span>🎯</span>
          <span>${isCalibrated ? 'Recalibrate' : 'Calibrate Personal Threshold'}</span>
        </button>
        <div id="fine-tune-status-msg" class="fine-tune-status ${data.calibration?.warning ? 'error' : 'info'}">
          ${escHtml(hint)}
        </div>
      </div>
    `;

    $('trigger-fine-tune-btn')?.addEventListener('click', () => runCalibration(speakerId));
  } catch (e) {
    statusEl.innerHTML = `<div class="empty-state-sm">Could not fetch model status: ${escHtml(e.message)}</div>`;
  }
}

function calibrationSummary(c) {
  if (!c) return '';
  const pct = v => `${Math.round(v * 100)}%`;
  let s = `Threshold ${c.threshold.toFixed(2)} · your held-out voice averaged ` +
          `${c.genuine_mean.toFixed(2)} and ${pct(c.genuine_accept_rate)} of windows passed.`;
  if (c.impostor_reject_rate != null) {
    s += ` Other enrolled voices rejected: ${pct(c.impostor_reject_rate)}.`;
  }
  if (c.warning) s += ` ⚠️ ${c.warning}`;
  return s;
}

async function runCalibration(speakerId) {
  const btn = $('trigger-fine-tune-btn');
  const msg = $('fine-tune-status-msg');

  if (btn) btn.disabled = true;
  if (msg) {
    msg.className = 'fine-tune-status info';
    msg.textContent = '🎯 Scoring each voice note against the others…';
  }

  try {
    const res  = await fetch(`${API_BASE}/speakers/${speakerId}/calibrate`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Calibration failed');

    await Promise.all([
      fetchSpeakers(),
      fetchModelStatus(speakerId),
    ]);
  } catch (e) {
    if (msg) {
      msg.className = 'fine-tune-status error';
      msg.textContent = `❌ Calibration failed: ${e.message}`;
    }
    if (btn) btn.disabled = false;
  }
}

function toggleVnRecord() {
  if (state.vnRecording) return;
  startVnRecording();
}

async function startVnRecording() {
  const speakerId = $('vn-speaker-select')?.value;
  if (!speakerId) return;

  state.vnRecording = true;
  state.vnRecordedBlob = null;

  const btn = $('vn-record-btn');
  btn.classList.add('recording');
  $('vn-record-btn-text').textContent = 'Recording…';
  $('vn-record-timer').classList.remove('hidden');
  $('vn-save-btn').classList.add('hidden');
  $('vn-waveform-placeholder').style.display = 'none';
  hideVnStatus();

  const startTime = Date.now();
  vnTimerInt = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    $('vn-timer-value').textContent = `${elapsed}s`;
    if (parseFloat(elapsed) >= state.vnDuration) {
      clearInterval(vnTimerInt);
    }
  }, 50);

  try {
    await state.audioCapture.init();

    const canvas = $('vn-waveform');
    resizeCanvas(canvas);

    function drawLive() {
      if (!state.vnRecording) return;
      const data = state.audioCapture.getWaveformData();
      if (data) state.visualizer.drawStaticWaveform(canvas, data);
      vnRafId = requestAnimationFrame(drawLive);
    }
    drawLive();

    const blob = await state.audioCapture.record(state.vnDuration);
    state.vnRecordedBlob = blob;

    state.vnRecording = false;
    if (vnRafId) cancelAnimationFrame(vnRafId);

    blob.arrayBuffer().then(buf => {
      const audioCtx = new AudioContext();
      audioCtx.decodeAudioData(buf, decoded => {
        const samples = decoded.getChannelData(0);
        state.visualizer.drawStaticWaveform(canvas, samples);
        audioCtx.close();
      });
    });

    btn.classList.remove('recording');
    $('vn-record-btn-text').textContent = 'Re-record';
    $('vn-record-timer').classList.add('hidden');
    $('vn-save-btn').classList.remove('hidden');
    showVnStatus('✅ Recording complete! Click "Save Voice Note" to save.', 'success');

  } catch (err) {
    state.vnRecording = false;
    if (vnRafId) cancelAnimationFrame(vnRafId);
    clearInterval(vnTimerInt);
    btn.classList.remove('recording');
    $('vn-record-btn-text').textContent = 'Record';
    $('vn-record-timer').classList.add('hidden');
    showVnStatus(`❌ ${err.message}`, 'error');
  }
}

async function saveVoiceNote() {
  const speakerId = $('vn-speaker-select')?.value;
  if (!speakerId || !state.vnRecordedBlob) return;

  showVnStatus('⏳ Processing audio & extracting ECAPA-TDNN embedding…', 'info');
  $('vn-save-btn').disabled = true;

  try {
    const wavBlob = await blobToWav(state.vnRecordedBlob, 16000);
    const formData = new FormData();
    formData.append('file', wavBlob, 'note.wav');

    const res  = await fetch(`${API_BASE}/speakers/${speakerId}/voice-notes`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Save voice note failed');

    showVnStatus('✅ Voice note saved successfully!', 'success');
    $('vn-record-btn-text').textContent = 'Record';
    $('vn-save-btn').classList.add('hidden');
    state.vnRecordedBlob = null;

    await Promise.all([
      fetchSpeakers(),
      loadVoiceNotesSection(),
    ]);
  } catch (err) {
    showVnStatus(`❌ ${err.message}`, 'error');
  } finally {
    $('vn-save-btn').disabled = false;
  }
}

async function deleteVoiceNote(speakerId, noteId) {
  if (!confirm('Delete this voice note?')) return;
  try {
    const res = await fetch(`${API_BASE}/speakers/${speakerId}/voice-notes/${noteId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    await Promise.all([
      fetchSpeakers(),
      loadVoiceNotesSection(),
    ]);
  } catch (e) {
    alert(`Failed to delete voice note: ${e.message}`);
  }
}

function showVnStatus(msg, type) {
  const el = $('vn-status');
  if (!el) return;
  el.textContent = msg;
  el.className = `enroll-status ${type}`;
  el.classList.remove('hidden');
}
function hideVnStatus() {
  $('vn-status')?.classList.add('hidden');
}

// ─── Door Page ────────────────────────────────────────────────────────────────
let monitorPageInited = false;

function initMonitorPage() {
  refreshLockStatus();
  if (monitorPageInited) return;
  monitorPageInited = true;

  const fftCanvas   = $('fft-canvas');
  const levelCanvas = $('level-meter');
  resizeCanvas(fftCanvas);
  resizeCanvas(levelCanvas);

  state.visualizer.mount({
    fftCanvas,
    levelCanvas,
    getAudioData: () => ({
      waveform:  state.audioCapture.getWaveformData(),
      frequency: state.audioCapture.getFrequencyData(),
    })
  });

  $('listen-btn')?.addEventListener('click', toggleListening);
  $('lock-now-btn')?.addEventListener('click', () => {
    fetch(`${API_BASE}/lock/lock`, { method: 'POST' }).catch(e => console.error(e));
  });
  $('clear-log-btn')?.addEventListener('click', () => {
    $('match-log').innerHTML = '<div class="empty-state-sm">Log cleared.</div>';
  });

  loadAccessLog();
}

async function loadAccessLog() {
  try {
    const data = await (await fetch(`${API_BASE}/access-log`)).json();
    const el = $('access-log');
    el.innerHTML = '<div class="empty-state-sm">No attempts yet.</div>';
    // events arrive newest-first; addAccessEntry prepends
    [...data.events].reverse().forEach(addAccessEntry);
  } catch (e) {
    console.warn('Could not load access log:', e);
  }
}

async function refreshLockStatus() {
  try {
    renderLock(await (await fetch(`${API_BASE}/lock/status`)).json(), false);
  } catch (_) {}
}

async function toggleListening() {
  if (state.listening) stopListening();
  else await startListening();
}

async function startListening() {
  if (state.speakers.length === 0) {
    addLogEntry({ type: 'error', message: 'No speakers enrolled yet — enroll someone first.' });
    return;
  }

  try {
    await state.audioCapture.init();
  } catch (err) {
    addLogEntry({ type: 'error', message: `Mic error: ${err.message}` });
    return;
  }

  state.ws = new WebSocket(`${WS_BASE}/listen`);

  state.ws.onopen = () => {
    state.ws.send(JSON.stringify({ type: 'start' }));
    state.listening = true;
    updateListenBtn(true);
    state.visualizer.start();

    $('live-transcript-bar')?.classList.remove('hidden');
    updateLiveTranscript('', false);
    startDoorSpeechRecognition();

    const nativeSR = state.audioCapture.audioContext?.sampleRate ?? 16000;
    state.audioCapture.startStreaming((chunk) => {
      if (state.ws?.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
          type: 'audio_chunk',
          data: float32ToBase64(chunk),
          sample_rate: nativeSR,
        }));
      }
    });
  };

  state.ws.onmessage = (evt) => handleWsMessage(JSON.parse(evt.data));
  state.ws.onclose   = () => stopListening(false);
  state.ws.onerror   = (e) => {
    console.error('WebSocket error', e);
    stopListening(false);
  };
}

function stopListening(sendStop = true) {
  if (sendStop && state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'stop' }));
  }
  state.ws?.close();
  state.ws = null;
  state.listening = false;
  state.audioCapture.stopStreaming();
  state.visualizer.stop();
  stopDoorSpeechRecognition();
  $('live-transcript-bar')?.classList.add('hidden');
  $('challenge-box')?.classList.add('hidden');
  $('matching-mode-wrap')?.classList.add('hidden');
  updateListenBtn(false);
  updateVAD(false);
  if (!state.doorUnlocked) $('lock-detail').textContent = 'Press “Start Listening”, then say the phrase.';
}

function updateListenBtn(listening) {
  const btn  = $('listen-btn');
  const icon = $('listen-btn-icon');
  const text = $('listen-btn-text');
  if (!btn) return;
  btn.classList.toggle('listening', listening);
  if (icon) icon.textContent = listening ? '■' : '▶';
  if (text) text.textContent = listening ? 'Stop Listening' : 'Start Listening';
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'status':
      if (!msg.listening) break;
      state.requireChallenge = msg.require_challenge;
      updateEnrolledBadge(msg.speakers);
      if (msg.require_challenge && !(window.SpeechRecognition || window.webkitSpeechRecognition)) {
        addLogEntry({
          type: 'error',
          message: 'This browser has no speech recognition, so the phrase can never be heard. ' +
                   'Use Chrome or Edge, or turn the challenge phrase off in Settings.',
        });
      }
      break;

    case 'challenge':
      showChallenge(msg.phrase);
      break;

    case 'phrase_heard':
      state.phraseHeard = true;
      updateLiveTranscript(state.lastTranscript, true);
      $('lock-detail').textContent = 'Phrase heard — checking voice…';
      break;

    case 'level':
      updateVAD(msg.voice);
      state.visualizer.setLevel(msg.rms);
      break;

    case 'match':
      updateScoreGauges(msg);
      addLogEntry(msg);
      if (msg.matched && msg.best_speaker !== state.spectrumSpeaker) {
        state.spectrumSpeaker = msg.best_speaker;
        loadSpeakerSpectrum(msg.best_speaker);
      }
      break;

    case 'lock':
      renderLock(msg, true);
      break;

    case 'access':
      addAccessEntry(msg.event);
      if (msg.event.result === 'denied') showDenied(msg.event);
      break;

    case 'error':
      addLogEntry({ type: 'error', message: msg.message });
      break;
  }
}

function updateEnrolledBadge(speakers) {
  const wrap  = $('matching-mode-wrap');
  const badge = $('matching-mode-badge');
  if (!wrap || !badge) return;
  wrap.classList.remove('hidden');
  badge.textContent = `${speakers.length} enrolled: ${speakers.map(s => s.name).join(', ')}`;
}

function showChallenge(phrase) {
  state.phraseHeard = false;
  updateLiveTranscript('', false);
  const box = $('challenge-box');
  if (!phrase) {
    box.classList.add('hidden');
    if (!state.doorUnlocked) $('lock-detail').textContent = 'Speak to unlock.';
    return;
  }
  $('challenge-phrase').textContent = phrase;
  box.classList.remove('hidden');
  if (!state.doorUnlocked) $('lock-detail').textContent = 'Say the phrase below in your normal voice.';
}

function renderLock(s, withSound) {
  const unlocked = !s.locked;
  const opened   = unlocked && !state.doorUnlocked;
  state.doorUnlocked = unlocked;

  $('door-panel').className = `door-panel ${unlocked ? 'unlocked' : 'locked'}`;
  $('lock-icon').textContent  = unlocked ? '🔓' : '🔒';
  $('lock-state').textContent = unlocked ? 'UNLOCKED' : 'LOCKED';
  $('lock-now-btn').classList.toggle('hidden', !unlocked);

  clearInterval(state.relockTimer);
  if (unlocked) {
    const who   = s.unlocked_by;
    const until = Date.now() + (s.relock_in ?? 0) * 1000;
    const tick  = () => {
      const left = Math.max(0, Math.ceil((until - Date.now()) / 1000));
      $('lock-detail').textContent =
        `Welcome, ${who?.name ?? ''}! (similarity ${who?.score?.toFixed(2) ?? '—'}) — locks again in ${left} s`;
    };
    tick();
    state.relockTimer = setInterval(tick, 250);
    if (opened && withSound) playTone('granted');
  } else {
    $('lock-detail').textContent = !state.listening ? 'Press “Start Listening”, then say the phrase.'
      : state.requireChallenge ? 'Say the phrase below in your normal voice.'
      : 'Speak to unlock.';
  }
}

function showDenied(event) {
  const panel = $('door-panel');
  panel.classList.add('denied');
  const why = event.closest
    ? `Voice not recognised (closest: ${event.closest}, ${event.score.toFixed(2)} < ${event.threshold.toFixed(2)})`
    : 'No clear speech heard — speak a little louder';
  $('lock-detail').textContent = `⛔ Access denied. ${why}`;
  playTone('denied');
  setTimeout(() => panel.classList.remove('denied'), 1500);
}

function addAccessEntry(e) {
  const log = $('access-log');
  if (!log) return;
  log.querySelector('.empty-state-sm')?.remove();

  const entry = document.createElement('div');
  entry.className = `log-entry ${e.result === 'granted' ? 'match' : 'denied'}`;
  const text = e.result === 'granted'
    ? `🔓 ${escHtml(e.name)} — similarity ${e.score.toFixed(2)}`
    : `⛔ ${escHtml(e.reason)}${e.closest ? ` — closest ${escHtml(e.closest)} ${e.score.toFixed(2)}` : ''}`;
  entry.innerHTML = `
    <span class="log-time">${new Date(e.time).toLocaleTimeString()}</span>
    <span class="log-scores">${text}</span>
  `;
  log.prepend(entry);
  while (log.children.length > 50) log.removeChild(log.lastChild);
}

function updateVAD(voice) {
  const indicator = $('vad-indicator');
  const label     = $('vad-label');
  if (!indicator || !label) return;
  indicator.classList.toggle('voice', !!voice);
  label.textContent = voice ? 'Voice Detected' : 'Silence';
}

function updateScoreGauges(msg) {
  setGauge('dl', msg.similarity_score, msg.similarity_score.toFixed(3));
  setGauge('classifier', msg.threshold, msg.threshold.toFixed(3));
  setGauge(
    'fusion',
    msg.recent_matches / msg.matches_required,
    `${msg.recent_matches} / ${msg.matches_required}${msg.leader ? ` — ${msg.leader}` : ''}`,
  );

  const label = $('match-mode-label');
  if (label) {
    label.textContent = !msg.phrase_required ? 'No phrase needed'
                      : msg.phrase_ok        ? 'Phrase ✓'
                      :                        'Waiting for phrase';
  }

  const who = $('live-speaker');
  if (who) {
    who.textContent = msg.matched
      ? `🗣 ${msg.best_name}`
      : `❓ Unknown voice (closest: ${msg.best_name})`;
    who.className = `live-speaker ${msg.matched ? 'known' : 'unknown'}`;
  }

  const all = $('all-scores');
  if (all) {
    all.innerHTML = Object.entries(msg.scores)
      .sort((a, b) => b[1] - a[1])
      .map(([name, v]) => `
        <div class="score-row">
          <span>${escHtml(name)}</span>
          <span class="score-row-bar"><span style="width:${Math.round(Math.max(0, Math.min(1, v)) * 100)}%"></span></span>
          <span class="score-row-val">${v.toFixed(2)}</span>
        </div>`)
      .join('');
  }
}

function setGauge(name, value, text) {
  const pct = `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
  const bar = $(`${name}-score-bar`);
  const val = $(`${name}-score-val`);
  if (bar) bar.style.width = pct;
  if (val) val.textContent = text;
}

function addLogEntry(msg) {
  const log = $('match-log');
  if (!log) return;
  log.querySelector('.empty-state-sm')?.remove();

  const matched = msg.matched || false;
  const entry   = document.createElement('div');
  entry.className = `log-entry ${matched ? 'match' : 'no-match'}`;

  let text;
  if (msg.type === 'error') {
    text = `⚠️ ${escHtml(msg.message)}`;
  } else {
    const cmp = matched ? '≥' : '<';
    text = `${matched ? '✅' : '·'} ${escHtml(msg.best_name)} ${msg.similarity_score.toFixed(3)} ${cmp} ` +
           `${msg.threshold.toFixed(2)}  [${msg.recent_matches}/${msg.matches_required} recent]`;
  }

  entry.innerHTML = `
    <span class="log-time">${new Date().toLocaleTimeString()}</span>
    <span class="log-scores">${text}</span>
  `;
  log.prepend(entry);
  while (log.children.length > 30) log.removeChild(log.lastChild);
}

/** Short feedback sounds: rising chime when granted, low buzz when denied. */
function playTone(kind) {
  try {
    const ctx  = new AudioContext();
    const gain = ctx.createGain();
    gain.gain.value = 0.25;
    gain.connect(ctx.destination);
    const notes = kind === 'granted'
      ? [[660, 0.12, 'sine'], [880, 0.12, 'sine'], [1320, 0.2, 'sine']]
      : [[180, 0.25, 'square'], [140, 0.35, 'square']];
    let t = ctx.currentTime;
    for (const [freq, dur, type] of notes) {
      const osc = ctx.createOscillator();
      osc.type = type;
      osc.frequency.value = freq;
      osc.connect(gain);
      osc.start(t);
      osc.stop(t + dur);
      t += dur + 0.03;
    }
    setTimeout(() => ctx.close(), (t - ctx.currentTime) * 1000 + 200);
  } catch (_) {}
}

// ─── Settings Page ────────────────────────────────────────────────────────────
let settingsPageInited = false;

function initSettingsPage() {
  checkHealth().then(data => {
    if (!data) return;
    const model = data.encoder_loaded
      ? '<span class="info-val ok">ECAPA-TDNN (loaded)</span>'
      : `<span class="info-val">${data.model_info?.encoder_error ? 'Load failed — see server log' : 'Loading…'}</span>`;
    $('system-info').innerHTML = `
      <div class="info-row"><span class="info-key">Backend Version</span><span class="info-val ok">VoiceGuard v3</span></div>
      <div class="info-row"><span class="info-key">Speaker Model</span>${model}</div>
      <div class="info-row"><span class="info-key">Embedding Dimension</span><span class="info-val">192-dim</span></div>
      <div class="info-row"><span class="info-key">Analysis Window</span><span class="info-val">1.5 s (0.75 s hop)</span></div>
      <div class="info-row"><span class="info-key">Unlock Rule</span><span class="info-val">Same person in 2 of last 3 windows + phrase</span></div>
      <div class="info-row"><span class="info-key">API Base</span><span class="info-val">${API_BASE}</span></div>
    `;
  });
  loadDoorConfig();

  if (settingsPageInited) return;
  settingsPageInited = true;

  $('settings-speaker-select')?.addEventListener('change', loadSpeakerSettings);
  $('similarity-threshold')?.addEventListener('input', () => {
    $('similarity-threshold-val').textContent = parseFloat($('similarity-threshold').value).toFixed(2);
  });
  $('save-settings-btn')?.addEventListener('click', saveSpeakerSettings);

  $('unlock-seconds')?.addEventListener('input', () => {
    $('unlock-seconds-val').textContent = `${$('unlock-seconds').value} s`;
  });
  $('save-door-btn')?.addEventListener('click', saveDoorConfig);
}

async function loadDoorConfig() {
  try {
    const cfg = await (await fetch(`${API_BASE}/config`)).json();
    $('require-challenge').checked = cfg.require_challenge;
    $('unlock-seconds').value = cfg.unlock_seconds;
    $('unlock-seconds-val').textContent = `${cfg.unlock_seconds} s`;
  } catch (e) {
    console.warn('Could not load door config:', e);
  }
}

async function saveDoorConfig() {
  const msgEl = $('door-saved-msg');
  try {
    const res = await fetch(`${API_BASE}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        require_challenge: $('require-challenge').checked,
        unlock_seconds: parseFloat($('unlock-seconds').value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    msgEl.textContent = state.listening
      ? '✅ Saved — stop and start listening to apply.'
      : '✅ Door settings saved!';
  } catch (e) {
    msgEl.textContent = `❌ ${e.message}`;
  }
  msgEl.classList.remove('hidden');
  setTimeout(() => msgEl.classList.add('hidden'), 3000);
}

async function loadSpeakerSettings() {
  const id = $('settings-speaker-select').value;
  if (!id) { $('threshold-controls')?.classList.add('hidden'); return; }

  const speaker = state.speakers.find(s => s.id === id);
  if (!speaker) return;

  const simVal = speaker.similarity_threshold ?? 0.50;
  $('similarity-threshold').value = simVal;
  $('similarity-threshold-val').textContent = simVal.toFixed(2);

  $('threshold-controls')?.classList.remove('hidden');
  $('settings-saved-msg')?.classList.add('hidden');
}

async function saveSpeakerSettings() {
  const id = $('settings-speaker-select').value;
  if (!id) return;

  const msgEl = $('settings-saved-msg');
  try {
    const res = await fetch(`${API_BASE}/settings/${id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ similarity_threshold: parseFloat($('similarity-threshold').value) }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    msgEl.textContent = '✅ Settings saved!';
    await fetchSpeakers();
  } catch (e) {
    msgEl.textContent = `❌ ${e.message}`;
  }
  msgEl.classList.remove('hidden');
  setTimeout(() => msgEl.classList.add('hidden'), 3000);
}

// ─── Speech Recognition (Web Speech API) ─────────────────────────────────────
function updateLiveTranscript(text, phraseHeard) {
  const el    = $('live-transcript-text');
  const badge = $('keyword-match-badge');
  if (!el) return;

  el.textContent = text ? `"${text}"` : 'Listening for speech…';
  el.classList.toggle('has-text', !!text);
  badge?.classList.toggle('hidden', !phraseHeard);
}

/**
 * Stream what the browser hears (interim + final text) to the backend,
 * which checks it against the current challenge phrase.
 */
function startDoorSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;

  stopDoorSpeechRecognition();

  const recog = new SpeechRecognition();
  recog.continuous      = true;
  recog.interimResults  = true;
  recog.lang            = 'en-US';
  recog.maxAlternatives = 1;

  recog.onresult = (event) => {
    let interim = '';
    let finalText = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript.trim().toLowerCase();
      if (event.results[i].isFinal) finalText += t + ' ';
      else interim = t;
    }

    const text = (finalText || interim).trim();
    if (!text) return;
    state.lastTranscript = text;
    updateLiveTranscript(text, state.phraseHeard);

    if (state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: 'transcript', text }));
    }

    if (finalText.trim()) {
      const log = $('match-log');
      log?.querySelector('.empty-state-sm')?.remove();
      const entry = document.createElement('div');
      entry.className = 'log-entry transcript-entry';
      entry.innerHTML = `
        <span class="log-time">${new Date().toLocaleTimeString()}</span>
        <span class="log-scores" style="color:var(--c-accent);">🗣 "${escHtml(finalText.trim())}"</span>
      `;
      log?.prepend(entry);
      while (log?.children.length > 30) log.removeChild(log.lastChild);
    }
  };

  recog.onerror = (e) => {
    if (e.error !== 'no-speech') console.warn('[STT] error:', e.error);
    if (e.error === 'not-allowed') {
      addLogEntry({ type: 'error', message: 'Speech recognition blocked — allow microphone access for this page.' });
    }
  };

  // Chrome stops recognition after a while; restart it while listening
  recog.onend = () => {
    if (state.listening && state.speechRecognizer === recog) {
      setTimeout(() => { try { recog.start(); } catch (_) {} }, 250);
    }
  };

  try {
    recog.start();
    state.speechRecognizer = recog;
  } catch (e) {
    console.warn('[STT] could not start:', e);
  }
}

function stopDoorSpeechRecognition() {
  const recog = state.speechRecognizer;
  state.speechRecognizer = null;
  if (recog) {
    try { recog.abort(); } catch (_) {}
  }
}

function startEnrollSpeechRecognition(onTranscript) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;

  stopEnrollSpeechRecognition();

  const recog = new SpeechRecognition();
  recog.continuous      = true;
  recog.interimResults  = true;
  recog.lang            = 'en-US';
  recog.maxAlternatives = 1;

  let accumulated = '';
  recog.onresult = (event) => {
    let text = '';
    for (let i = 0; i < event.results.length; i++) {
      text += event.results[i][0].transcript;
    }
    accumulated = text.trim();
    onTranscript(accumulated);
  };

  recog.onerror = () => {};
  recog.onend   = () => {
    if (state.recording) {
      try { recog.start(); } catch(e) {}
    }
  };

  try {
    recog.start();
    state.enrollRecognizer = recog;
  } catch(e) {}
}

function stopEnrollSpeechRecognition() {
  if (state.enrollRecognizer) {
    try { state.enrollRecognizer.abort(); } catch(e) {}
    state.enrollRecognizer = null;
  }
}

// ─── Utilities ────────────────────────────────────────────────────────────────
async function blobToWav(blob, targetSr = 16000) {
  const arrayBuf = await blob.arrayBuffer();
  let decoded;
  try {
    decoded = await new AudioContext().decodeAudioData(arrayBuf);
  } catch (e) {
    throw new Error(`Audio decode failed: ${e.message}`);
  }

  const numFrames = Math.ceil(decoded.duration * targetSr);
  const offCtx    = new OfflineAudioContext(1, numFrames, targetSr);
  const srcNode   = offCtx.createBufferSource();
  srcNode.buffer  = decoded;
  srcNode.connect(offCtx.destination);
  srcNode.start(0);
  const rendered  = await offCtx.startRendering();
  const samples   = rendered.getChannelData(0);

  return encodeWAV(samples, targetSr);
}

function encodeWAV(samples, sr) {
  const numSamples = samples.length;
  const byteRate   = sr * 2;
  const blockAlign = 2;
  const dataBytes  = numSamples * 2;
  const buffer     = new ArrayBuffer(44 + dataBytes);
  const view       = new DataView(buffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeStr(0, 'RIFF');
  view.setUint32(4,  36 + dataBytes, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16,         true);
  view.setUint16(20, 1,          true);
  view.setUint16(22, 1,          true);
  view.setUint32(24, sr,         true);
  view.setUint32(28, byteRate,   true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16,         true);
  writeStr(36, 'data');
  view.setUint32(40, dataBytes,  true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

async function loadSpeakerSpectrum(speakerId) {
  if (!speakerId) { state.visualizer.setReferenceSpectrum(null); return; }
  try {
    const res = await fetch(`${API_BASE}/speakers/${speakerId}/spectrum`);
    if (!res.ok) return;
    const data = await res.json();
    state.visualizer.setReferenceSpectrum(data.spectrum);
  } catch (e) {
    console.warn('[Monitor] Could not load reference spectrum:', e);
  }
}

function float32ToBase64(float32Array) {
  const bytes = new Uint8Array(float32Array.buffer);
  let binary  = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function resizeCanvas(canvas) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  initEnrollPage();
  await checkHealth();
  await fetchSpeakers();

  setInterval(checkHealth, 10000);

  window.addEventListener('resize', () => {
    resizeCanvas($('enroll-waveform'));
    resizeCanvas($('fft-canvas'));
    resizeCanvas($('level-meter'));
  });
}

init();
