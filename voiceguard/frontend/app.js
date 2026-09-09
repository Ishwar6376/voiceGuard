/**
 * app.js — VoiceGuard v2 Main Frontend Application
 * Handles: routing, enrollment, incremental voice notes, fine-tuning,
 * live WebSocket monitoring with ECAPA-TDNN + classifier head, and settings.
 */

import { AudioCapture } from './audio_capture.js';
import { Visualizer }   from './visualizer.js';

const API_BASE = `http://${window.location.hostname}:8000`;
const WS_BASE  = `ws://${window.location.hostname}:8000`;

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  page:            'enroll',
  speakers:        [],
  recording:       false,
  recordDuration:  3,
  recordedBlob:    null,
  vnRecording:     false,
  vnDuration:      3,
  vnRecordedBlob:  null,
  listening:       false,
  alarmActive:     false,
  ws:              null,
  audioCapture:    new AudioCapture(),
  visualizer:      new Visualizer(),
  speechRecognizer: null,   // Web Speech API instance for monitor
  enrollRecognizer: null,   // Web Speech API instance for enroll recording
  activeKeyword:   '',      // keyword configured for current speaker
  keywordGateOpen: false,   // whether keyword was just detected
  keywordGateTimer: null,  // auto-close timer handle
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
          ${s.has_model ? '<span class="badge badge-classifier" title="Fine-tuned MLP model ready">Fine-Tuned</span>' : ''}
          <span class="badge badge-dtw" title="Voice Notes">${s.voice_note_count || 0} Notes</span>
          ${s.keyword ? `<span class="badge badge-keyword" title="Trigger phrase">🔑 ${escHtml(s.keyword)}</span>` : ''}
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

  ['vn-speaker-select', 'monitor-speaker-select', 'settings-speaker-select'].forEach(id => {
    const sel = $(id);
    if (!sel) return;
    const cur = sel.value;
    sel.innerHTML = `<option value="">— Select Speaker —</option>${opts}`;
    if (cur && state.speakers.some(s => s.id === cur)) sel.value = cur;
  });

  const listenBtn = $('listen-btn');
  if (listenBtn) {
    listenBtn.disabled = !$('monitor-speaker-select')?.value && state.speakers.length === 0;
  }
}

async function deleteSpeaker(id) {
  if (!confirm('Delete this speaker and all associated voice notes and model checkpoints?')) return;
  try {
    await fetch(`${API_BASE}/speakers/${id}`, { method: 'DELETE' });
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
  const keyword = $('speaker-keyword')?.value.trim() ?? '';

  showEnrollStatus('⏳ Computing ECAPA-TDNN embedding & saving voice note…', 'info');
  $('save-btn').disabled = true;

  try {
    const wavBlob = await blobToWav(state.recordedBlob, 16000);

    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', wavBlob, `${name}.wav`);
    if (keyword) formData.append('keyword', keyword);

    const res  = await fetch(`${API_BASE}/enroll`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Enrollment failed');

    const kwMsg = keyword ? ` Keyword: "${keyword}"` : '';
    showEnrollStatus(`✅ "${name}" enrolled successfully with ECAPA-TDNN embedding!${kwMsg}`, 'success');
    $('record-btn-text').textContent = 'Record';
    $('save-btn').classList.add('hidden');
    state.recordedBlob = null;
    $('speaker-name').value = '';
    if ($('speaker-keyword')) $('speaker-keyword').value = '';
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
    const notes = data.notes || [];

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
            <div class="note-title">Voice Note #${idx + 1} (${n.duration}s)</div>
            <div class="note-meta">${new Date(n.created_at).toLocaleString()}</div>
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

    const minNotes  = data.min_notes_required || 5;
    const count     = data.note_count || 0;
    const progress  = Math.min(100, Math.round((count / minNotes) * 100));
    const isTrained = data.model_exists;
    const isReady   = data.ready_for_finetune;

    statusEl.innerHTML = `
      <div class="model-status-header">
        <span class="model-status-badge ${isTrained ? 'trained' : 'untrained'}">
          ${isTrained ? '✅ Model Fine-Tuned' : '⚠️ Embedding Matching Only'}
        </span>
        <span class="note-meta">${count} / ${minNotes} notes required</span>
      </div>

      <div class="model-progress-bar" title="${progress}% of minimum required voice notes">
        <div class="model-progress-fill" style="width: ${progress}%"></div>
      </div>

      <div class="fine-tune-box">
        <button id="trigger-fine-tune-btn" class="btn-fine-tune" ${isReady ? '' : 'disabled'}>
          <span>⚡</span>
          <span>${isTrained ? 'Re-Train Personal Classifier' : 'Fine-Tune Personal Classifier'}</span>
        </button>
        <div id="fine-tune-status-msg" class="fine-tune-status info ${isReady ? '' : 'hidden'}">
          ${isReady
            ? `Ready to fine-tune! You have ${count} voice notes.`
            : `Add ${minNotes - count} more voice note(s) to unlock personal model fine-tuning.`
          }
        </div>
      </div>
    `;

    $('trigger-fine-tune-btn')?.addEventListener('click', () => runFineTuning(speakerId));
  } catch (e) {
    statusEl.innerHTML = `<div class="empty-state-sm">Could not fetch model status: ${escHtml(e.message)}</div>`;
  }
}

async function runFineTuning(speakerId) {
  const btn = $('trigger-fine-tune-btn');
  const msg = $('fine-tune-status-msg');

  if (btn) btn.disabled = true;
  if (msg) {
    msg.className = 'fine-tune-status info';
    msg.classList.remove('hidden');
    msg.textContent = '⚡ Fine-tuning personal MLP classifier head on ECAPA-TDNN embeddings…';
  }

  try {
    const res  = await fetch(`${API_BASE}/speakers/${speakerId}/fine-tune`, { method: 'POST' });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Fine-tuning failed');

    if (msg) {
      msg.className = 'fine-tune-status success';
      msg.textContent = `✅ ${data.status}! Final Loss: ${data.loss?.toFixed(4) ?? '0.0000'}`;
    }

    await Promise.all([
      fetchSpeakers(),
      fetchModelStatus(speakerId),
    ]);
  } catch (e) {
    if (msg) {
      msg.className = 'fine-tune-status error';
      msg.textContent = `❌ Fine-tuning failed: ${e.message}`;
    }
  } finally {
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
    formData.append('audio', wavBlob, `note.wav`);

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
    await fetch(`${API_BASE}/speakers/${speakerId}/voice-notes/${noteId}`, { method: 'DELETE' });
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

// ─── Monitor Page ─────────────────────────────────────────────────────────────
let monitorPageInited = false;

function initMonitorPage() {
  const speakerSel = $('monitor-speaker-select');
  if ($('listen-btn')) {
    $('listen-btn').disabled = !speakerSel?.value && state.speakers.length === 0;
  }

  if (monitorPageInited) return;
  monitorPageInited = true;

  speakerSel?.addEventListener('change', () => {
    const id = speakerSel.value;
    if ($('listen-btn')) $('listen-btn').disabled = !id;
    if (id) {
      loadSpeakerSpectrum(id);
      const speaker = state.speakers.find(s => s.id === id);
      updateKeywordBadge(speaker?.keyword || '');
    } else {
      state.visualizer.setReferenceSpectrum(null);
      updateKeywordBadge('');
    }
  });

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
  $('clear-log-btn')?.addEventListener('click', () => {
    $('match-log').innerHTML = '<div class="empty-state-sm">Log cleared.</div>';
  });
}

async function toggleListening() {
  if (state.listening) {
    stopListening();
    return;
  }
  const speakerId = $('monitor-speaker-select').value;
  if (!speakerId) {
    $('monitor-speaker-select').focus();
    $('monitor-speaker-select').style.borderColor = 'var(--c-danger)';
    setTimeout(() => { $('monitor-speaker-select').style.borderColor = ''; }, 2000);
    return;
  }
  await startListening();
}

async function startListening() {
  const speakerId = $('monitor-speaker-select').value;
  if (!speakerId) return;

  const speaker = state.speakers.find(s => s.id === speakerId);
  state.activeKeyword = (speaker?.keyword || '').trim().toLowerCase();

  try {
    await state.audioCapture.init();
  } catch (err) {
    addLogEntry({ type: 'error', message: `Mic error: ${err.message}` });
    return;
  }

  await loadSpeakerSpectrum(speakerId);

  state.ws = new WebSocket(`${WS_BASE}/ws/monitor?speaker_id=${encodeURIComponent(speakerId)}`);

  state.ws.onopen = () => {
    state.listening = true;
    updateListenBtn(true);
    state.visualizer.start();

    $('live-transcript-bar')?.classList.remove('hidden');
    updateLiveTranscript('', false);

    startMonitorSpeechRecognition(state.activeKeyword);

    const nativeSR = state.audioCapture.audioContext?.sampleRate ?? 16000;

    state.audioCapture.startStreaming((chunk) => {
      if (state.ws?.readyState === WebSocket.OPEN) {
        const b64 = float32ToBase64(chunk);
        state.ws.send(JSON.stringify({
          type: 'audio_chunk',
          data: b64,
          sample_rate: nativeSR
        }));
      }
    });
  };

  state.ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    handleWsMessage(msg);
  };

  state.ws.onclose = () => {
    stopListening(false);
  };

  state.ws.onerror = (e) => {
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
  stopMonitorSpeechRecognition();
  $('live-transcript-bar')?.classList.add('hidden');
  updateKeywordGate(false, '');
  updateListenBtn(false);
  updateVAD(false);
}

function updateListenBtn(listening) {
  const btn  = $('listen-btn');
  const icon = $('listen-btn-icon');
  const text = $('listen-btn-text');
  if (!btn) return;
  if (listening) {
    btn.classList.add('listening');
    if (icon) icon.textContent = '■';
    if (text) text.textContent = 'Stop Listening';
  } else {
    btn.classList.remove('listening');
    if (icon) icon.textContent = '▶';
    if (text) text.textContent = 'Start Listening';
  }
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'status':
      if (msg.matching_mode) {
        updateMatchingModeBadge(msg.matching_mode);
      }
      break;

    case 'result':
      updateVAD(msg.is_vad_active);
      if (msg.rms != null) state.visualizer.setLevel(msg.rms);
      updateScoreGauges(msg);
      addLogEntry(msg);
      if (msg.verified) {
        triggerAlarmUI(msg);
      }
      break;

    case 'text_gate':
      break;

    case 'error':
      addLogEntry({ type: 'error', message: msg.message });
      break;
  }
}

function updateMatchingModeBadge(mode) {
  const wrap  = $('matching-mode-wrap');
  const badge = $('matching-mode-badge');
  if (!wrap || !badge) return;

  wrap.classList.remove('hidden');
  if (mode === 'classifier') {
    badge.textContent = 'Fine-Tuned Classifier';
    badge.className   = 'matching-mode-badge classifier';
  } else {
    badge.textContent = 'ECAPA-TDNN Similarity';
    badge.className   = 'matching-mode-badge';
  }
}

function updateVAD(voice) {
  const indicator = $('vad-indicator');
  const label     = $('vad-label');
  if (!indicator || !label) return;
  if (voice) {
    indicator.classList.add('voice');
    label.textContent = 'Voice Detected';
  } else {
    indicator.classList.remove('voice');
    label.textContent = 'Silence';
  }
}

function updateScoreGauges(msg) {
  const simScore        = msg.similarity_score ?? 0;
  const classifierScore = msg.classifier_score;
  const finalScore      = msg.score ?? 0;
  const matchMode       = msg.match_mode || 'similarity';

  setGauge('dl', simScore);

  const clfWrap = $('classifier-gauge-wrap');
  if (classifierScore != null) {
    if (clfWrap) clfWrap.style.display = 'block';
    setGauge('classifier', classifierScore);
  } else {
    if (clfWrap) clfWrap.style.display = 'none';
  }

  setGauge('fusion', finalScore);

  const matchLabel = $('match-mode-label');
  if (matchLabel) {
    matchLabel.textContent = matchMode === 'classifier' ? 'Classifier' : 'Similarity';
  }
}

function setGauge(name, value) {
  const pct = `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
  const bar = $(`${name}-score-bar`);
  const val = $(`${name}-score-val`);
  if (bar) bar.style.width = pct;
  if (val) val.textContent = value.toFixed(3);
}

function addLogEntry(msg) {
  const log = $('match-log');
  if (!log) return;
  const emptyState = log.querySelector('.empty-state-sm');
  if (emptyState) emptyState.remove();

  const verified = msg.verified || false;
  const time     = new Date().toLocaleTimeString();
  const entry    = document.createElement('div');
  entry.className = `log-entry ${verified ? 'match' : 'no-match'}`;

  const simStr  = msg.similarity_score != null ? `Sim:${msg.similarity_score.toFixed(3)} ` : '';
  const clfStr  = msg.classifier_score != null ? `Clf:${msg.classifier_score.toFixed(3)} ` : '';
  const scoreStr = msg.score != null ? `Final:${msg.score.toFixed(3)}` : '';
  const modeStr = msg.match_mode ? ` [${msg.match_mode}]` : '';
  const statusStr = verified ? '🚨 VERIFIED' : '·';

  entry.innerHTML = `
    <span class="log-time">${time}</span>
    <span class="log-scores">${statusStr} ${simStr}${clfStr}${scoreStr}${modeStr}${msg.message ? ' ' + escHtml(msg.message) : ''}</span>
  `;
  log.prepend(entry);

  while (log.children.length > 30) log.removeChild(log.lastChild);
}

// ─── Alarm UI ─────────────────────────────────────────────────────────────────
let alarmSound = null;

function triggerAlarmUI(msg) {
  if (state.alarmActive) return;
  state.alarmActive = true;
  const overlay = $('alarm-overlay');
  if (!overlay) return;
  overlay.classList.remove('hidden');

  $('alarm-speaker-name').textContent = `Voice Verified — Speaker Match!`;
  const simStr = msg.similarity_score != null ? `Sim: ${msg.similarity_score.toFixed(3)}` : '';
  const clfStr = msg.classifier_score != null ? `  |  Classifier: ${msg.classifier_score.toFixed(3)}` : '';
  $('alarm-scores').textContent = `${simStr}${clfStr}  |  ${new Date().toLocaleTimeString()}`;
  playAlarmSound();
}

function hideAlarmUI() {
  state.alarmActive = false;
  $('alarm-overlay')?.classList.add('hidden');
  stopAlarmSound();
}

function playAlarmSound() {
  stopAlarmSound();
  try {
    const ctx  = new AudioContext();
    const gain = ctx.createGain();
    gain.gain.value = 0.4;
    gain.connect(ctx.destination);

    let t = ctx.currentTime;
    function beep(freq, dur) {
      const osc = ctx.createOscillator();
      osc.type = 'square';
      osc.frequency.value = freq;
      osc.connect(gain);
      osc.start(t);
      osc.stop(t + dur);
      t += dur + 0.05;
    }
    for (let i = 0; i < 6; i++) { beep(880, 0.12); beep(660, 0.08); }
    alarmSound = { ctx, gain };
  } catch(e) {}
}

function stopAlarmSound() {
  if (alarmSound) {
    try {
      alarmSound.gain.gain.cancelScheduledValues(0);
      alarmSound.gain.gain.setTargetAtTime(0, alarmSound.ctx.currentTime, 0.01);
      setTimeout(() => alarmSound?.ctx?.close(), 100);
    } catch(e) {}
    alarmSound = null;
  }
}

$('alarm-stop-btn')?.addEventListener('click', () => {
  hideAlarmUI();
});

// ─── Settings Page ────────────────────────────────────────────────────────────
function initSettingsPage() {
  checkHealth().then(data => {
    if (!data) return;
    $('system-info').innerHTML = `
      <div class="info-row"><span class="info-key">Backend Version</span><span class="info-val ok">VoiceGuard v2</span></div>
      <div class="info-row"><span class="info-key">Base Model</span><span class="info-val ok">ECAPA-TDNN (SpeechBrain)</span></div>
      <div class="info-row"><span class="info-key">Embedding Dimension</span><span class="info-val">192-dim</span></div>
      <div class="info-row"><span class="info-key">Personal Classifier</span><span class="info-val">3-Layer MLP Head</span></div>
      <div class="info-row"><span class="info-key">API Base</span><span class="info-val">${API_BASE}</span></div>
      <div class="info-row"><span class="info-key">Sample Rate</span><span class="info-val">16000 Hz</span></div>
    `;
  });

  const speakerSel = $('settings-speaker-select');
  speakerSel?.addEventListener('change', loadSpeakerSettings);

  [
    ['similarity-threshold', 'similarity-threshold-val'],
    ['classifier-threshold', 'classifier-threshold-val'],
  ].forEach(([sliderId, valId]) => {
    $(sliderId)?.addEventListener('input', () => {
      const valEl = $(valId);
      if (valEl) valEl.textContent = parseFloat($(sliderId).value).toFixed(2);
    });
  });

  $('save-settings-btn')?.addEventListener('click', saveSpeakerSettings);
}

async function loadSpeakerSettings() {
  const id = $('settings-speaker-select').value;
  if (!id) { $('threshold-controls')?.classList.add('hidden'); return; }

  const speaker = state.speakers.find(s => s.id === id);
  if (!speaker) return;

  const simVal = speaker.similarity_threshold ?? 0.65;
  const clfVal = speaker.classifier_threshold ?? 0.70;

  if ($('similarity-threshold')) $('similarity-threshold').value = simVal;
  if ($('similarity-threshold-val')) $('similarity-threshold-val').textContent = simVal.toFixed(2);
  if ($('classifier-threshold')) $('classifier-threshold').value = clfVal;
  if ($('classifier-threshold-val')) $('classifier-threshold-val').textContent = clfVal.toFixed(2);

  $('threshold-controls')?.classList.remove('hidden');
  $('settings-saved-msg')?.classList.add('hidden');
}

async function saveSpeakerSettings() {
  const id = $('settings-speaker-select').value;
  if (!id) return;

  const settings = {
    similarity_threshold: parseFloat($('similarity-threshold').value),
    classifier_threshold: parseFloat($('classifier-threshold').value),
  };

  try {
    const res = await fetch(`${API_BASE}/speakers/${id}/thresholds`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    });
    if (res.ok) {
      $('settings-saved-msg')?.classList.remove('hidden');
      await fetchSpeakers();
      setTimeout(() => $('settings-saved-msg')?.classList.add('hidden'), 3000);
    }
  } catch (e) {
    console.error('Save settings failed:', e);
  }
}

// ─── Speech Recognition (Web Speech API) ─────────────────────────────────────
function updateKeywordBadge(keyword) {
  state.activeKeyword = (keyword || '').trim().toLowerCase();
  const wrap = $('keyword-badge-wrap');
  const text = $('active-keyword-text');
  if (!wrap || !text) return;
  if (state.activeKeyword) {
    text.textContent = `"${state.activeKeyword}"`;
    wrap.classList.remove('hidden');
  } else {
    wrap.classList.add('hidden');
  }
}

function updateLiveTranscript(text, keywordMatched) {
  const el    = $('live-transcript-text');
  const badge = $('keyword-match-badge');
  if (!el) return;

  if (text) {
    el.textContent = `"${text}"`;
    el.classList.add('has-text');
  } else {
    el.textContent = 'Listening for speech…';
    el.classList.remove('has-text');
  }

  if (badge) {
    if (keywordMatched) {
      badge.classList.remove('hidden');
    } else {
      badge.classList.add('hidden');
    }
  }
}

function updateKeywordGate(open, transcript) {
  state.keywordGateOpen = open;
  updateLiveTranscript(transcript, open);

  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({
      type:       'text_match',
      matched:    open,
      transcript: transcript
    }));
  }

  if (state.keywordGateTimer) clearTimeout(state.keywordGateTimer);
  if (open) {
    state.keywordGateTimer = setTimeout(() => {
      state.keywordGateOpen = false;
      updateLiveTranscript(transcript, false);
      if (state.ws?.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: 'text_match', matched: false, transcript }));
      }
      state.keywordGateTimer = null;
    }, 3000);
  }
}

function startMonitorSpeechRecognition(keyword) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn('[STT] Web Speech API not supported in this browser.');
    if (state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: 'text_match', matched: true, transcript: '' }));
    }
    return;
  }

  stopMonitorSpeechRecognition();

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
      if (event.results[i].isFinal) {
        finalText += t + ' ';
      } else {
        interim = t;
      }
    }

    const text = (finalText || interim).trim();
    updateLiveTranscript(text, state.keywordGateOpen);

    if (text) {
      const log = $('match-log');
      const emptyState = log?.querySelector('.empty-state-sm');
      if (emptyState) emptyState.remove();
      const entry = document.createElement('div');
      entry.className = 'log-entry transcript-entry';
      entry.innerHTML = `
        <span class="log-time">${new Date().toLocaleTimeString()}</span>
        <span class="log-scores" style="color:var(--c-accent);">🗣 "${escHtml(text)}"</span>
      `;
      log?.prepend(entry);
      while (log?.children.length > 30) log.removeChild(log.lastChild);
    }

    if (finalText.trim()) {
      const matched = !keyword || finalText.trim().includes(keyword);
      updateKeywordGate(matched, finalText.trim());
    }
  };

  recog.onerror = (e) => {
    if (e.error !== 'no-speech') console.warn('[STT Monitor] error:', e.error);
    if (['network', 'audio-capture'].includes(e.error)) {
      setTimeout(() => {
        if (state.listening) recog.start();
      }, 1000);
    }
  };

  recog.onend = () => {
    if (state.listening) {
      try { recog.start(); } catch(e) {}
    }
  };

  try {
    recog.start();
    state.speechRecognizer = recog;
  } catch(e) {
    console.warn('[STT Monitor] could not start:', e);
  }
}

function stopMonitorSpeechRecognition() {
  if (state.speechRecognizer) {
    try { state.speechRecognizer.abort(); } catch(e) {}
    state.speechRecognizer = null;
  }
  if (state.keywordGateTimer) {
    clearTimeout(state.keywordGateTimer);
    state.keywordGateTimer = null;
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
