/**
 * visualizer.js — Canvas-based waveform, FFT spectrum, and level meter renderers
 *
 * Key design: canvas dimensions are synced to CSS layout on EVERY draw call.
 * This means we never need to manually resize — it just works after any CSS
 * transition, page navigation, or window resize.
 */

const DPR = window.devicePixelRatio || 1;

export class Visualizer {
  constructor() {
    this._waveformCanvas    = null;
    this._fftCanvas         = null;
    this._levelCanvas       = null;
    this._waveCtx           = null;
    this._fftCtx            = null;
    this._levelCtx          = null;
    this._rafId             = null;
    this._getAudioData      = null;
    this._currentLevel      = 0;
    this._referenceSpectrum = null;  // enrolled speaker's avg spectrum (Float32Array, 0–1)
  }

  /** Attach canvases + data getter. Safe to call before canvas is visible. */
  mount({ waveformCanvas, fftCanvas, levelCanvas, getAudioData }) {
    this._getAudioData = getAudioData;

    if (waveformCanvas) {
      this._waveformCanvas = waveformCanvas;
      this._waveCtx        = waveformCanvas.getContext('2d');
    }
    if (fftCanvas) {
      this._fftCanvas = fftCanvas;
      this._fftCtx    = fftCanvas.getContext('2d');
    }
    if (levelCanvas) {
      this._levelCanvas = levelCanvas;
      this._levelCtx    = levelCanvas.getContext('2d');
    }
  }

  /** Draw a waveform from an already-recorded Float32Array (enrollment) */
  drawStaticWaveform(canvas, samples) {
    const ctx = canvas.getContext('2d');
    this._syncSize(canvas, ctx);
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0,   '#6c63ff');
    gradient.addColorStop(0.5, '#a855f7');
    gradient.addColorStop(1,   '#ec4899');

    ctx.strokeStyle = gradient;
    ctx.lineWidth   = 2;
    ctx.beginPath();

    const step = Math.max(1, Math.floor(samples.length / width));
    for (let x = 0; x < width; x++) {
      const sample = samples[x * step] ?? 0;
      const y = (1 - (sample * 0.5 + 0.5)) * height;
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  /** Set (or clear) the enrolled speaker's reference spectrum for overlay */
  setReferenceSpectrum(spectrum) {
    // spectrum: number[] 0–1, or null to clear
    this._referenceSpectrum = spectrum ? Float32Array.from(spectrum) : null;
  }

  /** Start the animation loop */
  start() {
    if (this._rafId) return;
    const loop = () => {
      this._draw();
      this._rafId = requestAnimationFrame(loop);
    };
    this._rafId = requestAnimationFrame(loop);
  }

  /** Stop the animation loop and clear canvases */
  stop() {
    if (this._rafId) { cancelAnimationFrame(this._rafId); this._rafId = null; }
    this._clearAll();
  }

  /** Update current RMS level (0–1) — called from WebSocket "level" events */
  setLevel(rms) {
    // Smooth the meter with a small attack/decay
    const target = Math.min(1, rms * 8);   // amplify slightly for visibility
    this._currentLevel = this._currentLevel * 0.55 + target * 0.45;
  }

  // ── Private ──────────────────────────────────────────────────────────────

  _draw() {
    if (!this._getAudioData) return;
    const { waveform, frequency } = this._getAudioData();
    if (this._levelCanvas)    this._drawLevelMeter();
    if (frequency && this._fftCanvas)    this._drawFFT(frequency);
    if (waveform  && this._waveformCanvas) this._drawLiveWaveform(waveform);
  }

  /** Sync canvas pixel dimensions to its CSS layout size (handles init + resize) */
  _syncSize(canvas, ctx) {
    const w = Math.round(canvas.clientWidth  * DPR);
    const h = Math.round(canvas.clientHeight * DPR);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width  = w;
      canvas.height = h;
    }
  }

  _drawLevelMeter() {
    const canvas = this._levelCanvas;
    const ctx    = this._levelCtx;
    this._syncSize(canvas, ctx);
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    const segments = 36;
    const segW     = width / segments;
    const gap      = Math.max(2, 2 * DPR);
    const filled   = Math.round(this._currentLevel * segments);

    for (let i = 0; i < segments; i++) {
      const t = i / (segments - 1);

      // Colour gradient: teal → yellow → red
      let r, g, b;
      if (t < 0.5) {
        r = Math.round(t * 2 * 255);
        g = Math.round(180 + t * 2 * 50);
        b = Math.round(180 * (1 - t * 2));
      } else {
        r = 255;
        g = Math.round(200 * (1 - (t - 0.5) * 2));
        b = 0;
      }

      ctx.fillStyle = i < filled
        ? `rgba(${r},${g},${b},0.95)`
        : 'rgba(255,255,255,0.07)';

      const x = i * segW + gap / 2;
      const barH = i < filled
        ? height - gap * 2
        : height * 0.3;
      const y = height - gap - barH;

      ctx.beginPath();
      ctx.roundRect(x, y, segW - gap, barH, 3);
      ctx.fill();
    }
  }

  _drawFFT(freqData) {
    const canvas = this._fftCanvas;
    const ctx    = this._fftCtx;
    this._syncSize(canvas, ctx);
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    if (!freqData || freqData.length === 0) return;

    const barCount = Math.min(freqData.length, Math.floor(width / 3));
    const barW     = width / barCount;

    // ── Grid lines ──────────────────────────────────────────────────────────────
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth   = 1;
    for (let i = 1; i < 4; i++) {
      const y = Math.round(height * (i / 4));
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    // ── Reference (enrolled) spectrum ghost overlay ─────────────────────────────
    const ref = this._referenceSpectrum;
    if (ref && ref.length > 0) {
      // Filled area
      const refGrad = ctx.createLinearGradient(0, 0, 0, height);
      refGrad.addColorStop(0,   'rgba(108, 99, 255, 0.22)');
      refGrad.addColorStop(1,   'rgba(108, 99, 255, 0.04)');

      ctx.beginPath();
      ctx.moveTo(0, height);
      for (let i = 0; i < barCount; i++) {
        // Map live bar index → reference index (they can be different lengths)
        const ri = Math.round((i / barCount) * (ref.length - 1));
        const v  = ref[ri] ?? 0;
        const x  = i * barW + barW / 2;
        const y  = height - v * height;
        i === 0 ? ctx.lineTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.lineTo(width, height);
      ctx.closePath();
      ctx.fillStyle = refGrad;
      ctx.fill();

      // Bright outline of the reference curve
      ctx.beginPath();
      for (let i = 0; i < barCount; i++) {
        const ri = Math.round((i / barCount) * (ref.length - 1));
        const v  = ref[ri] ?? 0;
        const x  = i * barW + barW / 2;
        const y  = height - v * height;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(168, 99, 255, 0.7)';
      ctx.lineWidth   = 1.5 * DPR;
      ctx.stroke();
    }

    // ── Live glow layer ────────────────────────────────────────────────────────────────
    for (let i = 0; i < barCount; i++) {
      const value = freqData[i] / 255;
      if (value < 0.02) continue;
      const barH = value * height;
      const x    = i * barW;
      const hue  = 260 - (i / barCount) * 80;
      ctx.fillStyle = `hsla(${hue}, 90%, 65%, ${value * 0.25})`;
      ctx.fillRect(x, height - barH - 4, barW, barH + 8);
    }

    // ── Live bars ────────────────────────────────────────────────────────────────────
    for (let i = 0; i < barCount; i++) {
      const value = freqData[i] / 255;
      if (value < 0.005) continue;

      const barH = value * height;
      const x    = i * barW + (barW > 4 ? 1 : 0);
      const w    = barW > 4 ? barW - 2 : barW;
      const hue  = 260 - (i / barCount) * 80;

      const grad = ctx.createLinearGradient(0, height - barH, 0, height);
      grad.addColorStop(0,   `hsla(${hue}, 95%, 75%, 1)`);
      grad.addColorStop(0.6, `hsla(${hue}, 80%, 55%, 0.9)`);
      grad.addColorStop(1,   `hsla(${hue}, 70%, 40%, 0.6)`);
      ctx.fillStyle = grad;

      ctx.beginPath();
      if (barH > 4) {
        ctx.roundRect(x, height - barH, w, barH, [3, 3, 0, 0]);
      } else {
        ctx.rect(x, height - barH, w, barH);
      }
      ctx.fill();

      if (value > 0.05) {
        ctx.fillStyle = `hsla(${hue}, 100%, 90%, ${value})`;
        ctx.fillRect(x, height - barH, w, Math.min(3, barH * 0.15));
      }
    }

    // ── Legend ──────────────────────────────────────────────────────────────────────
    if (height > 50) {
      const fs = Math.round(10 * DPR);
      ctx.font = `500 ${fs}px Inter, sans-serif`;

      if (ref) {
        // Enrolled dot + label
        ctx.fillStyle = 'rgba(168, 99, 255, 0.85)';
        ctx.beginPath();
        ctx.arc(8 * DPR, 10 * DPR, 4 * DPR, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.textAlign = 'left';
        ctx.fillText('Enrolled', 16 * DPR, 14 * DPR);
      }

      // Live dot + label
      const liveX = ref ? 80 * DPR : 8 * DPR;
      ctx.fillStyle = 'rgba(0, 229, 255, 0.85)';
      ctx.beginPath();
      ctx.arc(liveX, 10 * DPR, 4 * DPR, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.fillText('Live', liveX + 8 * DPR, 14 * DPR);

      // Freq labels
      ctx.fillStyle = 'rgba(255,255,255,0.2)';
      ctx.textAlign = 'left';
      const labels = [[0.1, '500Hz'], [0.3, '2kHz'], [0.55, '4kHz'], [0.85, '8kHz']];
      labels.forEach(([frac, label]) => {
        ctx.fillText(label, frac * width, height - 4 * DPR);
      });
    }
  }

  _drawLiveWaveform(waveformData) {
    const canvas = this._waveformCanvas;
    const ctx    = this._waveCtx;
    this._syncSize(canvas, ctx);
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    if (!waveformData || waveformData.length === 0) return;

    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0,   '#6c63ff');
    gradient.addColorStop(0.5, '#a855f7');
    gradient.addColorStop(1,   '#ec4899');

    ctx.strokeStyle = gradient;
    ctx.lineWidth   = 1.5 * DPR;
    ctx.shadowBlur  = 6 * DPR;
    ctx.shadowColor = 'rgba(108,99,255,0.6)';
    ctx.beginPath();

    const sliceW = width / waveformData.length;
    let x = 0;
    for (let i = 0; i < waveformData.length; i++) {
      const v = (waveformData[i] * 0.5 + 0.5) * height;
      i === 0 ? ctx.moveTo(x, v) : ctx.lineTo(x, v);
      x += sliceW;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  _clearAll() {
    [
      [this._waveCtx,  this._waveformCanvas],
      [this._fftCtx,   this._fftCanvas],
      [this._levelCtx, this._levelCanvas],
    ].forEach(([ctx, canvas]) => {
      if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
  }
}
