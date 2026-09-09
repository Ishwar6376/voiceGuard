/**
 * audio_capture.js — Web Audio API + MediaRecorder + WebSocket streaming
 */

const SAMPLE_RATE = 16000;

export class AudioCapture {
  constructor() {
    this._stream      = null;
    this._mediaRecorder = null;
    this._audioContext  = null;
    this._analyser      = null;
    this._sourceNode    = null;
    this._scriptNode    = null;
    this._onChunk       = null;
  }

  get analyser()     { return this._analyser; }
  get audioContext() { return this._audioContext; }

  /**
   * Initialize microphone + AudioContext.
   * MUST be called directly inside a user-gesture handler (button click)
   * so the browser allows AudioContext to start and mic access is granted.
   */
  async init() {
    // ── Mic permission ───────────────────────────────────────────────────────
    if (!this._stream) {
      this._stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          // Don't force sampleRate — let the browser negotiate;
          // we'll handle resampling on the server side.
        }
      });
    }

    // ── AudioContext ─────────────────────────────────────────────────────────
    if (!this._audioContext) {
      // Use the browser's native sample rate — forcing 16kHz can cause
      // AudioContext to fail or downsample incorrectly on some systems.
      this._audioContext = new AudioContext();
    }

    // CRITICAL: Resume the context. Browsers start AudioContext in
    // "suspended" state as part of the autoplay policy. Calling resume()
    // here works because we're still inside the synchronous user-gesture.
    if (this._audioContext.state === 'suspended') {
      await this._audioContext.resume();
    }

    // ── Audio graph: mic → analyser ──────────────────────────────────────────
    if (!this._sourceNode) {
      this._sourceNode = this._audioContext.createMediaStreamSource(this._stream);

      this._analyser = this._audioContext.createAnalyser();
      this._analyser.fftSize = 1024;
      this._analyser.smoothingTimeConstant = 0.75;

      this._sourceNode.connect(this._analyser);
    }

    console.log(
      `[AudioCapture] Ready — SR: ${this._audioContext.sampleRate}Hz, ` +
      `state: ${this._audioContext.state}`
    );
  }

  /**
   * Record N seconds → returns Blob (browser-native codec, e.g. WebM).
   * The blob is later converted to WAV by blobToWav() in app.js before upload.
   */
  async record(durationSeconds = 3) {
    await this.init();
    return new Promise((resolve, reject) => {
      const chunks   = [];
      const mimeType = this._getBestMime();
      const recorder = new MediaRecorder(this._stream, mimeType ? { mimeType } : {});

      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.onstop  = () => resolve(new Blob(chunks, { type: recorder.mimeType }));
      recorder.onerror = reject;

      recorder.start(100);
      setTimeout(() => recorder.stop(), durationSeconds * 1000);
      this._mediaRecorder = recorder;
    });
  }

  /** Stop any active recording */
  stopRecording() {
    if (this._mediaRecorder?.state !== 'inactive') {
      this._mediaRecorder?.stop();
    }
  }

  /**
   * Start streaming raw PCM Float32 chunks to a callback.
   * Uses ScriptProcessorNode — deprecated but works on local HTTP (no HTTPS needed).
   * AudioWorklet requires HTTPS, which is impractical for local RPi dev.
   */
  startStreaming(onChunk) {
    if (!this._audioContext || !this._analyser) {
      console.error('[AudioCapture] startStreaming called before init()');
      return;
    }
    if (this._scriptNode) return; // already streaming

    this._onChunk = onChunk;

    // 4096-sample buffer at native SR ≈ 85–93ms per callback
    const BUFFER_SIZE = 4096;
    this._scriptNode = this._audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

    this._scriptNode.onaudioprocess = (e) => {
      if (!this._onChunk) return;
      // getChannelData returns a live reference — copy it before sending
      const live = e.inputBuffer.getChannelData(0);
      this._onChunk(new Float32Array(live));
    };

    // analyser → scriptProcessor → destination (silent output required to
    // keep ScriptProcessorNode alive — browsers garbage-collect unconnected nodes)
    this._analyser.connect(this._scriptNode);
    this._scriptNode.connect(this._audioContext.destination);

    console.log(
      `[AudioCapture] Streaming started — ` +
      `buffer: ${BUFFER_SIZE}, SR: ${this._audioContext.sampleRate}Hz`
    );
  }

  /** Stop streaming and disconnect the ScriptProcessorNode */
  stopStreaming() {
    this._onChunk = null;
    if (this._scriptNode) {
      try {
        this._scriptNode.disconnect();
        this._analyser?.disconnect(this._scriptNode);
      } catch (_) {}
      this._scriptNode = null;
    }
    console.log('[AudioCapture] Streaming stopped.');
  }

  /** Time-domain data for waveform canvas (Float32, range -1…1) */
  getWaveformData() {
    if (!this._analyser) return null;
    const buf = new Float32Array(this._analyser.fftSize);
    this._analyser.getFloatTimeDomainData(buf);
    return buf;
  }

  /** Frequency-domain data for FFT canvas (Uint8, 0…255) */
  getFrequencyData() {
    if (!this._analyser) return null;
    const buf = new Uint8Array(this._analyser.frequencyBinCount);
    this._analyser.getByteFrequencyData(buf);
    return buf;
  }

  _getBestMime() {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'];
    return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
  }

  /** Release all resources */
  destroy() {
    this.stopStreaming();
    this._stream?.getTracks().forEach(t => t.stop());
    this._audioContext?.close();
    this._stream      = null;
    this._audioContext = null;
    this._analyser    = null;
    this._sourceNode  = null;
  }
}
