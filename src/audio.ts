export const SAMPLE_RATE = 16000;

/** Decodes any browser-supported audio/video file to 16 kHz mono PCM. */
export async function decodeToMono16k(data: ArrayBuffer): Promise<Float32Array> {
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  try {
    const buf = await ctx.decodeAudioData(data);
    if (buf.numberOfChannels === 1) return buf.getChannelData(0).slice();
    const out = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const ch = buf.getChannelData(c);
      for (let i = 0; i < ch.length; i++) out[i] += ch[i] / buf.numberOfChannels;
    }
    return out;
  } finally {
    void ctx.close();
  }
}

export interface RecordingResult {
  blob: Blob;
  wallClockSec: number;
  hiddenSec: number;
}

/** Microphone recorder that also tracks how long the page was hidden, to detect background gaps. */
export class MicRecorder {
  private recorder?: MediaRecorder;
  private stream?: MediaStream;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private hiddenSince = 0;
  private hiddenMs = 0;
  private onVisibility = () => {
    if (document.hidden) this.hiddenSince = performance.now();
    else if (this.hiddenSince) {
      this.hiddenMs += performance.now() - this.hiddenSince;
      this.hiddenSince = 0;
    }
  };

  get elapsedSec(): number {
    return this.startedAt ? (performance.now() - this.startedAt) / 1000 : 0;
  }

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: true, autoGainControl: true },
    });
    this.chunks = [];
    this.hiddenMs = 0;
    this.recorder = new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.recorder.start(1000);
    this.startedAt = performance.now();
    document.addEventListener('visibilitychange', this.onVisibility);
  }

  stop(): Promise<RecordingResult> {
    return new Promise((resolve) => {
      const rec = this.recorder!;
      rec.onstop = () => {
        this.onVisibility();
        document.removeEventListener('visibilitychange', this.onVisibility);
        this.stream?.getTracks().forEach((t) => t.stop());
        const wallClockSec = this.elapsedSec;
        this.startedAt = 0;
        resolve({ blob: new Blob(this.chunks, { type: rec.mimeType }), wallClockSec, hiddenSec: this.hiddenMs / 1000 });
      };
      rec.stop();
    });
  }
}

/**
 * Splits audio into windows of at most `maxSec`, cutting at the quietest 100 ms frame
 * within the last `searchSec` of each window so words are less likely to be cut in half.
 * Windows that are essentially silent are dropped (Whisper hallucinates on silence).
 */
export function splitAtSilence(
  audio: Float32Array,
  maxSec = 28,
  searchSec = 6,
  silenceRms = 0.002,
): Array<[number, number]> {
  const frame = SAMPLE_RATE / 10;
  const maxLen = maxSec * SAMPLE_RATE;
  const windows: Array<[number, number]> = [];
  let start = 0;
  while (start < audio.length) {
    let end = Math.min(start + maxLen, audio.length);
    if (end < audio.length) {
      let best = end;
      let bestRms = Infinity;
      for (let f = end - searchSec * SAMPLE_RATE; f + frame <= end; f += frame) {
        const r = rms(audio, f, f + frame);
        if (r < bestRms) {
          bestRms = r;
          best = f + frame / 2;
        }
      }
      end = best;
    }
    if (end - start >= SAMPLE_RATE / 2 && rms(audio, start, end) > silenceRms) windows.push([start, end]);
    start = end;
  }
  return windows;
}

function rms(a: Float32Array, from: number, to: number): number {
  let s = 0;
  for (let i = from; i < to; i++) s += a[i] * a[i];
  return Math.sqrt(s / Math.max(1, to - from));
}

export function formatTime(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n: number) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}
