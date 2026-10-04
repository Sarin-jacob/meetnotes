/// <reference lib="webworker" />
import {
  AutoFeatureExtractor,
  AutoModel,
  AutoModelForAudioFrameClassification,
  AutoProcessor,
  env,
  pipeline,
  type PyAnnoteProcessor,
  type Tensor,
} from '@huggingface/transformers';
import { SAMPLE_RATE } from './audio.ts';
import { l2normalize, type SpeakerGroup, type SpeechSegment, type TextChunk } from './diarize.ts';

env.allowLocalModels = false;

export type WorkerRequest =
  | {
      type: 'asr';
      audio: Float32Array;
      windows: Array<[number, number]>;
      model: string;
      device: 'webgpu' | 'wasm';
      task: 'translate' | 'transcribe';
      language: string;
    }
  | { type: 'diarize'; audio: Float32Array };

export type WorkerEvent =
  | { type: 'progress'; text: string; ratio: number }
  | { type: 'log'; text: string }
  | { type: 'asr-done'; loadMs: number; firstWindowMs: number; processMs: number; chunks: TextChunk[] }
  | { type: 'diarize-done'; loadMs: number; segmentMs: number; embedMs: number; segments: SpeechSegment[]; groups: SpeakerGroup[] }
  | { type: 'error'; message: string };

const SEG_MODEL = 'onnx-community/pyannote-segmentation-3.0';
const EMB_MODEL = 'onnx-community/wespeaker-voxceleb-resnet34-LM';
// pyannote powerset classes: 0 silence, 1–3 single speaker, 4–6 two speakers at once.
const OVERLAP_FIRST: Record<number, number> = { 4: 1, 5: 1, 6: 2 };
const MIN_EMBED_SEC = 1;

const post = (e: WorkerEvent) => self.postMessage(e);

function downloadProgress(label: string) {
  const files = new Map<string, number>();
  return (p: { status: string; file?: string; progress?: number }) => {
    if (p.status !== 'progress' || !p.file) return;
    files.set(p.file, p.progress ?? 0);
    const avg = [...files.values()].reduce((a, b) => a + b, 0) / files.size / 100;
    post({ type: 'progress', text: `Downloading ${label}… ${Math.round(avg * 100)}%`, ratio: avg });
  };
}

async function runAsr(req: Extract<WorkerRequest, { type: 'asr' }>) {
  const dtype =
    req.device === 'webgpu'
      ? { encoder_model: 'fp32', decoder_model_merged: 'q4' }
      : { encoder_model: 'q8', decoder_model_merged: 'q8' };
  let t = performance.now();
  const asr = await pipeline('automatic-speech-recognition', req.model, {
    device: req.device,
    dtype: dtype as never,
    progress_callback: downloadProgress('Whisper') as never,
  });
  // Warm up on a second of silence so GPU shader compilation isn't counted as processing time.
  post({ type: 'progress', text: 'Warming up…', ratio: 0 });
  await asr(new Float32Array(SAMPLE_RATE), { task: req.task, language: req.language } as never);
  const loadMs = performance.now() - t;

  const chunks: TextChunk[] = [];
  let firstWindowMs = 0;
  t = performance.now();
  for (const [i, [s, e]] of req.windows.entries()) {
    const w0 = performance.now();
    const out = (await asr(req.audio.subarray(s, e), {
      task: req.task,
      language: req.language,
      return_timestamps: true,
    } as never)) as { text: string; chunks?: Array<{ timestamp: [number, number | null]; text: string }> };
    if (i === 0) firstWindowMs = performance.now() - w0;

    const offset = s / SAMPLE_RATE;
    const windowEnd = e / SAMPLE_RATE;
    for (const c of out.chunks ?? [{ timestamp: [0, null], text: out.text }]) {
      const text = c.text.trim();
      if (!text) continue;
      chunks.push({ start: offset + c.timestamp[0], end: Math.min(windowEnd, offset + (c.timestamp[1] ?? windowEnd - offset)), text });
    }
    post({ type: 'progress', text: `Transcribing ${i + 1}/${req.windows.length}`, ratio: (i + 1) / req.windows.length });
  }
  const processMs = performance.now() - t;
  await asr.dispose();
  post({ type: 'asr-done', loadMs, firstWindowMs, processMs, chunks });
}

async function runDiarize(audio: Float32Array) {
  let t = performance.now();
  const segProcessor = await AutoProcessor.from_pretrained(SEG_MODEL);
  const segModel = await AutoModelForAudioFrameClassification.from_pretrained(SEG_MODEL, {
    device: 'wasm',
    dtype: 'fp32',
    progress_callback: downloadProgress('speaker models') as never,
  });
  const embExtractor = await AutoFeatureExtractor.from_pretrained(EMB_MODEL);
  const embModel = await AutoModel.from_pretrained(EMB_MODEL, {
    device: 'wasm',
    dtype: 'fp32',
    progress_callback: downloadProgress('speaker models') as never,
  });
  const loadMs = performance.now() - t;

  // 1. Segmentation: who-is-talking per 10 s window (speaker ids are local to each window).
  const win = 10 * SAMPLE_RATE;
  const segments: SpeechSegment[] = [];
  const groupKey = new Map<string, number>();
  const groupAudio: Array<Float32Array[]> = [];
  const groups: SpeakerGroup[] = [];
  const windowCount = Math.ceil(audio.length / win);
  t = performance.now();
  for (let w = 0; w < windowCount; w++) {
    const start = w * win;
    const chunk = audio.subarray(start, Math.min(start + win, audio.length));
    if (chunk.length < SAMPLE_RATE / 2) break;
    const inputs = await segProcessor(chunk);
    const { logits } = (await segModel(inputs)) as { logits: Tensor };
    const local = (segProcessor as unknown as PyAnnoteProcessor).post_process_speaker_diarization(logits, chunk.length)[0];
    for (const seg of local) {
      if (seg.id === 0 || seg.end - seg.start < 0.25) continue;
      const overlap = seg.id >= 4;
      const localSpeaker = overlap ? OVERLAP_FIRST[seg.id] : seg.id;
      const key = `${w}:${localSpeaker}`;
      let g = groupKey.get(key);
      if (g === undefined) {
        g = groups.length;
        groupKey.set(key, g);
        groups.push({ embedding: null, speechSec: 0, firstStart: start / SAMPLE_RATE + seg.start });
        groupAudio.push([]);
      }
      segments.push({ start: start / SAMPLE_RATE + seg.start, end: start / SAMPLE_RATE + seg.end, group: g, overlap });
      if (!overlap) {
        groupAudio[g].push(chunk.subarray(Math.floor(seg.start * SAMPLE_RATE), Math.ceil(seg.end * SAMPLE_RATE)));
        groups[g].speechSec += seg.end - seg.start;
      }
    }
    post({ type: 'progress', text: `Finding speech ${w + 1}/${windowCount}`, ratio: ((w + 1) / windowCount) * 0.5 });
  }
  const segmentMs = performance.now() - t;

  // 2. Voiceprints: one embedding per local speaker with enough clean (non-overlapping) speech.
  t = performance.now();
  for (const [g, parts] of groupAudio.entries()) {
    if (groups[g].speechSec >= MIN_EMBED_SEC) {
      const joined = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
      let o = 0;
      for (const p of parts) {
        joined.set(p, o);
        o += p.length;
      }
      const feats = await embExtractor(joined);
      const out = (await embModel(feats)) as Record<string, Tensor>;
      groups[g].embedding = l2normalize(Object.values(out)[0].data as Float32Array);
    }
    post({ type: 'progress', text: `Voiceprints ${g + 1}/${groups.length}`, ratio: 0.5 + ((g + 1) / groups.length) * 0.5 });
  }
  const embedMs = performance.now() - t;

  await segModel.dispose();
  await embModel.dispose();
  post({ type: 'diarize-done', loadMs, segmentMs, embedMs, segments, groups });
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  try {
    if (e.data.type === 'asr') await runAsr(e.data);
    else await runDiarize(e.data.audio);
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err) });
  }
};
