import { decodeToMono16k, formatTime, MicRecorder, SAMPLE_RATE, splitAtSilence } from './audio.ts';
import { getDeviceInfo, type DeviceInfo } from './device.ts';
import { assignSpeakers, labelChunks, toTurns, type LabelledChunk, type SpeakerGroup, type SpeechSegment, type TextChunk } from './diarize.ts';
import { estimateHourSec, llmChoices, summarise, type LlmResult } from './llm.ts';
import type { WorkerEvent, WorkerRequest } from './ml.worker.ts';
import { DarkScreen, playHandoverAlarm } from './screen.ts';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const SAMPLE_BASE = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/';
const SPEAKER_COLORS = ['#3b7dd8', '#e0803a', '#3aa676', '#c4495d', '#8a63d2', '#c9a227', '#2fa3b5', '#a0602f'];

type Fact = [label: string, value: string, cls?: 'good' | 'warn' | 'bad'];

interface Results {
  app: string;
  testedAt: string;
  device?: DeviceInfo;
  audio?: { source: string; durationSec: number; wallClockSec?: number; hiddenSec?: number; lostSec?: number };
  asr?: Record<string, unknown>;
  diarization?: Record<string, unknown>;
  llm?: Omit<LlmResult, 'text'> & { estHourMin: number; outputChars: number };
  screen?: Record<string, unknown>;
  estimateForOneHourMeetingMin?: number;
}

const results: Results = { app: 'meetnotes-benchmark@1', testedAt: new Date().toISOString() };
let device: DeviceInfo | undefined;
let audio: Float32Array | null = null;
let asrChunks: TextChunk[] | null = null;
let diar: { segments: SpeechSegment[]; groups: SpeakerGroup[] } | null = null;
let labelled: LabelledChunk[] | null = null;
let busy = false;

// ---------- small UI helpers ----------

function setFacts(id: string, facts: Fact[]) {
  $(id).replaceChildren(
    ...facts.flatMap(([label, value, cls]) => {
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = value;
      if (cls) dd.className = cls;
      return [dt, dd];
    }),
  );
}

function progress(id: string, text: string, ratio: number) {
  const el = $(id);
  el.hidden = false;
  (el.firstElementChild as HTMLElement).style.width = `${Math.round(Math.min(1, ratio) * 100)}%`;
  el.querySelector('span')!.textContent = text;
}

function log(text: string) {
  const el = $('log');
  el.textContent += `[${new Date().toLocaleTimeString()}] ${text}\n`;
}

function renderResults() {
  $('results-json').textContent = JSON.stringify(results, null, 2);
}

const sec = (ms: number) => ms / 1000;
const fmtSec = (s: number) => (s < 90 ? `${s.toFixed(1)} s` : `${(s / 60).toFixed(1)} min`);
const speedClass = (x: number): Fact[2] => (x >= 2 ? 'good' : x >= 1 ? 'warn' : 'bad');
const hourClass = (min: number): Fact[2] => (min <= 30 ? 'good' : min <= 60 ? 'warn' : 'bad');

function refreshButtons() {
  const hasAudio = !!audio;
  $<HTMLButtonElement>('btn-asr').disabled = busy || !hasAudio;
  $<HTMLButtonElement>('btn-diar').disabled = busy || !hasAudio;
  $<HTMLButtonElement>('btn-recluster').disabled = busy || !diar;
  $<HTMLButtonElement>('btn-llm').disabled = busy || !asrChunks?.length || device?.webgpu !== 'yes';
  $<HTMLButtonElement>('btn-all').disabled = busy || !hasAudio;
  $<HTMLButtonElement>('btn-rec').disabled = busy && !recorder;
}

async function guard(label: string, fn: () => Promise<void>) {
  if (busy) return;
  busy = true;
  refreshButtons();
  try {
    await fn();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log(`${label} failed: ${msg}`);
    alert(`${label} failed:\n${msg.slice(0, 400)}`);
  } finally {
    busy = false;
    refreshButtons();
    renderResults();
  }
}

// ---------- 1. device ----------

async function initDevice() {
  device = await getDeviceInfo();
  results.device = device;
  setFacts('device-facts', [
    ['Phone', [device.model, device.os].filter(Boolean).join(' · ') || 'unknown'],
    ['Browser', device.browser],
    ['RAM', device.ramGB ? `${device.ramGB} GB${device.ramGB >= 8 ? '+' : ''} (browser rounds down)` : 'unknown'],
    ['CPU cores', String(device.cpuCores)],
    ['GPU (WebGPU)', device.webgpu === 'yes' ? device.gpu ?? 'yes' : device.webgpu === 'no-api' ? 'not supported' : 'no GPU adapter', device.webgpu === 'yes' ? 'good' : 'bad'],
    ['Half-precision GPU', device.shaderF16 ? 'yes' : 'no (slower LLM builds)', device.shaderF16 ? 'good' : 'warn'],
    ['Multi-thread CPU', device.crossOriginIsolated ? 'yes' : 'no (CPU path will be slower)', device.crossOriginIsolated ? 'good' : 'warn'],
    ['Free storage', device.storageFreeGB ? `${device.storageFreeGB} GB available to the app` : 'unknown'],
    ['Wake lock / vibrate', `${device.wakeLock ? 'yes' : 'no'} / ${device.vibrate ? 'yes' : 'no'}`],
  ]);
  if (device.webgpu !== 'yes') $<HTMLSelectElement>('asr-device').value = 'wasm';

  const choices = llmChoices(device.shaderF16);
  $('llm-model').replaceChildren(
    ...choices.map((c) => Object.assign(document.createElement('option'), { value: c.id, textContent: c.label })),
  );
  log(`Device: ${JSON.stringify(device)}`);
  renderResults();
}

// ---------- 2. audio ----------

const recorder = new MicRecorder();
let recording = false;
let recTimer = 0;
const dark = new DarkScreen();
dark.onExit = () => {
  if (recording) void toggleRecord();
};

async function useAudio(data: ArrayBuffer, source: string, blob: Blob, extra: Partial<NonNullable<Results['audio']>> = {}) {
  log(`Decoding ${source} (${(data.byteLength / 1e6).toFixed(1)} MB)…`);
  audio = await decodeToMono16k(data);
  asrChunks = diar = labelled = null;
  const durationSec = audio.length / SAMPLE_RATE;
  if (extra.wallClockSec !== undefined) extra.lostSec = +Math.max(0, extra.wallClockSec - durationSec).toFixed(1);
  results.audio = { source, durationSec: +durationSec.toFixed(1), ...extra };
  delete results.asr;
  delete results.diarization;
  delete results.llm;

  const player = $<HTMLAudioElement>('player');
  if (player.src) URL.revokeObjectURL(player.src);
  player.src = URL.createObjectURL(blob);
  player.hidden = false;

  const facts: Fact[] = [
    ['Source', source],
    ['Length', formatTime(durationSec)],
  ];
  if (extra.wallClockSec !== undefined) {
    const lost = extra.lostSec ?? 0;
    facts.push(['Screen-off / app-switch time', fmtSec(extra.hiddenSec ?? 0)]);
    facts.push(['Audio lost', lost > 1.5 ? `${fmtSec(lost)} — recording paused in background` : 'none', lost > 1.5 ? 'bad' : 'good']);
  }
  if (durationSec < 120) facts.push(['Note', 'Use 3–5 min of real conversation for meaningful numbers', 'warn']);
  setFacts('audio-facts', facts);
  for (const id of ['asr-facts', 'diar-facts', 'llm-facts']) setFacts(id, []);
  $('timeline').hidden = $('transcript-wrap').hidden = $('llm-out-wrap').hidden = true;
  renderResults();
}

async function toggleRecord() {
  const btn = $<HTMLButtonElement>('btn-rec');
  const status = $('rec-status');
  if (!recording) {
    await recorder.start();
    recording = true;
    btn.textContent = '■ Stop';
    btn.classList.add('recording');
    status.hidden = false;
    recTimer = window.setInterval(() => (status.textContent = `Recording ${formatTime(recorder.elapsedSec)}`), 500);
    if ($<HTMLInputElement>('chk-dark').checked) {
      const locked = await dark.enter(() => recorder.elapsedSec);
      log(`Dark screen on, wake lock ${locked ? 'acquired' : 'FAILED'}`);
    }
    return;
  }
  recording = false;
  clearInterval(recTimer);
  btn.textContent = '● Record';
  btn.classList.remove('recording');
  status.hidden = true;
  if (!document.getElementById('dark-overlay')!.hidden) await dark.exit();
  const rec = await recorder.stop();
  const data = await rec.blob.arrayBuffer();
  await guard('Loading recording', () =>
    useAudio(data, 'microphone', rec.blob, { wallClockSec: +rec.wallClockSec.toFixed(1), hiddenSec: +rec.hiddenSec.toFixed(1) }),
  );
}

// ---------- worker plumbing ----------

function runMlWorker<T extends WorkerEvent['type']>(req: WorkerRequest, done: T, progressId: string): Promise<Extract<WorkerEvent, { type: T }>> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<WorkerEvent>) => {
      const m = e.data;
      if (m.type === 'progress') progress(progressId, m.text, m.ratio);
      else if (m.type === 'log') log(m.text);
      else if (m.type === 'error') {
        worker.terminate();
        reject(new Error(m.message));
      } else if (m.type === done) {
        worker.terminate();
        resolve(m as Extract<WorkerEvent, { type: T }>);
      }
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message || 'worker crashed (possibly out of memory)'));
    };
    worker.postMessage(req);
  });
}

// ---------- 3. speech-to-text ----------

async function runAsr() {
  if (!audio) return;
  const model = $<HTMLSelectElement>('asr-model').value;
  const deviceSel = $<HTMLSelectElement>('asr-device').value as 'webgpu' | 'wasm';
  const task = $<HTMLSelectElement>('asr-task').value as 'translate' | 'transcribe';
  const language = $<HTMLSelectElement>('asr-lang').value;
  const windows = splitAtSilence(audio);
  const audioSec = audio.length / SAMPLE_RATE;
  log(`ASR ${model} on ${deviceSel}, ${task}, lang=${language}, ${windows.length} windows`);
  progress('asr-progress', 'Starting…', 0);

  const r = await runMlWorker({ type: 'asr', audio: audio.slice(), windows, model, device: deviceSel, task, language }, 'asr-done', 'asr-progress');
  asrChunks = r.chunks;
  const speed = audioSec / sec(r.processMs);
  const hourMin = 60 / speed;
  results.asr = {
    model,
    device: deviceSel,
    task,
    language,
    loadSec: +sec(r.loadMs).toFixed(1),
    firstWindowSec: +sec(r.firstWindowMs).toFixed(1),
    processSec: +sec(r.processMs).toFixed(1),
    xRealtime: +speed.toFixed(2),
    estHourMin: +hourMin.toFixed(1),
    chunks: r.chunks.length,
    sampleText: r.chunks.map((c) => c.text).join(' ').slice(0, 600),
  };
  progress('asr-progress', 'Done', 1);
  setFacts('asr-facts', [
    ['Model load + warm-up', fmtSec(sec(r.loadMs))],
    ['Processing', fmtSec(sec(r.processMs))],
    ['Speed', `${speed.toFixed(2)}× real-time`, speedClass(speed)],
    ['1-hour meeting ≈', `${hourMin.toFixed(0)} min`, hourClass(hourMin)],
  ]);
  updateTranscript();
  updateOverall();
}

// ---------- 4. speakers ----------

async function runDiar() {
  if (!audio) return;
  const audioSec = audio.length / SAMPLE_RATE;
  progress('diar-progress', 'Starting…', 0);
  const r = await runMlWorker({ type: 'diarize', audio: audio.slice() }, 'diarize-done', 'diar-progress');
  diar = { segments: r.segments, groups: r.groups };
  const processSec = sec(r.segmentMs + r.embedMs);
  const speed = audioSec / processSec;
  results.diarization = {
    loadSec: +sec(r.loadMs).toFixed(1),
    segmentSec: +sec(r.segmentMs).toFixed(1),
    voiceprintSec: +sec(r.embedMs).toFixed(1),
    xRealtime: +speed.toFixed(2),
    estHourMin: +(60 / speed).toFixed(1),
    voiceprints: r.groups.filter((g) => g.embedding).length,
  };
  progress('diar-progress', 'Done', 1);
  recluster();
  updateOverall();
}

function recluster() {
  if (!diar || !results.diarization) return;
  const threshold = Number($<HTMLInputElement>('diar-threshold').value) || 0.5;
  const groupSpeaker = assignSpeakers(diar.groups, threshold);
  const turns = toTurns(diar.segments, groupSpeaker);
  const speakers = new Set(groupSpeaker).size;
  const d = results.diarization;
  d.threshold = threshold;
  d.speakersFound = speakers;
  d.turns = turns.length;
  const speed = d.xRealtime as number;
  setFacts('diar-facts', [
    ['Model load', fmtSec(d.loadSec as number)],
    ['Processing', fmtSec((d.segmentSec as number) + (d.voiceprintSec as number))],
    ['Speed', `${speed.toFixed(1)}× real-time`, speedClass(speed)],
    ['1-hour meeting ≈', `${(60 / speed).toFixed(0)} min`, hourClass(60 / speed)],
    ['Speakers found', `${speakers} (is that right? adjust threshold and re-group)`],
  ]);

  const total = audio!.length / SAMPLE_RATE;
  const tl = $('timeline');
  tl.hidden = false;
  const bars: HTMLElement[] = [];
  let cursor = 0;
  for (const t of turns) {
    if (t.start > cursor) bars.push(bar(t.start - cursor, 'transparent'));
    bars.push(bar(t.end - Math.max(cursor, t.start), SPEAKER_COLORS[t.speaker % SPEAKER_COLORS.length], `Speaker ${t.speaker + 1}`));
    cursor = Math.max(cursor, t.end);
  }
  tl.replaceChildren(...bars);
  function bar(dur: number, color: string, title = '') {
    const el = document.createElement('div');
    el.style.width = `${(Math.max(0, dur) / total) * 100}%`;
    el.style.background = color;
    el.title = title;
    return el;
  }
  labelled = asrChunks ? labelChunks(asrChunks, turns) : null;
  updateTranscript();
  renderResults();
}

function updateTranscript() {
  const wrap = $('transcript-wrap');
  const rows = labelled ?? asrChunks?.map((c) => ({ ...c, speaker: -1 }));
  if (!rows?.length) return;
  wrap.hidden = false;
  $('transcript').replaceChildren(
    ...mergeTurns(rows).map((r) => {
      const p = document.createElement('div');
      const ts = Object.assign(document.createElement('span'), { className: 'ts', textContent: formatTime(r.start) });
      p.append(ts);
      if (r.speaker >= 0) {
        const who = Object.assign(document.createElement('span'), { className: 'who', textContent: `Speaker ${r.speaker + 1}` });
        who.style.color = SPEAKER_COLORS[r.speaker % SPEAKER_COLORS.length];
        p.append(who);
      }
      p.append(r.text);
      return p;
    }),
  );
}

function mergeTurns(rows: LabelledChunk[]): LabelledChunk[] {
  const out: LabelledChunk[] = [];
  for (const r of rows) {
    const last = out.at(-1);
    if (last && last.speaker === r.speaker) {
      last.text += ' ' + r.text;
      last.end = r.end;
    } else out.push({ ...r });
  }
  return out;
}

function transcriptForLlm(): string {
  const rows = labelled ?? asrChunks?.map((c) => ({ ...c, speaker: -1 })) ?? [];
  return mergeTurns(rows)
    .map((r) => `[${formatTime(r.start)}] ${r.speaker >= 0 ? `Speaker ${r.speaker + 1}: ` : ''}${r.text}`)
    .join('\n');
}

// ---------- 5. summary ----------

async function runLlm() {
  if (!asrChunks?.length) return;
  const model = $<HTMLSelectElement>('llm-model').value;
  progress('llm-progress', 'Starting…', 0);
  $('llm-out-wrap').hidden = false;
  const out = $('llm-out');
  out.textContent = '';
  const r = await summarise(model, transcriptForLlm(), (text, ratio) => progress('llm-progress', text, ratio), (text) => (out.textContent = text));
  const hourMin = estimateHourSec(r) / 60;
  const { text, ...metrics } = r;
  results.llm = { ...metrics, estHourMin: +hourMin.toFixed(1), outputChars: text.length };
  progress('llm-progress', 'Done', 1);
  setFacts('llm-facts', [
    ['Model load', fmtSec(sec(r.loadMs))],
    ['Reading speed', `${r.prefillTokPerSec.toFixed(0)} tokens/s`],
    ['Writing speed', `${r.decodeTokPerSec.toFixed(1)} tokens/s`, r.decodeTokPerSec >= 8 ? 'good' : r.decodeTokPerSec >= 4 ? 'warn' : 'bad'],
    ['1-hour meeting ≈', Number.isFinite(hourMin) ? `${hourMin.toFixed(0)} min` : 'n/a', hourClass(hourMin)],
    ...(r.transcriptTruncated ? ([['Note', 'Transcript was cut to fit this single-pass test', 'warn']] as Fact[]) : []),
  ]);
  updateOverall();
}

function updateOverall() {
  const parts = [results.asr?.estHourMin, results.diarization?.estHourMin, results.llm?.estHourMin].filter((x): x is number => typeof x === 'number');
  if (parts.length) results.estimateForOneHourMeetingMin = +parts.reduce((a, b) => a + b, 0).toFixed(1);
  renderResults();
}

// ---------- 6. screen & alerts ----------

async function testDark() {
  const start = performance.now();
  const ok = await dark.enter(() => (performance.now() - start) / 1000);
  results.screen = { ...results.screen, wakeLock: ok ? 'acquired' : 'failed' };
  setFacts('screen-facts', [['Wake lock', ok ? 'acquired — screen will stay on' : 'failed', ok ? 'good' : 'bad']]);
  renderResults();
}

async function testAlarm() {
  const { vibrated } = await playHandoverAlarm();
  results.screen = { ...results.screen, alarmPlayed: true, vibrated };
  setFacts('screen-facts', [
    ...(results.screen.wakeLock ? ([['Wake lock', String(results.screen.wakeLock)]] as Fact[]) : []),
    ['Vibration', vibrated ? 'triggered' : 'not supported', vibrated ? 'good' : 'warn'],
    ['Alarm sound', 'played — was it loud enough?'],
  ]);
  renderResults();
}

// ---------- results export ----------

async function copyResults() {
  await navigator.clipboard.writeText(JSON.stringify(results, null, 2));
  $('btn-copy').textContent = 'Copied ✓';
  setTimeout(() => ($('btn-copy').textContent = 'Copy'), 1500);
}

async function shareResults() {
  const text = JSON.stringify(results, null, 2);
  if (navigator.share) await navigator.share({ title: 'MeetNotes benchmark', text }).catch(() => {});
  else await copyResults();
}

function downloadResults() {
  const name = `meetnotes-benchmark-${(device?.model || 'phone').replace(/\W+/g, '-')}-${Date.now()}.json`;
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })),
    download: name,
  });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------- wiring ----------

$('btn-rec').onclick = () => void toggleRecord().catch((e) => alert(`Microphone error: ${e.message}`));
$<HTMLInputElement>('file-in').onchange = (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) void guard('Loading file', async () => useAudio(await file.arrayBuffer(), file.name, file));
};
$('btn-sample').onclick = () =>
  void guard('Loading sample', async () => {
    const pick = $<HTMLSelectElement>('sample-pick');
    const blob = await (await fetch(SAMPLE_BASE + pick.value)).blob();
    await useAudio(await blob.arrayBuffer(), `sample: ${pick.value}`, blob);
  });
$('btn-asr').onclick = () => void guard('Speech-to-text', runAsr);
$('btn-diar').onclick = () => void guard('Speaker separation', runDiar);
$('btn-recluster').onclick = () => recluster();
$('btn-llm').onclick = () => void guard('Summary', runLlm);
$('btn-all').onclick = () =>
  void guard('Full run', async () => {
    await runAsr();
    await runDiar();
    if (device?.webgpu === 'yes') await runLlm();
  });
$('btn-dark').onclick = () => void testDark();
$('btn-alarm').onclick = () => void testAlarm();
$('btn-copy').onclick = () => void copyResults();
$('btn-share').onclick = () => void shareResults();
$('btn-download').onclick = downloadResults;

void initDevice().then(refreshButtons);
renderResults();
