import { CreateWebWorkerMLCEngine, prebuiltAppConfig } from '@mlc-ai/web-llm';

export interface LlmChoice {
  id: string;
  label: string;
}

/** Candidates for 6–8 GB phones, best first. f16 builds need the GPU's `shader-f16` feature. */
const CANDIDATES: Array<{ base: string; name: string }> = [
  { base: 'Qwen3.5-2B-q4f16_1-MLC', name: 'Qwen3.5 2B' },
  { base: 'Qwen3-1.7B-q4f16_1-MLC', name: 'Qwen3 1.7B' },
  { base: 'gemma-2-2b-it-q4f16_1-MLC', name: 'Gemma 2 2B' },
  { base: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', name: 'Llama 3.2 1B' },
  { base: 'gemma3-1b-it-q4f16_1-MLC', name: 'Gemma 3 1B' },
  { base: 'Qwen3.5-0.8B-q4f16_1-MLC', name: 'Qwen3.5 0.8B' },
];

export function llmChoices(hasF16: boolean): LlmChoice[] {
  const byId = new Map(prebuiltAppConfig.model_list.map((m) => [m.model_id, m]));
  return CANDIDATES.flatMap(({ base, name }) => {
    const id = hasF16 ? base : base.replace('q4f16_1', 'q4f32_1');
    const record = byId.get(id);
    if (!record) return [];
    const gb = record.vram_required_MB ? ` · needs ≈${(record.vram_required_MB / 1024).toFixed(1)} GB` : '';
    return [{ id, label: `${name}${hasF16 ? '' : ' (f32)'}${gb}` }];
  });
}

// Keep the prompt inside the 4k-token context window with room for the answer.
const MAX_TRANSCRIPT_CHARS = 10_000;
const MAX_ANSWER_TOKENS = 900;

const SYSTEM = `You write meeting minutes from transcripts. The transcript was machine-translated to English from Hindi/English speech, so expect some errors. Be factual: never invent names, numbers or dates. If something is unclear, say so briefly.`;

const INSTRUCTIONS = `Write the minutes in Markdown with exactly these sections:

## Summary
3–5 sentences.

## Key decisions
- One bullet per decision.

## Action items
- [ ] Owner — task (due date if mentioned)

## Minutes
Topic-by-topic bullet points, each starting with the [mm:ss] time it was discussed.

If a section has nothing, write "None noted."`;

export interface LlmResult {
  model: string;
  loadMs: number;
  totalMs: number;
  transcriptTruncated: boolean;
  promptTokens: number;
  completionTokens: number;
  prefillTokPerSec: number;
  decodeTokPerSec: number;
  timeToFirstTokenSec: number;
  text: string;
}

export async function summarise(
  model: string,
  transcript: string,
  onProgress: (text: string, ratio: number) => void,
  onText: (text: string) => void,
): Promise<LlmResult> {
  const worker = new Worker(new URL('./llm.worker.ts', import.meta.url), { type: 'module' });
  try {
    let t = performance.now();
    const engine = await CreateWebWorkerMLCEngine(worker, model, {
      initProgressCallback: (r) => onProgress(r.text, r.progress),
    });
    const loadMs = performance.now() - t;

    const transcriptTruncated = transcript.length > MAX_TRANSCRIPT_CHARS;
    const body = transcriptTruncated ? transcript.slice(0, MAX_TRANSCRIPT_CHARS) + '\n[…transcript truncated for this test…]' : transcript;

    t = performance.now();
    onProgress('Writing minutes…', 1);
    const stream = await engine.chat.completions.create({
      messages: [
        { role: 'system', content: SYSTEM },
        { role: 'user', content: `Transcript:\n\n${body}\n\n${INSTRUCTIONS}` },
      ],
      stream: true,
      stream_options: { include_usage: true },
      temperature: 0.3,
      max_tokens: MAX_ANSWER_TOKENS,
      extra_body: model.startsWith('Qwen3') ? { enable_thinking: false } : undefined,
    });

    let text = '';
    let usage: { prompt_tokens: number; completion_tokens: number; extra: { prefill_tokens_per_s: number; decode_tokens_per_s: number; time_to_first_token_s: number } } | undefined;
    for await (const chunk of stream) {
      text += chunk.choices[0]?.delta?.content ?? '';
      if (chunk.usage) usage = chunk.usage as typeof usage;
      onText(stripThinking(text));
    }
    const totalMs = performance.now() - t;
    await engine.unload();

    return {
      model,
      loadMs,
      totalMs,
      transcriptTruncated,
      promptTokens: usage?.prompt_tokens ?? 0,
      completionTokens: usage?.completion_tokens ?? 0,
      prefillTokPerSec: usage?.extra.prefill_tokens_per_s ?? 0,
      decodeTokPerSec: usage?.extra.decode_tokens_per_s ?? 0,
      timeToFirstTokenSec: usage?.extra.time_to_first_token_s ?? 0,
      text: stripThinking(text),
    };
  } finally {
    worker.terminate();
  }
}

function stripThinking(s: string): string {
  return s.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim();
}

/**
 * Rough time to summarise a 60-minute meeting with map-reduce: ~10.4k transcript tokens
 * split into 5 chunks (≈2.7k prompt + 250 notes each), then one combine pass (≈1.5k in, 900 out).
 */
export function estimateHourSec(r: LlmResult): number {
  if (!r.prefillTokPerSec || !r.decodeTokPerSec) return NaN;
  const prefill = 5 * 2700 + 1500;
  const decode = 5 * 250 + 900;
  return prefill / r.prefillTokPerSec + decode / r.decodeTokPerSec;
}
