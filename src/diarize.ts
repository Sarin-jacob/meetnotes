/** A stretch of speech from the segmentation model, tied to a local (per-window) speaker group. */
export interface SpeechSegment {
  start: number;
  end: number;
  group: number;
  /** True when the segmentation model detected two people talking at once. */
  overlap: boolean;
}

/** One local speaker inside one segmentation window, with its voiceprint (null when too little clean audio). */
export interface SpeakerGroup {
  embedding: Float32Array | null;
  speechSec: number;
  firstStart: number;
}

export interface Turn {
  start: number;
  end: number;
  speaker: number;
}

export interface TextChunk {
  start: number;
  end: number;
  text: string;
}

export interface LabelledChunk extends TextChunk {
  speaker: number;
}

export function l2normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na * nb) || 1);
}

/**
 * Agglomerative clustering on cosine similarity with centroid linkage
 * (clusters are represented by the sum of their unit-length members).
 * Merging stops once the most similar pair falls below `threshold`.
 * Returns a cluster index per input, numbered 0..k-1 in input order.
 */
export function agglomerate(vectors: Float32Array[], threshold: number): number[] {
  const n = vectors.length;
  const sums = vectors.map((v) => l2normalize(v));
  const members = vectors.map((_, i) => [i]);
  const alive = new Array<boolean>(n).fill(true);
  const sim = vectors.map(() => new Float32Array(n));
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) sim[i][j] = sim[j][i] = cosine(sums[i], sums[j]);

  for (;;) {
    let bi = -1, bj = -1, best = -Infinity;
    for (let i = 0; i < n; i++) {
      if (!alive[i]) continue;
      for (let j = i + 1; j < n; j++) {
        if (alive[j] && sim[i][j] > best) {
          best = sim[i][j];
          bi = i;
          bj = j;
        }
      }
    }
    if (bi < 0 || best < threshold) break;
    for (let d = 0; d < sums[bi].length; d++) sums[bi][d] += sums[bj][d];
    members[bi].push(...members[bj]);
    alive[bj] = false;
    for (let k = 0; k < n; k++) if (alive[k] && k !== bi) sim[bi][k] = sim[k][bi] = cosine(sums[bi], sums[k]);
  }

  const labels = new Array<number>(n);
  let next = 0;
  for (let i = 0; i < n; i++) {
    if (!alive[i]) continue;
    for (const m of members[i]) labels[m] = next;
    next++;
  }
  return labels;
}

/**
 * Maps each local speaker group to a global speaker number. Speakers are numbered
 * by first appearance; groups without a voiceprint borrow the nearest group in time.
 */
export function assignSpeakers(groups: SpeakerGroup[], threshold: number): number[] {
  const withEmb = groups.map((g, i) => ({ g, i })).filter(({ g }) => g.embedding);
  const clusters = agglomerate(withEmb.map(({ g }) => g.embedding!), threshold);
  const raw = new Array<number>(groups.length).fill(-1);
  withEmb.forEach(({ i }, k) => (raw[i] = clusters[k]));

  for (let i = 0; i < groups.length; i++) {
    if (raw[i] >= 0 || !withEmb.length) continue;
    let nearest = withEmb[0].i;
    for (const { i: j } of withEmb) {
      if (Math.abs(groups[j].firstStart - groups[i].firstStart) < Math.abs(groups[nearest].firstStart - groups[i].firstStart)) nearest = j;
    }
    raw[i] = clusters[withEmb.findIndex(({ i: j }) => j === nearest)];
  }

  // Renumber by first appearance so "Speaker 1" is whoever talks first.
  const order = groups.map((g, i) => ({ t: g.firstStart, c: raw[i] })).sort((a, b) => a.t - b.t);
  const renumber = new Map<number, number>();
  for (const { c } of order) if (c >= 0 && !renumber.has(c)) renumber.set(c, renumber.size);
  return raw.map((c) => renumber.get(c) ?? 0);
}

/** Collapses segments into speaker turns, joining same-speaker segments separated by short pauses. */
export function toTurns(segments: SpeechSegment[], groupSpeaker: number[], maxGapSec = 1): Turn[] {
  const turns: Turn[] = [];
  for (const s of [...segments].sort((a, b) => a.start - b.start)) {
    const speaker = groupSpeaker[s.group];
    const last = turns.at(-1);
    if (last && last.speaker === speaker && s.start - last.end <= maxGapSec) last.end = Math.max(last.end, s.end);
    else turns.push({ start: s.start, end: s.end, speaker });
  }
  return turns;
}

/** Gives each transcript chunk the speaker whose turns overlap it the most. */
export function labelChunks(chunks: TextChunk[], turns: Turn[]): LabelledChunk[] {
  return chunks.map((c) => {
    const talk = new Map<number, number>();
    for (const t of turns) {
      const o = Math.min(c.end, t.end) - Math.max(c.start, t.start);
      if (o > 0) talk.set(t.speaker, (talk.get(t.speaker) ?? 0) + o);
    }
    let speaker = -1, most = 0;
    for (const [s, d] of talk) if (d > most) [speaker, most] = [s, d];
    if (speaker < 0) {
      // No overlap (e.g. speech the segmenter missed): use the closest turn.
      const mid = (c.start + c.end) / 2;
      let dist = Infinity;
      for (const t of turns) {
        const d = Math.min(Math.abs(t.start - mid), Math.abs(t.end - mid));
        if (d < dist) [dist, speaker] = [d, t.speaker];
      }
    }
    return { ...c, speaker: Math.max(0, speaker) };
  });
}
