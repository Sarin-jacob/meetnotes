import { test } from 'node:test';
import assert from 'node:assert/strict';
import { agglomerate, assignSpeakers, labelChunks, toTurns, type SpeakerGroup } from './diarize.ts';

const v = (...xs: number[]) => Float32Array.from(xs);

test('agglomerate groups similar vectors and keeps dissimilar ones apart', () => {
  const labels = agglomerate([v(1, 0, 0), v(0.95, 0.1, 0), v(0, 1, 0), v(0.05, 0.98, 0), v(0, 0, 1)], 0.8);
  assert.deepEqual(labels, [0, 0, 1, 1, 2]);
});

test('agglomerate with an empty input returns nothing', () => {
  assert.deepEqual(agglomerate([], 0.5), []);
});

test('assignSpeakers numbers by first appearance and fills groups without voiceprints', () => {
  const groups: SpeakerGroup[] = [
    { embedding: v(0, 1), speechSec: 3, firstStart: 5 },
    { embedding: v(1, 0), speechSec: 3, firstStart: 0 },
    { embedding: null, speechSec: 0.4, firstStart: 6 },
    { embedding: v(0.99, 0.05), speechSec: 3, firstStart: 12 },
  ];
  assert.deepEqual(assignSpeakers(groups, 0.7), [1, 0, 1, 0]);
});

test('toTurns joins short same-speaker gaps but splits on speaker change', () => {
  const turns = toTurns(
    [
      { start: 0, end: 2, group: 0, overlap: false },
      { start: 2.5, end: 4, group: 0, overlap: false },
      { start: 4.2, end: 6, group: 1, overlap: false },
    ],
    [0, 1],
  );
  assert.deepEqual(turns, [
    { start: 0, end: 4, speaker: 0 },
    { start: 4.2, end: 6, speaker: 1 },
  ]);
});

test('labelChunks picks the speaker with the most overlap, else the nearest', () => {
  const turns = [
    { start: 0, end: 4, speaker: 0 },
    { start: 4, end: 10, speaker: 1 },
  ];
  const out = labelChunks(
    [
      { start: 3, end: 8, text: 'a' },
      { start: 0, end: 2, text: 'b' },
      { start: 11, end: 12, text: 'c' },
    ],
    turns,
  );
  assert.deepEqual(out.map((c) => c.speaker), [1, 0, 1]);
});
