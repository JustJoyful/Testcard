import { test } from 'node:test';
import assert from 'node:assert/strict';
import { scoreText, secondsText, shortHash, middleEllipsis, clockText } from '../src/format.js';

test('scores round for display only and are never clamped', () => {
  assert.equal(scoreText(5.25), '5.25');
  assert.equal(scoreText(3.596115361254694), '3.60');
  assert.equal(scoreText(0.4), '0.40');
  assert.equal(scoreText(-0.25), '-0.25');
  for (const bad of [NaN, Infinity, null, '3.6']) assert.throws(() => scoreText(bad), TypeError);
});

test('seconds use two decimals and a floor marker for the head', () => {
  assert.equal(secondsText(0.003, { head: true }), '< 0.01 s');
  assert.equal(secondsText(0.00724809399980586, { head: true }), '< 0.01 s');
  assert.equal(secondsText(23.892053884992492), '23.89 s');
  assert.equal(secondsText(0.003), '0.00 s');
  assert.equal(secondsText(undefined), '');
});

test('hashes, filenames and clocks', () => {
  assert.equal(shortHash('4c9782aefca4b81da4d15da3'), '4c9782ae…');
  const name = middleEllipsis('Zeke Scream Aot final final cut.mp4', 22);
  assert.equal(Array.from(name).length, 22);
  assert.ok(name.startsWith('Zeke') && name.endsWith('.mp4') && name.includes('…'));
  assert.equal(middleEllipsis('short.mp4'), 'short.mp4');
  assert.equal(Array.from(middleEllipsis('🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬🎬.mp4')).length, 22);
  assert.equal(clockText(7000), '00:00:07');
  assert.equal(clockText(3723000), '01:02:03');
});
