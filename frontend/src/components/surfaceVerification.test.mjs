import assert from 'node:assert/strict';
import test from 'node:test';
import { displayToRaw, rawToDisplay, swapSurfaceRecords } from './surfaceVerification.js';

test('raw coordinates survive zoom and resize round-trip', () => {
  const raw = { x: 311.25, y: 517.75 };
  for (const scale of [0.25, 0.75, 1, 2.5]) {
    const transform = { scale, offsetX: 18, offsetY: 27, imageWidth: 885, horizontalFlip: false };
    const restored = displayToRaw(rawToDisplay(raw, transform), transform);
    assert.ok(Math.abs(restored.x - raw.x) < 1e-9 && Math.abs(restored.y - raw.y) < 1e-9);
  }
});

test('horizontal display flip maps pixels but preserves canonical raw coordinate', () => {
  const raw = { x: 100, y: 200 };
  const transform = { scale: 0.5, offsetX: 4, offsetY: 6, imageWidth: 800, horizontalFlip: true };
  assert.equal(rawToDisplay(raw, transform).x, 354);
  assert.deepEqual(displayToRaw(rawToDisplay(raw, transform), transform), raw);
});

test('swap changes associations only and retains linked landmarks', () => {
  const surfaces = [{ surface: 'mesial', cej: { x: 12, y: 20 } }, { surface: 'distal', cej: { x: 88, y: 20 } }];
  const swapped = swapSurfaceRecords(surfaces);
  assert.equal(swapped[0].surface, 'distal');
  assert.deepEqual(swapped[0].cej, surfaces[0].cej);
  assert.equal(surfaces[0].surface, 'mesial');
});
