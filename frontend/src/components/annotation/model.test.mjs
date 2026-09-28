import assert from 'node:assert/strict';
import test from 'node:test';
import { fit, imageToScreen, screenToImage, clamp } from './model.ts';

test('fit preserves aspect ratio inside display bounds', () => {
  assert.deepEqual(fit(1024, 512, 768, 768), { width: 768, height: 384, scale: .75 });
});

test('coordinates round trip through zoom, pan, and offset', () => {
  const original = { x: 412.5, y: 228.3 };
  const offset = { x: 173, y: -29 };
  const scale = 2.125;
  const result = screenToImage(imageToScreen(original, offset, scale), offset, scale);
  assert.ok(Math.abs(result.x - original.x) < 1e-9);
  assert.ok(Math.abs(result.y - original.y) < 1e-9);
  assert.deepEqual(clamp({ x: -10, y: 2000 }, 1024, 768), { x: 0, y: 768 });
});
