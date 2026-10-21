// The app must compute what the research code computes. Fixtures come from
// web/test/make_fixtures.py (run it first).

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import * as ort from 'onnxruntime-web';
import { preprocess } from '../src/preprocess.js';
import { heatmap, scoreDecomposition } from '../src/xai.js';

const dir = new URL('./fixtures/', import.meta.url);
const read = (name) => readFileSync(new URL(name, dir));
const f32 = (name) => { const b = read(name); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const fx = JSON.parse(read('fixtures.json'));
const meta = JSON.parse(read('bundle/model.json'));
const classifier = (() => { const b = read('bundle/classifier.bin'); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); })();
const [k] = meta.feature_shape;
const n = meta.classes.length;
const weight = classifier.subarray(0, n * k);
const bias = classifier.subarray(n * k);

function maxAbsDiff(a, b) {
  assert.equal(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

test('preprocessing matches albumentations (resize, centre crop, normalise)', () => {
  const { width, height } = fx.photo;
  const { crop, input } = preprocess(read('photo_rgba.bin'), width, height, meta);
  const ref = read('crop_rgb.bin');
  // A few values round differently from OpenCV's optimised kernels: allow one level on <0.5% of them.
  assert.ok(maxAbsDiff(crop, ref) <= 1, 'crop pixels differ by more than one level');
  let differing = 0;
  for (let i = 0; i < ref.length; i++) differing += crop[i] !== ref[i];
  assert.ok(differing / ref.length < 0.005, `${differing} of ${ref.length} values differ`);
  assert.ok(maxAbsDiff(input, f32('input.bin')) < 1.1 / 255 / 0.224);
});

test('ONNX Runtime Web reproduces the PyTorch logits and feature maps', async () => {
  const session = await ort.InferenceSession.create(read('bundle/model.onnx'));
  const s = fx.image_size;
  const out = await session.run({ image: new ort.Tensor('float32', f32('input.bin'), [1, 3, s, s]) });
  assert.ok(maxAbsDiff(out.logits.data, f32('logits.bin')) < 1e-3);
  assert.ok(maxAbsDiff(out.features.data, f32('features.bin')) < 1e-3);
});

test('closed-form Grad-CAM++ heatmaps match the Python implementation', () => {
  const s = fx.image_size;
  const features = f32('features.bin');
  const ref = f32('heatmaps.bin');
  fx.explained_classes.forEach((cls, i) => {
    const ours = heatmap(features, meta.feature_shape, weight, cls, s);
    const diff = maxAbsDiff(ours, ref.subarray(i * s * s, (i + 1) * s * s));
    assert.ok(diff < 1e-4, `class ${cls}: max difference ${diff}`);
  });
});

test('score decomposition adds up to the model logit', () => {
  const logits = f32('logits.bin');
  for (const cls of fx.explained_classes) {
    const d = scoreDecomposition(f32('features.bin'), meta.feature_shape, weight, bias, cls);
    assert.ok(Math.abs(d.logit - logits[cls]) < 1e-3, `class ${cls}: ${d.logit} vs ${logits[cls]}`);
  }
});
