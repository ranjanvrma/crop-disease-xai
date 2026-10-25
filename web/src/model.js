// Loads an exported bundle (web/public/models/<run>/, written by scripts/export_web.py)
// and runs it with ONNX Runtime Web on the CPU through WebAssembly, which every
// current phone and desktop browser supports.

import * as ort from 'onnxruntime-web/wasm';

// The WebAssembly runtime is bundled by Vite and served with the app (no CDN),
// so it works offline.
const BASE = new URL('./', document.baseURI).href;
// Threads need cross-origin isolation, which most static hosts don't give; ORT
// falls back to one thread on its own, so ask for threads only when available.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export async function loadBundle(base = `${BASE}models/mnv3_ft_field/`) {
  // The service worker serves these from its own cache after the first visit, so the
  // browser's HTTP cache is skipped (it can refuse a file this large in private windows).
  const get = (name) => fetch(`${base}${name}`, { cache: 'no-store' });
  const [meta, remedies, classifier, onnx] = await Promise.all([
    get('model.json').then(okJson),
    get('remedies.json').then(okJson),
    get('classifier.bin').then(okBuffer),
    get('model.onnx').then(okBuffer),
  ]);
  const [k] = meta.feature_shape;
  const n = meta.classes.length;
  const all = new Float32Array(classifier);
  if (all.length !== n * k + n) throw new Error('classifier.bin does not match model.json');
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(new Uint8Array(onnx), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return {
    meta,
    remedies,
    weight: all.subarray(0, n * k),
    bias: all.subarray(n * k),
    session,
    loadMs: performance.now() - t0,
  };
}

/** One forward pass -> { logits: Float32Array, features: Float32Array (K*H*W), ms }. */
export async function run(bundle, input) {
  const s = bundle.meta.image_size;
  const t0 = performance.now();
  const out = await bundle.session.run({ image: new ort.Tensor('float32', input, [1, 3, s, s]) });
  return { logits: out.logits.data, features: out.features.data, ms: performance.now() - t0 };
}

async function okJson(r) {
  if (!r.ok) throw new Error(`${r.url}: ${r.status}`);
  return r.json();
}

async function okBuffer(r) {
  if (!r.ok) throw new Error(`${r.url}: ${r.status}`);
  return r.arrayBuffer();
}
