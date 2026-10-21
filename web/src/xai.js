// Explanations computed on the device from ONE forward pass.
//
// JavaScript port of src/cropxai/explain/closed_form.py. The exported model
// returns logits and the final feature maps A (K x H x W); the head is
// global-average-pool -> one Linear layer, so the gradient of class score y_c
// w.r.t. A^k is the constant g_k = W[c,k] / (H*W) and Grad-CAM++ has a closed
// form (no backward pass). test/parity.test.mjs checks this file against the
// Python implementation on the same model and image.

export const EPS = 1e-6; // same constant pytorch-grad-cam uses

/** Raw Grad-CAM++ map (H*W, before normalisation) for class `cls`. */
export function closedFormGradCamPP(features, [k, h, w], weight, cls) {
  const hw = h * w;
  const cam = new Float64Array(hw);
  for (let c = 0; c < k; c++) {
    const g = weight[cls * k + c] / hw;
    if (g <= 0) continue; // relu(g) = 0: this channel gets no weight
    const fmap = features.subarray(c * hw, (c + 1) * hw);
    let s = 0;
    for (let i = 0; i < hw; i++) s += fmap[i];
    const alpha = (g * g) / (2 * g * g + s * g * g * g + EPS);
    const wk = hw * g * alpha;
    for (let i = 0; i < hw; i++) cam[i] += wk * fmap[i];
  }
  for (let i = 0; i < hw; i++) if (cam[i] < 0) cam[i] = 0;
  return cam;
}

/** Min-max normalise to [0, 1] in place (same 1e-7 guard as the library). */
export function minMax(x) {
  let lo = Infinity;
  for (const v of x) if (v < lo) lo = v;
  let hi = -Infinity;
  for (let i = 0; i < x.length; i++) {
    x[i] -= lo;
    if (x[i] > hi) hi = x[i];
  }
  for (let i = 0; i < x.length; i++) x[i] /= 1e-7 + hi;
  return x;
}

/** Bilinear resize of one channel, PyTorch `interpolate(..., align_corners=False)` semantics. */
export function resizeBilinear(src, sh, sw, dh, dw) {
  const out = new Float64Array(dh * dw);
  const index = (d, inSize, outSize) => {
    const s = Math.max(0, ((d + 0.5) * inSize) / outSize - 0.5);
    const i0 = Math.floor(s);
    const i1 = i0 < inSize - 1 ? i0 + 1 : i0;
    return [i0, i1, s - i0];
  };
  const cols = Array.from({ length: dw }, (_, x) => index(x, sw, dw));
  for (let y = 0; y < dh; y++) {
    const [y0, y1, ly] = index(y, sh, dh);
    for (let x = 0; x < dw; x++) {
      const [x0, x1, lx] = cols[x];
      const top = src[y0 * sw + x0] * (1 - lx) + src[y0 * sw + x1] * lx;
      const bottom = src[y1 * sw + x0] * (1 - lx) + src[y1 * sw + x1] * lx;
      out[y * dw + x] = top * (1 - ly) + bottom * ly;
    }
  }
  return out;
}

/** Heatmap in [0, 1] at `size` x `size`, exactly as `to_heatmap` in Python. */
export function heatmap(features, shape, weight, cls, size) {
  const [, h, w] = shape;
  const cam = minMax(closedFormGradCamPP(features, shape, weight, cls));
  return Float32Array.from(minMax(resizeBilinear(cam, h, w, size, size)));
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function topK(probs, k) {
  return Array.from(probs, (p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, k);
}

/**
 * Exact decomposition of the class score: logit_c = sum_k W[c,k] * mean(A^k) + b_c.
 * Because the head is linear this is not an approximation: it is how the model
 * computed the score. Returns every channel's contribution, largest first.
 */
export function scoreDecomposition(features, [k, h, w], weight, bias, cls) {
  const hw = h * w;
  const channels = [];
  let total = bias[cls];
  for (let c = 0; c < k; c++) {
    let s = 0;
    for (let i = c * hw; i < (c + 1) * hw; i++) s += features[i];
    const pooled = s / hw;
    const contribution = weight[cls * k + c] * pooled;
    total += contribution;
    channels.push({ channel: c, pooled, weight: weight[cls * k + c], contribution });
  }
  channels.sort((a, b) => b.contribution - a.contribution);
  return { logit: total, bias: bias[cls], channels };
}

/** One feature map, min-max normalised and upsampled, for display. */
export function featureMap(features, [, h, w], channel, size) {
  const hw = h * w;
  const fmap = Float64Array.from(features.subarray(channel * hw, (channel + 1) * hw));
  return Float32Array.from(minMax(resizeBilinear(minMax(fmap), h, w, size, size)));
}

/** Deterministic PRNG so the random-baseline check is reproducible. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Pixel indices to hide for the deletion check: the `fraction` most highlighted
 * pixels of `heat`, or the same number chosen at random (the baseline).
 */
export function pixelsToHide(heat, fraction, random = false, seed = 0) {
  const n = Math.round(heat.length * fraction);
  const order = Array.from(heat.keys());
  if (random) {
    const rand = mulberry32(seed);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
  } else {
    order.sort((a, b) => heat[b] - heat[a]);
  }
  return order.slice(0, n);
}

/**
 * Copy of a normalised CHW input with the given pixels set to 0, i.e. the
 * dataset mean colour after ImageNet normalisation (the deletion baseline).
 */
export function hidePixels(input, size, pixels) {
  const out = Float32Array.from(input);
  const plane = size * size;
  for (const p of pixels) for (let c = 0; c < 3; c++) out[c * plane + p] = 0;
  return out;
}
