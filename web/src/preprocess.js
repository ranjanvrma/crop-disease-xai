// Image preprocessing identical to the Python evaluation pipeline
// (src/cropxai/data/transforms.py: eval_transform):
//   resize the short side to `resize_short` (cv2 INTER_LINEAR, half-pixel centres)
//   -> centre-crop `image_size` -> /255 -> ImageNet mean/std -> CHW float32.
// test/parity.test.mjs checks it against albumentations on the same image.

/** Output size of albumentations SmallestMaxSize: short side == maxSize, aspect kept, truncated like int(). */
export function shortSideSize(w, h, maxSize) {
  const scale = maxSize / Math.min(w, h);
  return [Math.max(1, Math.trunc(w * scale)), Math.max(1, Math.trunc(h * scale))];
}

/**
 * Source indices and 11-bit fixed-point weights for one output coordinate,
 * as cv2.resize INTER_LINEAR computes them for uint8 images (float32 maths).
 */
function linearCoeffs(d, inSize, outSize) {
  let f = Math.fround((d + 0.5) * (inSize / outSize) - 0.5);
  let i = Math.floor(f);
  f = Math.fround(f - i);
  if (i < 0) { i = 0; f = 0; }
  if (i >= inSize - 1) { i = inSize - 1; f = 0; }
  const w0 = Math.round(Math.fround(1 - f) * 2048);
  return [i, Math.min(i + 1, inSize - 1), w0, 2048 - w0];
}

/**
 * RGBA uint8 -> RGB uint8 resize with cv2.resize INTER_LINEAR fixed-point
 * arithmetic (the vectorised path OpenCV takes on x86 and ARM), so the model
 * sees the same pixels as in evaluation. Matches to within one intensity level.
 */
export function resizeLinearRGB(rgba, sw, sh, dw, dh) {
  const out = new Uint8ClampedArray(dw * dh * 3);
  const cols = Array.from({ length: dw }, (_, x) => linearCoeffs(x, sw, dw));
  const p = (y, x, c) => rgba[(y * sw + x) * 4 + c];
  for (let y = 0; y < dh; y++) {
    const [y0, y1, b0, b1] = linearCoeffs(y, sh, dh);
    for (let x = 0; x < dw; x++) {
      const [x0, x1, a0, a1] = cols[x];
      for (let c = 0; c < 3; c++) {
        const top = p(y0, x0, c) * a0 + p(y0, x1, c) * a1;
        const bottom = p(y1, x0, c) * a0 + p(y1, x1, c) * a1;
        out[(y * dw + x) * 3 + c] = (((b0 * (top >> 4)) >> 16) + ((b1 * (bottom >> 4)) >> 16) + 2) >> 2;
      }
    }
  }
  return out;
}

/** Top-left corner of the centre crop, as albumentations CenterCrop computes it. */
export function centreCropOrigin(w, h, size) {
  return [Math.floor((w - size) / 2), Math.floor((h - size) / 2)];
}

/**
 * RGBA pixels of any size -> { input: Float32Array(3*S*S), crop: RGB uint8 S*S, box }.
 * `box` is the crop's position in the source image (fractions), for drawing it.
 */
export function preprocess(rgba, w, h, meta) {
  const s = meta.image_size;
  const [rw, rh] = shortSideSize(w, h, meta.resize_short);
  const resized = resizeLinearRGB(rgba, w, h, rw, rh);
  const [ox, oy] = centreCropOrigin(rw, rh, s);
  const crop = new Uint8ClampedArray(s * s * 3);
  for (let y = 0; y < s; y++) {
    const from = ((oy + y) * rw + ox) * 3;
    crop.set(resized.subarray(from, from + s * 3), y * s * 3);
  }
  const plane = s * s;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * plane + i] = (crop[i * 3 + c] / 255 - meta.mean[c]) / meta.std[c];
    }
  }
  const box = { x: ox / rw, y: oy / rh, w: s / rw, h: s / rh };
  return { input, crop, box };
}
