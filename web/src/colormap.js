// Heatmap colours and canvas helpers.

// Inferno-like ramp (dark purple -> red -> yellow): stays readable over green
// leaves and does not rely on red/green alone.
const STOPS = [
  [0.0, [0, 0, 4]],
  [0.25, [87, 16, 110]],
  [0.5, [188, 55, 84]],
  [0.75, [249, 142, 9]],
  [1.0, [252, 255, 164]],
];

export function colour(v) {
  const t = Math.min(1, Math.max(0, v));
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i];
    const [t0, c0] = STOPS[i - 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return c0.map((a, j) => Math.round(a + (c1[j] - a) * f));
    }
  }
  return STOPS.at(-1)[1];
}

/** Draw an RGB uint8 image (size x size) to a canvas. */
export function drawRGB(canvas, rgb, size) {
  canvas.width = canvas.height = size;
  const img = new ImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    img.data.set([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255], i * 4);
  }
  canvas.getContext('2d').putImageData(img, 0, 0);
}

/** Photo with the heatmap blended on top; weak areas stay see-through. */
export function drawOverlay(canvas, rgb, heat, size, opacity) {
  canvas.width = canvas.height = size;
  const img = new ImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const a = opacity * Math.min(1, heat[i] * 1.25);
    const [r, g, b] = colour(heat[i]);
    img.data[i * 4] = rgb[i * 3] * (1 - a) + r * a;
    img.data[i * 4 + 1] = rgb[i * 3 + 1] * (1 - a) + g * a;
    img.data[i * 4 + 2] = rgb[i * 3 + 2] * (1 - a) + b * a;
    img.data[i * 4 + 3] = 255;
  }
  canvas.getContext('2d').putImageData(img, 0, 0);
}

/** A map in [0, 1] drawn on its own with the colour ramp. */
export function drawMap(canvas, map, size) {
  canvas.width = canvas.height = size;
  const img = new ImageData(size, size);
  for (let i = 0; i < size * size; i++) img.data.set([...colour(map[i]), 255], i * 4);
  canvas.getContext('2d').putImageData(img, 0, 0);
}

/** Image pixels with some of them hidden (shown grey), to show what the check removed. */
export function drawHidden(canvas, rgb, size, pixels) {
  const copy = Uint8ClampedArray.from(rgb);
  for (const p of pixels) copy.set([124, 116, 104], p * 3); // ImageNet mean colour
  drawRGB(canvas, copy, size);
}

export function legendGradient() {
  const parts = STOPS.map(([t, c]) => `rgb(${c.join(',')}) ${t * 100}%`);
  return `linear-gradient(to right, ${parts.join(', ')})`;
}
