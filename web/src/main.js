import { drawHidden, drawMap, drawOverlay, drawRGB, legendGradient } from './colormap.js';
import { animate } from 'motion';
import { icon } from './icons.js';
import { loadBundle, run } from './model.js';
import { preprocess, shortSideSize } from './preprocess.js';
import {
  featureMap, heatmap, hidePixels, oodScore, pixelsToHide, scoreDecomposition, softmax, topK,
} from './xai.js';
import { enter, growBars, parallax, reduceMotion, tilt } from './ui.js';

const $ = (id) => document.getElementById(id);
const HIDE_FRACTION = 0.2;
// Very large photos are first scaled down by the browser so a phone never holds a
// 50-megapixel bitmap; the exact training resize (preprocess.js) runs after that.
const MAX_DECODE_FACTOR = 4;
// Larger files are refused before decoding, so a phone is never asked to hold a huge bitmap.
const MAX_FILE_MB = 40;
const HERO_SAMPLE = './samples/tomato-early-blight-lab.jpg';

let bundle = null;
let current = null; // the last analysed image
let lastFile = null; // re-analysed when the user switches model
let models = { default: null, models: [] };

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

/** A drop in confidence as text, e.g. "-42 points" or "no change". */
function change(drop) {
  const v = Math.round(-100 * drop);
  return v === 0 ? 'no change' : `${v > 0 ? '+' : ''}${v} points`;
}

const pct = (p) => `${(100 * p).toFixed(p < 0.1 ? 1 : 0)}%`;

function label(i) {
  const name = bundle.meta.classes[i];
  const r = bundle.remedies.classes[name];
  return r ? `${r.crop}: ${r.name}` : name.replaceAll('_', ' ');
}

// ---------- start-up ----------

function showNetwork() {
  const net = $('net');
  net.textContent = navigator.onLine ? 'Online' : 'Offline: still works';
  net.className = `net-chip ${navigator.onLine ? '' : 'warn'}`;
}

function showModelInfo() {
  const m = bundle.meta;
  const acc = m.measured_accuracy;
  const banner = $('banner');
  if (m.demo) {
    banner.className = 'banner danger';
    banner.innerHTML = '<strong>Demo model.</strong> This build uses an untrained model, so every diagnosis is meaningless. It only shows how the app works.';
  } else if (acc) {
    banner.className = 'banner';
    banner.innerHTML = `<strong>Works best on close-up photos of one leaf.</strong> Right ${pct(acc.test_top1)} of the time on lab photos, but only ${pct(acc.field_strict_top1)} on real field photos, so always check the heatmap. <a href="./explain.html#field-gap">Why?</a>`;
  } else {
    banner.className = 'banner';
    banner.innerHTML = '<strong>Works best on close-up photos of one leaf.</strong> Its accuracy on real field photos has not been measured for this model, so always check the heatmap.';
  }
  banner.hidden = false;
  $('modelinfo').textContent = `Model: ${m.arch} (${m.run_name}, seed ${m.seed}, commit ${m.git_commit})${m.demo ? ', demo' : ''}. Loaded in ${Math.round(bundle.loadMs)} ms.`;
}

// models/index.json (scripts/export_site_data.py) lists every exported bundle.
// The choice is kept in the URL (?model=...) so a demo link opens the same model.
async function loadModelList() {
  const r = await fetch('./models/index.json', { cache: 'no-store' });
  if (!r.ok) throw new Error(`models/index.json: ${r.status}`);
  models = await r.json();
  const wanted = new URLSearchParams(location.search).get('model');
  const chosen = models.models.some((m) => m.id === wanted) ? wanted : models.default;
  $('model').innerHTML = models.models.map((m) => `<option value="${esc(m.id)}" ${m.id === chosen ? 'selected' : ''}>${esc(m.label)} (${m.size_mb} MB)</option>`).join('');
  return chosen;
}

async function useModel(id) {
  const info = models.models.find((m) => m.id === id);
  $('modelnote').textContent = info ? info.note : '';
  setStatus(`Loading ${id}…`, 'loading');
  document.body.dataset.ready = 'false';
  $('model').disabled = true;
  try {
    bundle = await loadBundle(new URL(`./models/${id}/`, document.baseURI).href);
    showModelInfo();
    renderSupported();
    setStatus('Ready. Take a photo, choose an image or try a sample.');
    document.body.dataset.ready = 'true';
    heroScan();
    if (lastFile) await analyse(lastFile);
  } catch (err) {
    setStatus(`Could not load the model (${err.message}). Check your connection and reload the page.`, 'error');
  } finally {
    $('model').disabled = false;
  }
}

async function init() {
  $('legend').style.background = legendGradient();
  $('hero-legend').style.background = legendGradient();
  tilt($('scan-card'));
  parallax($('scan-card'), 36);
  showNetwork();
  addEventListener('online', showNetwork);
  addEventListener('offline', showNetwork);
  if (import.meta.env.PROD && 'serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  }
  try {
    await useModel(await loadModelList());
  } catch (err) {
    setStatus(`Could not load the model list (${err.message}). Check your connection and reload the page.`, 'error');
  }
}

/** One status line with an optional state: 'loading' (spinner) or 'error'. */
function setStatus(text, state = '') {
  const el = $('status');
  el.textContent = text;
  el.classList.toggle('loading', state === 'loading');
  el.classList.toggle('error', state === 'error');
}

// ---------- image input ----------

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('This file is not an image the browser can read.')); };
    img.src = url;
  });
}

async function pixelsOf(file) {
  const { img, url } = await loadImage(file);
  const limit = bundle.meta.resize_short * MAX_DECODE_FACTOR;
  const f = Math.min(1, limit / Math.min(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * f));
  const h = Math.max(1, Math.round(img.naturalHeight * f));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, w, h);
  URL.revokeObjectURL(url);
  return { rgba: ctx.getImageData(0, 0, w, h).data, w, h, original: [img.naturalWidth, img.naturalHeight] };
}

async function analyse(file) {
  if (!bundle || !file) return;
  if (file.type && !file.type.startsWith('image/')) {
    setStatus(`"${file.name}" is not an image. Choose a JPG, PNG or WebP photo of a leaf.`, 'error');
    return;
  }
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    setStatus(`That file is ${Math.round(file.size / 1048576)} MB; the limit is ${MAX_FILE_MB} MB. Use a smaller photo.`, 'error');
    return;
  }
  lastFile = file;
  setStatus('Looking at the leaf…', 'loading');
  try {
    const t0 = performance.now();
    const px = await pixelsOf(file);
    const pre = preprocess(px.rgba, px.w, px.h, bundle.meta);
    const tPre = performance.now() - t0;
    const out = await run(bundle, pre.input);
    const probs = softmax(out.logits);
    const ranked = topK(probs, 5);
    const ood = bundle.ood ? oodScore(out.features, bundle.meta.feature_shape, bundle.ood) : null;
    current = { px, pre, out, probs, ranked, ood, tPre, shown: ranked[0][0] };
    render();
    setStatus('Done. The result is below.');
    $('result').hidden = false;
    $('verdict').scrollIntoView({ behavior: reduceMotion.matches ? 'auto' : 'smooth', block: 'start' });
    enter([$('verdict'), $('remedy'), $('why'), $('expert')]);
    growBars($('verdict'));
    await faithfulnessCheck();
  } catch (err) {
    setStatus(err.message, 'error');
  }
}

// ---------- rendering ----------

function render() {
  renderVerdict();
  renderHeatmap();
  renderClassPicker();
  renderLookFor();
  renderRemedy();
  renderHood();
}

/** True when the photo does not look like any of the training photos (see scripts/fit_ood.py). */
function flagged() {
  return !!current.ood && current.ood.score > bundle.ood.threshold;
}

/** Flagged, and the user has not asked to see the model's guess anyway. */
function isUnsupported() {
  return flagged() && !current.showAnyway;
}

/** The 14 crops, each with its conditions, from the bundle's remedies. */
function cropsList() {
  const crops = new Map();
  for (const name of bundle.meta.classes) {
    const r = bundle.remedies.classes[name];
    const crop = r?.crop ?? name.split('___')[0].replaceAll('_', ' ');
    if (!crops.has(crop)) crops.set(crop, []);
    crops.get(crop).push(r ? (r.kind === 'healthy' ? 'Healthy' : r.name) : name.split('___')[1].replaceAll('_', ' '));
  }
  return [...crops].sort(([a], [b]) => a.localeCompare(b));
}

function cropChips() {
  return `<ul class="cropchips">${cropsList().map(([crop]) => `<li>${esc(crop)}</li>`).join('')}</ul>`;
}

function renderSupported() {
  const crops = cropsList();
  $('supported').innerHTML = `
    <h2>What it can check</h2>
    <p>${crops.length} crops and ${bundle.meta.classes.length} conditions (healthy included). Other plants, fruits, flowers, whole fields and non-plant photos are <b>not supported</b>.
    The app warns you when a photo looks unlike its training leaves, but it can miss a leaf of another plant, so check the list.</p>
    ${cropChips()}
    <details class="more"><summary>Show all ${bundle.meta.classes.length} conditions</summary>
      <dl class="croplist">${crops.map(([crop, conds]) => `<div><dt>${esc(crop)}</dt><dd>${conds.map(esc).join(' · ')}</dd></div>`).join('')}</dl>
    </details>`;
}

function isUnsure() {
  return current.ranked[0][1] < bundle.meta.unsure_threshold;
}

function renderVerdict() {
  const [best, p] = current.ranked[0];
  const r = bundle.remedies.classes[bundle.meta.classes[best]];
  const el = $('verdict');
  if (isUnsupported()) {
    el.className = 'card verdict unsupported';
    el.innerHTML = `
      <p class="eyebrow">Not a supported crop?</p>
      <h2>This doesn't look like a leaf from one of the supported crops</h2>
      <p>The photo is very different from every leaf photo the model learned from, so a diagnosis would be unreliable. The model only knows these ${cropsList().length} crops:</p>
      ${cropChips()}
      <p>If it <em>is</em> one of them, the photo is probably too different from the training photos (whole plant, busy background, several leaves).
      Retake it: one leaf, close up, filling most of the frame, on a plain background, in daylight.</p>
      <p><button type="button" class="btn small" id="anyway">Show the model's guess anyway</button></p>
      <p class="muted small">Unusualness ${current.ood.score.toFixed(2)}; the warning shows above ${bundle.ood.threshold.toFixed(2)}. <a href="./explain.html#unsupported">How this check works</a></p>`;
    $('anyway').onclick = () => { current.showAnyway = true; render(); };
    return;
  }
  if (isUnsure()) {
    el.className = 'card verdict unsure';
    el.innerHTML = `
      ${flagged() ? '<p class="banner">Shown on request: this photo was flagged as unlike the training photos.</p>' : ''}
      <p class="eyebrow">Not sure</p>
      <h2>The model is not confident enough to name a problem</h2>
      <p>Its best guess, <strong>${esc(label(best))}</strong>, has only ${pct(p)} confidence (it needs ${pct(bundle.meta.unsure_threshold)}).
      Retake the photo of a single leaf, close up, in daylight, or ask a local expert.</p>
      <p class="muted small">Closest matches: ${current.ranked.slice(0, 3).map(([i, q]) => `${esc(label(i))} (${pct(q)})`).join(', ')}</p>`;
    return;
  }
  const healthy = r?.kind === 'healthy';
  el.className = `card verdict ${healthy ? 'healthy' : `urgency-${r?.urgency ?? 'medium'}`}`;
  el.innerHTML = `
    <p class="eyebrow">${esc(r?.crop ?? '')}</p>
    <h2>${healthy ? 'Looks healthy' : esc(r?.name ?? label(best))}</h2>
    <p class="confidence"><span class="bar"><i style="width:${(100 * p).toFixed(1)}%"></i></span> ${pct(p)} confidence</p>
    ${flagged() ? '<p class="banner">Shown on request: this photo was flagged as unlike the training photos, so this answer is unreliable.</p>' : ''}
    ${healthy ? '' : `<p><span class="chip urgency">${esc(r.urgency)} urgency</span> <span class="chip">${esc(r.kind)}</span>${r.pathogen ? ` <em class="muted small">${esc(r.pathogen)}</em>` : ''}</p>`}
    <div class="verdict-actions"><button type="button" class="btn small" data-copy="${esc(summaryText(best, p, healthy))}" data-copy-done="Result copied">${icon('copy', { size: 18 })} Copy result</button></div>`;
}

/** Plain-text summary of a diagnosis, for sharing. */
function summaryText(best, p, healthy) {
  const r = bundle.remedies.classes[bundle.meta.classes[best]];
  const what = healthy ? `${r?.crop ?? ''}: looks healthy` : label(best);
  return `${what} (${pct(p)} confidence). Checked with CropXAI (${bundle.meta.run_name}), a research prototype: always confirm with a local expert. ${location.origin}${location.pathname}`;
}

function currentHeat() {
  const { out, shown } = current;
  const s = bundle.meta.image_size;
  return heatmap(out.features, bundle.meta.feature_shape, bundle.weight, shown, s);
}

function renderHeatmap() {
  const s = bundle.meta.image_size;
  current.heat = currentHeat();
  drawOverlay($('overlay'), current.pre.crop, current.heat, s, Number($('opacity').value));
  const same = current.shown === current.ranked[0][0];
  $('heatnote').textContent = same
    ? 'Bright = the parts of the photo that pushed the model towards its answer (Grad-CAM++, computed on this device).'
    : `Showing the evidence for "${label(current.shown)}" instead of the top answer. Compare the two to see what the model would need to change its mind.`;
}

function renderClassPicker() {
  const el = $('classpick');
  el.innerHTML = '<span class="muted small">Show evidence for:</span>';
  for (const [i, p] of current.ranked.slice(0, 3)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = `pill ${i === current.shown ? 'on' : ''}`;
    b.setAttribute('aria-pressed', i === current.shown);
    b.textContent = `${label(i)} · ${pct(p)}`;
    b.onclick = () => { current.shown = i; renderHeatmap(); renderClassPicker(); };
    el.append(b);
  }
}

function renderLookFor() {
  if (isUnsupported()) {
    $('lookfor').innerHTML = '<li class="lead">The photo was flagged as not a supported crop.</li><li>If the colour sits on something that is not a leaf, the model was looking at the wrong thing.</li>';
    return;
  }
  const r = bundle.remedies.classes[bundle.meta.classes[current.ranked[0][0]]];
  $('lookfor').innerHTML = `<li class="lead">Signs of ${esc(label(current.ranked[0][0]))}:</li>`
    + (r?.look_for ?? []).map((s) => `<li>${esc(s)}</li>`).join('');
}

function list(title, items) {
  if (!items?.length) return '';
  return `<h3>${title}</h3><ul>${items.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>`;
}

function renderRemedy() {
  const el = $('remedy');
  const r = bundle.remedies.classes[bundle.meta.classes[current.ranked[0][0]]];
  if (isUnsupported()) {
    el.innerHTML = '<h2>What to do</h2><p>No advice: this photo does not look like one of the supported crops. If it is one, retake the photo of a single leaf, close up, in daylight.</p>';
    return;
  }
  if (isUnsure() || !r) {
    el.innerHTML = '<h2>What to do</h2><p>No advice until the diagnosis is clear. Retake the photo, or compare the leaf with the possible problems above.</p>';
    return;
  }
  const healthy = r.kind === 'healthy';
  el.innerHTML = `
    <h2>${healthy ? 'Keep it healthy' : 'What to do'}</h2>
    ${list(healthy ? 'Good practice' : 'First steps (no chemicals)', r.cultural)}
    ${list('Organic options', r.organic)}
    ${list('Chemical options (active ingredients)', r.chemical)}
    <p class="note small">${esc(bundle.remedies.disclaimer)}${bundle.remedies.reviewed ? '' : ' This advice has not yet been reviewed by a plant pathologist.'}</p>`;
}

async function faithfulnessCheck() {
  const el = $('faithful');
  $('faithbadge').textContent = 'Heatmap check: running…';
  $('faithbadge').className = 'chip';
  el.innerHTML = '<p class="muted small">Running the check…</p>';
  const s = bundle.meta.image_size;
  const cls = current.ranked[0][0];
  const heat = heatmap(current.out.features, bundle.meta.feature_shape, bundle.weight, cls, s);
  const top = pixelsToHide(heat, HIDE_FRACTION);
  const rand = pixelsToHide(heat, HIDE_FRACTION, true, 0);
  const pTop = softmax((await run(bundle, hidePixels(current.pre.input, s, top))).logits)[cls];
  const pRand = softmax((await run(bundle, hidePixels(current.pre.input, s, rand))).logits)[cls];
  const p = current.probs[cls];
  const dropTop = p - pTop;
  const dropRand = p - pRand;
  const passes = dropTop > dropRand + 0.05;
  $('faithbadge').textContent = `Heatmap check: ${passes ? 'passes' : 'doubtful'}`;
  $('faithbadge').className = `chip ${passes ? 'ok' : 'warn'}`;
  el.innerHTML = `
    <div class="pair">
      <figure><canvas id="hidTop"></canvas><figcaption>Bright 20% hidden<br><strong>${pct(pTop)}</strong></figcaption></figure>
      <figure><canvas id="hidRand"></canvas><figcaption>Random 20% hidden<br><strong>${pct(pRand)}</strong></figcaption></figure>
    </div>
    <p class="${passes ? 'ok' : 'warn'}"><strong>${passes ? 'Passes.' : 'Doubtful.'}</strong>
      Confidence in "${esc(label(cls))}" was ${pct(p)}. Hiding the bright area: ${change(dropTop)};
      hiding a random area: ${change(dropRand)}.
      ${passes ? 'The highlighted area really drove the answer.' : 'The heatmap may not show what really drove this answer; trust it less.'}</p>`;
  drawHidden($('hidTop'), current.pre.crop, s, top);
  drawHidden($('hidRand'), current.pre.crop, s, rand);
}

function renderHood() {
  const m = bundle.meta;
  const [k, h, w] = m.feature_shape;
  const { px, out, ranked, tPre } = current;
  const cls = ranked[0][0];
  const d = scoreDecomposition(out.features, m.feature_shape, bundle.weight, bundle.bias, cls);
  const positive = d.channels.filter((c) => c.weight > 0).length;
  const top = d.channels.slice(0, 6);
  const rest = d.channels.slice(6).reduce((a, c) => a + c.contribution, 0);
  const [rw, rh] = shortSideSize(px.w, px.h, m.resize_short);

  $('hoodbody').innerHTML = `
    <h3>1. What happened to your photo</h3>
    <ol class="steps">
      <li><b>Photo</b> ${px.original[0]}×${px.original[1]} pixels${px.w !== px.original[0] ? ` (scaled to ${px.w}×${px.h} first)` : ''}</li>
      <li><b>Resize</b> short side to ${m.resize_short} px → ${rw}×${rh}</li>
      <li><b>Centre crop</b> ${m.image_size}×${m.image_size}, the square the model sees <canvas id="cropview" class="thumb"></canvas></li>
      <li><b>Normalise</b> colours with the ImageNet mean and spread (what the network was pre-trained on) <span class="muted">${tPre.toFixed(0)} ms</span></li>
      <li><b>Network</b> ${esc(m.arch)}: ${k} feature maps of ${h}×${w}, each a detector for some visual pattern <span class="muted">${out.ms.toFixed(0)} ms on this device</span></li>
      <li><b>Average</b> each map to one number (global average pooling) → ${k} numbers</li>
      <li><b>Score</b> each of the ${m.classes.length} classes: a weighted sum of those ${k} numbers plus a bias</li>
      <li><b>Softmax</b> turns scores into confidences that add up to 100%</li>
    </ol>

    <h3>2. Where the score for "${esc(label(cls))}" came from</h3>
    <p class="small">Because the last layer is a plain weighted sum, this split is exact, not an estimate:
      score <b>${d.logit.toFixed(3)}</b> = bias ${d.bias.toFixed(3)} + the contributions of all ${k} channels (the six largest are below)
      (model's own score: ${out.logits[cls].toFixed(3)}).</p>
    <div class="channels">
      ${top.map((c) => `
        <figure>
          <canvas data-channel="${c.channel}"></canvas>
          <figcaption>channel ${c.channel}<br>+${c.contribution.toFixed(3)}<br><span class="muted">${c.pooled.toFixed(2)} × ${c.weight.toFixed(3)}</span></figcaption>
        </figure>`).join('')}
    </div>
    <p class="small muted">The other ${k - top.length} channels add ${rest >= 0 ? '+' : ''}${rest.toFixed(3)} in total. Each picture shows where that pattern detector fired.</p>

    <h3>3. How the heatmap is made (Grad-CAM++ without a backward pass)</h3>
    <p class="small">Grad-CAM++ normally needs gradients, which phones can't compute. Here the gradient of the score with respect to every
      cell of feature map <i>k</i> is the constant g<sub>k</sub> = W<sub>c,k</sub> / ${h * w}, so the method reduces to a formula on the feature maps alone:</p>
    <p class="formula">α<sub>k</sub> = g<sub>k</sub>² / (2g<sub>k</sub>² + g<sub>k</sub>³ Σ<sub>ij</sub>A<sup>k</sup><sub>ij</sub>), &nbsp;
      w<sub>k</sub> = ${h * w} · ReLU(g<sub>k</sub>) · α<sub>k</sub>, &nbsp; map = ReLU(Σ<sub>k</sub> w<sub>k</sub>A<sup>k</sup>)</p>
    <p class="small">${positive} of ${k} channels have a positive weight for this class and take part. The ${h}×${w} map is scaled to 0–1 and enlarged to ${m.image_size}×${m.image_size}.
      This is the same calculation as the research code, checked to match it in the project's tests.</p>
    <canvas id="rawcam" class="thumb big" aria-label="Heatmap before blending"></canvas>

    <h3>4. All top answers</h3>
    <ul class="probs">
      ${ranked.map(([i, p]) => `<li><span>${esc(label(i))}</span><span class="bar"><i style="width:${(100 * p).toFixed(1)}%"></i></span><b>${pct(p)}</b></li>`).join('')}
    </ul>
    <p class="small muted">Unsure below ${pct(m.unsure_threshold)}. Threshold source: ${esc(m.unsure_threshold_source)}.</p>

    <h3>5. Is this a supported crop?</h3>
    ${current.ood ? `<p class="small">The ${k} averaged feature values are compared with the average of each of the ${m.classes.length} classes in the training photos
      (Mahalanobis distance, ${bundle.ood.dims} main directions). Distance to the nearest class (${esc(label(current.ood.nearest))}):
      <b>${current.ood.score.toFixed(2)}</b>; the warning shows above <b>${bundle.ood.threshold.toFixed(2)}</b>
      (${esc(bundle.ood.threshold_source)}). ${isUnsupported() ? '<b>Flagged as not a supported crop.</b>' : 'Not flagged.'}</p>
      <p class="small muted">It catches photos that look nothing like the training leaves (people, objects, scenes). It can miss a leaf of another plant that looks like a supported one,
      and it also flags most real field photos of supported crops, because those look unlike the lab training photos too. <a href="./explain.html#unsupported">Measured results</a></p>`
    : '<p class="small muted">This model has no supported-crop check yet.</p>'}

    <h3>6. About this model</h3>
    <p class="small muted">Run ${esc(m.run_name)}, seed ${m.seed}, code ${esc(m.git_commit)}, exported ${esc(m.exported_at)}. File fingerprint ${esc(m.onnx_sha256.slice(0, 16))}…</p>`;

  const s = m.image_size;
  drawRGB($('cropview'), current.pre.crop, s);
  drawMap($('rawcam'), heatmap(out.features, m.feature_shape, bundle.weight, cls, s), s);
  for (const c of document.querySelectorAll('canvas[data-channel]')) {
    drawMap(c, featureMap(out.features, m.feature_shape, Number(c.dataset.channel), 112), 112);
  }
}

// ---------- hero: the model scanning a real sample leaf ----------

let heroDone = false;
function heroScan() {
  if (heroDone || !bundle) return;
  heroDone = true;
  const idle = window.requestIdleCallback ?? ((f) => setTimeout(f, 200));
  idle(async () => {
    const chip = $('hero-result');
    try {
      const r = await fetch(HERO_SAMPLE);
      if (!r.ok) throw new Error(String(r.status));
      const px = await pixelsOf(await r.blob());
      const pre = preprocess(px.rgba, px.w, px.h, bundle.meta);
      const out = await run(bundle, pre.input);
      const [[best, p]] = topK(softmax(out.logits), 1);
      const s = bundle.meta.image_size;
      const canvas = $('hero-canvas');
      drawOverlay(canvas, pre.crop, heatmap(out.features, bundle.meta.feature_shape, bundle.weight, best, s), s, 0.62);
      chip.querySelector('strong').textContent = label(best);
      chip.querySelector('.muted').textContent = `${pct(p)} confidence, computed just now`;
      if (reduceMotion.matches) { canvas.style.clipPath = 'none'; return; }
      const line = document.querySelector('.scan-card .scanline');
      animate(line, { top: ['0%', '100%'], opacity: [1, 1, 0] }, { duration: 1.6, ease: 'easeInOut' });
      animate(canvas, { clipPath: ['inset(0 0 100% 0)', 'inset(0 0 0% 0)'] }, { duration: 1.6, ease: 'easeInOut' });
      animate(chip, { scale: [0.92, 1], opacity: [0.6, 1] }, { type: 'spring', stiffness: 260, damping: 18, delay: 1.4 });
    } catch {
      chip.querySelector('strong').textContent = 'Sample leaf';
      chip.querySelector('.muted').textContent = 'the live example could not run';
    }
  });
}

// ---------- events ----------

for (const id of ['camera', 'gallery']) {
  $(id).addEventListener('change', (e) => { analyse(e.target.files[0]); e.target.value = ''; });
}
$('opacity').addEventListener('input', () => {
  if (current) drawOverlay($('overlay'), current.pre.crop, current.heat, bundle.meta.image_size, Number($('opacity').value));
});
$('model').addEventListener('change', (e) => {
  const url = new URL(location.href);
  url.searchParams.set('model', e.target.value);
  history.replaceState(null, '', url);
  useModel(e.target.value);
});
for (const b of document.querySelectorAll('[data-sample]')) {
  b.addEventListener('click', async () => {
    try {
      const r = await fetch(b.dataset.sample);
      if (!r.ok) throw new Error(`${b.dataset.sample}: ${r.status}`);
      const blob = await r.blob();
      analyse(new File([blob], b.dataset.sample.split('/').pop(), { type: blob.type }));
    } catch (err) {
      setStatus(`Could not load the sample (${err.message}).`, 'error');
    }
  });
}
$('again').addEventListener('click', () => {
  $('result').hidden = true;
  $('pick').scrollIntoView({ behavior: reduceMotion.matches ? 'auto' : 'smooth' });
  $('gallery').focus({ preventScroll: true });
});
const drop = document.body;
drop.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragging'); });
drop.addEventListener('dragleave', () => document.body.classList.remove('dragging'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.classList.remove('dragging');
  analyse(e.dataTransfer.files[0]);
});

init();
