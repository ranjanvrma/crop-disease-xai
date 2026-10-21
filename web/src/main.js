import { drawHidden, drawMap, drawOverlay, drawRGB, legendGradient } from './colormap.js';
import { loadBundle, run } from './model.js';
import { preprocess, shortSideSize } from './preprocess.js';
import {
  featureMap, heatmap, hidePixels, pixelsToHide, scoreDecomposition, softmax, topK,
} from './xai.js';

const $ = (id) => document.getElementById(id);
const HIDE_FRACTION = 0.2;
// Very large photos are first scaled down by the browser so a phone never holds a
// 50-megapixel bitmap; the exact training resize (preprocess.js) runs after that.
const MAX_DECODE_FACTOR = 4;

let bundle = null;
let current = null; // the last analysed image

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
  net.className = `chip ${navigator.onLine ? '' : 'warn'}`;
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
    banner.innerHTML = `<strong>Lab-trained model.</strong> In testing it was right ${pct(acc.test_top1)} of the time on lab photos but only ${pct(acc.field_strict_top1)} on ${acc.field_n} real field photos. Use the heatmap and the checklist to judge each result.`;
  } else {
    banner.className = 'banner';
    banner.innerHTML = '<strong>Lab-trained model.</strong> Its accuracy on real field photos has not been measured for this build. Use the heatmap and the checklist to judge each result.';
  }
  banner.hidden = false;
  $('modelinfo').textContent = `Model: ${m.arch} (${m.run_name}, seed ${m.seed}, commit ${m.git_commit})${m.demo ? ', demo' : ''}. Loaded in ${Math.round(bundle.loadMs)} ms.`;
}

async function init() {
  $('legend').style.background = legendGradient();
  showNetwork();
  addEventListener('online', showNetwork);
  addEventListener('offline', showNetwork);
  if (import.meta.env.PROD && 'serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  }
  try {
    bundle = await loadBundle();
    showModelInfo();
    $('status').textContent = 'Ready.';
    document.body.dataset.ready = 'true';
  } catch (err) {
    $('status').textContent = `Could not load the model (${err.message}). Run scripts/export_web.py first.`;
    $('status').classList.add('error');
  }
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
  $('status').textContent = 'Looking at the leaf…';
  $('status').classList.remove('error');
  try {
    const t0 = performance.now();
    const px = await pixelsOf(file);
    const pre = preprocess(px.rgba, px.w, px.h, bundle.meta);
    const tPre = performance.now() - t0;
    const out = await run(bundle, pre.input);
    const probs = softmax(out.logits);
    const ranked = topK(probs, 5);
    current = { px, pre, out, probs, ranked, tPre, shown: ranked[0][0] };
    render();
    $('status').textContent = '';
    $('result').hidden = false;
    $('verdict').scrollIntoView({ behavior: 'smooth', block: 'start' });
    await faithfulnessCheck();
  } catch (err) {
    $('status').textContent = err.message;
    $('status').classList.add('error');
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

function isUnsure() {
  return current.ranked[0][1] < bundle.meta.unsure_threshold;
}

function renderVerdict() {
  const [best, p] = current.ranked[0];
  const r = bundle.remedies.classes[bundle.meta.classes[best]];
  const el = $('verdict');
  if (isUnsure()) {
    el.className = 'card verdict unsure';
    el.innerHTML = `
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
    ${healthy ? '' : `<p><span class="chip urgency">${esc(r.urgency)} urgency</span> <span class="chip">${esc(r.kind)}</span>${r.pathogen ? ` <em class="muted small">${esc(r.pathogen)}</em>` : ''}</p>`}`;
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

    <h3>5. About this model</h3>
    <p class="small muted">Run ${esc(m.run_name)}, seed ${m.seed}, code ${esc(m.git_commit)}, exported ${esc(m.exported_at)}. File fingerprint ${esc(m.onnx_sha256.slice(0, 16))}…</p>`;

  const s = m.image_size;
  drawRGB($('cropview'), current.pre.crop, s);
  drawMap($('rawcam'), heatmap(out.features, m.feature_shape, bundle.weight, cls, s), s);
  for (const c of document.querySelectorAll('canvas[data-channel]')) {
    drawMap(c, featureMap(out.features, m.feature_shape, Number(c.dataset.channel), 112), 112);
  }
}

// ---------- events ----------

for (const id of ['camera', 'gallery']) {
  $(id).addEventListener('change', (e) => { analyse(e.target.files[0]); e.target.value = ''; });
}
$('opacity').addEventListener('input', () => {
  if (current) drawOverlay($('overlay'), current.pre.crop, current.heat, bundle.meta.image_size, Number($('opacity').value));
});
$('again').addEventListener('click', () => { $('result').hidden = true; $('pick').scrollIntoView({ behavior: 'smooth' }); });
const drop = document.body;
drop.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragging'); });
drop.addEventListener('dragleave', () => document.body.classList.remove('dragging'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.classList.remove('dragging');
  analyse(e.dataTransfer.files[0]);
});

init();
