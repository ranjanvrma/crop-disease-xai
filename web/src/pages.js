// Results, Explainability and About pages. Every number comes from
// public/data/site.json (scripts/export_site_data.py, which reads results/tables/
// and results/metrics/) or from the default model's remedies.json; none is typed here.

import { countUp, enter, growBars, smoothDetails } from './ui.js';

const $ = (id) => document.getElementById(id);
const page = document.body.dataset.page;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// ---------- table helpers ----------

/** Value of column `col` in the first row whose leading cells equal `keys`, as {mean, std} or null. */
function cell(table, keys, col) {
  const c = table.columns.indexOf(col);
  if (c < 0) throw new Error(`no column ${col}`);
  const i = table.rows.findIndex((r) => keys.every((k, j) => r[j] === k));
  return i < 0 ? null : table.numbers[i][c];
}

const fmt = (v, d = 1) => (v == null ? '—' : v.mean.toFixed(d));

function renderTable(el, table, highlight) {
  el.innerHTML = `
    <table>
      <thead><tr>${table.columns.map((c) => `<th scope="col">${esc(c)}</th>`).join('')}</tr></thead>
      <tbody>${table.rows.map((r) => `<tr class="${r[0] === highlight ? 'hl' : ''}">${r.map((c, j) => (j === 0 ? `<th scope="row">${esc(c)}</th>` : `<td>${esc(c)}</td>`)).join('')}</tr>`).join('')}</tbody>
    </table>
    <p class="muted small">Source: <code>${esc(table.source)}</code></p>`;
}

// ---------- charts (plain SVG, colours from CSS variables) ----------

/**
 * Horizontal grouped bar chart.
 * rows: [{label, values: [{mean, std}|null, ...]}], series: [{name, cls}], max: axis maximum.
 */
function barChart(el, rows, series, { max, unit = '', digits = 1 } = {}) {
  const labelW = 132, barH = 14, gap = 4, groupGap = 14, plotW = 250, right = 52;
  const groupH = series.length * (barH + gap) - gap;
  const h = rows.length * (groupH + groupGap);
  const x = (v) => (Math.min(max, Math.max(0, v)) / max) * plotW;
  let y = 6;
  let body = '';
  for (const row of rows) {
    body += `<text x="${labelW - 8}" y="${y + groupH / 2 + 4}" text-anchor="end" class="lbl">${esc(row.label)}</text>`;
    row.values.forEach((v, i) => {
      const yy = y + i * (barH + gap);
      if (v == null) {
        body += `<text x="${labelW + 4}" y="${yy + barH - 3}" class="val">not measured</text>`;
        return;
      }
      const w = x(v.mean);
      body += `<rect x="${labelW}" y="${yy}" width="${w.toFixed(1)}" height="${barH}" rx="3" class="${series[i].cls}"><title>${esc(series[i].name)}: ${v.mean}${unit}${v.std != null ? ` ± ${v.std}` : ''}</title></rect>`;
      if (v.std) {
        const a = x(v.mean - v.std), b = x(v.mean + v.std);
        body += `<line x1="${labelW + a}" x2="${labelW + b}" y1="${yy + barH / 2}" y2="${yy + barH / 2}" class="err" />`;
      }
      body += `<text x="${labelW + w + 6}" y="${yy + barH - 3}" class="val">${v.mean.toFixed(digits)}${unit}</text>`;
    });
    y += groupH + groupGap;
  }
  // The legend is HTML under the chart, so long series names wrap instead of being clipped.
  const legend = `<ul class="chart-legend">${series.map((s) => `<li><i class="${s.cls}"></i>${esc(s.name)}</li>`).join('')}</ul>`;
  el.innerHTML = `<svg viewBox="0 0 ${labelW + plotW + right} ${h}" role="img" aria-label="${esc(series.map((s) => s.name).join(' vs '))}">${body}</svg>${legend}`;
  growBars(el);
}

// ---------- pages ----------

const LABELS = {
  mnv3_ft_field: 'MNV3 fine-tuned (app)',
  mnv3_fe: 'MNV3 frozen',
  effnet_b0_fe: 'EfficientNet-B0',
  scratch_cnn: 'Small CNN (n=1)',
  mnv3_fe_int8: 'MNV3 INT8',
  mnv3_fe_p50: 'MNV3 p50',
  mnv3_fe_p50_ft8: 'MNV3 p50 + FT',
  mnv3_fe_p50_int8: 'MNV3 p50 + INT8',
  mnv3_fe_p70: 'MNV3 p70',
  mnv3_fe_p70_ft8: 'MNV3 p70 + FT',
  mnv3_fe_p70_int8: 'MNV3 p70 + INT8',
};
const name = (run) => LABELS[run] ?? run;
const APP_MODEL = 'mnv3_ft_field';

function keypoints(items) {
  const el = $('keypoints');
  el.innerHTML = items.map(([big, text], i) => `<section class="card stat${i === 0 ? ' ink' : ''}"><p class="big">${esc(big)}</p><p>${text}</p></section>`).join('');
  el.querySelectorAll('.big').forEach(countUp);
  enter([...el.children]);
}

function gapChart(el, t1) {
  const runs = t1.rows.map((r) => r[0]);
  barChart(el, runs.map((run) => ({
    label: name(run),
    values: [cell(t1, [run], 'Top-1 PV test (%)'), cell(t1, [run], 'Field top-1 strict (%)')],
  })), [{ name: 'Lab test (PlantVillage)', cls: 's1' }, { name: 'Field (PlantDoc)', cls: 's2' }], { max: 100, unit: '%' });
}

async function results() {
  const site = await json('./data/site.json');
  const { table1_accuracy: t1, table2_explanation: t2, table3_stability: t3 } = site.tables;
  $('datasets').textContent = `Lab test set: ${site.field.test_n.toLocaleString()} PlantVillage images never used in training. Field test set: ${site.field.field_n.toLocaleString()} PlantDoc photos covering ${site.field.field_classes} of the 38 classes, never used for training or tuning.`;
  const v = (run, col) => cell(t1, [run], col);
  keypoints([
    [`${fmt(v(APP_MODEL, 'Top-1 PV test (%)'))}%`, 'correct on held-out <b>lab photos</b> (the default model in the app).'],
    [`${fmt(v(APP_MODEL, 'Field top-1 strict (%)'))}%`, 'correct on <b>real field photos</b>: the open problem, explained on the <a href="./explain.html#field-gap">Explainability</a> page.'],
    [`${fmt(v('mnv3_fe_int8', 'Weights (MB)'), 1)} MB`, `INT8 model, ${Math.round(v('mnv3_fe', 'Weights (MB)').mean / v('mnv3_fe_int8', 'Weights (MB)').mean)}× smaller than its parent, still ${fmt(v('mnv3_fe_int8', 'Top-1 PV test (%)'))}% on lab photos.`],
  ]);
  gapChart($('gapchart'), t1);
  renderTable($('table1'), t1, APP_MODEL);
  renderTable($('table2'), t2, APP_MODEL);
  renderTable($('table3'), t3);
}

async function explain() {
  const site = await json('./data/site.json');
  const { table1_accuracy: t1, table2_explanation: t2, table3_stability: t3 } = site.tables;

  const worst = Math.max(...site.closed_form.map((r) => r.max_abs_diff));
  keypoints([
    ['1 pass', `The heatmap is computed <b>in your browser</b> from a single forward pass, and matches the standard library exactly (largest difference about ${worst.toExponential(0)}).`],
    [fmt(cell(t3, ['mnv3_fe_int8'], 'Spearman ↑ (test)'), 2), 'similarity of heatmaps after <b>INT8 compression</b> (1 = identical). Pruning changes them far more.'],
    [`${fmt(cell(t1, [APP_MODEL], 'Top-1 PV test (%)'), 0)}% → ${fmt(cell(t1, [APP_MODEL], 'Field top-1 strict (%)'), 0)}%`, 'lab vs. field accuracy. The heatmaps help explain why: <a href="#field-gap">the background shortcut</a>.'],
  ]);

  // 2. closed form vs library
  $('closedform').innerHTML = `
    <table>
      <thead><tr><th scope="col">Model</th><th scope="col">Seeds × images</th><th scope="col">Max. abs. difference</th><th scope="col">Min. Pearson r</th><th scope="col">Library (ms/img)</th><th scope="col">Closed form (ms/img)</th></tr></thead>
      <tbody>${site.closed_form.map((r) => `<tr class="${r.run === APP_MODEL ? 'hl' : ''}"><th scope="row">${esc(r.run)}</th><td>${r.seeds} × ${r.n_images}</td><td>${r.max_abs_diff.toExponential(1)}</td><td>${r.pearson_r_min.toFixed(6)}</td><td>${r.ms_library.toFixed(1)}</td><td>${r.ms_closed_form.toFixed(1)}</td></tr>`).join('')}</tbody>
    </table>
    <p class="muted small">Heatmaps scaled to 0–1; timings on ${esc(site.closed_form[0].device)}. Source: <code>results/metrics/closed_form/</code></p>`;

  // 3. deletion AUC by method, for models that have all three methods
  const methods = ['Grad-CAM', 'Grad-CAM++', 'Score-CAM'];
  const full = [...new Set(t2.rows.map((r) => r[0]))].filter((run) => methods.every((m) => cell(t2, [run, m], 'Del. AUC ↓ (test)')));
  barChart($('delchart'), full.map((run) => ({ label: name(run), values: methods.map((m) => cell(t2, [run, m], 'Del. AUC ↓ (test)')) })),
    [{ name: 'Grad-CAM', cls: 's1' }, { name: 'Grad-CAM++', cls: 's2' }, { name: 'Score-CAM', cls: 's3' }], { max: 0.7, digits: 3 });
  const camWins = full.filter((run) => cell(t2, [run, 'Grad-CAM'], 'Del. AUC ↓ (test)').mean < cell(t2, [run, 'Grad-CAM++'], 'Del. AUC ↓ (test)').mean);
  $('delfinding').innerHTML = `<b>Finding:</b> plain Grad-CAM had a lower (better) deletion AUC than Grad-CAM++ on ${camWins.length} of ${full.length} models
    (${camWins.map(name).join(', ')}). We report this as measured rather than assume Grad-CAM++ is better; its advantage is expected on leaves
    with several separate lesions, which needs lesion masks to test. For the app model, Grad-CAM++ and Grad-CAM are close
    (${fmt(cell(t2, [APP_MODEL, 'Grad-CAM++'], 'Del. AUC ↓ (test)'), 3)} vs ${fmt(cell(t2, [APP_MODEL, 'Grad-CAM'], 'Del. AUC ↓ (test)'), 3)}).`;

  // 4. stability under compression
  barChart($('stabchart'), t3.rows.map((r) => ({ label: name(r[0]), values: [cell(t3, [r[0]], 'Spearman ↑ (test)'), cell(t3, [r[0]], 'Spearman ↑ (field)')] })),
    [{ name: 'Lab test', cls: 's1' }, { name: 'Field', cls: 's2' }], { max: 1, digits: 2 });
  const int8 = cell(t3, ['mnv3_fe_int8'], 'Spearman ↑ (test)');
  const pruned = t3.rows.filter((r) => r[0].includes('_p')).map((r) => cell(t3, [r[0]], 'Spearman ↑ (test)').mean);
  $('stabfinding').innerHTML = `<b>Finding:</b> INT8 quantization keeps the explanation almost unchanged (Spearman ${int8.mean.toFixed(2)} on lab photos, at a quarter of the size),
    while pruning changes where the model looks a lot (Spearman ${Math.min(...pruned).toFixed(2)}–${Math.max(...pruned).toFixed(2)}), even after fine-tuning recovers part of the accuracy.
    So for a phone, quantize first; prune only with care.`;

  // 5. calibration
  const eceRuns = t1.rows.map((r) => r[0]).filter((run) => !run.includes('_p'));
  barChart($('ecechart'), eceRuns.map((run) => ({ label: name(run), values: [cell(t1, [run], 'ECE PV test')] })),
    [{ name: 'ECE on lab test set (lower is better)', cls: 's1' }], { max: 0.15, digits: 3 });
  const ece = (run) => fmt(cell(t1, [run], 'ECE PV test'), 3);
  $('ecefinding').innerHTML = `<b>Finding:</b> the fine-tuned app model is the most accurate MobileNetV3 on lab photos but less well calibrated
    (ECE ${ece(APP_MODEL)} vs ${ece('mnv3_fe')} for the frozen version and ${ece('effnet_b0_fe')} for EfficientNet-B0): its confidence numbers should not be read as probabilities.`;

  // 6. field gap
  const acc = (run, col) => fmt(cell(t1, [run], col), 1);
  $('gapfinding').innerHTML = `
    <p>Every model is accurate on lab photos and weak on real field photos (${site.field.field_n.toLocaleString()} PlantDoc images):</p>
    <ul>
      <li>MobileNetV3 frozen: <b>${acc('mnv3_fe', 'Top-1 PV test (%)')}%</b> lab → <b>${acc('mnv3_fe', 'Field top-1 strict (%)')}%</b> field</li>
      <li>EfficientNet-B0: <b>${acc('effnet_b0_fe', 'Top-1 PV test (%)')}%</b> → <b>${acc('effnet_b0_fe', 'Field top-1 strict (%)')}%</b></li>
      <li>MobileNetV3 fine-tuned with field-style augmentation (random crops, rotation, perspective, blur, noise, shadows, brightness, JPEG artefacts): <b>${acc(APP_MODEL, 'Top-1 PV test (%)')}%</b> → <b>${acc(APP_MODEL, 'Field top-1 strict (%)')}%</b></li>
    </ul>
    <p><b>What the heatmaps show.</b> PlantVillage photos are one centred leaf on a plain background, so "leaf vs. background" is trivially easy and
      the model never has to learn to ignore clutter. On lab photos the heatmaps mostly sit on the leaf, but some peak at the leaf edge or the image border.
      On field photos, with soil and other leaves in view, the heatmaps often find a lesion and the class is still wrong, or they spread over other leaves and the background.</p>
    <p><b>What the field-augmentation experiment adds.</b> Fully fine-tuning the network on training photos made to look like field photos raised lab accuracy but moved field accuracy (vs. the frozen MobileNetV3) by only
      ${(cell(t1, [APP_MODEL], 'Field top-1 strict (%)').mean - cell(t1, ['mnv3_fe'], 'Field top-1 strict (%)').mean).toFixed(1)} points. So image quality is not what is missing: the models depend on the
      <em>lab setting</em> itself (one leaf, plain background), a form of <b>shortcut learning</b>. The likely fix is training on leaves pasted onto real backgrounds, or on real labelled field photos.</p>
    <p class="note small"><b>Caveats.</b> Border peaks can also come from CNN zero-padding, so we treat the background explanation as the most likely one, not a proof.
      Some PlantDoc images are not single leaves (e.g. a watermarked peach-fruit stock photo and a whole cornfield, both visible below); they were kept on purpose so the field
      set is never hand-tuned. The field set was never used for training or for choosing models.</p>`;

  // 7. supported-crop check (models/<id>/ood.json, scripts/fit_ood.py)
  await oodFinding();
}

async function oodFinding() {
  const index = await json('./models/index.json');
  const r = await fetch(`./models/${index.default}/ood.json`);
  if (!r.ok) { $('oodfinding').innerHTML = '<p class="muted small">The default model has no supported-crop check yet.</p>'; return; }
  const o = await r.json();
  const rep = o.report;
  const p = (v) => `${Math.round(100 * v)}%`;
  const hold = Object.entries(rep.holdout_crop_flagged).sort(([, a], [, b]) => a - b);
  const crop = (c) => c.split('___')[0].replace('_(including_sour)', '').replace('_(maize)', '').replace(',_bell', ' pepper').replaceAll('_', ' ');
  const fa = rep.field_accuracy;
  $('oodfinding').innerHTML = `
    <h3>How well it works (default model, ${esc(index.default)})</h3>
    <div class="tablewrap"><table>
      <thead><tr><th scope="col">Photos</th><th scope="col">How many</th><th scope="col">Flagged as not a supported crop</th><th scope="col">Caught by "Not sure" alone</th></tr></thead>
      <tbody>
        <tr><th scope="row">PlantVillage lab photos (validation, used to set the threshold)</th><td>${rep.n.val}</td><td>${p(rep.flagged.val)}</td><td>${p(rep.unsure_flagged.val)}</td></tr>
        <tr class="hl"><th scope="row">Not plants (people, a cat, objects, space, coins…)</th><td>${rep.n.nonplant}</td><td>${p(rep.flagged.nonplant)}</td><td>${p(rep.unsure_flagged.nonplant)}</td></tr>
        <tr class="hl"><th scope="row">Other vegetation (aerial tree canopy, grass)</th><td>${rep.n.vegetation}</td><td>${p(rep.flagged.vegetation)}</td><td>${p(rep.unsure_flagged.vegetation)}</td></tr>
        <tr><th scope="row">PlantDoc field photos of <em>supported</em> crops</th><td>${rep.n.field.toLocaleString()}</td><td>${p(rep.flagged.field)}</td><td>${p(rep.unsure_flagged.field)}</td></tr>
      </tbody>
    </table></div>
    <p class="small"><b>Not plants and other vegetation:</b> almost all caught, while only ${p(rep.flagged.val)} of lab photos are wrongly flagged. The confidence threshold alone lets most of them through with a confident-looking diagnosis.</p>
    <p class="small"><b>Field photos:</b> most are flagged too, because they look unlike the lab photos. That is not wrong as a warning: on the ${fa.n_passed} field photos that pass,
      the model is right ${p(fa.passed)} of the time, against ${p(fa.flagged)} on the ${fa.n_flagged.toLocaleString()} flagged ones. That is why the Diagnose page lets you see the guess anyway.</p>
    <p class="small"><b>A leaf of a crop it does not know</b> is the hard case. We have no labelled photos of unsupported crops, so we used a proxy: refit the check with one crop left out and see
      how many of that crop's lab photos get flagged. On average ${p(rep.holdout_crop_flagged_mean)}, from ${p(hold[0][1])} (${esc(crop(hold[0][0]))}) to ${p(hold.at(-1)[1])} (${esc(crop(hold.at(-1)[0]))}).
      So about ${p(1 - rep.holdout_crop_flagged_mean)} of leaves of an unknown crop may still get a diagnosis, and the proxy is probably optimistic because the network itself did learn those crops.</p>
    <p class="muted small">Threshold: ${esc(o.threshold_source)}. Fitted on ${esc(o.fitted_on)}, ${o.dims} directions. The non-plant and vegetation photos are public sample images
      (scikit-image, matplotlib, DeepForest), a small set; treat those rates as indicative. Source: <code>scripts/fit_ood.py</code>.</p>`;
}

async function about() {
  const index = await json('./models/index.json');
  const rem = await json(`./models/${index.default}/remedies.json`);
  $('remedynote').textContent = `${rem.disclaimer}${rem.reviewed ? '' : ' This advice has not yet been reviewed by a plant pathologist.'}`;
  const entries = Object.entries(rem.classes).sort(([, a], [, b]) => a.crop.localeCompare(b.crop) || (a.kind === 'healthy') - (b.kind === 'healthy') || a.name.localeCompare(b.name));
  const list = (title, items) => (items?.length ? `<h4>${title}</h4><ul>${items.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>` : '');
  $('remedylist').innerHTML = entries.map(([key, r]) => `
    <details class="remedy" data-search="${esc(`${r.crop} ${r.name} ${r.kind} ${r.pathogen ?? ''}`.toLowerCase())}">
      <summary><b>${esc(r.crop)}</b>: ${esc(r.kind === 'healthy' ? 'Healthy' : r.name)} <span class="chip">${esc(r.kind)}</span>${r.urgency && r.urgency !== 'none' ? ` <span class="chip">${esc(r.urgency)} urgency</span>` : ''}</summary>
      ${r.pathogen ? `<p class="muted small"><em>${esc(r.pathogen)}</em> · class <code>${esc(key)}</code></p>` : `<p class="muted small">class <code>${esc(key)}</code></p>`}
      ${list('Look for', r.look_for)}
      ${list(r.kind === 'healthy' ? 'Good practice' : 'First steps (no chemicals)', r.cultural)}
      ${list('Organic options', r.organic)}
      ${list('Chemical options (active ingredients)', r.chemical)}
    </details>`).join('');
  smoothDetails($('remedylist'));
  $('remedyfilter').addEventListener('input', (e) => {
    const q = e.target.value.trim().toLowerCase();
    let shown = 0;
    for (const d of document.querySelectorAll('.remedy')) {
      d.hidden = !!q && !q.split(/\s+/).every((w) => d.dataset.search.includes(w));
      if (!d.hidden) shown++;
    }
    const empty = $('remedyempty');
    empty.hidden = shown > 0;
    empty.textContent = shown ? '' : `No crop or disease matches "${e.target.value.trim()}". Try a crop name such as tomato or potato.`;
  });
}

const pages = { results, explain, about };
pages[page]?.().catch((err) => {
  const main = document.querySelector('main');
  console.error('Site data failed to load; run scripts/export_site_data.py', err);
  const target = main.querySelector('.page-head') ?? main;
  target.insertAdjacentHTML('afterend', `<p class="banner danger" role="alert">Could not load the results (${esc(err.message)}). Check your connection and reload the page.</p>`);
  for (const sk of document.querySelectorAll('.skeleton-card')) sk.remove();
});

if (import.meta.env.PROD && 'serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
