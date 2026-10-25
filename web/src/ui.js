// Shared site behaviour: navigation, theme, back-to-top, scroll progress,
// accordions, copy-to-clipboard, toasts and all animation (Motion, vanilla API).
// Every animation is skipped when the visitor prefers reduced motion; content is
// never hidden unless this script is running to reveal it (see style.css).

import { animate, hover, inView, press, scroll, stagger } from 'motion';
import { icon } from './icons.js';

const root = document.documentElement;
export const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const canHover = matchMedia('(hover: hover) and (pointer: fine)');
export const spring = { type: 'spring', stiffness: 260, damping: 26, mass: 0.9 };
const soft = { type: 'spring', stiffness: 170, damping: 24 };

const still = () => reduceMotion.matches;

// ---------- reveal on scroll ----------

/** Fade/slide in every [data-reveal] element (or its children for [data-stagger]) when it enters view. */
export function reveal(scope = document) {
  const targets = scope.querySelectorAll('[data-reveal]:not([data-revealed])');
  for (const el of targets) {
    el.dataset.revealed = '';
    if (still()) { el.style.opacity = 1; continue; }
    inView(el, () => {
      const kids = el.hasAttribute('data-stagger') ? [...el.children] : null;
      animate(el, { opacity: [0, 1], y: kids ? 0 : [18, 0] }, { duration: 0.55, ease: [0.22, 1, 0.36, 1] });
      if (kids) animate(kids, { opacity: [0, 1], y: [22, 0], scale: [0.98, 1] }, { ...soft, delay: stagger(0.07) });
    }, { amount: 0.15 });
  }
}

/** Animate freshly rendered elements into place right away (e.g. a new diagnosis). */
export function enter(elements) {
  const els = elements.filter(Boolean);
  if (still() || !els.length) return;
  animate(els, { opacity: [0, 1], y: [24, 0] }, { ...soft, delay: stagger(0.08) });
}

/** Grow bars (confidence bars, chart rects) from zero width. */
export function growBars(scope) {
  if (still()) return;
  const bars = scope.querySelectorAll('.bar i, .chart rect[class^="s"]');
  if (!bars.length) return;
  inView(scope, () => {
    animate(bars, { scaleX: [0, 1] }, { duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: stagger(0.025) });
  }, { amount: 0.2 });
}

/** Count the leading number of an element's text up from 0, keeping its exact final text. */
export function countUp(el) {
  const text = el.textContent;
  const m = text.match(/^(\d+(?:\.\d+)?)(.*)$/s);
  if (still() || !m) return;
  const target = Number(m[1]);
  const decimals = (m[1].split('.')[1] ?? '').length;
  inView(el, () => {
    animate(0, target, {
      duration: 1.1, ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => { el.textContent = `${v.toFixed(decimals)}${m[2]}`; },
      onComplete: () => { el.textContent = text; },
    });
  });
}

// ---------- gestures ----------

function gestures() {
  if (still()) return;
  press('.btn, .pill, .sample, .icon-btn, .to-top', (el) => {
    animate(el, { scale: 0.96 }, { type: 'spring', stiffness: 500, damping: 30 });
    return () => animate(el, { scale: 1 }, spring);
  });
  if (!canHover.matches) return;
  hover('.card.lift, .sample', (el) => {
    animate(el, { y: -4 }, spring);
    return () => animate(el, { y: 0 }, spring);
  });
}

/** 3D tilt that follows the pointer (fine pointers only), springing back on leave. */
export function tilt(el, { max = 9 } = {}) {
  if (still() || !canHover.matches || !el) return;
  el.addEventListener('pointermove', (e) => {
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    animate(el, { rotateY: x * max * 2, rotateX: -y * max * 2 }, soft);
  });
  el.addEventListener('pointerleave', () => animate(el, { rotateY: 0, rotateX: 0 }, { ...soft, stiffness: 120 }));
}

/** Gentle parallax: the element drifts upward a little as it scrolls past. */
export function parallax(el, distance = 40) {
  if (still() || !el) return;
  scroll(animate(el, { y: [0, -distance] }, { ease: 'linear' }), { target: el, offset: ['start end', 'end start'] });
}

// ---------- navigation ----------

function navIndicator() {
  const nav = document.getElementById('site-nav');
  const ind = nav?.querySelector('.nav-indicator');
  if (!nav || !ind) return;
  const links = [...nav.querySelectorAll('.nav-list a')];
  const current = links.find((a) => a.getAttribute('aria-current') === 'page');
  const place = (a, instant) => {
    if (!a || getComputedStyle(ind).display === 'none') return;
    const nr = nav.getBoundingClientRect(), r = a.getBoundingClientRect();
    const to = { x: r.left - nr.left, width: r.width, opacity: 1 };
    if (instant || still()) { Object.assign(ind.style, { transform: `translateX(${to.x}px)`, width: `${to.width}px`, opacity: 1 }); return; }
    animate(ind, to, spring);
  };
  place(current, true);
  for (const a of links) {
    a.addEventListener('pointerenter', () => place(a));
    a.addEventListener('focus', () => place(a));
  }
  nav.addEventListener('pointerleave', () => (current ? place(current) : animate(ind, { opacity: 0 })));
  addEventListener('resize', () => place(current, true));
}

function mobileMenu() {
  const btn = document.getElementById('menu-btn');
  const nav = document.getElementById('site-nav');
  if (!btn || !nav) return;
  const set = (open) => {
    nav.classList.toggle('open', open);
    btn.setAttribute('aria-expanded', String(open));
    btn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    if (open && !still()) {
      const list = nav.querySelector('.nav-list');
      animate(list, { opacity: [0, 1], y: [-8, 0] }, { duration: 0.22 });
      animate(list.querySelectorAll('li'), { opacity: [0, 1], x: [-12, 0] }, { ...soft, delay: stagger(0.04) });
    }
  };
  btn.addEventListener('click', () => set(!nav.classList.contains('open')));
  nav.addEventListener('click', (e) => { if (e.target.closest('a')) set(false); });
  addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && nav.classList.contains('open')) { set(false); btn.focus(); }
  });
  document.addEventListener('click', (e) => {
    if (nav.classList.contains('open') && !e.target.closest('.site-header')) set(false);
  });
  matchMedia('(min-width: 861px)').addEventListener('change', (e) => { if (e.matches) set(false); });
}

// ---------- theme ----------

const THEME_KEY = 'cropxai-theme';
const effectiveTheme = () => root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

function themeToggle() {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;
  const meta = document.querySelector('meta[name="theme-color"]:not([media])');
  const sync = () => {
    const t = effectiveTheme();
    btn.dataset.mode = t;
    const next = t === 'dark' ? 'light' : 'dark';
    btn.setAttribute('aria-label', `Theme: ${root.dataset.theme ? t : `${t} (follows your device)`}. Switch to ${next}`);
    if (meta) meta.content = t === 'dark' ? '#0b140f' : '#f4f2e9';
  };
  btn.addEventListener('click', () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch { /* not saved; still applied */ }
    sync();
    if (!still()) animate(btn.querySelectorAll('.icon'), { rotate: [-90, 0], scale: [0.6, 1] }, spring);
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', sync);
  sync();
}

// ---------- scroll helpers ----------

function scrollProgress() {
  const bar = document.querySelector('.scroll-progress i');
  if (!bar || document.body.dataset.page === 'diagnose') return;
  if (still()) {
    addEventListener('scroll', () => {
      const max = root.scrollHeight - innerHeight;
      bar.style.transform = `scaleX(${max > 0 ? scrollY / max : 0})`;
    }, { passive: true });
    return;
  }
  scroll(animate(bar, { scaleX: [0, 1] }, { ease: 'linear' }));
}

function backToTop() {
  const btn = document.getElementById('to-top');
  if (!btn) return;
  let shown = false;
  const update = () => {
    const show = scrollY > 900;
    if (show === shown) return;
    shown = show;
    if (show) {
      btn.hidden = false;
      if (!still()) animate(btn, { opacity: [0, 1], y: [12, 0] }, spring);
    } else {
      btn.hidden = true;
    }
  };
  addEventListener('scroll', update, { passive: true });
  btn.addEventListener('click', () => {
    scrollTo({ top: 0, behavior: still() ? 'auto' : 'smooth' });
    document.getElementById('main')?.focus({ preventScroll: true });
  });
  update();
}

// ---------- accordions (details/summary with a height animation) ----------

export function smoothDetails(scope = document) {
  for (const d of scope.querySelectorAll('details.more, .faq details, details.hood, details.advanced')) {
    if (d.dataset.smooth) continue;
    d.dataset.smooth = '';
    const summary = d.querySelector(':scope > summary');
    summary?.addEventListener('click', (e) => {
      if (still()) return;
      e.preventDefault();
      const start = d.offsetHeight;
      if (d.open) {
        const end = summary.offsetHeight + (parseFloat(getComputedStyle(d).paddingTop) || 0) + (parseFloat(getComputedStyle(d).paddingBottom) || 0) + (parseFloat(getComputedStyle(d).borderTopWidth) || 0) * 2;
        d.style.overflow = 'hidden';
        animate(d, { height: [`${start}px`, `${end}px`] }, { duration: 0.28, ease: [0.4, 0, 0.2, 1] }).then(() => {
          d.open = false;
          d.style.height = d.style.overflow = '';
        });
      } else {
        d.open = true;
        const end = d.offsetHeight;
        d.style.overflow = 'hidden';
        animate(d, { height: [`${start}px`, `${end}px`] }, { duration: 0.34, ease: [0.22, 1, 0.36, 1] }).then(() => {
          d.style.height = d.style.overflow = '';
        });
      }
    });
  }
}

// ---------- feedback ----------

let toastTimer;
/** Short status message at the bottom of the screen (also announced to screen readers). */
export function toast(message, iconName = 'check') {
  let el = document.querySelector('.toast');
  if (!el) {
    el = document.createElement('div');
    el.className = 'toast';
    el.setAttribute('role', 'status');
    document.body.append(el);
  }
  el.innerHTML = `${icon(iconName, { size: 18 })}<span></span>`;
  el.querySelector('span').textContent = message;
  el.hidden = false;
  if (!still()) animate(el, { opacity: [0, 1], y: [16, 0] }, spring);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

/** Copy text to the clipboard with visible feedback; falls back when the API is unavailable. */
export async function copyText(text, done = 'Copied to clipboard') {
  try {
    await navigator.clipboard.writeText(text);
    toast(done);
  } catch {
    const ta = Object.assign(document.createElement('textarea'), { value: text });
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;opacity:0';
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand?.('copy');
    ta.remove();
    toast(ok ? done : 'Could not copy: select the text and copy it yourself', ok ? 'check' : 'alert');
  }
}

function copyButtons() {
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-copy]');
    if (b) copyText(b.dataset.copy, b.dataset.copyDone || 'Copied to clipboard');
  });
}

// ---------- boot ----------

function boot() {
  root.classList.add('motion-ready');
  themeToggle();
  mobileMenu();
  navIndicator();
  scrollProgress();
  backToTop();
  copyButtons();
  smoothDetails();
  gestures();
  reveal();
  if (!still()) {
    const header = document.querySelector('.site-header');
    if (header) animate(header, { y: [-12, 0], opacity: [0, 1] }, { duration: 0.4 });
  }
}

boot();
