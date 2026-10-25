// Shared site behaviour: navigation, theme, back-to-top, scroll progress,
// accordions, copy-to-clipboard, toasts and all animation (Motion, vanilla API).
// Every animation is skipped when the visitor prefers reduced motion; content is
// never hidden unless this script is running to reveal it (see style.css).

import { animate, hover, inView, press, scroll, stagger } from 'motion';
import { icon } from './icons.js';

const root = document.documentElement;
export const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const canHover = matchMedia('(hover: hover) and (pointer: fine)');
const wideNav = matchMedia('(min-width: 861px)'); // style.css switches to the menu button below this
export const spring = { type: 'spring', stiffness: 260, damping: 26, mass: 0.9 };
const soft = { type: 'spring', stiffness: 170, damping: 24 };
const ease = [0.22, 1, 0.36, 1];

const still = () => reduceMotion.matches;

// ---------- reveal on scroll ----------

/** Fade/slide in every [data-reveal] element (or its children for [data-stagger]) when it enters view. */
export function reveal(scope = document) {
  for (const el of scope.querySelectorAll('[data-reveal]:not([data-revealed])')) {
    el.dataset.revealed = '';
    if (still()) { el.style.opacity = 1; continue; }
    inView(el, () => {
      const kids = el.hasAttribute('data-stagger') ? [...el.children] : null;
      animate(el, { opacity: [0, 1], y: kids ? 0 : [18, 0] }, { duration: 0.6, ease });
      if (kids) animate(kids, { opacity: [0, 1], y: [22, 0] }, { ...soft, delay: stagger(0.08) });
    }, { amount: 0.12 });
  }
}

/** Animate freshly rendered elements into place right away (e.g. a new diagnosis). */
export function enter(elements) {
  const els = elements.filter(Boolean);
  if (still() || !els.length) return;
  animate(els, { opacity: [0, 1], y: [24, 0] }, { ...soft, delay: stagger(0.08) });
}

/** Grow bars (confidence bars, chart rects) from zero width once they are in view. */
export function growBars(scope) {
  if (still()) return;
  const bars = scope.querySelectorAll('.bar i, .chart rect[class^="s"]');
  if (!bars.length) return;
  inView(scope, () => {
    animate(bars, { scaleX: [0, 1] }, { duration: 0.8, ease, delay: stagger(0.025) });
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
      duration: 1.1, ease,
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

// ---------- navigation ----------

function navIndicator() {
  const nav = document.getElementById('site-nav');
  const ind = nav?.querySelector('.nav-indicator');
  if (!nav || !ind) return;
  const links = [...nav.querySelectorAll('.nav-list a')];
  const current = links.find((a) => a.getAttribute('aria-current') === 'page');
  const place = (a, instant) => {
    if (!a || !wideNav.matches) return;
    const nr = nav.getBoundingClientRect();
    const r = a.getBoundingClientRect();
    const to = { x: r.left - nr.left, y: r.top - nr.top, width: r.width, opacity: 1 };
    animate(ind, to, instant || still() ? { duration: 0 } : spring);
  };
  const home = () => (current ? place(current) : animate(ind, { opacity: 0 }, { duration: 0.2 }));
  place(current, true);
  // The web fonts change the link widths, so measure again once they have loaded.
  document.fonts?.ready.then(() => place(current, true));
  for (const a of links) {
    a.addEventListener('pointerenter', () => place(a));
    a.addEventListener('focus', () => place(a));
    a.addEventListener('blur', home);
  }
  nav.addEventListener('pointerleave', home);
  addEventListener('resize', () => place(current, true));
  wideNav.addEventListener('change', () => place(current, true));
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
      animate(list, { opacity: [0, 1], y: [-8, 0], scale: [0.98, 1] }, { duration: 0.22, ease });
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
  wideNav.addEventListener('change', (e) => { if (e.matches) set(false); });
}

/** Mark the contents link of the section being read (aria-current="true"). */
function tocSpy() {
  const links = [...document.querySelectorAll('.toc a[href^="#"]')];
  const sections = links.map((a) => document.getElementById(a.hash.slice(1))).filter(Boolean);
  if (!sections.length || !('IntersectionObserver' in window)) return;
  const visible = new Set();
  const mark = () => {
    const first = sections.find((s) => visible.has(s));
    for (const a of links) {
      if (first && a.hash === `#${first.id}`) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    }
  };
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) (e.isIntersecting ? visible.add(e.target) : visible.delete(e.target));
    mark();
  }, { rootMargin: '-25% 0px -55% 0px' });
  for (const s of sections) io.observe(s);
}

// ---------- theme ----------

const THEME_KEY = 'cropxai-theme';
const THEME_COLOR = { light: '#e9f1e8', dark: '#06110b' };
const prefersDark = matchMedia('(prefers-color-scheme: dark)');
const effectiveTheme = () => root.dataset.theme || (prefersDark.matches ? 'dark' : 'light');

function themeToggle() {
  const btn = document.getElementById('theme-toggle');
  const meta = document.querySelector('meta[name="theme-color"]');
  const sync = () => {
    const t = effectiveTheme();
    if (meta) meta.content = THEME_COLOR[t];
    if (!btn) return;
    btn.dataset.mode = t;
    const next = t === 'dark' ? 'light' : 'dark';
    btn.setAttribute('aria-label', `Switch to ${next} theme (now ${t}${root.dataset.theme ? '' : ', following your device'})`);
  };
  btn?.addEventListener('click', () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch { /* not saved; still applied */ }
    sync();
    if (!still()) animate(btn.querySelectorAll('.icon'), { rotate: [-90, 0], scale: [0.6, 1] }, spring);
  });
  prefersDark.addEventListener('change', sync);
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
    btn.hidden = !show;
    if (show && !still()) animate(btn, { opacity: [0, 1], y: [12, 0] }, spring);
  };
  addEventListener('scroll', update, { passive: true });
  btn.addEventListener('click', () => {
    scrollTo({ top: 0, behavior: still() ? 'auto' : 'smooth' });
    document.getElementById('main')?.focus({ preventScroll: true });
  });
  update();
}

// ---------- accordions (details/summary with a height animation) ----------

const ACCORDIONS = 'details.more, .faq details, details.hood, details.advanced, details.remedy';

/** Animate opening and closing of the accordions in `scope`; safe to call again on new markup. */
export function smoothDetails(scope = document) {
  for (const d of scope.querySelectorAll(ACCORDIONS)) {
    if (d.dataset.smooth) continue;
    d.dataset.smooth = '';
    const summary = d.querySelector(':scope > summary');
    summary?.addEventListener('click', (e) => {
      if (still()) return;
      e.preventDefault();
      if ('animating' in d.dataset) return; // ignore clicks while it is still moving
      d.dataset.animating = '';
      const done = () => { d.style.height = d.style.overflow = ''; delete d.dataset.animating; };
      const start = d.offsetHeight;
      d.style.overflow = 'hidden';
      if (d.open) {
        const cs = getComputedStyle(d);
        const end = summary.offsetHeight + parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom)
          + parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth);
        animate(d, { height: [`${start}px`, `${end}px`] }, { duration: 0.28, ease: [0.4, 0, 0.2, 1] })
          .then(() => { d.open = false; done(); });
      } else {
        d.open = true;
        const end = d.offsetHeight;
        animate(d, { height: [`${start}px`, `${end}px`] }, { duration: 0.34, ease }).then(done);
      }
    });
  }
}

// ---------- feedback ----------

let toastEl;
let toastTimer;
/** Short status message at the bottom of the screen (also announced to screen readers). */
export function toast(message, iconName = 'check') {
  toastEl.innerHTML = `${icon(iconName, { size: 18 })}<span></span>`;
  toastEl.querySelector('span').textContent = message;
  toastEl.hidden = false;
  if (!still()) animate(toastEl, { opacity: [0, 1], y: [16, 0] }, spring);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.hidden = true; }, 2600);
}

/** Copy text to the clipboard with visible feedback; falls back when the API is unavailable. */
export async function copyText(text, done = 'Copied to clipboard') {
  try {
    await navigator.clipboard.writeText(text);
    toast(done);
  } catch {
    const ta = Object.assign(document.createElement('textarea'), { value: text });
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
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
  // The live region exists from the start, so screen readers announce the first toast too.
  toastEl = Object.assign(document.createElement('div'), { className: 'toast', hidden: true });
  toastEl.setAttribute('role', 'status');
  document.body.append(toastEl);

  root.classList.add('motion-ready');
  themeToggle();
  mobileMenu();
  navIndicator();
  tocSpy();
  scrollProgress();
  backToTop();
  copyButtons();
  smoothDetails();
  gestures();
  reveal();
  if (!still()) {
    const header = document.querySelector('.header-bar');
    if (header) animate(header, { y: [-12, 0], opacity: [0, 1] }, { duration: 0.45, ease });
  }
}

boot();
