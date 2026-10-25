// Lucide icons (ISC licence) rendered to inline SVG strings. Used at build time
// by vite.config.js for the static HTML and at runtime for generated markup, so
// icons never need an icon font or a client-side icon pass.

import {
  ArrowUp, BookOpen, Camera, ChartColumn, Check, ChevronRight, Code, Copy, Cpu, FileText, Gauge,
  ImageUp, Info, Layers, Leaf, Menu, Microscope, Moon, ScanSearch, ShieldCheck, Sprout, Sun,
  TriangleAlert, WifiOff, X,
} from 'lucide';

const ICONS = {
  'arrow-up': ArrowUp, 'book-open': BookOpen, camera: Camera, chart: ChartColumn, check: Check,
  'chevron-right': ChevronRight, code: Code, copy: Copy, cpu: Cpu, 'file-text': FileText, gauge: Gauge,
  'image-up': ImageUp, info: Info, layers: Layers, leaf: Leaf, menu: Menu, microscope: Microscope,
  moon: Moon, scan: ScanSearch, shield: ShieldCheck, sprout: Sprout, sun: Sun, alert: TriangleAlert,
  'wifi-off': WifiOff, x: X,
};

const attrs = (o) => Object.entries(o).map(([k, v]) => `${k}="${String(v).replace(/"/g, '&quot;')}"`).join(' ');

/** `<svg>` markup for a named icon; decorative (aria-hidden) unless a label is given. */
export function icon(name, { size = 20, cls = '', label = '' } = {}) {
  const node = ICONS[name];
  if (!node) throw new Error(`unknown icon "${name}"`);
  const a11y = label ? `role="img" aria-label="${label.replace(/"/g, '&quot;')}"` : 'aria-hidden="true" focusable="false"';
  const body = node.map(([tag, a]) => `<${tag} ${attrs(a)}/>`).join('');
  return `<svg class="icon ${cls}" ${a11y} width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
}
