import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { defineConfig } from 'vite';
import { icon } from './src/icons.js';
import { SECURITY_HEADERS } from './security-headers.js';

const PAGES = ['index', 'explain', 'results', 'about', 'privacy', 'terms', '404'];

// Shared header and footer (partials/*.html) and {{icon:name}} placeholders are
// expanded into every page at build time, so each page is complete, crawlable
// static HTML with no client-side templating.
function partials() {
  const read = (name) => readFileSync(resolve(import.meta.dirname, 'partials', name), 'utf8');
  const icons = (html) => html.replace(/{{icon:([a-z-]+)}}/g, (_, name) => icon(name));
  return {
    name: 'partials',
    transformIndexHtml(html) {
      return icons(html
        .replace(/<!--@header ([a-z0-9]+)-->/, (_, page) => read('header.html')
          .replace(`data-nav="${page}"`, `data-nav="${page}" aria-current="page"`)
          .replace('{{net}}', page === 'diagnose' ? '<span id="net" class="net-chip" aria-live="polite"></span>' : ''))
        .replace(/<!--@footer ([a-z0-9]+)-->/, (_, page) => read('footer.html')
          .replace('{{modelinfo}}', page === 'diagnose' ? '<p id="modelinfo" class="model-info"></p>' : '')));
    },
  };
}

// Lists every built file so the service worker (public/sw.js) can cache the
// whole app, default model included, on first visit. The other models in the
// picker are cached the first time they are chosen instead, so a phone does not
// download all of them up front. The hash names the cache, so a new build
// replaces the old copy.
function precacheManifest() {
  let outDir;
  return {
    name: 'precache-manifest',
    configResolved(config) { outDir = config.build.outDir; },
    closeBundle() {
      const files = [];
      const walk = (dir) => {
        for (const name of readdirSync(dir)) {
          const path = join(dir, name);
          if (statSync(path).isDirectory()) walk(path);
          else if (!['sw.js', 'precache.json'].includes(name) && !name.endsWith('.map')) files.push(path);
        }
      };
      walk(outDir);
      const { default: def } = JSON.parse(readFileSync(join(outDir, 'models', 'index.json'), 'utf8'));
      const precached = files.filter((f) => {
        const parts = relative(outDir, f).split(/[\\/]/);
        return parts[0] !== 'models' || parts.length === 2 || parts[1] === def;
      });
      const hash = createHash('sha256');
      for (const f of files.sort()) hash.update(relative(outDir, f)).update(readFileSync(f));
      const version = hash.digest('hex').slice(0, 12);
      const urls = precached.map((f) => relative(outDir, f).split('\\').join('/'));
      writeFileSync(join(outDir, 'precache.json'), JSON.stringify({ version, urls: ['./', ...urls] }));
      const sw = join(outDir, 'sw.js');
      writeFileSync(sw, readFileSync(sw, 'utf8').replace('__PRECACHE_VERSION__', version));
    },
  };
}

// Static hosts (Render included) answer a missing path with 404.html, so
// `vite preview` does the same instead of falling back to index.html.
function notFoundPage() {
  return {
    name: 'not-found-page',
    configurePreviewServer(server) {
      const dist = resolve(server.config.root, server.config.build.outDir);
      return () => server.middlewares.use((req, res) => {
        const home = ['/', '/index.html'].includes(req.url.split('?')[0]); // render.yaml rewrites "/" to /index.html
        res.statusCode = home ? 200 : 404;
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.end(readFileSync(join(dist, home ? 'index.html' : '404.html')));
      });
    },
  };
}

export default defineConfig({
  base: './', // works from any sub-path, e.g. GitHub Pages
  appType: 'mpa',
  build: {
    target: 'es2020',
    assetsInlineLimit: 0,
    rollupOptions: {
      input: Object.fromEntries(PAGES.map((p) => [p, resolve(import.meta.dirname, `${p}.html`)])),
    },
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  plugins: [partials(), precacheManifest(), notFoundPage()],
  preview: { headers: SECURITY_HEADERS },
});
