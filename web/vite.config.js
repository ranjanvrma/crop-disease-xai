import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { defineConfig } from 'vite';

// Lists every built file so the service worker (public/sw.js) can cache the
// whole app, model included, on first visit. The hash names the cache, so a
// new build replaces the old copy.
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
      const hash = createHash('sha256');
      for (const f of files.sort()) hash.update(f).update(readFileSync(f));
      const version = hash.digest('hex').slice(0, 12);
      const urls = files.map((f) => relative(outDir, f).split('\\').join('/'));
      writeFileSync(join(outDir, 'precache.json'), JSON.stringify({ version, urls: ['./', ...urls] }));
      const sw = join(outDir, 'sw.js');
      writeFileSync(sw, readFileSync(sw, 'utf8').replace('__PRECACHE_VERSION__', version));
    },
  };
}

export default defineConfig({
  base: './', // works from any sub-path, e.g. GitHub Pages
  build: { target: 'es2020', assetsInlineLimit: 0 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  plugins: [precacheManifest()],
});
