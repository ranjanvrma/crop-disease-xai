// Offline support: cache the whole app (code, ONNX Runtime, model, remedies) on
// first visit, then serve it from the cache. The build (vite.config.js) writes
// precache.json and stamps VERSION below, so every new build installs a fresh
// copy and deletes the old one.

const VERSION = '__PRECACHE_VERSION__';
const CACHE = `cropxai-${VERSION}`;

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const { urls } = await (await fetch('./precache.json', { cache: 'no-store' })).json();
    await (await caches.open(CACHE)).addAll(urls);
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET' || new URL(event.request.url).origin !== location.origin) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    return (await cache.match(event.request, { ignoreSearch: true, ignoreVary: true })) ?? fetch(event.request);
  })());
});
