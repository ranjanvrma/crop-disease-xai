// Offline support: cache the whole app (code, ONNX Runtime, default model,
// remedies, all pages) on first visit. The build (vite.config.js) writes
// precache.json and stamps VERSION below, so every new build installs a fresh
// copy and deletes the old one.
//
// Pages and data files are fetched from the network first and fall back to the
// cache when offline, so a new version of the site shows up on the next visit
// instead of one visit later. Everything else (hashed build assets, models,
// images) is served from the cache first.

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

const networkFirst = (request) => request.mode === 'navigate' || /\.json$/.test(new URL(request.url).pathname);

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET' || new URL(request.url).origin !== location.origin) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    if (networkFirst(request)) {
      try {
        const response = await fetch(request);
        if (response.ok) event.waitUntil(cache.put(request, response.clone()));
        return response;
      } catch {
        const hit = await cache.match(request, { ignoreSearch: true, ignoreVary: true });
        if (hit) return hit;
        if (request.mode === 'navigate') return (await cache.match('./404.html')) ?? Response.error();
        return Response.error();
      }
    }
    const hit = await cache.match(request, { ignoreSearch: true, ignoreVary: true });
    if (hit) return hit;
    const response = await fetch(request);
    // Models other than the default are cached the first time they are used.
    if (response.ok && new URL(request.url).pathname.includes('/models/')) {
      event.waitUntil(cache.put(request, response.clone()));
    }
    return response;
  })());
});
