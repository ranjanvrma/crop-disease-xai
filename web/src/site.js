// Entry point for the plain content pages (privacy, terms, 404): shared
// navigation and animation, plus the offline service worker.
import './ui.js';

if (import.meta.env.PROD && 'serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
