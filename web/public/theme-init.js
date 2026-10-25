// Runs before first paint (loaded without defer in <head>): applies the saved
// theme so there is no flash, and marks the page for scroll-reveal animations.
// A separate file, not inline, because the Content-Security-Policy forbids inline scripts.
(function () {
  var root = document.documentElement;
  try {
    var t = localStorage.getItem('cropxai-theme');
    if (t === 'light' || t === 'dark') root.setAttribute('data-theme', t);
  } catch (e) { /* storage blocked: follow the device setting */ }
  if (!window.matchMedia || !matchMedia('(prefers-reduced-motion: reduce)').matches) root.classList.add('js-motion');
})();
