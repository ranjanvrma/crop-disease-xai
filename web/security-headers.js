// HTTP security headers for the site. The same values are set in ../render.yaml
// (production) and on `vite preview` (local testing); test/headers.test.mjs
// checks that the two stay in sync.
//
// The site is static and has no logins, forms that post, cookies or third-party
// scripts. The CSP therefore allows only this origin, plus:
//   'wasm-unsafe-eval'  ONNX Runtime Web compiles its WebAssembly model runtime
//   style 'unsafe-inline'  inline style attributes (bar widths) in rendered markup
//   blob: / data:       canvases and images created in the page
export const SECURITY_HEADERS = {
  'Content-Security-Policy': [
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self'",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "manifest-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ].join('; '),
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), payment=(), usb=()',
  'Cross-Origin-Opener-Policy': 'same-origin',
  'X-Frame-Options': 'DENY',
};
