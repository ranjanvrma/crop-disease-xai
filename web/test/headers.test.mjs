// The security headers tested locally (vite preview) must be the ones deployed
// on Render. This fails if web/security-headers.js and render.yaml drift apart.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { SECURITY_HEADERS } from '../security-headers.js';

const yaml = readFileSync(new URL('../../render.yaml', import.meta.url), 'utf8');

test('render.yaml sets every security header with the same value', () => {
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) {
    const block = yaml.match(new RegExp(`name: ${name}\\s*\\n\\s*value: "?([^"\\n]+)"?`));
    assert.ok(block, `${name} missing from render.yaml`);
    assert.equal(block[1].trim(), value, `${name} differs between security-headers.js and render.yaml`);
  }
});

test('the CSP allows nothing from other origins', () => {
  const csp = SECURITY_HEADERS['Content-Security-Policy'];
  assert.doesNotMatch(csp, /https?:|\*/, 'no external origins or wildcards');
  assert.match(csp, /frame-ancestors 'none'/);
  assert.match(csp, /object-src 'none'/);
});
