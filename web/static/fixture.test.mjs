// Shared by the node tests: the Python fixture and a tolerance comparison (NaN/None must match).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

export const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/demo_cases.json', import.meta.url), 'utf8'));
export let checks = 0;
export function close(a, b, what, tol = 1e-9) {
  checks++;
  if (b === null || Number.isNaN(b)) return assert.ok(a === null || Number.isNaN(a), `${what}: expected NaN, got ${a}`);
  assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);
}
export function closeAll(a, b, what, tol) {
  assert.equal(a.length, b.length, `${what}: length`);
  b.forEach((v, i) => close(a[i], v, `${what}[${i}]`, tol));
}
