// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { sha256 } from './data.js';
import { configureDevelopmentCatalog, loadStartupCatalog, validateStartupCatalog } from './startup.js';
import { gzipSync } from './vendor/fflate/esm/browser.js';

const base = new URL('http://localhost/prefix/');
const catalog = { schema: 1, revision: 'a'.repeat(64), metadata: { tabs: [] },
  sources: { archives: [], unified: [{ path: 'capture', format: 'typed-json-v1' }] },
  transport: { schema: 2, raw_buckets: {} } };
catalog.runtime_sha256 = await sha256(new TextEncoder().encode(JSON.stringify(catalog) + '\n'));
const responseForText = value => new Response(gzipSync(new TextEncoder().encode(value)));
const responseFor = value => responseForText(JSON.stringify(value) + '\n');
const unorderedSource = JSON.stringify({ ...catalog, runtime_sha256: undefined }).replace('"tabs":[]', '"tabs":[],"group_labels":{"v1_split":"Split","1":"One","2":"Two","10":"Ten"}') + '\n';
const unorderedHash = await sha256(new TextEncoder().encode(unorderedSource));
const canonicalSource = unorderedSource.slice(0, -2) + `,"runtime_sha256":"${unorderedHash}"}\n`;
const orderedResult = await loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => responseForText(canonicalSource) });
assert.equal(orderedResult.catalog.metadata.group_labels.v1_split, 'Split');
assert.equal(orderedResult.catalog.metadata.group_labels['10'], 'Ten');
assert.notEqual(await sha256(new TextEncoder().encode(JSON.stringify(JSON.parse(unorderedSource)) + '\n')), unorderedHash,
  'fixture must exercise JavaScript integer-key reordering relative to Rust source');
for (const malformed of [canonicalSource.trimEnd(), canonicalSource.replace(',"runtime_sha256"', ', "runtime_sha256"'),
  canonicalSource.replace('"Ten"', '"Changed"'), canonicalSource.replace(`,"runtime_sha256":"${unorderedHash}"`, '')]) {
  await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => responseForText(malformed) }), /integrity/);
}
for (const query of ['', 'startup=compressed', 'startup=external', 'startup=bootstrap', 'startup=inline',
  'startup=bootstrap&raw=conformance/source.yaml', 'startup=external&slots_tab-unified=old']) {
  const requests = [];
  const result = await loadStartupCatalog(base, new URLSearchParams(query), {
    fetchImpl: async (url, options) => { requests.push(url.pathname); assert.equal(options.cache, 'no-store'); return responseFor(catalog); },
  });
  assert.deepEqual(result.catalog, catalog);
  assert.deepEqual(requests, ['/prefix/conformance/viewer-resolved-catalog.json.gz']);
  assert.equal(result.store.memory.size, 0, 'catalog load must not preload captures or originals');
  assert.equal(result.store.allowSourceFiles, false);
}
for (const mutate of [
  value => { value.revision = 'b'.repeat(64); },
  value => { value.runtime_sha256 = '0'.repeat(64); },
  value => { delete value.runtime_sha256; },
  value => { value.transport.raw_buckets.ff = {}; },
  value => { value.sources.unified[0].path = 'changed'; },
]) {
  const changed = structuredClone(catalog); mutate(changed);
  await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => responseFor(changed) }), /integrity/);
}
configureDevelopmentCatalog(null);
await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => new Response(null, { status: 404 }) }), /404/);
const yamlCatalog = structuredClone(catalog);
delete yamlCatalog.runtime_sha256;
delete yamlCatalog.transport;
delete yamlCatalog.sources.unified[0].format;
let fallbackCalls = 0;
configureDevelopmentCatalog(async (actualBase, fetchImpl) => {
  fallbackCalls++;
  assert.equal(actualBase, base);
  const response = await fetchImpl(new URL('conformance/viewer-unified-catalog.json', actualBase));
  return response.json();
});
try {
  const requests = [];
  const loaded = await loadStartupCatalog(base, new URLSearchParams('startup=inline'), { fetchImpl: async url => {
    requests.push(url.pathname);
    return url.pathname.endsWith('.gz') ? new Response(null, { status: 404 }) : Response.json(yamlCatalog);
  } });
  assert.deepEqual(loaded.catalog, yamlCatalog);
  assert.equal(loaded.store.allowSourceFiles, true);
  assert.equal(fallbackCalls, 1);
  assert.deepEqual(requests, ['/prefix/conformance/viewer-resolved-catalog.json.gz', '/prefix/conformance/viewer-unified-catalog.json']);
  for (const response of [new Response(null, { status: 503 }), responseFor({}), responseFor({ ...catalog, runtime_sha256: '0'.repeat(64) }), new Response(new Uint8Array([1, 2]))]) {
    await assert.rejects(loadStartupCatalog(base, new URLSearchParams(), { fetchImpl: async () => response }));
    assert.equal(fallbackCalls, 1, 'development fallback must not hide corrupt or failed published data');
  }
} finally { configureDevelopmentCatalog(null); }
assert.throws(() => validateStartupCatalog(yamlCatalog), /regeneration/);
assert.deepEqual(validateStartupCatalog(yamlCatalog, { resolved: false }), yamlCatalog);
const unknown = structuredClone(yamlCatalog);
unknown.sources.unified[0].format = 'unknown';
assert.throws(() => validateStartupCatalog(unknown, { resolved: false }), /Unsupported/);
for (const invalid of [null, false, 0, '', [], {}]) {
  assert.throws(() => validateStartupCatalog(invalid), /Unsupported/);
}
console.log('canonical startup modes, runtime identity, lazy loading, fail-closed publication, and development fallback passed');
