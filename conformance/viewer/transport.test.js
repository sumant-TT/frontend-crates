// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { HttpResourceStore, sha256 } from './data.js';
import { boundedGunzip, ChunkTransport, MAX_PACK_SIZE } from './transport.js';

const first = new Uint8Array([0xef, 0xbb, 0xbf, 97, 13, 10]);
const second = new Uint8Array([0, 255, 42]);
const bytes = new Uint8Array([...first, ...second]);
const compressed = new Uint8Array(gzipSync(bytes));
const pack = { path: 'packs/example.gz', sha256: await sha256(compressed), size: compressed.length, decoded_sha256: await sha256(bytes), decoded_size: bytes.length };
const recipe = { schema: 2, sha256: await sha256(bytes), size: bytes.length, chunks: [
  { sha256: await sha256(first), size: first.length, offset: 0, pack },
  { sha256: await sha256(second), size: second.length, offset: first.length, pack },
] };
const resource = { path: 'source/original.yaml', sha256: recipe.sha256, size: recipe.size, transport: 'chunked-v2', recipe };
const other = { path: 'source/second.yaml', sha256: recipe.chunks[1].sha256, size: second.length, transport: 'chunked-v2', recipe: { schema: 2, sha256: recipe.chunks[1].sha256, size: second.length, chunks: [recipe.chunks[1]] } };
function storeFor(fetchImpl) {
  return new HttpResourceStore({ baseUrl: 'http://test/', fetchImpl });
}
let requests = 0;
const store = storeFor(async url => {
  requests++;
  assert.equal(url.pathname, '/packs/example.gz');
  return new Response(compressed);
});
const values = await Promise.all([store.read(resource), store.read(other), store.read(resource)]);
assert.deepEqual(values, [bytes, second, bytes]);
assert.equal(requests, 1, 'concurrent resources share a single pack fetch');
await assert.rejects(store.read({ ...resource, size: 1024 * 1024 * 1024 + 1,
  recipe: { ...recipe, size: 1024 * 1024 * 1024 + 1 } }), /size|limit|large|recipe/,
  'oversized logical content must fail before output allocation');
values[0][0] = 0;
assert.deepEqual(await store.read(resource), bytes, 'reader cannot mutate cached bytes');

for (const change of [
  r => { r.schema = 1; },
  r => { r.size++; },
  r => { r.sha256 = '0'.repeat(64); },
  r => { r.chunks[0].offset = -1; },
  r => { r.chunks[0].offset = 999; },
  r => { r.chunks[0].size++; },
  r => { r.chunks[0].pack.decoded_size = MAX_PACK_SIZE + 1; },
]) {
  const invalidRecipe = structuredClone(recipe);
  change(invalidRecipe);
  await assert.rejects(store.read({ ...resource, recipe: invalidRecipe }), /recipe|chunk/, 'cached bytes must not bypass recipe admission');
}
for (const change of [
  r => { r.chunks.reverse(); },
  r => { r.chunks[0].sha256 = '0'.repeat(64); },
  r => { r.chunks.forEach(chunk => { chunk.pack.decoded_sha256 = '0'.repeat(64); }); },
  r => { r.chunks.forEach(chunk => { chunk.pack.decoded_size++; }); },
]) {
  const invalidRecipe = structuredClone(recipe);
  change(invalidRecipe);
  const invalidStore = storeFor(async () => new Response(compressed));
  await assert.rejects(invalidStore.read({ ...resource, recipe: invalidRecipe }), /hash|length/);
}
for (const initial of ['missing', 'corrupt']) {
  let attempt = 0;
  const retryStore = storeFor(async () => {
    attempt++;
    if (attempt === 1) return initial === 'missing' ? new Response(null, { status: 404 }) : new Response(new Uint8Array(compressed.length));
    return new Response(compressed);
  });
  await assert.rejects(retryStore.read(resource), /fetch failed|SHA-256 mismatch/);
  assert.deepEqual(await retryStore.read(resource), bytes);
  assert.equal(attempt, 2);
}
const cachedStore = storeFor(async () => new Response(compressed));
cachedStore.memory.set(pack.sha256, new Uint8Array(compressed.length));
assert.deepEqual(await cachedStore.read(resource), bytes, 'corrupt cache entries must be replaced');

const indexBytes = new TextEncoder().encode(JSON.stringify({ schema: 2, resources: { [resource.path]: recipe, [other.path]: other.recipe } }));
const indexCompressed = new Uint8Array(gzipSync(indexBytes));
const index = { path: 'indexes/captures.gz', sha256: await sha256(indexCompressed), size: indexCompressed.length, decoded_size: indexBytes.length, decoded_sha256: await sha256(indexBytes) };
let indexRequests = 0;
let packRequests = 0;
const indexStore = storeFor(async url => {
  if (url.pathname === '/indexes/captures.gz') { indexRequests++; return new Response(indexCompressed); }
  packRequests++;
  return new Response(compressed);
});
const indexed = ({ recipe: ignored, ...descriptor }) => ({ ...descriptor, recipe_index: index });
assert.deepEqual(await Promise.all([indexStore.read(indexed(resource)), indexStore.read(indexed(other))]), [bytes, second]);
assert.equal(indexRequests, 1);
assert.equal(packRequests, 1);
await assert.rejects(indexStore.read({ ...indexed(resource), path: 'missing.yaml' }), /recipe missing/);
let indexAttempts = 0;
const retryIndexStore = storeFor(async url => {
  if (url.pathname !== '/indexes/captures.gz') return new Response(compressed);
  indexAttempts++;
  return indexAttempts === 1 ? new Response(null, { status: 404 }) : new Response(indexCompressed);
});
await assert.rejects(retryIndexStore.read(indexed(resource)), /fetch failed/);
assert.deepEqual(await retryIndexStore.read(indexed(resource)), bytes);
assert.equal(indexAttempts, 2, 'failed lazy indexes are evicted for retry');
await assert.rejects(indexStore.read({ ...indexed(resource), recipe_index: { ...index, decoded_sha256: '0'.repeat(64) } }), /decoded hash/);
await assert.rejects(indexStore.read({ ...indexed(resource), recipe_index: { ...index, decoded_size: 64 * 1024 * 1024 + 1 } }), /decoded size/);
const oversized = new Uint8Array(gzipSync(new Uint8Array(1000)));
new DataView(oversized.buffer).setUint32(oversized.length - 4, 1, true);
assert.throws(() => boundedGunzip(oversized, 1), /length mismatch/, 'forged ISIZE cannot truncate oversized output silently');
const largePacks = new Map();
const largeResources = [];
for (const value of [1, 2, 3]) {
  const data = new Uint8Array(600 * 1024).fill(value);
  const encoded = new Uint8Array(gzipSync(data));
  const descriptor = { path: `packs/${value}.gz`, sha256: await sha256(encoded), size: encoded.length, decoded_sha256: await sha256(data), decoded_size: data.length };
  const chunks = [];
  for (let offset = 0; offset < data.length; offset += 128 * 1024) {
    const piece = data.subarray(offset, Math.min(data.length, offset + 128 * 1024));
    chunks.push({ sha256: await sha256(piece), size: piece.length, offset, pack: descriptor });
  }
  largePacks.set(descriptor.path, encoded);
  largeResources.push({ path: `source/${value}`, sha256: await sha256(data), size: data.length,
    recipe: { schema: 2, sha256: await sha256(data), size: data.length, chunks } });
}
let decodedLoads = 0;
const boundedTransport = new ChunkTransport({ sha256, validateResource() {}, maxDecodedPackBytes: MAX_PACK_SIZE,
  async readBytes(descriptor) {
    decodedLoads++;
    const encoded = largePacks.get(descriptor.path);
    assert.equal(await sha256(encoded), descriptor.sha256);
    return encoded;
  },
});
for (const item of largeResources) {
  const decoded = await boundedTransport.read(item);
  assert.equal(await sha256(decoded), item.sha256);
  assert.ok(boundedTransport.decodedPackBytes <= MAX_PACK_SIZE);
  assert.equal(boundedTransport.packs.size, 1);
}
assert.equal(decodedLoads, 3);
await Promise.all([boundedTransport.read(largeResources[0]), boundedTransport.read(largeResources[0])]);
assert.equal(decodedLoads, 4, 'evicted packs reload once for concurrent readers');
await boundedTransport.read(largeResources[1]);
largePacks.set('packs/1.gz', new Uint8Array(largePacks.get('packs/1.gz').length));
await assert.rejects(boundedTransport.read(largeResources[0]), /assert|equal/i, 'evicted packs cannot bypass compressed-byte validation');
assert.ok(boundedTransport.decodedPackBytes <= MAX_PACK_SIZE);
console.log('chunk transport integrity, concurrency, cache admission, eviction, and retry checks passed');
