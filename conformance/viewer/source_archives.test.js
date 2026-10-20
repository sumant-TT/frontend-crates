// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { HttpResourceStore, sha256 } from './data.js';
import './development_parser.js';

function archiveBytes(files) {
  const blocks = [];
  for (const [name, data] of files) {
    const header = Buffer.alloc(512);
    header.write(name);
    header.write('0000644\0', 100);
    header.write('0000000\0', 108);
    header.write('0000000\0', 116);
    header.write(data.length.toString(8).padStart(11, '0') + '\0', 124);
    header.write('00000000000\0', 136);
    header.fill(32, 148, 156);
    header[156] = 48;
    const sum = header.reduce((total, byte) => total + byte, 0);
    header.write(sum.toString(8).padStart(6, '0') + '\0 ', 148);
    blocks.push(header, Buffer.from(data), Buffer.alloc((512 - data.length % 512) % 512));
  }
  return new Uint8Array(gzipSync(Buffer.concat([...blocks, Buffer.alloc(1024)])));
}

const name = 'families/qwen3/vllm_python-1.2.3.yaml';
const bytes = new TextEncoder().encode('typed: [null, "null", 1, 1.0, true]\r\n');
const compressed = archiveBytes([[name, bytes]]);
const archive = { path: 'conformance/fixtures-unified-v2/versions/vllm-1.2.3.tar.gz', sha256: await sha256(compressed), size: compressed.length };
const resource = { path: `conformance/fixtures-unified-v2/${name}`, sha256: await sha256(bytes), size: bytes.length, archive };
let requests = 0;
const store = new HttpResourceStore({ baseUrl: 'http://local/', allowSourceFiles: true, retainBytes: false,
  fetchImpl: async url => { requests++; assert.equal(url.pathname, '/' + archive.path); return new Response(compressed); } });
const readers = await Promise.all(Array.from({ length: 12 }, () => store.read(resource)));
assert.equal(requests, 1, 'concurrent members share their archive download without nesting scheduler slots');
for (const result of readers) assert.deepEqual(result, bytes);
readers[0][0] = 0;
assert.deepEqual(await store.read(resource), bytes);
await assert.rejects(store.read({ ...resource, sha256: '0'.repeat(64) }), /integrity mismatch/);
await assert.rejects(store.read({ ...resource, path: resource.path.replace(name, 'families/qwen3/missing.yaml') }), /member missing/);
await assert.rejects(store.read({ ...resource, path: 'outside/file.yaml' }), /binding/);
const production = new HttpResourceStore({ baseUrl: 'http://local/', fetchImpl: async () => { throw new Error('must not fetch'); } });
production.memory.set(resource.sha256, bytes);
await assert.rejects(production.read(resource), /Source files disabled/);
let attempts = 0;
const retry = new HttpResourceStore({ baseUrl: 'http://local/', allowSourceFiles: true,
  fetchImpl: async () => ++attempts === 1 ? new Response(new Uint8Array(compressed.length)) : new Response(compressed) });
await assert.rejects(retry.read(resource), /SHA-256 mismatch/);
assert.deepEqual(await retry.read(resource), bytes);
assert.equal(attempts, 2);
console.log('engine/version archive checkout reads, exact bytes, concurrent readers, retry, and production rejection passed');
