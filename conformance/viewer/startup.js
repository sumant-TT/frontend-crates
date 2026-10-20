// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { HttpResourceStore, gunzipSync, sha256, resourceMetrics } from './data.js';
const decoder = new TextDecoder('utf-8', { fatal: true });
let developmentFallback = null;
export function configureDevelopmentCatalog(loader) { developmentFallback = loader; }

export const startupMetrics = { phases: {}, resources: resourceMetrics };

export async function timedStartupPhase(name, operation) {
  const start = performance.now();
  try { return await operation(); }
  finally { startupMetrics.phases[name] = (startupMetrics.phases[name] || 0) + performance.now() - start; }
}

export function validateStartupCatalog(catalog, { resolved = true } = {}) {
  if (!catalog || catalog.schema !== 1 || !Array.isArray(catalog.sources?.archives)) throw new Error('Unsupported conformance catalog');
  if (!Array.isArray(catalog.sources.unified) || catalog.sources.unified.some(resource =>
    resource.format !== 'typed-json-v1' && (resolved || resource.format !== undefined))) {
    throw new Error(resolved ? 'Browser data needs regeneration' : 'Unsupported conformance resource format');
  }
  return catalog;
}

export async function loadStartupCatalog(baseUrl, params, { fetchImpl = globalThis.fetch.bind(globalThis) } = {}) {
  const store = new HttpResourceStore({ baseUrl, fetchImpl });
  const mode = params.get('startup') || 'compressed';
  startupMetrics.mode = mode;
  const response = await fetchImpl(new URL('conformance/viewer-resolved-catalog.json.gz', baseUrl), { cache: 'no-store' });
  if (response.ok) {
    const text = decoder.decode(gunzipSync(new Uint8Array(await response.arrayBuffer())));
    const catalog = validateStartupCatalog(JSON.parse(text));
    if (catalog.transport?.schema === 2) {
      const trailer = /,"runtime_sha256":"([a-f0-9]{64})"}\n$/.exec(text);
      const content = trailer ? text.slice(0, trailer.index) + '}\n' : '';
      if (!trailer || trailer[1] !== catalog.runtime_sha256 || await sha256(new TextEncoder().encode(content)) !== trailer[1]) throw new Error('Runtime catalog integrity mismatch');
    }
    store.allowSourceFiles = catalog.transport?.schema !== 2 && Boolean(developmentFallback);
    return { catalog, store };
  }
  if (response.status !== 404 || !developmentFallback) throw new Error(`Catalog fetch failed (${response.status})`);
  store.allowSourceFiles = true;
  return { catalog: validateStartupCatalog(await developmentFallback(baseUrl, fetchImpl), { resolved: false }), store };
}
