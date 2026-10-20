# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify the native site's actual Chromium journey under the Pages prefix."""
import hashlib
import gzip
import json
import tarfile
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

selenium = pytest.importorskip("selenium")
from selenium.webdriver.support.ui import Select, WebDriverWait
from browser_helpers import raw_view_download

from test_browser_viewer import ROOT, _driver, _ready, _server
from unified_archives import read_source


def test_native_publication_prefix_switch_raw_downloads_and_legacy_routes(published_viewer):
    prefix = "/frontend-crates"
    server, requests = _server(prefix=prefix, root=published_viewer)
    driver = _driver()
    base = f"http://127.0.0.1:{server.server_port}{prefix}/"
    catalog = json.loads(gzip.decompress((published_viewer / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    source_catalog = json.loads((ROOT / "conformance/viewer-catalog.json").read_bytes())
    assert catalog["transport"]["schema"] == 2
    try:
        driver.get(base + "?tab=tab-unified&keep=native-publication")
        _ready(driver, "tab-unified")
        assert not any("/indexes/raw-" in path for path in requests), "default startup must not request original lookup indexes"
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == ["dynamo"]
        driver.find_element("css selector", "[data-slot-open='0']").click()
        select = WebDriverWait(driver, 15).until(lambda page: page.find_element("css selector", "[data-slot-capture='0']"))
        choices = [option.get_attribute("value") for option in select.find_elements("tag name", "option")]
        historical = next(value for value in choices if value.startswith("dynamo@"))
        Select(select).select_by_value(historical)
        WebDriverWait(driver, 180).until(lambda page: page.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected.includes(arguments[0]) && window.__conformanceLoader.pending === 0", historical))
        _ready(driver, "tab-unified")
        restored = driver.current_url
        driver.get(restored)
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == [historical]
        assert parse_qs(urlsplit(driver.current_url).query)["keep"] == ["native-publication"]
        hidden = driver.execute_script("""
            return window.__CONFORMANCE_PAGE.tabs.find(tab => tab.id === 'tab-unified').candidates
                .find(item => item.key.startsWith('dynamo@') && !arguments[0].includes(item.key)
                    && !item.source_blocker && item.has_captured_results !== false
                    && (item.native_unified_families || item.capture_families || []).length).key;
        """, choices)
        driver.get(base + "?" + urlencode({"tab": "tab-unified", "slots_tab-unified": hidden}))
        _ready(driver, "tab-unified")
        normalized = driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected")
        assert normalized[0] in choices
        driver.refresh()
        _ready(driver, "tab-unified")
        assert driver.execute_script("return window.__conformanceController.selection(document.getElementById('tab-unified')).selected") == normalized
        # The picker keeps the latest patch in a release line; older data must
        # still load and paint through the same browser resource/model pipeline.
        driver.execute_async_script("""
            const done = arguments[arguments.length - 1];
            window.__conformanceLoader.select('tab-unified', {selected: [arguments[0]], reference: null}).then(done);
        """, hidden)
        _ready(driver, "tab-unified")
        assert driver.execute_script("return document.getElementById('tab-unified')._comparisonSelection.keys") == [hidden]
        originals = source_catalog["sources"]["unified"]
        probes = []
        for kind in ("inputs-and-golden", "history"):
            resource = next(item for item in originals if item["kind"] == kind)
            probes.append((resource["path"], read_source(ROOT, resource)))
        # Execution proofs and historical manifests now travel inside snapshots.
        for version, field in (("0.4.10.post1", b"execution_proof:"), ("0.4.9.post4", b"historical_evidence:")):
            resource = next(item for item in originals if item["family"] == "qwen3"
                            and item["implementation"] == "sglang_python_batch" and item["version"] == version)
            data = read_source(ROOT, resource)
            assert b"\n" + field in data
            probes.append((resource["path"], data))
        archive = source_catalog["sources"]["archives"][0]
        member = next(item for item in archive["members"] if item["kind"] == "file")
        with tarfile.open(ROOT / archive["path"]) as stream:
            member_bytes = stream.extractfile(member["path"]).read()
            probes.extend([(member["path"], member_bytes), ("conformance/" + member["path"], member_bytes)])
        probes.append((archive["path"], (ROOT / archive["path"]).read_bytes()))
        for path, expected in probes:
            driver.get(base + "?" + urlencode({"raw": path}))
            WebDriverWait(driver, 180).until(lambda page: page.find_elements("css selector", "#viewer pre") or page.execute_script("return !!window.__conformanceError"))
            assert not driver.execute_script("return window.__conformanceError || null")
            rendered = driver.find_element("css selector", "#viewer pre").get_property("textContent")
            if path.endswith((".yaml", ".yml", ".json", ".txt")):
                assert rendered.encode() == expected
            else:
                assert rendered == f"Binary resource ({len(expected)} bytes). Use Download to save the original bytes."
            link = driver.find_element("link text", "Download original resource")
            assert link.get_attribute("href").startswith("blob:")
            downloaded = raw_view_download(driver)
            assert downloaded['hash'] == hashlib.sha256(expected).hexdigest()
            assert downloaded['size'] == len(expected)
        driver.get(base + "conformance/CONFORMANCE_v2.html?tab=tab-unified&keep=native#case")
        WebDriverWait(driver, 20).until(lambda page: urlsplit(page.current_url).path in {prefix + "/", prefix + "/index.html"})
        assert urlsplit(driver.current_url).fragment == "case"
        assert parse_qs(urlsplit(driver.current_url).query)["keep"] == ["native"]
        _ready(driver, "tab-unified")
        assert any(path.endswith("/viewer/worker.js") for path in requests)
        assert not [path for path in requests if path.endswith((".yaml", ".yml", ".tar.gz"))]
        assert all(path.startswith(prefix + "/") or path == "/favicon.ico" for path in requests)
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_checkout_raw_route_retains_original_source_fallback():
    server, requests = _server(prefix="/frontend-crates", root=ROOT)
    driver = _driver()
    catalog = json.loads((ROOT / "conformance/viewer-catalog.json").read_bytes())
    resource = next(item for item in catalog["sources"]["unified"] if item["kind"] == "history" and item.get("archive"))
    try:
        driver.get(f"http://127.0.0.1:{server.server_port}/frontend-crates/?" + urlencode({"raw": resource["path"]}))
        WebDriverWait(driver, 180).until(lambda page: page.find_elements("css selector", "#viewer pre") or page.execute_script("return !!window.__conformanceError"))
        assert not driver.execute_script("return window.__conformanceError || null")
        assert driver.find_element("css selector", "#viewer pre").get_property("textContent").encode() == read_source(ROOT, resource)
        assert "/frontend-crates/" + resource["archive"]["path"] in requests
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()


def test_startup_url_aliases_use_the_packed_loader(published_viewer):
    server, requests = _server(prefix="/frontend-crates", root=published_viewer)
    driver = _driver()
    try:
        baseline = None
        for alias in ("external", "bootstrap", "inline"):
            requests.clear()
            driver.get(f"http://127.0.0.1:{server.server_port}/frontend-crates/?tab=tab-unified&startup={alias}")
            _ready(driver, "tab-unified")
            current = driver.execute_script("return {selection: window.__conformanceModel.candidates.map(item => item.key), rows: window.__conformanceModel.rows.map(row => row.family || row.section)}")
            if baseline is None:
                baseline = current
            assert current == baseline
            assert "/frontend-crates/conformance/viewer-resolved-catalog.json.gz" in requests
            assert not any(path.endswith((".yaml", ".yml", ".tar.gz", "viewer-startup.json")) or "/bootstrap/" in path for path in requests)
    finally:
        driver.quit()
        server.shutdown()
        server.server_close()
