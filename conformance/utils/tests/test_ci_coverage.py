# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import ci_coverage
import ci_publication_cache
import unified_history
import verify_ci_selection


@pytest.mark.parametrize("path", [
    "conformance/fixtures-unified-v2/versions/dynamo-0.7.18.tar.gz",
    "conformance/viewer/model.js", "conformance/viewer/values.js",
    "conformance/viewer/snapshots.js", "conformance/viewer/development_parser.js",
    "conformance/publisher/src/yaml.rs", "conformance/utils/src/viewer_catalog.py",
    "conformance/utils/src/ci_coverage.py", "conformance/parser_families.yaml",
    "conformance/fixtures-manifest.json", "Cargo.lock", "unknown-file",
])
def test_evidence_readers_and_unclassified_changes_require_exhaustive(path):
    assert ci_coverage.classify(["index.html", path]) == "exhaustive"
    assert ci_coverage.classify([]) == "exhaustive"
    assert ci_coverage.classify(["index.html"]) == "latest-two"


@pytest.mark.parametrize("status", ["A", "D", "T", "U", "R100", "C100"])
def test_structural_changes_cannot_select_the_smaller_profile(status):
    assert ci_coverage.classify_changes(f"{status}\0index.html\0") == "exhaustive"
    assert ci_coverage.classify_changes("R100\0conformance/viewer/model.js\0index.html\0") == "exhaustive"
    assert ci_coverage.classify_changes("M\0index.html\0") == "latest-two"


@pytest.mark.parametrize("profile", ["exhaustive", "latest-two"])
@pytest.mark.parametrize("shard_count", [4, verify_ci_selection.QUALIFICATION_SHARDS])
def test_capture_shards_have_exact_disjoint_union_and_newest_versions(profile, shard_count):
    rows = [{"path": f"{implementation}/{mode}/{family}/{version}", "implementation": implementation,
             "mode": mode, "family": family, "version": version}
            for implementation in ("dynamo_v2", "sglang") for mode in ("unified", "batch")
            for family in ("a", "b") for version in ("0.7.9", "0.7.10", "0.7.18")]
    whole = ci_coverage.select_resources(rows, profile=profile, shard=0, shards=1)
    shards = [ci_coverage.select_resources(rows, profile=profile, shard=index, shards=shard_count) for index in range(shard_count)]
    paths = [row["path"] for shard in shards for row in shard]
    assert len(paths) == len(set(paths))
    assert sorted(paths) == sorted(row["path"] for row in whole)
    assert len(whole) == (24 if profile == "exhaustive" else 16)
    if profile == "latest-two":
        assert {row["version"] for row in whole} == {"0.7.10", "0.7.18"}


def test_authored_reader_does_not_open_historical_archives(tmp_path, monkeypatch):
    family_path = tmp_path / "families/sample/inputs_and_golden.yaml"
    family_path.parent.mkdir(parents=True)
    family_path.write_text(yaml.safe_dump({"input_document": {"family": "sample"},
                                         "golden_document": {"family": "sample"}, "cases": {}}))
    monkeypatch.setattr(unified_history, "_load_store_unlocked", lambda *_: pytest.fail("full store loaded"))
    versions = tmp_path / "versions"
    versions.mkdir()
    (versions / "dynamo-0.7.18.tar.gz").write_bytes(b"must never be opened")
    store = unified_history.load_authored_store(tmp_path)
    assert set(store.families) == {"sample"}
    assert not store.histories


@pytest.mark.parametrize("linked", ["families", "family", "document"])
def test_authored_reader_rejects_links_in_selected_inputs(tmp_path, linked):
    root = tmp_path / "store"
    family = root / "families/sample"
    family.mkdir(parents=True)
    source = family / "inputs_and_golden.yaml"
    source.write_text("input_document: {family: sample}\ngolden_document: {family: sample}\ncases: {}\n")
    target = {"families": root / "families", "family": family, "document": source}[linked]
    outside = tmp_path / "outside"
    target.rename(outside)
    target.symlink_to(outside, target_is_directory=linked != "document")
    with pytest.raises(ValueError):
        unified_history.load_authored_store(root)


def test_cache_miss_hit_corruption_and_identity_invalidation(tmp_path):
    assert not ci_publication_cache.valid(tmp_path, "source-reader-policy")
    for name in ci_publication_cache.FILES:
        (tmp_path / name).write_bytes(name.encode())
    ci_publication_cache.seal(tmp_path, "source-reader-policy")
    assert ci_publication_cache.valid(tmp_path, "source-reader-policy")
    for identity in ("changed-source", "changed-reader", "changed-authored", "changed-policy"):
        assert not ci_publication_cache.valid(tmp_path, identity)
    (tmp_path / ci_publication_cache.FILES[0]).write_bytes(b"corrupt")
    assert not ci_publication_cache.valid(tmp_path, "source-reader-policy")


def test_cache_key_changes_for_source_reader_authored_and_configuration(tmp_path, monkeypatch):
    paths = ["capture.yaml", "reader.rs", "authored.yaml", "policy.yaml"]
    for path in paths:
        (tmp_path / path).write_bytes(b"original")

    def run(arguments, **kwargs):
        if arguments[:2] == ["git", "ls-files"]:
            output = b"" if "--others" in arguments else "\0".join(paths).encode() + b"\0"
        elif arguments[:2] == ["git", "rev-parse"]:
            output = b"a" * 40
        elif arguments[:2] == ["git", "show-ref"]:
            output = b"tag-identity"
        else:
            assert arguments == ["rustc", "--version"]
            output = b"rustc identity"
        return SimpleNamespace(stdout=output, returncode=0)

    monkeypatch.setattr(ci_publication_cache.subprocess, "run", run)
    original = ci_publication_cache.cache_key(tmp_path)
    for path in paths:
        (tmp_path / path).write_bytes(b"changed")
        assert ci_publication_cache.cache_key(tmp_path) != original
        (tmp_path / path).write_bytes(b"original")
        assert ci_publication_cache.cache_key(tmp_path) == original


def test_cache_identity_does_not_follow_tracked_directory_links(tmp_path):
    outside = tmp_path / "external"
    outside.mkdir()
    link = tmp_path / "skills"
    link.symlink_to(outside, target_is_directory=True)
    before = ci_publication_cache.input_identity(link)
    (outside / "unrelated.yaml").write_bytes(b"not a publication input")
    assert ci_publication_cache.input_identity(link) == before
    link.unlink()
    link.symlink_to(tmp_path / "different", target_is_directory=True)
    assert ci_publication_cache.input_identity(link) != before


def test_workflow_publishes_before_qualification_and_requires_every_lane():
    root = Path(__file__).resolve().parents[3]
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
    assert jobs["conformance-fixtures"]["needs"] == "changes"
    assert jobs["rust"].get("needs") is None
    required = set(jobs["conformance-table"]["needs"])
    for name, job in jobs.items():
        if name.startswith("conformance-") and name != "conformance-table":
            assert name in required
            assert job["timeout-minutes"] == 20
    publication = json.dumps(jobs["pages-build"])
    assert "snapshot_parity" not in publication
    assert "pytest" not in "\n".join(step.get("run", "") for step in jobs["pages-build"]["steps"] if "pip" not in step.get("run", ""))
    qualification = jobs["conformance-qualification"]
    assert qualification["strategy"]["matrix"]["shard"] == "${{ fromJSON(needs.changes.outputs.qualification_matrix) }}"
    assert qualification["env"]["CONFORMANCE_PARITY_SHARDS"] == "${{ strategy.job-total }}"
    assert jobs["changes"]["outputs"]["qualification_matrix"] == "${{ steps.qualification.outputs.matrix }}"
    step = next(step for step in jobs["changes"]["steps"] if step.get("id") == "qualification")
    assert "verify_ci_selection.py --shard-matrix" in step["run"]
    result = subprocess.run([sys.executable, str(root / "conformance/utils/src/verify_ci_selection.py"), "--shard-matrix"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == list(range(verify_ci_selection.QUALIFICATION_SHARDS))
    assert verify_ci_selection.QUALIFICATION_SHARDS == 8


@pytest.mark.parametrize("arguments", [[], ["--baseline", "."], ["--shard-matrix", "--selections", "."]])
def test_ci_accounting_cli_rejects_incomplete_or_mixed_modes(arguments):
    result = subprocess.run([sys.executable, verify_ci_selection.__file__, *arguments],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "missing-shard", "extra"])
def test_ci_accounting_rejects_lost_tests_and_incomplete_qualification(fault):
    qualification = sorted(verify_ci_selection.QUALIFICATION)
    baseline = [{"selected": ["fixture", "browser", *qualification]}]
    assignments = [{"selected": ["fixture"]}, {"selected": ["browser"]}]
    assignments.extend({"selected": qualification} for _ in range(verify_ci_selection.QUALIFICATION_SHARDS))
    if fault == "missing":
        assignments.pop(0)
    elif fault == "duplicate":
        assignments.append({"selected": ["browser"]})
    elif fault == "missing-shard":
        assignments.pop()
    elif fault == "extra":
        assignments.append({"selected": ["uncollected"]})
    if fault:
        with pytest.raises(ValueError, match="CI selection mismatch"):
            verify_ci_selection.verify(baseline, assignments)
    else:
        assert verify_ci_selection.verify(baseline, assignments)["unique_tests"] == 2 + len(qualification)
