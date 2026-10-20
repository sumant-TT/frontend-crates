# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-candidate publication caching with checked payload identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


FILES = ("native-pages.tar", "source-catalogs.tar", "conformance-publish")


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def input_identity(path):
    if path.is_symlink():
        # Agent configuration links are tracked but can target directories
        # outside the checkout. Their link text belongs to this tree.
        return ["link", hashlib.sha256(os.readlink(path).encode()).hexdigest()]
    return ["file", file_hash(path)]


def cache_key(root):
    # Hash actual LFS content and working files, never pointer text or only an
    # archive's semantic fingerprint. Reader, tests and configuration all count.
    paths = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    new_sources = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z",
                                  "conformance/utils/src", "conformance/viewer", "conformance/publisher/src", ".github"],
                                 cwd=root, capture_output=True, check=True).stdout
    paths += new_sources
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True).stdout.decode().strip()
    rust = subprocess.run(["rustc", "--version"], capture_output=True, check=True).stdout.decode().strip()
    tags = subprocess.run(["git", "show-ref", "--tags"], cwd=root, capture_output=True, check=False)
    if tags.returncode not in (0, 1):
        raise RuntimeError("cannot inventory release tags")
    inputs = {path: input_identity(root / path) for path in paths.decode().split("\0") if path}
    identity = {"schema": 1, "revision": revision, "files": inputs, "rust": rust,
                "tags": tags.stdout.decode(), "platform": [platform.system(), platform.machine()]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def seal(directory, identity):
    manifest = {"identity": identity, "files": {name: file_hash(directory / name) for name in FILES}}
    (directory / "cache-manifest.json").write_text(json.dumps(manifest, sort_keys=True))


def valid(directory, identity):
    try:
        manifest = json.loads((directory / "cache-manifest.json").read_bytes())
        return (manifest["identity"] == identity and set(manifest["files"]) == set(FILES)
                and all((directory / name).is_file() and not (directory / name).is_symlink()
                        and file_hash(directory / name) == manifest["files"][name] for name in FILES))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("key", "seal", "check"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--identity")
    args = parser.parse_args()
    if args.action == "key":
        print(cache_key(args.root))
        return 0
    if args.directory is None or not args.identity:
        parser.error("--directory and --identity are required")
    if args.action == "seal":
        seal(args.directory, args.identity)
        return 0
    return 0 if valid(args.directory, args.identity) else 1


if __name__ == "__main__":
    raise SystemExit(main())
