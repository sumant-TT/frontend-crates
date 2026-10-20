// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated source generation, web-only assembly and transactional installation.
use crate::{
    codec::{json_bytes, sha256},
    pipeline,
    resources::{MemoryBudget, Policy},
    source::{self, STORE},
    transaction::{self, Lease},
};
use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    time::Instant,
};

pub struct Options {
    pub root: PathBuf,
    pub output: PathBuf,
    pub revision: String,
    pub candidate_tree: Option<String>,
    pub data_only: bool,
    pub workers: String,
}

fn absolute(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                ensure!(normalized.pop(), "path escapes root");
            }
            Component::CurDir => (),
            other => normalized.push(other.as_os_str()),
        }
    }
    source::no_links(&path)?;
    source::no_links(&normalized)?;
    Ok(normalized)
}

fn capture_inventory(root: &Path, catalog: &Value) -> Result<BTreeSet<String>> {
    let inventory = source::inventory(&root.join(STORE))?;
    let expected = catalog["sources"]["unified"]
        .as_array()
        .context("missing unified inventory")?
        .iter()
        .map(|r| {
            let name = r.get("archive").unwrap_or(r)["path"]
                .as_str()
                .context("missing resource path")?;
            Ok(name
                .strip_prefix(&format!("{STORE}/"))
                .context("unified resource outside store")?
                .to_string())
        })
        .collect::<Result<BTreeSet<_>>>()?;
    ensure!(
        catalog["sources"]["unified"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["path"].as_str())
            .collect::<BTreeSet<_>>()
            .len()
            == catalog["sources"]["unified"].as_array().unwrap().len(),
        "duplicate inventory resource"
    );
    for name in &inventory {
        let path = root.join(STORE).join(name);
        let parts = name.split('/').collect::<Vec<_>>();
        if path.is_dir() {
            ensure!(
                matches!(
                    parts.as_slice(),
                    ["families"] | ["executions"] | ["receipts"] | ["versions"] | ["families", _]
                ),
                "unknown store directory: {name}"
            );
            if parts.len() == 2 {
                ensure!(
                    path.join("inputs_and_golden.yaml").is_file(),
                    "family has no authored input: {name}"
                );
            }
        } else {
            ensure!(
                matches!(
                    parts.as_slice(),
                    ["families", _, _] | ["executions", _] | ["receipts", _]
                ) && name.ends_with(".yaml")
                    || matches!(parts.as_slice(), ["versions", _]) && name.ends_with(".tar.gz"),
                "unknown store file: {name}"
            );
        }
    }
    let files = inventory
        .iter()
        .filter(|name| root.join(STORE).join(name).is_file())
        .cloned()
        .collect::<BTreeSet<_>>();
    ensure!(files == expected, "source inventory differs from catalog");
    let mut logical = BTreeSet::new();
    let mut checked = BTreeMap::new();
    for resource in catalog["sources"]["unified"].as_array().unwrap() {
        if let Some(archive) = resource.get("archive") {
            let name = archive["path"]
                .as_str()
                .context("missing source archive path")?;
            checked.entry(name).or_insert(resource);
        } else {
            ensure!(
                logical.insert(resource["path"].as_str().unwrap().to_string()),
                "duplicate loose source"
            );
        }
    }
    let admitted = checked
        .values()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|resource| source::archived_inventory(root, resource))
        .collect::<Result<Vec<_>>>()?;
    for members in admitted {
        for member in members {
            ensure!(
                logical.insert(format!("{STORE}/{member}")),
                "duplicate archived source"
            );
        }
    }
    let declared = catalog["sources"]["unified"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["path"].as_str().unwrap().to_string())
        .collect::<BTreeSet<_>>();
    ensure!(
        logical == declared,
        "archived source inventory differs from catalog"
    );
    Ok(inventory)
}

fn manifest_identity(root: &Path, catalog: &Value) -> Result<Vec<u8>> {
    let bytes = source::read(root, "conformance/fixtures-manifest.json")?;
    let manifest: Value = serde_json::from_slice(&bytes)?;
    let shards = manifest["shards"]
        .as_array()
        .context("missing fixture manifest shards")?;
    let mut archives = BTreeMap::new();
    for shard in shards {
        let name = shard["path"].as_str().context("missing shard path")?;
        source::relative(name)?;
        if shard["format"] == "unified-history" {
            ensure!(
                &catalog["sources"]["unified_history"] == shard,
                "history manifest differs from catalog"
            );
        } else {
            archives.insert(format!("conformance/fixtures/{name}"), shard);
        }
    }
    ensure!(
        archives.len()
            == catalog["sources"]["archives"]
                .as_array()
                .context("missing archive sources")?
                .len(),
        "manifest archive membership differs from catalog"
    );
    for resource in catalog["sources"]["archives"].as_array().unwrap() {
        let shard = archives
            .get(resource["path"].as_str().context("missing archive path")?)
            .context("archive missing from manifest")?;
        ensure!(
            shard["sha256"] == resource["sha256"] && shard["size"] == resource["size"],
            "archive manifest identity differs from catalog"
        );
    }
    if let Some(shard) = catalog["sources"]
        .get("unified_history")
        .filter(|v| !v.is_null())
    {
        let mut rows = catalog["sources"]["unified"].as_array().context("missing Unified sources")?.iter().map(|r| Ok(json!({"path":r["path"].as_str().context("missing source path")?.strip_prefix(&format!("{STORE}/")).context("source outside store")?,"sha256":r["sha256"],"size":r["size"]}))).collect::<Result<Vec<_>>>()?;
        rows.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
        ensure!(
            shard["sha256"] == sha256(&catalog_identity(&json!(rows))?)
                && shard["size"].as_u64()
                    == Some(rows.iter().map(|r| r["size"].as_u64().unwrap()).sum()),
            "history inventory digest differs from manifest"
        );
    }
    Ok(bytes)
}

fn metadata_sources(root: &Path, catalog: &Value) -> Result<BTreeMap<String, String>> {
    let fixed = [
        (
            "parser_families",
            "conformance/utils/src/parser_families.yaml",
        ),
        ("legacy_taxonomy", "conformance/case-taxonomy.yaml"),
        (
            "unified_taxonomy",
            "conformance/utils/src/unified_taxonomy.py",
        ),
        (
            "viewer_metadata",
            "conformance/utils/src/viewer_metadata.py",
        ),
        (
            "reasoning_metadata",
            "conformance/utils/src/reasoning_metadata.py",
        ),
        ("implementations", "conformance/utils/src/impls.py"),
        ("peer_versions", "conformance/utils/src/pyproject.stub.toml"),
        ("parser_config", "parsers/v1/src/tool_calling/config.rs"),
        (
            "parser_metadata",
            "conformance/utils/src/parser_metadata.py",
        ),
        (
            "divergence_notes",
            "conformance/toolcalling/known-divergences.yaml",
        ),
        (
            "unified_authoring",
            "conformance/utils/src/gen_unified_golden.py",
        ),
        (
            "unified_v1_authoring",
            "conformance/utils/src/gen_unified_v1_cases.py",
        ),
        ("unified_tools", "conformance/utils/src/unified_tools.json"),
        (
            "parser_registrations",
            "parsers/v1/src/tool_calling/parsers.rs",
        ),
    ];
    let hashes = &catalog["metadata"]["source_hashes"];
    let mut sources = BTreeMap::new();
    for (key, path) in fixed {
        if let Some(digest) = hashes.get(key) {
            sources.insert(
                path.to_string(),
                digest
                    .as_str()
                    .context("invalid metadata hash")?
                    .to_string(),
            );
        }
    }
    if let Some(docs) = hashes["documentation"].as_object() {
        for (path, digest) in docs {
            sources.insert(
                path.clone(),
                digest
                    .as_str()
                    .context("invalid document hash")?
                    .to_string(),
            );
        }
    }
    for (key, path) in [
        ("projection", "conformance/utils/src/viewer_catalog.py"),
        ("predicate", "conformance/viewer/comparison.js"),
        ("indexer", "conformance/viewer/unavailable_controls.js"),
    ] {
        if let Some(digest) = hashes["availability_controls"].get(key) {
            sources.insert(
                path.to_string(),
                digest.as_str().context("invalid control hash")?.to_string(),
            );
        }
    }
    for (path, digest) in &sources {
        ensure!(
            sha256(&source::read(root, path)?) == *digest,
            "metadata source identity mismatch: {path}"
        );
    }
    Ok(sources)
}

fn asset_graph(root: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let html = regex::Regex::new(r#"(?:src|href)=["']([^"']+)["']"#)?;
    let js = regex::Regex::new(r#"(?:from\s+|import\s*(?:\(\s*)?|new URL\(\s*)["'](\.[^"']+)"#)?;
    let css = regex::Regex::new(r#"url\(["']?([^"')]+)"#)?;
    let mut pending = vec![
        "index.html".to_string(),
        "conformance/CONFORMANCE_v2.html".to_string(),
    ];
    let mut files = BTreeMap::new();
    while let Some(name) = pending.pop() {
        if files.contains_key(&name) {
            continue;
        }
        ensure!(
            ![".yaml", ".yml", ".tar.gz", ".md", ".txt", ".test.js"]
                .iter()
                .any(|suffix| name.ends_with(suffix)),
            "forbidden web artifact: {name}"
        );
        let mut bytes = source::read(root, &name)?;
        if name == "index.html" {
            bytes = String::from_utf8(bytes)?
                .replace(
                    "conformance/viewer/development.js",
                    "conformance/viewer/app.js",
                )
                .into_bytes();
        }
        if [".html", ".js", ".css"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
        {
            let text = std::str::from_utf8(&bytes)?;
            let pattern = if name.ends_with(".html") {
                &html
            } else if name.ends_with(".js") {
                &js
            } else {
                &css
            };
            for capture in pattern.captures_iter(text) {
                let link = capture[1].split(['?', '#']).next().unwrap();
                if link.is_empty() || link.contains(':') || link.starts_with("//") {
                    continue;
                }
                ensure!(
                    !link.starts_with('/') && !link.contains('%'),
                    "nonrelative or encoded runtime asset: {link}"
                );
                let parent = Path::new(&name).parent().context("asset has no parent")?;
                let mut normalized = PathBuf::new();
                for component in parent.join(link).components() {
                    match component {
                        Component::CurDir => (),
                        Component::ParentDir => {
                            ensure!(normalized.pop(), "runtime asset escapes site");
                        }
                        Component::Normal(c) => normalized.push(c),
                        _ => anyhow::bail!("runtime asset escapes site"),
                    }
                }
                let child = normalized.to_str().context("nonUTF8 asset")?.to_string();
                if root.join(&child).is_dir() && name.ends_with(".html") {
                    continue;
                }
                if root.join(&child).is_file() {
                    pending.push(child);
                } else {
                    anyhow::bail!("referenced runtime asset missing: {child}");
                }
            }
        }
        files.insert(name, bytes);
    }
    Ok(files)
}

fn bundle_assets(
    root: &Path,
    assets: &BTreeMap<String, Vec<u8>>,
) -> Result<BTreeMap<String, Vec<u8>>> {
    if !assets.contains_key("conformance/viewer/app.js") {
        return Ok(assets.clone());
    }
    let output = Command::new("node")
        .arg(root.join("conformance/viewer/build.mjs"))
        .current_dir(root)
        .output()
        .context("launch locked production JavaScript bundler")?;
    ensure!(
        output.status.success(),
        "production bundler failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout)?;
    ensure!(
        result["schema"] == 1,
        "unsupported production bundle schema"
    );
    for input in result["inputs"]
        .as_array()
        .context("missing bundle dependency graph")?
    {
        let path = input.as_str().context("invalid bundle dependency")?;
        ensure!(
            assets.contains_key(path),
            "bundle consumed an undeclared development dependency: {path}"
        );
    }
    let bundles = result["files"]
        .as_object()
        .context("missing production bundles")?;
    ensure!(bundles.len() == 2, "unexpected production bundles");
    let mut published = assets
        .iter()
        .filter(|(path, _)| !path.ends_with(".js"))
        .map(|(path, bytes)| (path.clone(), bytes.clone()))
        .collect::<BTreeMap<_, _>>();
    for path in ["conformance/viewer/app.js", "conformance/viewer/worker.js"] {
        let bytes = STANDARD.decode(
            bundles
                .get(path)
                .and_then(Value::as_str)
                .context("missing production bundle")?,
        )?;
        ensure!(!bytes.is_empty(), "empty production bundle");
        published.insert(path.to_string(), bytes);
    }
    Ok(published)
}

fn catalog_identity(value: &Value) -> Result<Vec<u8>> {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), sorted(v)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    let text = serde_json::to_string(&sorted(value))?;
    let mut escaped = String::new();
    for ch in text.chars() {
        if ch.is_ascii() {
            escaped.push(ch);
        } else {
            for unit in ch.encode_utf16(&mut [0; 2]) {
                escaped.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    Ok(escaped.into_bytes())
}

fn validate_catalog(catalog: &Value) -> Result<()> {
    ensure!(catalog["schema"] == 1, "unsupported catalog schema");
    let mut identity = catalog.clone();
    let revision = identity
        .as_object_mut()
        .context("invalid catalog")?
        .remove("revision")
        .context("missing catalog revision")?;
    ensure!(
        revision == sha256(&catalog_identity(&identity)?),
        "catalog revision mismatch"
    );
    let mut paths = BTreeSet::new();
    for kind in ["unified", "archives"] {
        for resource in catalog["sources"][kind]
            .as_array()
            .context("missing sources")?
        {
            let path = resource["path"].as_str().context("missing source path")?;
            source::relative(path)?;
            ensure!(paths.insert(path), "duplicate source path");
            if kind == "archives" {
                ensure!(
                    path.starts_with("conformance/fixtures/")
                        && path.ends_with(".tar.gz")
                        && resource["members"].is_array(),
                    "invalid archive source descriptor"
                );
            }

            ensure!(
                resource["sha256"]
                    .as_str()
                    .is_some_and(|v| source::hex(v, 64))
                    && resource["size"]
                        .as_u64()
                        .is_some_and(|n| n < source::MAX_ARTIFACT),
                "invalid source descriptor"
            );
        }
    }
    for resource in catalog["sources"]["unified"].as_array().unwrap() {
        let name = resource["path"]
            .as_str()
            .unwrap()
            .strip_prefix(&format!("{STORE}/"))
            .context("Unified source outside store")?;
        let parts = name.split('/').collect::<Vec<_>>();
        match parts.as_slice() {
            ["executions", file] | ["receipts", file] => {
                let kind = if parts[0] == "executions" {
                    "execution"
                } else {
                    "receipt"
                };
                ensure!(
                    resource["kind"] == kind
                        && resource["family"].is_null()
                        && resource["implementation"].is_null()
                        && resource["version"].is_null()
                        && file.ends_with(".yaml"),
                    "shared source identity mismatch"
                );
            }
            ["families", family, file] => {
                ensure!(
                    resource["family"] == *family,
                    "source family identity mismatch"
                );
                if *file == "inputs_and_golden.yaml" {
                    ensure!(
                        resource["kind"] == "inputs-and-golden"
                            && resource["implementation"].is_null()
                            && resource["version"].is_null(),
                        "authored source identity mismatch"
                    );
                } else {
                    let stem = file
                        .strip_suffix(".yaml")
                        .context("unknown family source extension")?;
                    let (implementation, version) = stem
                        .split_once('-')
                        .context("invalid capture source identity")?;
                    crate::history::Release::parse(version)?;
                    ensure!(
                        implementation
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                            && resource["kind"] == "history"
                            && resource["implementation"] == implementation
                            && resource["version"] == version,
                        "capture source identity mismatch"
                    );
                }
            }
            _ => anyhow::bail!("invalid Unified source topology"),
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PhaseClock {
    wall: Instant,
    cpu_seconds: f64,
}

fn usage_cpu_seconds(usage: &libc::rusage) -> f64 {
    usage.ru_utime.tv_sec as f64
        + usage.ru_utime.tv_usec as f64 / 1e6
        + usage.ru_stime.tv_sec as f64
        + usage.ru_stime.tv_usec as f64 / 1e6
}

impl PhaseClock {
    fn now() -> Result<Self> {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        ensure!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
            "cannot measure publication CPU: {}",
            std::io::Error::last_os_error()
        );
        Ok(Self {
            wall: Instant::now(),
            cpu_seconds: usage_cpu_seconds(&unsafe { usage.assume_init() }),
        })
    }
}

fn statistics(start: PhaseClock, phase: &str, extra: Value) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    let mut result = json!({"phase":phase,"wall_seconds":start.wall.elapsed().as_secs_f64(),"timing_scope":"inclusive","details":extra});
    if status == 0 {
        let usage = unsafe { usage.assume_init() };
        let cpu_seconds = usage_cpu_seconds(&usage);
        result["process_cpu_seconds"] = json!(cpu_seconds);
        result["phase_cpu_seconds"] = json!(cpu_seconds - start.cpu_seconds);
        result["peak_process_memory_bytes"] = json!(usage.ru_maxrss as u64 * 1024);
    } else {
        result["statistics_error"] = json!(std::io::Error::last_os_error().to_string());
    }
    eprintln!("{result}");
}

fn publication_time() -> Result<String> {
    let now = unsafe { libc::time(std::ptr::null_mut()) };
    let mut local = std::mem::MaybeUninit::<libc::tm>::uninit();
    ensure!(
        !unsafe { libc::localtime_r(&now, local.as_mut_ptr()) }.is_null(),
        "cannot format publication time"
    );
    let mut buffer = [0i8; 64];
    let format = c"%Y-%m-%d %H:%M %Z";
    let length = unsafe {
        libc::strftime(
            buffer.as_mut_ptr(),
            buffer.len(),
            format.as_ptr(),
            local.as_ptr(),
        )
    };
    ensure!(length > 0, "cannot format publication time");
    Ok(unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr()) }
        .to_str()?
        .to_string())
}

fn command(options: &Options, root: &Path, output: &Path) -> Result<String> {
    let mut args = vec![
        "conformance-publish".to_string(),
        "--root".into(),
        root.to_str().context("non UTF8 root")?.into(),
        "--output".into(),
        output.to_str().context("non UTF8 output")?.into(),
        "--revision".into(),
        options.revision.clone(),
    ];
    if let Some(tree) = &options.candidate_tree {
        args.extend(["--candidate-tree".into(), tree.clone()]);
    }
    if options.data_only {
        args.push("--data-only".into());
    }
    if options.workers != "auto" {
        args.extend(["--workers".into(), options.workers.clone()]);
    }
    Ok(args
        .iter()
        .map(|arg| {
            if arg
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&b))
            {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" "))
}

/// Validate permanent evidence without preparing or installing report resources.
pub fn validate(root: &Path, workers: &str) -> Result<Value> {
    let root = absolute(root)?;
    let _session = source::archive_session();
    let policy = Policy::detect(workers)?;
    let pool = policy.pool()?;
    let catalog: Value =
        serde_json::from_slice(&source::read(&root, "conformance/viewer-catalog.json")?)?;
    validate_catalog(&catalog)?;
    manifest_identity(&root, &catalog)?;
    let inventory = pool.install(|| capture_inventory(&root, &catalog))?;
    let reserved = source::source_size(&root, "conformance/viewer-catalog.json")?
        .checked_mul(32)
        .context("catalog reservation overflow")?;
    let budget = MemoryBudget::new(
        policy
            .memory_bytes
            .checked_sub(reserved)
            .context("catalog exceeds validation memory budget")?,
    );
    let stage = tempfile::tempdir()?;
    pool.install(|| {
        pipeline::snapshots(
            &root,
            stage.path(),
            &catalog,
            &budget,
            pipeline::SnapshotEncoding::ValidateOnly,
        )
    })?;
    ensure!(
        pool.install(|| capture_inventory(&root, &catalog))? == inventory,
        "source changed during validation"
    );
    for resource in catalog["sources"]["unified"].as_array().unwrap() {
        source::verified(&root, resource)?;
    }
    Ok(
        json!({"validated":true,"source_revision":catalog["revision"],"resources":catalog["sources"]["unified"].as_array().unwrap().len()}),
    )
}

pub fn publish(options: Options) -> Result<()> {
    let _archives = source::archive_session();
    let started = PhaseClock::now()?;
    ensure!(
        source::hex(&options.revision, 40),
        "revision must be a full immutable commit SHA"
    );
    if let Some(tree) = &options.candidate_tree {
        ensure!(
            source::hex(tree, 40),
            "candidate tree must be a full Git tree SHA"
        );
    }
    let root = absolute(&options.root)?.canonicalize()?;
    let output = absolute(&options.output)?;
    ensure!(
        !root.starts_with(&output) || (options.data_only && output == root),
        "output overlaps source checkout"
    );
    ensure!(
        !options.data_only || output == root,
        "data-only destination must be source checkout"
    );
    let protected = [
        "conformance/fixtures-manifest.json",
        "conformance/viewer",
        "conformance/utils",
        "conformance/fixtures",
        "conformance/fixtures-unified-v2",
        "index.html",
        "conformance/CONFORMANCE_v2.html",
        "conformance/viewer-catalog.json",
        "conformance/viewer-unified-catalog.json",
    ];
    if !options.data_only {
        for name in protected {
            let path = root.join(name);
            ensure!(
                !output.starts_with(&path) && !path.starts_with(&output),
                "output overlaps maintained source"
            );
        }
    }
    fs::create_dir_all(output.parent().context("missing output parent")?)?;
    let transaction_parent = if options.data_only {
        root.clone()
    } else {
        output.parent().unwrap().to_path_buf()
    };
    let source_parent = root.join("conformance");
    let _source_lease = if options.data_only || transaction_parent == source_parent {
        Lease::acquire(&source_parent)?
    } else {
        Lease::read(&source_parent)?
    };
    let _output_lease = if transaction_parent != root.join("conformance") {
        Some(Lease::acquire(&transaction_parent)?)
    } else {
        None
    };
    if output.exists() && !options.data_only {
        ensure!(
            output.is_dir(),
            "existing site destination must be a directory"
        );
        ensure!(
            !output.starts_with(&root)
                || output.join("conformance/viewer-publication.json").is_file(),
            "existing destination overlaps source checkout"
        );
        transaction::sync_tree(&output)?;
    }
    let policy = Policy::detect(&options.workers)?;
    let pool = policy.pool()?;
    let control_bytes = source::source_size(&root, "conformance/viewer-catalog.json")?
        .checked_mul(32)
        .and_then(|n| n.checked_add(256 * 1024 * 1024))
        .context("catalog memory reservation overflow")?;
    let control_budget = MemoryBudget::new(control_bytes);
    let _control = control_budget.reserve(control_bytes)?;
    let job_memory = policy
        .memory_bytes
        .checked_sub(control_bytes)
        .filter(|n| *n > 0)
        .context("catalog and assets exceed publication memory budget")?;
    let budget = MemoryBudget::new(job_memory);
    let catalog_bytes = source::read(&root, "conformance/viewer-catalog.json")?;
    let catalog: Value = serde_json::from_slice(&catalog_bytes)?;
    validate_catalog(&catalog)?;
    let manifest = manifest_identity(&root, &catalog)?;
    let projection_path = root.join("conformance/viewer-unified-catalog.json");
    let original_projection = if projection_path.exists() {
        let bytes = source::read(&root, "conformance/viewer-unified-catalog.json")?;
        ensure!(
            serde_json::from_slice::<Value>(&bytes)? == pipeline::projection(&catalog)?,
            "source projection mismatch"
        );
        Some(bytes)
    } else {
        None
    };
    let metadata = metadata_sources(&root, &catalog)?;
    let inventory = pool.install(|| capture_inventory(&root, &catalog))?;
    let assets = asset_graph(&root)?;
    let build_inputs = if !options.data_only && assets.contains_key("conformance/viewer/app.js") {
        [
            "conformance/viewer/build.mjs",
            "conformance/viewer/package.json",
            "conformance/viewer/package-lock.json",
            "conformance/viewer/vendor/fflate/LICENSE",
        ]
        .into_iter()
        .map(|path| Ok((path, source::read(&root, path)?)))
        .collect::<Result<BTreeMap<_, _>>>()?
    } else {
        BTreeMap::new()
    };
    let web_assets = if options.data_only {
        assets.clone()
    } else {
        bundle_assets(&root, &assets)?
    };
    for path in metadata.keys().chain(assets.keys()) {
        let path = root.join(path);
        ensure!(
            options.data_only || (!path.starts_with(&output) && !output.starts_with(&path)),
            "output overlaps source input"
        );
    }
    statistics(
        started,
        "admission",
        json!({"detected_cpus":policy.detected_cpus,"workers":policy.workers,"memory_budget":policy.memory_bytes,"control_reserved_bytes":control_bytes,"job_memory_budget":job_memory,"allocator_arenas":policy.allocator_arenas}),
    );
    let stage = transaction::unique(&transaction_parent, ".native-publication-stage-")?;
    let result = (|| -> Result<()> {
        let phase = PhaseClock::now()?;
        let resolved = pool.install(|| {
            pipeline::snapshots(
                &root,
                &stage,
                &catalog,
                &budget,
                if options.data_only {
                    pipeline::SnapshotEncoding::CheckoutGzip
                } else {
                    pipeline::SnapshotEncoding::CanonicalSpool
                },
            )
        })?;
        statistics(
            phase,
            "resolve-encode",
            json!({"resolved_resources":resolved["sources"]["unified"].as_array().map(Vec::len)}),
        );
        let phase = PhaseClock::now()?;
        if options.data_only {
            let projection = pipeline::projection(&resolved)?;
            pipeline::write(
                &stage,
                "conformance/viewer-resolved-catalog.json",
                &json_bytes(&projection)?,
            )?;
            pipeline::write(
                &stage,
                "conformance/viewer-resolved-catalog.json.gz",
                &crate::codec::gzip(&json_bytes(&projection)?)?,
            )?;
        } else {
            pool.install(|| {
                crate::web_transport::assemble(&root, &stage, &catalog, &resolved, &budget)
            })?;
        }
        let mut publication = json!({"schema":1,"revision":options.revision,"catalog_revision":if options.data_only {&catalog["revision"]} else {&resolved["revision"]},"generated_on":publication_time()?,"command":command(&options,&root,&output)?,"output":output.to_str(),"resolved_catalog_revision":resolved["revision"]});
        if let Some(tree) = &options.candidate_tree {
            publication["candidate_tree"] = json!(tree);
        }
        if options.data_only {
            publication["source_checkout"] = json!(true);
        }
        pipeline::write(
            &stage,
            "conformance/viewer-publication.json",
            &json_bytes(&publication)?,
        )?;
        statistics(phase, "resolved-assembly", json!({}));
        if !options.data_only {
            let phase = PhaseClock::now()?;
            for (path, bytes) in &web_assets {
                pipeline::write(&stage, path, bytes)?;
            }
            let index = String::from_utf8(web_assets["index.html"].clone())?;
            ensure!(
                index.contains("</head>") && !index.contains("id=\"conformance-publication\""),
                "unexpected publication index"
            );
            let inline = serde_json::to_string(&publication)?.replace('<', "\\u003c");
            pipeline::write(&stage,"index.html",index.replacen("</head>",&format!("<script type=\"application/json\" id=\"conformance-publication\">{inline}</script></head>"),1).as_bytes())?;
            statistics(
                phase,
                "originals-assembly",
                json!({"runtime_assets":web_assets.len()}),
            );
        }
        if !options.data_only {
            // Recheck the compiled graph: source membership cannot prove that
            // bundling left no dangling imports or unused published modules.
            let published_graph = asset_graph(&stage)?;
            ensure!(
                published_graph.keys().eq(web_assets.keys()),
                "compiled runtime dependency graph differs from declared assets"
            );
            let inventory = crate::web_transport::inventory(&stage, web_assets.keys().cloned())?;
            let audit_path = output.with_extension("inventory.json");
            fs::write(audit_path, json_bytes(&inventory)?)?;
        }
        let phase = PhaseClock::now()?;
        let paths = source::inventory(&stage)?;
        let mut bytes = 0u64;
        let mut count = 0;
        for path in paths {
            let path = stage.join(path);
            if path.is_file() {
                let size = fs::metadata(&path)?.len();
                ensure!(size < source::MAX_ARTIFACT, "artifact is at least 1 GB");
                bytes += size;
                count += 1;
            }
        }
        statistics(
            phase,
            "artifact-inventory",
            json!({"files":count,"output_bytes":bytes}),
        );
        let pairs = if options.data_only {
            let mut pairs = Vec::new();
            for name in [
                "conformance/viewer-data",
                "conformance/viewer-resolved-catalog.json",
                "conformance/viewer-resolved-catalog.json.gz",
                "conformance/viewer-publication.json",
            ] {
                if stage.join(name).exists() {
                    source::no_links(&output.join(name))?;
                    pairs.push((stage.join(name), output.join(name)));
                }
            }
            pairs
        } else {
            vec![(stage.clone(), output.clone())]
        };
        let install_started = PhaseClock::now()?;
        transaction::publish_checked(&transaction_parent, &pairs, || {
            let phase = PhaseClock::now()?;
            ensure!(
                source::read(&root, "conformance/fixtures-manifest.json")? == manifest,
                "fixture manifest changed during publication"
            );
            ensure!(
                source::read(&root, "conformance/viewer-catalog.json")? == catalog_bytes,
                "catalog changed during publication"
            );
            if let Some(bytes) = &original_projection {
                ensure!(
                    source::read(&root, "conformance/viewer-unified-catalog.json")? == *bytes,
                    "projection changed during publication"
                );
            }
            ensure!(
                pool.install(|| capture_inventory(&root, &catalog))? == inventory,
                "source layout changed during publication"
            );
            pool.install(|| {
                catalog["sources"]["unified"]
                    .as_array()
                    .unwrap()
                    .par_iter()
                    .chain(
                        catalog["sources"]["archives"]
                            .as_array()
                            .unwrap()
                            .par_iter(),
                    )
                    .try_for_each(|r| -> Result<()> {
                        let _reservation = budget.reserve(
                            r["size"]
                                .as_u64()
                                .context("invalid resource size")?
                                .checked_mul(2)
                                .context("reread reservation overflow")?
                                .max(1024 * 1024),
                        )?;
                        source::verified(&root, r).map(|_| ())
                    })
            })?;
            for (path, digest) in metadata {
                ensure!(
                    sha256(&source::read(&root, &path)?) == digest,
                    "metadata changed during publication"
                );
            }
            ensure!(
                asset_graph(&root)? == assets,
                "runtime assets changed during publication"
            );
            for (path, bytes) in &build_inputs {
                ensure!(
                    source::read(&root, path)? == *bytes,
                    "production build input changed during publication: {path}"
                );
            }
            statistics(phase, "source-recheck", json!({}));
            Ok(())
        })?;
        statistics(install_started, "durable-install", json!({}));
        statistics(
            started,
            "complete",
            json!({"files":count,"output_bytes":bytes,"workers":policy.workers}),
        );
        Ok(())
    })();
    let phase = PhaseClock::now()?;
    // Never remove a recovery candidate after an indeterminate commit.
    if !transaction_parent
        .join(".conformance-transaction.json")
        .exists()
        && stage.exists()
    {
        fs::remove_dir_all(&stage)?;
    }
    statistics(phase, "cleanup", json!({}));
    let phase = PhaseClock::now()?;
    drop(pool);
    statistics(phase, "pool-shutdown", json!({}));
    result
}
