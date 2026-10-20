// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Assemble only runtime-consumed indexes and shared payload packs.
use crate::{
    codec::{self, json_bytes, sha256},
    history::Release,
    pipeline,
    resources::MemoryBudget,
    source,
    transport::PackWriter,
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn index(stage: &Path, path: &str, value: &Value) -> Result<Value> {
    let decoded = json_bytes(value)?;
    let bytes = codec::gzip(&decoded)?;
    pipeline::write(stage, path, &bytes)?;
    Ok(
        json!({"path":path,"sha256":sha256(&bytes),"size":bytes.len(),"decoded_size":decoded.len(),"decoded_sha256":sha256(&decoded)}),
    )
}

pub fn assemble(
    root: &Path,
    stage: &Path,
    original: &Value,
    resolved: &Value,
    budget: &MemoryBudget,
) -> Result<Value> {
    let mut catalog = pipeline::projection(resolved)?;
    let reference = catalog["metadata"]["selection_policy"]["unified"]["reference"]
        .as_str()
        .unwrap_or("");
    let selected = catalog["sources"]["candidates"]
        .as_array()
        .context("missing candidates")?
        .iter()
        .filter(|c| c["implementation"] == reference)
        .map(|c| {
            Ok((
                Release::parse(c["version"].as_str().context("missing version")?)?,
                c,
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max_by(|a, b| a.0.cmp(&b.0));
    let default_version = selected
        .as_ref()
        .and_then(|(_, c)| c["version"].as_str())
        .unwrap_or("");
    let default_families: BTreeSet<String> = selected
        .as_ref()
        .map(|(_, c)| {
            let key = if reference == "dynamo_v2" {
                "native_unified_families"
            } else {
                "capture_families"
            };
            c[key]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut resources = catalog["sources"]["unified"]
        .as_array()
        .context("missing resolved resources")?
        .clone();
    let priority = |r: &Value| {
        if (r["kind"] == "inputs-and-golden"
            && default_families.contains(r["family"].as_str().unwrap_or("")))
            || (r["implementation"] == reference && r["version"] == default_version)
        {
            0
        } else {
            1
        }
    };
    resources.sort_by_key(|r| (priority(r), r["path"].as_str().unwrap_or("").to_string()));
    let mut writer = PackWriter::new(stage);
    let mut tokens = Vec::new();
    let mut phase = 0;
    for resource in &mut resources {
        let next = priority(resource);
        if next != phase {
            writer.flush_phase()?;
            phase = next;
        }
        let path = resource["path"]
            .as_str()
            .context("missing bundle path")?
            .to_string();
        ensure!(
            resource["compression"] == "none",
            "production requires canonical envelopes"
        );
        let bytes = source::verified(&stage.join(".native-envelopes"), resource)?;
        tokens.push(writer.add(&bytes)?);
        resource["transport"] = json!("chunked-v2");
        fs::remove_file(stage.join(".native-envelopes").join(path))?;
    }
    writer.flush_phase()?;
    let mut raw = BTreeMap::<String, Value>::new();
    let mut original_tokens = Vec::new();
    let originals = ["unified", "archives"]
        .into_iter()
        .map(|kind| {
            original["sources"][kind]
                .as_array()
                .context("missing originals")
        })
        .collect::<Result<Vec<_>>>()?;
    let originals = originals
        .iter()
        .flat_map(|rows| rows.iter())
        .collect::<Vec<_>>();
    // Admission releases in each task before returning. Only descriptors are
    // collected, so bounded readers never retain a whole history in memory.
    originals
        .par_iter()
        .try_for_each(|resource| -> Result<()> {
            let path = resource["path"].as_str().context("missing original path")?;
            let size = if path.ends_with(".tar.gz") {
                source::archive_source_size(root, path)?.checked_mul(3)
            } else {
                resource["size"]
                    .as_u64()
                    .context("missing original size")?
                    .checked_mul(64)
            }
            .context("original validation reservation overflow")?;
            let _reservation = budget.reserve(size.max(16 * 1024 * 1024))?;
            let bytes = source::verified(root, resource)?;
            if path.ends_with(".yaml") {
                crate::yaml::load_str(std::str::from_utf8(&bytes)?)?;
            }
            if path.ends_with(".tar.gz") {
                source::archive(&bytes, resource)?;
            }
            Ok(())
        })?;
    for resource in originals {
        // Re-read and verify the same pinned identity admitted above. Pack
        // insertion order stays independent of validation completion order.
        let bytes = source::verified(root, resource)?;
        let path = resource["path"].as_str().context("missing original path")?;
        let token = writer.add(&bytes)?;
        let mut original_descriptor = resource.clone();
        // Source containers are a checkout concern. Published originals use
        // pack recipes and must not reference unpublished source archives.
        original_descriptor
            .as_object_mut()
            .unwrap()
            .remove("archive");
        original_tokens.push((path.to_string(), token, original_descriptor));
        if let Some(members) = resource["members"].as_array() {
            for member in members.iter().filter(|m| m["kind"] == "file") {
                let member = member["path"].as_str().context("missing archive member")?;
                for alias in [member.to_string(), format!("conformance/{member}")] {
                    ensure!(!raw.contains_key(&alias), "ambiguous raw alias: {alias}");
                    raw.insert(alias, json!({"path":path,"member":member}));
                }
            }
        }
    }
    writer.flush_phase()?;
    let mut inputs = json!({});
    for (resource, token) in resources.iter_mut().zip(tokens) {
        let path = resource["path"].as_str().unwrap().to_string();
        let recipe = writer.recipe(token)?;
        if resource["kind"] == "inputs-and-golden" {
            inputs[&path] = recipe;
        } else {
            let name = format!(
                "conformance/viewer-data/indexes/capture-{}.json.gz",
                resource["sha256"].as_str().unwrap()
            );
            resource["recipe_index"] =
                index(stage, &name, &json!({"schema":2,"resources":{path:recipe}}))?;
        }
    }
    let input_index = index(
        stage,
        "conformance/viewer-data/indexes/inputs.json.gz",
        &json!({"schema":2,"resources":inputs}),
    )?;
    for resource in resources
        .iter_mut()
        .filter(|r| r["kind"] == "inputs-and-golden")
    {
        resource["recipe_index"] = input_index.clone();
    }
    for (path, token, resource) in original_tokens {
        ensure!(!raw.contains_key(&path), "ambiguous original path");
        raw.insert(path.clone(), json!({"resource":{"path":path,"sha256":resource["sha256"],"size":resource["size"],"recipe":writer.recipe(token)?},"member":null}));
    }
    let mut buckets = BTreeMap::<String, BTreeMap<String, Value>>::new();
    for (path, entry) in raw {
        buckets
            .entry(sha256(path.as_bytes())[..2].to_string())
            .or_default()
            .insert(path, entry);
    }
    let mut lookup = json!({});
    for (bucket, entries) in buckets {
        let path = format!("conformance/viewer-data/indexes/raw-{bucket}.json.gz");
        lookup[&bucket] = index(stage, &path, &json!({"schema":2,"resources":entries}))?;
    }
    // The reader only needs these identities to validate resolved record members.
    catalog["sources"]["original_unified"] = json!(original["sources"]["unified"].as_array().unwrap().iter().filter(|r| r["kind"] == "history").map(|r| json!({"path":r["path"],"kind":r["kind"],"family":r["family"],"implementation":r["implementation"],"version":r["version"]})).collect::<Vec<_>>());
    for candidate in catalog["sources"]["candidates"].as_array_mut().unwrap() {
        candidate.as_object_mut().unwrap().remove("sources");
        candidate.as_object_mut().unwrap().remove("source");
    }
    catalog["sources"]["unified"] = json!(resources);
    catalog["transport"] = json!({"schema":2,"raw_buckets":lookup});
    // The source revision remains the provenance identity; this digest identifies
    // this exact runtime representation, independently of the old gzip bytes.
    catalog["runtime_sha256"] = json!(sha256(&json_bytes(&catalog)?));
    // Reject unreferenced chunks and damaged pack offsets before installation.
    let _validation_memory = budget.reserve(
        (crate::transport::VALIDATION_CACHE_BYTES + 4 * crate::transport::MAX_PACK) as u64,
    )?;
    crate::transport::validate_recipes(stage, &writer.finish()?)?;
    let spools = stage.join(".native-envelopes");
    if spools.exists() {
        fs::remove_dir_all(spools)?;
    }
    pipeline::write(
        stage,
        "conformance/viewer-resolved-catalog.json.gz",
        &codec::gzip(&json_bytes(&catalog)?)?,
    )?;
    Ok(catalog)
}

/// Traverse the actual runtime dependency graph, then reject every extra file.
/// This audit lives beside the site, never inside the browser publication.
pub fn inventory(stage: &Path, assets: impl Iterator<Item = String>) -> Result<Value> {
    let mut pending: Vec<String> = assets.collect();
    pending.extend([
        "conformance/viewer-resolved-catalog.json.gz".into(),
        "conformance/viewer-publication.json".into(),
    ]);
    let mut expected = BTreeMap::<String, Value>::new();
    fn references(
        value: &Value,
        stage: &Path,
        pending: &mut Vec<String>,
        checked: &mut BTreeMap<String, Value>,
    ) -> Result<()> {
        match value {
            Value::Object(fields) => {
                if value.get("sha256").is_some()
                    && value.get("size").is_some()
                    && value.get("path").is_some()
                    && value.get("recipe").is_none()
                    && value["transport"] != "chunked-v2"
                {
                    let path = value["path"].as_str().context("invalid dependency path")?;
                    source::relative(path)?;
                    if let Some(previous) = checked.get(path) {
                        ensure!(previous == value, "conflicting runtime dependency: {path}");
                        return Ok(());
                    }
                    checked.insert(path.to_string(), value.clone());
                    let bytes = fs::read(stage.join(path))
                        .with_context(|| format!("missing runtime dependency: {path}"))?;
                    ensure!(
                        value["sha256"] == sha256(&bytes)
                            && value["size"].as_u64() == Some(bytes.len() as u64),
                        "runtime dependency identity mismatch: {path}"
                    );
                    pending.push(path.to_string());
                }
                for child in fields.values() {
                    references(child, stage, pending, checked)?;
                }
            }
            Value::Array(values) => {
                for child in values {
                    references(child, stage, pending, checked)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    let mut checked = BTreeMap::new();
    while let Some(path) = pending.pop() {
        if expected.contains_key(&path) {
            continue;
        }
        let bytes = fs::read(stage.join(&path))?;
        expected.insert(
            path.clone(),
            json!({"sha256":sha256(&bytes),"size":bytes.len()}),
        );
        if path.ends_with(".json.gz") {
            let decoded =
                codec::gunzip(&bytes, &sha256(&bytes), source::MAX_ARTIFACT as usize - 1)?;
            references(
                &serde_json::from_slice::<Value>(&decoded)?,
                stage,
                &mut pending,
                &mut checked,
            )?;
        }
    }
    let actual: BTreeSet<String> = source::inventory(stage)?
        .into_iter()
        .filter(|path| stage.join(path).is_file())
        .collect();
    let declared: BTreeSet<String> = expected.keys().cloned().collect();
    ensure!(
        actual == declared,
        "publication dependency inventory mismatch; extra={:?}, missing={:?}",
        actual.difference(&declared).collect::<Vec<_>>(),
        declared.difference(&actual).collect::<Vec<_>>()
    );
    Ok(json!({"schema":2,"files":expected,"file_count":actual.len()}))
}
