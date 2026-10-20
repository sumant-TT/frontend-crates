// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Directory leases and durable journals compatible with retained Python writers.
use crate::{
    codec::json_bytes,
    source::{no_links, relative},
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::Write,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const JOURNAL: &str = ".conformance-transaction.json";

pub struct Lease {
    _file: File,
}
impl Lease {
    fn lock(parent: &Path, operation: i32) -> Result<Self> {
        no_links(parent)?;
        let file = File::open(parent)?;
        ensure!(file.metadata()?.is_dir(), "lease parent is not a directory");
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), operation) } == 0,
            "directory lease failed: {}",
            std::io::Error::last_os_error()
        );
        Ok(Self { _file: file })
    }

    pub fn acquire(parent: &Path) -> Result<Self> {
        let lease = Self::lock(parent, libc::LOCK_EX)?;
        recover(parent)?;
        Ok(lease)
    }

    /// Hold stable source bytes alongside other readers. Recovery always takes
    /// a fresh exclusive lease; never upgrade a held shared lease in place.
    pub fn read(parent: &Path) -> Result<Self> {
        loop {
            let lease = Self::lock(parent, libc::LOCK_SH)?;
            let journal = parent.join(JOURNAL);
            no_links(&journal)?;
            if !journal.exists() {
                return Ok(lease);
            }
            drop(lease);
            drop(Self::acquire(parent)?);
        }
    }
}

pub fn sync_tree(path: &Path) -> Result<()> {
    no_links(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            sync_tree(&entry?.path())?;
        }
    } else {
        ensure!(metadata.is_file(), "nonregular transaction payload");
    }
    File::open(path)?.sync_all()?;
    Ok(())
}

pub fn unique(parent: &Path, prefix: &str) -> Result<PathBuf> {
    let path = parent.join(format!(
        "{prefix}{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&path)?;
    Ok(path)
}

fn journal(parent: &Path, value: &Value) -> Result<()> {
    let temporary = parent.join(format!("{JOURNAL}.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&json_bytes(value)?)?;
    file.sync_all()?;
    fs::rename(&temporary, parent.join(JOURNAL))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn path(parent: &Path, value: &Value) -> Result<PathBuf> {
    let value = value.as_str().context("missing transaction path")?;
    relative(value)?;
    let path = parent.join(value);
    no_links(&path)?;
    Ok(path)
}

fn remove(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(!metadata.file_type().is_symlink(), "transaction symlink");
            if metadata.is_dir() {
                fs::remove_dir_all(path)?;
            } else {
                fs::remove_file(path)?;
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn exchange(left: &Path, right: &Path) -> Result<()> {
    let left = CString::new(left.as_os_str().as_encoded_bytes())?;
    let right = CString::new(right.as_os_str().as_encoded_bytes())?;
    ensure!(
        unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                left.as_ptr(),
                libc::AT_FDCWD,
                right.as_ptr(),
                libc::RENAME_EXCHANGE,
            )
        } == 0,
        "atomic directory exchange failed: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

fn backup_existing(source: &Path, destination: &Path) -> Result<()> {
    no_links(source)?;
    let metadata = fs::symlink_metadata(source)?;
    if metadata.is_dir() {
        fs::create_dir(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            backup_existing(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else {
        ensure!(metadata.is_file(), "nonregular existing publication");
        fs::hard_link(source, destination)?;
    }
    Ok(())
}

fn restore(backup: &Path, destination: &Path, candidate: &Path) -> Result<()> {
    // Keep the rollback snapshot immutable so recovery can itself be interrupted.
    remove(candidate)?;
    backup_existing(backup, candidate)?;
    sync_tree(candidate)?;
    if destination.exists() {
        exchange(candidate, destination)?;
    } else {
        fs::rename(candidate, destination)?;
    }
    File::open(
        destination
            .parent()
            .context("missing rollback destination parent")?,
    )?
    .sync_all()?;
    remove(candidate)
}

pub fn recover(parent: &Path) -> Result<()> {
    let journal_path = parent.join(JOURNAL);
    no_links(&journal_path)?;
    if !journal_path.exists() {
        return Ok(());
    }
    let value: Value = serde_json::from_slice(&fs::read(&journal_path)?)?;
    ensure!(
        value["schema_version"] == 1
            && [Some("prepared"), Some("committed")].contains(&value["phase"].as_str()),
        "invalid transaction journal"
    );
    let backup_root = path(parent, &value["backup_root"])?;
    ensure!(
        backup_root.parent() == Some(parent)
            && backup_root
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(".conformance-transaction-backup-")),
        "invalid backup root"
    );
    let states = value["states"]
        .as_array()
        .context("invalid transaction states")?;
    let mut destinations = Vec::new();
    for state in states {
        let destination = path(parent, &state["destination"])?;
        let candidate = path(parent, &state["candidate"])?;
        let backup = path(parent, &state["backup"])?;
        ensure!(
            backup.parent() == Some(backup_root.as_path())
                && destination != journal_path
                && !destination.starts_with(&backup_root)
                && !candidate.starts_with(&destination)
                && !destination.starts_with(&candidate),
            "invalid transaction paths"
        );
        ensure!(
            destinations
                .iter()
                .all(|p: &PathBuf| !p.starts_with(&destination) && !destination.starts_with(p)),
            "overlapping transaction destinations"
        );
        destinations.push(destination);
        ensure!(state["had_original"].is_boolean(), "invalid original state");
    }
    if value["phase"] == "prepared" {
        for (index, state) in states.iter().enumerate().rev() {
            let destination = path(parent, &state["destination"])?;
            let backup = path(parent, &state["backup"])?;
            if state["had_original"] == true {
                if backup.exists() {
                    restore(
                        &backup,
                        &destination,
                        &backup_root.join(format!(".restore-{index:02}")),
                    )?;
                } else {
                    ensure!(destination.exists(), "transaction original unavailable");
                }
            } else {
                ensure!(!backup.exists(), "unexpected transaction backup");
                remove(&destination)?;
            }
            File::open(destination.parent().context("missing destination parent")?)?.sync_all()?;
        }
    }
    for state in states {
        let candidate = path(parent, &state["candidate"])?;
        let first = candidate
            .strip_prefix(parent)?
            .components()
            .next()
            .context("empty candidate")?;
        if first
            .as_os_str()
            .to_str()
            .is_some_and(|name| name.starts_with(".native-publication-stage-"))
        {
            remove(&parent.join(first.as_os_str()))?;
        }
    }
    remove(&backup_root)?;
    fs::remove_file(journal_path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub fn publish(parent: &Path, pairs: &[(PathBuf, PathBuf)]) -> Result<()> {
    publish_checked(parent, pairs, || Ok(()))
}

pub fn publish_checked(
    parent: &Path,
    pairs: &[(PathBuf, PathBuf)],
    before_commit: impl FnOnce() -> Result<()>,
) -> Result<()> {
    recover(parent)?;
    ensure!(!pairs.is_empty(), "empty publication transaction");
    for (index, (candidate, destination)) in pairs.iter().enumerate() {
        for (other_candidate, other_destination) in &pairs[..index] {
            ensure!(
                !destination.starts_with(other_destination)
                    && !other_destination.starts_with(destination)
                    && !candidate.starts_with(other_candidate)
                    && !other_candidate.starts_with(candidate),
                "overlapping publication paths"
            );
        }
        sync_tree(candidate)?;
    }
    let backup_root = unique(parent, ".conformance-transaction-backup-")?;
    let mut states = Vec::new();
    for (index, (candidate, destination)) in pairs.iter().enumerate() {
        no_links(candidate)?;
        no_links(destination)?;
        ensure!(
            candidate.starts_with(parent) && destination.starts_with(parent),
            "transaction outside parent"
        );
        ensure!(
            !candidate.starts_with(destination) && !destination.starts_with(candidate),
            "overlapping candidate and destination"
        );
        let backup = backup_root.join(format!(
            "{index:02}-{}",
            destination
                .file_name()
                .context("invalid destination")?
                .to_string_lossy()
        ));
        if destination.exists() {
            backup_existing(destination, &backup)?;
        }
        states.push(json!({"candidate":candidate.strip_prefix(parent)?.to_str(),"destination":destination.strip_prefix(parent)?.to_str(),"backup":backup.strip_prefix(parent)?.to_str(),"had_original":destination.exists()}));
    }
    sync_tree(&backup_root)?;
    if let Err(error) = before_commit() {
        remove(&backup_root)?;
        return Err(error);
    }
    let mut document = json!({"schema_version":1,"phase":"prepared","backup_root":backup_root.strip_prefix(parent)?.to_str(),"states":states});
    journal(parent, &document)?;
    let result = (|| -> Result<()> {
        for state in &states {
            let candidate = path(parent, &state["candidate"])?;
            let destination = path(parent, &state["destination"])?;
            if state["had_original"] == true {
                exchange(&candidate, &destination)?;
            } else {
                fs::rename(&candidate, &destination)?;
            }
            File::open(candidate.parent().context("missing candidate parent")?)?.sync_all()?;
            File::open(destination.parent().context("missing destination parent")?)?.sync_all()?;
        }
        document["phase"] = json!("committed");
        journal(parent, &document)
    })();
    if let Err(error) = result {
        // A visible commit must retain its journal if durability is uncertain.
        let current: Value = serde_json::from_slice(&fs::read(parent.join(JOURNAL))?)?;
        if current["phase"] == "prepared" {
            recover(parent).context("publication rollback failed")?;
        }
        return Err(error);
    }
    recover(parent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_readers_share_ownership_and_exclude_writers() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let lease = Lease::read(parent.path())?;
        let peer = File::open(parent.path())?;
        assert_eq!(
            unsafe { libc::flock(peer.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) },
            0
        );
        assert_ne!(
            unsafe { libc::flock(peer.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        drop(lease);
        assert_eq!(
            unsafe { libc::flock(peer.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        Ok(())
    }

    #[test]
    fn interrupted_partial_data_recovery_is_repeatable() -> Result<()> {
        let parent = unique(&std::env::temp_dir(), "native-partial-recovery-")?;
        let result = (|| -> Result<()> {
            let stage = unique(&parent, ".native-publication-stage-")?;
            let backup_root = unique(&parent, ".conformance-transaction-backup-")?;
            let mut states = Vec::new();
            for index in 0..3 {
                let destination = parent.join(format!("data-{index}"));
                let candidate = stage.join(format!("data-{index}"));
                let backup = backup_root.join(format!("{index:02}-data"));
                fs::write(&candidate, b"new")?;
                if index != 0 {
                    fs::write(&destination, b"old")?;
                    backup_existing(&destination, &backup)?;
                    exchange(&candidate, &destination)?;
                } else {
                    fs::rename(&candidate, &destination)?;
                }
                states.push(json!({"candidate":candidate.strip_prefix(&parent)?.to_str(),"destination":destination.strip_prefix(&parent)?.to_str(),"backup":backup.strip_prefix(&parent)?.to_str(),"had_original":index != 0}));
            }
            journal(
                &parent,
                &json!({"schema_version":1,"phase":"prepared","backup_root":backup_root.file_name().unwrap().to_str(),"states":states}),
            )?;
            // Recovery restores in reverse order. Interrupt twice after its exchange,
            // leaving one restored output and the other outputs still installed.
            let saved = backup_root.join("02-data");
            let destination = parent.join("data-2");
            let disposable = backup_root.join(".restore-02");
            for _ in 0..2 {
                remove(&disposable)?;
                backup_existing(&saved, &disposable)?;
                sync_tree(&disposable)?;
                exchange(&disposable, &destination)?;
                assert_eq!(fs::read(&destination)?, b"old");
                assert_eq!(fs::read(&saved)?, b"old");
                assert_eq!(fs::read(parent.join("data-1"))?, b"new");
            }
            recover(&parent)?;
            recover(&parent)?;
            assert!(!parent.join("data-0").exists());
            for index in 1..3 {
                assert_eq!(fs::read(parent.join(format!("data-{index}")))?, b"old");
            }
            assert!(!stage.exists());
            assert!(!backup_root.exists());
            assert!(!parent.join(JOURNAL).exists());
            Ok(())
        })();
        fs::remove_dir_all(parent)?;
        result
    }

    #[test]
    fn exchange_keeps_complete_output_visible_and_prepared_journal_rolls_back() -> Result<()> {
        let parent = unique(&std::env::temp_dir(), "native-exchange-test-")?;
        let result = (|| -> Result<()> {
            let destination = parent.join("site");
            fs::create_dir(&destination)?;
            fs::write(destination.join("value"), b"old")?;
            let candidate = unique(&parent, ".native-publication-stage-")?;
            fs::write(candidate.join("value"), b"new")?;
            let backup_root = unique(&parent, ".conformance-transaction-backup-")?;
            let saved = backup_root.join("00-site");
            backup_existing(&destination, &saved)?;
            sync_tree(&backup_root)?;
            let document = json!({"schema_version":1,"phase":"prepared","backup_root":backup_root.file_name().unwrap().to_str(),"states":[{"candidate":candidate.file_name().unwrap().to_str(),"destination":"site","backup":saved.strip_prefix(&parent)?.to_str(),"had_original":true}]});
            journal(&parent, &document)?;
            exchange(&candidate, &destination)?;
            // This is the persistent state at an interruption after the install syscall.
            assert_eq!(fs::read(destination.join("value"))?, b"new");
            assert_eq!(fs::read(saved.join("value"))?, b"old");
            let interrupted = backup_root.join(".restore-00");
            backup_existing(&saved, &interrupted)?;
            exchange(&interrupted, &destination)?;
            assert_eq!(fs::read(destination.join("value"))?, b"old");
            assert_eq!(fs::read(saved.join("value"))?, b"old");
            recover(&parent)?;
            assert_eq!(fs::read(destination.join("value"))?, b"old");
            assert!(!candidate.exists());
            let rejected = unique(&parent, ".native-publication-stage-")?;
            fs::write(rejected.join("value"), b"rejected")?;
            assert!(
                publish_checked(
                    &parent,
                    &[(rejected, destination.clone())],
                    || anyhow::bail!("source changed after staging")
                )
                .is_err()
            );
            assert_eq!(fs::read(destination.join("value"))?, b"old");
            Ok(())
        })();
        fs::remove_dir_all(parent)?;
        result
    }
}
