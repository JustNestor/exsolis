use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{Error, Result};

pub(crate) fn usz(v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| Error::Corrupt("value exceeds address space"))
}

/// `a` must be a power of two.
#[inline]
pub(crate) fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

pub(crate) fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Runs `f(0..n)` on up to `threads` scoped threads (dynamic work stealing via an
/// atomic counter) and returns results in index order.
pub(crate) fn par_map<T, F>(n: usize, threads: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        return (0..n).map(&f).collect();
    }
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        local.push((i, f(i)));
                    }
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread panicked"))
            .collect()
    });
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for part in parts {
        for (i, t) in part {
            slots[i] = Some(t);
        }
    }
    slots
        .into_iter()
        .map(|t| t.expect("missing result"))
        .collect()
}

/// Writes to `<path>.tmp`, fsyncs, then renames over `path`, so a crash never
/// leaves a half-written .pak under the final name.
pub(crate) fn atomic_write<T>(path: &Path, f: impl FnOnce(File) -> Result<(File, T)>) -> Result<T> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);

    let res = (|| {
        let file = File::create(&tmp)?;
        let (file, t) = f(file)?;
        file.sync_all()?;
        Ok::<T, Error>(t)
    })();

    match res {
        Ok(t) => {
            if let Err(e) = fs::rename(&tmp, path) {
                let _ = fs::remove_file(&tmp);
                return Err(Error::Io(e));
            }
            Ok(t)
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}
