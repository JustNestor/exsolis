use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use fastpak::*;

fn tmp(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "fastpak-test-{}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed),
        name
    ));
    p
}

fn noise(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn text(len: usize) -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(len)
        .copied()
        .collect()
}

#[test]
fn roundtrip_layout_and_lookup() {
    let path = tmp("rt.pak");
    let files = [
        ("textures/b.png", noise(1, 5000)),
        ("models/a.glb", text(10_000)),
        ("textures/c.PNG", noise(2, 300)),
        ("shaders/x.wgsl", text(2000)),
        ("readme", text(500)),
    ];
    let mut b = PakBuilder::new();
    b.intent(Intent::Balanced).threads(2);
    for (p, d) in &files {
        b.add_bytes(p, d.clone()).unwrap();
    }
    let stats = b.build(&path).unwrap();
    assert_eq!(stats.entries, 5);

    let pak = Pak::open(&path).unwrap();
    assert_eq!(pak.len(), 5);
    for (p, d) in &files {
        assert_eq!(&*pak.get(p).unwrap(), &d[..], "{p}");
    }

    // Same-extension entries are contiguous.
    let mut seen = HashSet::new();
    let mut prev: Option<String> = None;
    for info in pak.iter() {
        let e = ext_of(info.path).to_ascii_lowercase();
        if prev.as_deref() != Some(e.as_str()) {
            assert!(seen.insert(e.clone()), "extension {e} is not contiguous");
            prev = Some(e);
        }
    }
    assert_eq!(pak.iter_ext("png").count(), 2);
    assert_eq!(pak.iter_ext(".PNG").count(), 2);
    assert_eq!(pak.iter_ext("nope").count(), 0);

    // Codec selection: png stored raw (zero-copy), text compressed.
    let raw = pak.info(pak.find("textures/b.png").unwrap()).unwrap();
    assert_eq!(raw.codec, Codec::Raw);
    assert!(matches!(pak.get("textures/b.png").unwrap(), std::borrow::Cow::Borrowed(_)));
    let z = pak.info(pak.find("models/a.glb").unwrap()).unwrap();
    assert_eq!(z.codec, Codec::Zstd);
    assert!(z.stored_len < z.raw_len);

    assert!(matches!(pak.get("nope"), Err(Error::NotFound(_))));
    pak.verify(2).unwrap();

    let names = vec!["readme", "nope", "models/a.glb"];
    let r = pak.get_many_par(&names, 2);
    assert_eq!(&*r[0].as_ref().unwrap(), &text(500)[..]);
    assert!(matches!(r[1], Err(Error::NotFound(_))));
    assert_eq!(&*r[2].as_ref().unwrap(), &text(10_000)[..]);

    drop(pak);
    let _ = std::fs::remove_file(path);
}

#[test]
fn dedup_stores_identical_blobs_once() {
    let path = tmp("dedup.pak");
    let mut b = PakBuilder::new();
    b.add_bytes("a/one.bin", text(4000)).unwrap();
    b.add_bytes("b/two.bin", text(4000)).unwrap();
    let stats = b.build(&path).unwrap();
    assert_eq!(stats.deduped, 1);

    let pak = Pak::open(&path).unwrap();
    let a = pak.info(pak.find("a/one.bin").unwrap()).unwrap();
    let b = pak.info(pak.find("b/two.bin").unwrap()).unwrap();
    assert_eq!(a.offset, b.offset);
    assert_eq!(&*pak.get("b/two.bin").unwrap(), &text(4000)[..]);
    drop(pak);
    let _ = std::fs::remove_file(path);
}

#[test]
fn merge_overlay_and_conflict() {
    let (pa, pb, out) = (tmp("a.pak"), tmp("b.pak"), tmp("m.pak"));

    let mut a = PakBuilder::new();
    a.add_bytes("x.txt", b"one".to_vec()).unwrap();
    a.add_bytes("only_a.bin", noise(3, 1000)).unwrap();
    a.build(&pa).unwrap();

    let mut b = PakBuilder::new();
    b.add_bytes("x.txt", b"two".to_vec()).unwrap();
    b.add_bytes("only_b.dds", text(3000)).unwrap();
    b.build(&pb).unwrap();

    let (a, b) = (Pak::open(&pa).unwrap(), Pak::open(&pb).unwrap());
    merge(&[&a, &b], &out, &MergeOptions::default()).unwrap();

    let m = Pak::open(&out).unwrap();
    assert_eq!(m.len(), 3);
    assert_eq!(&*m.get("x.txt").unwrap(), &b"two"[..]);
    assert_eq!(&*m.get("only_b.dds").unwrap(), &text(3000)[..]);
    assert_eq!(&*m.get("only_a.bin").unwrap(), &noise(3, 1000)[..]);
    m.verify(1).unwrap();

    let strict = MergeOptions { on_conflict: Conflict::Error, ..Default::default() };
    assert!(merge(&[&a, &b], tmp("never.pak"), &strict).is_err());

    drop((a, b, m));
    for p in [pa, pb, out] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn corruption_is_detected() {
    let path = tmp("bad.pak");
    let mut b = PakBuilder::new();
    b.add_bytes("a.txt", text(1000)).unwrap();
    b.build(&path).unwrap();

    let good = std::fs::read(&path).unwrap();

    // Flip a byte in the names blob (end of file) -> index checksum mismatch.
    let mut bad = good.clone();
    *bad.last_mut().unwrap() ^= 0xFF;
    std::fs::write(&path, &bad).unwrap();
    assert!(Pak::open(&path).is_err());

    // Truncated file.
    std::fs::write(&path, &good[..good.len() - 3]).unwrap();
    assert!(Pak::open(&path).is_err());

    let _ = std::fs::remove_file(path);
}

#[test]
fn rejects_bad_paths() {
    let mut b = PakBuilder::new();
    assert!(b.add_bytes("../evil", vec![1]).is_err());
    assert!(b.add_bytes("", vec![1]).is_err());
    b.add_bytes("dir\\file.bin", vec![1]).unwrap(); // normalized to dir/file.bin
    assert!(matches!(b.add_bytes("dir/file.bin", vec![2]), Err(Error::DuplicatePath(_))));
}
