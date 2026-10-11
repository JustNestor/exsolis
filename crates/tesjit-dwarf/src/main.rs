//! `tesjit-dwarf` CLI.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use tesjit_dwarf::{TypeDb, export_with_dwo};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut list: Option<String> = None;
    let mut show: Vec<String> = Vec::new();
    let mut dwo_dir: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => {
                output = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--list" => list = args.next(),
            "--show" => show.push(args.next().unwrap_or_else(|| usage())),
            "--dwo-dir" => {
                // Split-DWARF: the real type info lives in sibling `.dwo` files.
                dwo_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())));
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: tesjit-dwarf <binary> [-o types.tdb] [--dwo-dir DIR] [--list substr] [--show Type]"
                );
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => usage(),
            _ => {
                if input.is_some() {
                    usage();
                }
                input = Some(PathBuf::from(arg));
            }
        }
    }
    let input = input.unwrap_or_else(|| usage());

    let data = match std::fs::read(&input) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("tesjit-dwarf: cannot read {}: {e}", input.display());
            return ExitCode::FAILURE;
        }
    };

    let source = input.display().to_string();
    // Split-DWARF note: rustc's dev default on this toolchain puts every real
    // type in sibling `.dwo` files, so the default is to look next to the
    // binary. Pass `--dwo-dir .` to disable that.
    let dwo_dir = dwo_dir.unwrap_or_else(|| default_dwo_dir(&input));
    let (db, stats) = match export_with_dwo(&data, &source, &dwo_dir) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("tesjit-dwarf: export failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if db.types.is_empty() {
        eprintln!(
            "tesjit-dwarf: `{}` has no exported types; walked {} CUs / {} DIEs (stats: {stats:?})",
            input.display(),
            stats.compilation_units,
            stats.dies
        );
        return ExitCode::FAILURE;
    }

    let output = output.unwrap_or_else(|| PathBuf::from("types.tdb"));
    if let Some(parent) = output.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let encoded = match db.to_bytes() {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("tesjit-dwarf: encode failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(&output, &encoded) {
        eprintln!("tesjit-dwarf: cannot write {}: {e}", output.display());
        return ExitCode::FAILURE;
    }

    let round_tripped = TypeDb::from_bytes(&encoded)
        .map(|db| db.types.len())
        .unwrap_or(0);
    println!(
        "{}: {} types ({} structs, {} unions, {} enums, {} arrays, {} pointers, {} aliases, {} opaque) \
         from {} CUs / {} DIEs; wrote {} bytes to {} (reloaded {round_tripped} types)",
        input.display(),
        db.len(),
        stats.structs,
        stats.unions,
        stats.enums,
        stats.arrays,
        stats.pointers,
        stats.aliases,
        stats.opaque,
        stats.compilation_units,
        stats.dies,
        encoded.len(),
        output.display(),
    );

    if let Some(needle) = list {
        let hits: Vec<_> = db
            .iter()
            .filter(|(name, _)| name.contains(&needle))
            .take(200)
            .map(|(name, def)| format!("  {name} (size {}, align {})", def.size, def.align))
            .collect();
        println!("matches for {needle:?}: {}", hits.len());
        for hit in hits {
            println!("{hit}");
        }
    }

    for name in &show {
        match db.get(name) {
            Some(def) => println!("{def:#?}"),
            None => println!("unknown type `{name}`"),
        }
    }
    ExitCode::SUCCESS
}

/// Where split-DWARF objects for a binary usually live.
///
/// rustc writes them next to the *object*, one build-script directory deeper
/// than the final binary: `<target>/debug/build/<crate>-<hash>/out`. Since that
/// is not derivable from the binary alone, the default is the directory the
/// binary itself is in, which is also where a single-file dwo dump lands.
fn default_dwo_dir(binary: &Path) -> PathBuf {
    std::env::args()
        .position(|arg| arg == "--dwo-dir")
        .map(|_| PathBuf::from("."))
        .unwrap_or_else(|| {
            binary
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
        })
}

fn usage() -> ! {
    eprintln!("usage: tesjit-dwarf <binary> [-o types.tdb] [--list substr] [--show Type]");
    std::process::exit(2)
}
