//! `tesjit-dwarf` CLI.

use std::path::PathBuf;
use std::process::ExitCode;

use tesjit_dwarf::{export, TypeDb};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut list: Option<String> = None;
    let mut show: Vec<String> = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => {
                output = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--list" => list = args.next(),
            "--show" => show.push(args.next().unwrap_or_else(|| usage())),
            "-h" | "--help" => {
                eprintln!("usage: tesjit-dwarf <binary> [-o types.tdb] [--list substr] [--show Type]");
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

    let (db, stats) = match export(&data, &input.display().to_string()) {
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
    let bytes = match db.to_bytes() {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("tesjit-dwarf: encode failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(&output, &bytes) {
        eprintln!("tesjit-dwarf: cannot write {}: {e}", output.display());
        return ExitCode::FAILURE;
    }

    let round_tripped = TypeDb::from_bytes(&bytes).map(|db| db.types.len()).unwrap_or(0);
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
        bytes.len(),
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

fn usage() -> ! {
    eprintln!("usage: tesjit-dwarf <binary> [-o types.tdb] [--list substr] [--show Type]");
    std::process::exit(2)
}
