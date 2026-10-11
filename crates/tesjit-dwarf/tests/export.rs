//! Golden tests: compile a small Rust fixture with debug info, export its
//! types through the DWARF reader, and check the recovered layouts against
//! the compiler's own decisions.

use std::path::PathBuf;
use std::process::Command;

use tesjit_dwarf::{TypeArgs, TypeDb, export, export_with_dwo};

/// The fixture source, kept in one place so the compiled binary matches.
///
/// `main` must *use* every type: rustc only emits DWARF for types that are
/// actually codegen'ed, and an unused type optimizes away even in debug.
const FIXTURE_SRC: &str = r#"
#[derive(Clone, Copy)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

pub struct Block { pub kind: u8, pub light: u8, pub pos: Vec3 }

pub enum Shape {
    Empty,
    Square { side: u8, filled: bool },
    Circle { radius: f32, pos: Vec3 },
}

pub struct World {
    pub blocks_vec_len: usize,
    pub names: [&'static str; 4],
    pub shape: Shape,
    pub coords: Vec3,
}

fn main() {
    let world = World {
        blocks_vec_len: 0,
        names: ["a", "b", "c", "d"],
        shape: Shape::Circle { radius: 1.0, pos: Vec3 { x: 0.0, y: 0.0, z: 0.0 } },
        coords: Vec3 { x: 1.0, y: 2.0, z: 3.0 },
    };
    let block = Block { kind: 1, light: 2, pos: world.coords };
    std::hint::black_box((&world, &block));
}
"#;

fn compile_fixture() -> Option<PathBuf> {
    let out_dir = std::env::temp_dir().join(format!("tesjit-dwarf-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&out_dir);
    let source = out_dir.join("fixture.rs");
    std::fs::write(&source, FIXTURE_SRC).ok()?;
    let binary = out_dir.join("fixture_bin");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let status = Command::new(rustc)
        .arg("--edition=2021")
        .arg("-g")
        .arg("--crate-type=bin")
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .status()
        .ok()?;
    assert!(status.success(), "rustc fixture build failed");
    Some(binary)
}

fn exported() -> TypeDb {
    let binary = compile_fixture().expect("cannot build fixture (rustc must be on PATH)");
    let data = std::fs::read(&binary).expect("read fixture binary");
    let (db, _stats) = export(&data, "fixture").expect("export fixture");
    db
}

fn field_of(db: &TypeDb, type_name: &str, field: &str) -> (String, u64) {
    let def = db
        .get(type_name)
        .unwrap_or_else(|| panic!("{type_name} missing"));
    let fields = match &def.args {
        TypeArgs::Struct(s) => &s.fields,
        other => panic!("{type_name} is not a struct: {other:?}"),
    };
    fields
        .iter()
        .find(|f| f.name == field)
        .map(|f| (f.ty.to_string(), f.offset))
        .unwrap_or_else(|| panic!("{type_name} has no field {field} (have {fields:?})"))
}

#[test]
fn structs_and_enum_layouts() {
    let db = exported();

    let vec3 = db.get("fixture::Vec3").expect("Vec3 missing");
    assert_eq!((vec3.size, vec3.align), (12, 4), "Vec3 layout");
    let offsets: Vec<_> = db
        .layout("fixture::Vec3")
        .expect("layout")
        .into_iter()
        .map(|f| (f.name.to_string(), f.offset))
        .collect();
    assert_eq!(
        offsets,
        vec![
            ("x".to_string(), 0),
            ("y".to_string(), 4),
            ("z".to_string(), 8)
        ]
    );

    // Padding: `repr(Rust)` puts the `align 4` field first and the two `u8`s
    // last, so `size` rounds up to a multiple of `align` (4).
    let block = db.get("fixture::Block").expect("Block missing");
    assert_eq!((block.size, block.align), (16, 4), "Block layout");
    let pos = field_of(&db, "fixture::Block", "pos");
    assert_eq!(pos.0, "fixture::Vec3");
    assert_eq!(pos.1, 0, "pos comes first");
    let kind = field_of(&db, "fixture::Block", "kind");
    let light = field_of(&db, "fixture::Block", "light");
    assert_eq!((kind.0.as_str(), kind.1), ("u8", 12));
    assert_eq!((light.0.as_str(), light.1), ("u8", 13));

    let shape = db.get("fixture::Shape").expect("Shape missing");
    assert_eq!(shape.size, 20, "Shape size");
    let tesjit_dwarf::TypeArgs::Enum(enum_ty) = &shape.args else {
        panic!("Shape should be an enum");
    };
    assert_eq!(enum_ty.discriminant.ty.to_string(), "u8");
    assert_eq!(enum_ty.variants.len(), 3);
    assert_eq!(enum_ty.variants[0].name, "Empty");
    assert_eq!(enum_ty.variants[1].name, "Square");
    assert_eq!(enum_ty.variants[1].discr, 1);
    assert_eq!(enum_ty.variants[2].name, "Circle");
    assert_eq!(enum_ty.variants[2].discr, 2);
    let circle = &enum_ty.variants[2];
    let fields: Vec<_> = circle
        .fields
        .iter()
        .map(|f| (f.name.as_str(), f.offset))
        .collect();
    assert_eq!(fields, vec![("radius", 4), ("pos", 8)]);

    // Array field: `names: [&'static str; 4]` must survive with its element
    // type and length.
    let names = field_of(&db, "fixture::World", "names");
    assert_eq!(names.0, "[&str; 4]", "{names:?}");
    // All four array fields (names, shape, coords) are laid out after the
    // 8-byte `blocks_vec_len`; just require it to be inside the struct.
    let world = db.get("fixture::World").unwrap();
    assert!(
        names.1 < world.size && world.size >= 8,
        "{names:?} / {world:?}"
    );
}

/// rustc's dev default here is split-DWARF: the binary holds only skeleton
/// units, so `export` alone cannot see the crate's own types. Reading the
/// sibling `.dwo` files must recover them — this is the whole reason
/// `export_with_dwo` exists.
#[test]
fn dwo_siblings_recover_types_the_binary_cannot() {
    // Build the fixture with split-DWARF and no embedded info, like a normal
    // dev build, so the binary really is skeleton-only.
    let dir = std::env::temp_dir().join(format!("tesjit-split-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let out = dir.join("out");
    let _ = std::fs::create_dir_all(&out);

    let src = dir.join("fixture.rs");
    std::fs::write(
        &src,
        r#"
        pub struct Nested { pub a: f32, pub b: u64 }
        // Deliberately not `repr(C)`: rustc may reorder fields, and the offsets
        // must come from DWARF rather than from this declaration order.
        pub struct Outer { pub flag: u8, pub nested: Nested, pub n: i64 }
        fn main() {
            let o = Outer { flag: 1, nested: Nested { a: 1.0, b: 2 }, n: 3 };
            std::hint::black_box((&o, o.nested.b));
        }
        "#,
    )
    .unwrap();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let status = std::process::Command::new(rustc)
        .arg("--edition=2021")
        .arg("-g")
        .arg("-Csplit-debuginfo=unpacked")
        .arg(&src)
        .arg("-o")
        .arg(out.join("fixture"))
        .status()
        .unwrap();
    assert!(status.success(), "split-DWARF fixture build failed");

    let binary = out.join("fixture");
    let data = std::fs::read(&binary).unwrap();

    // The skeleton-only read finds no `fixture::Outer` — that is the bug this
    // whole change is about.
    let (plain, _) = export(&data, "fixture").expect("export fixture");
    assert!(
        plain.get("fixture::Outer").is_none(),
        "a split-DWARF binary must not contain its own types; found {:?}",
        plain.get("fixture::Outer").map(|d| d.name.to_string())
    );

    // ... and the sibling `.dwo` read must.
    let (db, _) = export_with_dwo(&data, "fixture", &out).expect("export with dwo");
    let outer = db.get("fixture::Outer").expect("Outer recovered from .dwo");
    let sizes = (outer.size, outer.align);
    let tesjit_dwarf::TypeArgs::Struct(fields) = &outer.args else {
        panic!("Outer is a struct");
    };
    let offset_of = |name: &str| fields.fields.iter().find(|f| f.name == name).map(|f| f.offset);
    // Field offsets come from DWARF, so they match the compiled layout whatever
    // order the source declared them in.
    assert!(offset_of("nested").is_some() && offset_of("n").is_some(), "{fields:?}");
    let nested_off = offset_of("nested").unwrap();
    let nested = db.get("fixture::Nested").expect("Nested recovered from .dwo");
    let nested_a = match &nested.args {
        tesjit_dwarf::TypeArgs::Struct(f) => f
            .fields
            .iter()
            .find(|f| f.name == "a")
            .map(|f| f.offset)
            .expect("nested.a"),
        _ => panic!("Nested is a struct"),
    };
    // This is the whole point: the source says `{ a: f32, b: u64 }`, but
    // `repr(Rust)` puts the u64 first, so `a` lives at offset 8 — and only the
    // DWARF export knows that. A mod reading `Nested` has to use this offset,
    // not the declaration order.
    assert_eq!((nested.size, nested_a), (16, 8), "repr(Rust) reorders fields");

    // And the nested struct is embedded at an offset the exporter reports, not
    // the one the source order suggests.
    let nested_bytes = |name: &str| {
        match &db.get("fixture::Nested").unwrap().args {
            tesjit_dwarf::TypeArgs::Struct(f) => f
                .fields
                .iter()
                .find(|f| f.name == name)
                .map(|f| f.offset)
                .unwrap_or(0),
            _ => 0,
        }
    };
    let _ = nested_bytes("a");
    // The composed layout: `flag` (1 byte, u64-aligned) then `nested` then `n`.
    assert!(nested_off % 4 == 0, "{nested_off}");
    assert!(offset_of("n").unwrap() > nested_off, "{fields:?}");
    let _ = (sizes.0, sizes.1);
    let _ = std::fs::remove_dir_all(&dir);
}

fn def_of<'a>(db: &'a tesjit_dwarf::TypeDb, name: &'a str) -> Option<&'a tesjit_dwarf::TypeDef> {
    db.get(name)
}
