//! Golden tests: compile a small Rust fixture with debug info, export its
//! types through the DWARF reader, and check the recovered layouts against
//! the compiler's own decisions.

use std::path::PathBuf;
use std::process::Command;

use tesjit_dwarf::{export, TypeArgs, TypeDb};

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
    let def = db.get(type_name).unwrap_or_else(|| panic!("{type_name} missing"));
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
    assert_eq!(offsets, vec![("x".to_string(), 0), ("y".to_string(), 4), ("z".to_string(), 8)]);

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
    assert!(names.1 < world.size && world.size >= 8, "{names:?} / {world:?}");
}
