//! Schema unit tests: round-trip through the on-disk format, aliasing and
//! layout flattening.

use tesjit_dwarf::{
    Field, Head, StructType, TypeArgs, TypeDb, TypeDef, TypeName, Variant, Discriminant,
    EnumType, FORMAT_VERSION,
};

fn struct_def(name: &str, size: u64, fields: Vec<(&str, &str, u64)>) -> TypeDef {
    TypeDef {
        name: TypeName::from(name),
        size,
        align: 0,
        args: TypeArgs::Struct(StructType {
            is_union: false,
            fields: fields
                .into_iter()
                .map(|(n, t, o)| Field {
                    name: n.to_owned(),
                    ty: TypeName::from(t),
                    offset: o,
                })
                .collect(),
        }),
    }
}

fn opaque(name: &str, size: u64, align: u64) -> TypeDef {
    TypeDef {
        name: TypeName::from(name),
        size,
        align,
        args: TypeArgs::Opaque(tesjit_dwarf::OpaqueKind::Foreign),
    }
}

fn sample() -> TypeDb {
    let mut db = TypeDb::new();
    db.types.insert("u8".into(), opaque("u8", 1, 1));
    db.types.insert("f32".into(), opaque("f32", 4, 4));
    db.types.insert(
        "inner::Point".into(),
        struct_def("inner::Point", 8, vec![("x", "f32", 0), ("y", "f32", 4)]),
    );
    db.types.insert(
        "outer::Padded".into(),
        struct_def("outer::Padded", 16, vec![("a", "u8", 0), ("p", "inner::Point", 4)]),
    );
    db.types.insert(
        "outer::Ref".into(),
        struct_def("outer::Ref", 16, vec![("p", "&inner::Point", 0), ("d", "u8", 8)]),
    );
    db.types.insert("&inner::Point".into(), TypeDef {
        name: TypeName::from("&inner::Point"),
        size: 8,
        align: 8,
        args: TypeArgs::Pointer { pointee: TypeName::from("inner::Point") },
    });
    db.aliases.insert("Point".into(), "inner::Point".into());
    db
}

#[test]
fn roundtrip_through_bincode() {
    let mut db = sample();
    db.head = Head {
        version: FORMAT_VERSION,
        ..Head::new()
    };
    let bytes = db.to_bytes().expect("encode");
    let back = TypeDb::from_bytes(&bytes).expect("decode");
    assert_eq!(db, back);
    assert_eq!(back.len(), db.len());
}

#[test]
fn rejects_empty_or_garbage() {
    assert!(TypeDb::from_bytes(b"").is_err());
    assert!(TypeDb::from_bytes(b"not a type db at all, just bytes").is_err());
}

#[test]
fn alias_lookup_and_canonicalize() {
    let db = sample();
    // Alias hop.
    let def = db.get("Point").expect("alias should resolve");
    assert_eq!(def.name.to_string(), "inner::Point");
    assert_eq!(db.canonicalize("Point"), "inner::Point");
    assert_eq!(db.aliases_of("inner::Point"), vec!["Point"]);
}

#[test]
fn flatten_nested_structs() {
    let db = sample();
    let leaves = db.layout("outer::Padded").expect("layout");
    let got: Vec<_> = leaves
        .iter()
        .map(|f| (f.name.as_str(), f.offset))
        .collect();
    assert_eq!(got, vec![("a", 0), ("p::x", 4), ("p::y", 8)]);
}

#[test]
fn pointer_fields_stop_the_walk() {
    let db = sample();
    let leaves = db.layout("outer::Ref").expect("layout");
    let got: Vec<_> = leaves.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(got, vec!["p", "d"]);
}

#[test]
fn resolve_reports_unknown_and_recursive() {
    let mut db = sample();
    assert!(db.get("nope").is_none());

    // A struct containing itself must be reported, not hang.
    db.types.insert(
        "rec::Loop".into(),
        struct_def("rec::Loop", 8, vec![("next", "rec::Loop", 0)]),
    );
    let err = db.layout("rec::Loop").unwrap_err();
    assert!(matches!(err, tesjit_dwarf::TypeDbError::RecursiveType(_)), "{err}");
}

#[test]
fn enum_variants_are_exported() {
    let mut db = sample();
    db.types.insert(
        "opt::Maybe".into(),
        TypeDef {
            name: TypeName::from("opt::Maybe"),
            size: 16,
            align: 4,
            args: TypeArgs::Enum(EnumType {
                discriminant: Discriminant {
                    name: Some("__tag".into()),
                    ty: TypeName::from("u8"),
                    offset: 0,
                },
                variants: vec![
                    Variant {
                        name: "None".into(),
                        discr: 0,
                        fields: vec![],
                    },
                    Variant {
                        name: "Some".into(),
                        discr: 1,
                        fields: vec![Field {
                            name: "0".into(),
                            ty: TypeName::from("inner::Point"),
                            offset: 8,
                        }],
                    },
                ],
            }),
        },
    );
    let leaves = db.layout("opt::Maybe").expect("layout");
    let got: Vec<_> = leaves
        .iter()
        .map(|f| (f.name.as_str(), f.offset))
        .collect();
    assert_eq!(got, vec![("Some::0::x", 8), ("Some::0::y", 12)]);
}

#[test]
fn remove_recursive_drops_dependents() {
    let mut db = sample();
    assert!(db.get("inner::Point").is_some());
    let removed = db.remove_recursive("inner::Point");
    // inner::Point + outer::Padded + outer::Ref + &inner::Point
    assert!(removed >= 3, "removed {removed}");
    assert!(db.get("inner::Point").is_none());
    assert!(db.get("outer::Padded").is_none());
}
