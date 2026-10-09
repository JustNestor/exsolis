fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap();
    let data = std::fs::read(&path).unwrap();
    let (db, _) = tesjit_dwarf::export(&data, &path).unwrap();
    let mut names: Vec<_> = db.iter().map(|(n, d)| format!("{n} size={}", d.size)).collect();
    names.sort();
    for n in names { println!("{n}"); }
}
