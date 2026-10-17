//! Shared memory layouts for host-provided ECS query inputs.

/// Version of the query descriptor ABI shared by the compiler and host.
pub const QUERY_DESC_VERSION: u32 = 1;

/// Header passed as the query parameter's machine-word value.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QueryInput {
    pub chunks_addr: u64,
    pub chunk_count: u64,
}

const _: () = assert!(std::mem::size_of::<usize>() == std::mem::size_of::<u64>());
const _: () = assert!(std::mem::size_of::<QueryInput>() == 16);

/// Number of u64 words in one chunk descriptor.
pub const fn desc_words(n_terms: usize) -> usize {
    1 + n_terms
}

/// Append one chunk descriptor: `[row_count, col_0, ..., col_n]`.
pub fn write_chunk(out: &mut Vec<u64>, rows: u64, cols: &[*mut u8]) {
    out.push(rows);
    out.extend(cols.iter().map(|ptr| *ptr as usize as u64));
}

/// Append one chunk descriptor from already-normalized pointer addresses.
pub fn write_chunk_addrs(out: &mut Vec<u64>, rows: u64, cols: &[u64]) {
    out.push(rows);
    out.extend_from_slice(cols);
}
