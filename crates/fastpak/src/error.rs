use std::{fmt, io};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// Path is empty, contains `..`, NUL, or is longer than 65535 bytes.
    InvalidPath(String),
    DuplicatePath(String),
    NotFound(String),
    /// The archive is structurally invalid (truncated, bad offsets, bad index...).
    Corrupt(&'static str),
    /// Unknown format version or codec id.
    Unsupported(String),
    /// Compression backend failure.
    Codec(String),
    ChecksumMismatch(String),
    /// Asset / archive limit exceeded (4 GiB per asset, `max_asset_size`, ...).
    TooLarge(String),
    /// Bug guard: writer invariants violated.
    Internal(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::InvalidPath(p) => write!(f, "invalid asset path: {p:?}"),
            Error::DuplicatePath(p) => write!(f, "duplicate asset path: {p:?}"),
            Error::NotFound(p) => write!(f, "asset not found: {p:?}"),
            Error::Corrupt(m) => write!(f, "corrupt pak: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Codec(m) => write!(f, "codec error: {m}"),
            Error::ChecksumMismatch(p) => write!(f, "checksum mismatch for {p:?}"),
            Error::TooLarge(m) => write!(f, "too large: {m}"),
            Error::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
