//! Diagnostics: one `CompileError` per problem, miette-rendered with rustc-style
//! source frames.
//!
//! The compiler collects as many errors as it can and reports them together
//! (mod authors fix a batch, not one error per run), which is why callers get a
//! [`Diagnostics`] list instead of a single error.

use std::sync::Arc;

use miette::{Diagnostic, NamedSource, SourceOffset, SourceSpan};
use oxc_span::Span;
use thiserror::Error;

/// A type/compile problem tied to a source span.
#[derive(Error, Diagnostic, Debug)]
pub enum CompileError {
    #[error("{msg}")]
    #[diagnostic(code(tsjit::compile))]
    Spanned {
        msg: String,
        #[source_code]
        src: NamedSource<Arc<str>>,
        #[label("{msg}")]
        span: SourceSpan,
    },

    #[error("{0}")]
    #[diagnostic(code(tsjit::plain))]
    Plain(String),

    #[error("syntax error(s)")]
    #[diagnostic(code(tsjit::syntax))]
    Syntax {
        #[source_code]
        src: NamedSource<Arc<str>>,
        #[related]
        diagnostics: Vec<SyntaxDiag>,
    },
}

/// One parse error, related to [`CompileError::Syntax`].
#[derive(Error, Diagnostic, Debug)]
#[error("{msg}")]
#[diagnostic(code(tsjit::syntax_item))]
pub struct SyntaxDiag {
    pub msg: String,
    #[label]
    pub span: SourceSpan,
}

impl SyntaxDiag {
    /// Wrap this parse error into the grouped syntax diagnostic for `src`.
    pub fn into_syntax_error(self, src: &Src) -> CompileError {
        CompileError::Syntax {
            src: src.clone(),
            diagnostics: vec![self],
        }
    }
}

/// Result of a compile/check run: all diagnostics found (possibly empty).
#[derive(Debug, Default)]
pub struct Diagnostics(pub Vec<CompileError>);

impl Diagnostics {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn ok(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn push(&mut self, err: CompileError) {
        self.0.push(err);
    }

    pub fn into_result<T>(self, value: T) -> Result<T, Self> {
        if self.0.is_empty() { Ok(value) } else { Err(self) }
    }
}

impl From<Vec<CompileError>> for Diagnostics {
    fn from(errors: Vec<CompileError>) -> Self {
        Self(errors)
    }
}

impl From<CompileError> for Diagnostics {
    fn from(err: CompileError) -> Self {
        Self(vec![err])
    }
}

impl IntoIterator for Diagnostics {
    type Item = CompileError;
    type IntoIter = std::vec::IntoIter<CompileError>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl std::fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} compile error(s)", self.0.len())
    }
}

impl std::error::Error for Diagnostics {}

/// Renders through miette: the first diagnostic carries the pretty source
/// frame, the rest are reported as `related`.
impl miette::Diagnostic for Diagnostics {
    fn code(&self) -> Option<Box<dyn std::fmt::Display + '_>> {
        self.0.first().and_then(|e| e.code())
    }

    fn severity(&self) -> Option<miette::Severity> {
        self.0.first().and_then(|e| e.severity())
    }

    fn help(&self) -> Option<Box<dyn std::fmt::Display + '_>> {
        self.0.first().and_then(|e| e.help())
    }

    fn url(&self) -> Option<Box<dyn std::fmt::Display + '_>> {
        self.0.first().and_then(|e| e.url())
    }

    fn source_code(&self) -> Option<&dyn miette::SourceCode> {
        self.0.first().and_then(|e| e.source_code())
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = miette::LabeledSpan> + '_>> {
        self.0.first().and_then(|e| e.labels())
    }

    fn related(&self) -> Option<Box<dyn Iterator<Item = &dyn miette::Diagnostic> + '_>> {
        Some(Box::new(self.0.iter().map(|e| e as &dyn miette::Diagnostic)))
    }
}

/// Everything the compiler knows about where a diagnostic came from.
pub type Src = NamedSource<Arc<str>>;

/// A spanned compile error in `src`.
pub fn err(src: &Src, span: Span, msg: impl Into<String>) -> CompileError {
    CompileError::Spanned {
        msg: msg.into(),
        src: src.clone(),
        span: SourceSpan::new(
            SourceOffset::from(span.start as usize),
            (span.end - span.start) as usize,
        ),
    }
}

/// A span-less error (bad configuration, unsupported host setup, ...).
pub fn plain(msg: impl Into<String>) -> CompileError {
    CompileError::Plain(msg.into())
}
