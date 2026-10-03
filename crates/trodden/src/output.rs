use std::{
    error::Error,
    fmt::{self, Display},
    io::{self, ErrorKind, Write},
};

use anyhow::{Context, Result};

#[derive(Debug)]
pub(crate) struct Output<W = io::Stdout> {
    writer: W,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Closed;

impl Output {
    pub(crate) fn stdout() -> Self {
        Self::new(io::stdout())
    }
}

impl Output<io::Stderr> {
    pub(crate) fn stderr() -> Self {
        Self::new(io::stderr())
    }
}

impl<W: Write> Output<W> {
    pub(crate) const fn new(writer: W) -> Self {
        Self { writer }
    }

    pub(crate) fn write_fmt(&mut self, text: fmt::Arguments<'_>) -> Result<()> {
        Self::check(self.writer.write_fmt(text))
    }

    pub(crate) fn flush(&mut self) -> Result<()> {
        Self::check(self.writer.flush())
    }

    fn check(result: io::Result<()>) -> Result<()> {
        match result {
            Err(error) if error.kind() == ErrorKind::BrokenPipe => Err(Closed.into()),
            other => other.context("write the output"),
        }
    }
}

impl Closed {
    pub(crate) fn caused(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| cause.is::<Self>())
    }
}

impl Display for Closed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the reader closed the output")
    }
}

impl Error for Closed {}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct Failing(pub(crate) ErrorKind);

#[cfg(test)]
impl Write for Failing {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(self.0.into())
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(self.0.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_reach_the_writer() {
        let mut output = Output::new(Vec::new());

        writeln!(output, "Procedures         {} ({} revisions)", 2, 3).expect("writes");
        writeln!(output).expect("writes");
        output.flush().expect("flushes");

        assert_eq!(output.writer, b"Procedures         2 (3 revisions)\n\n");
    }

    #[test]
    fn a_closed_reader_is_told_apart_from_other_failures() {
        let mut closed = Output::new(Failing(ErrorKind::BrokenPipe));
        let mut full = Output::new(Failing(ErrorKind::StorageFull));

        let closed_write = writeln!(closed, "ID").expect_err("broken pipes fail");
        let closed_flush = closed.flush().expect_err("broken pipes fail");
        let full_write = writeln!(full, "ID").expect_err("full disks fail");

        assert!(Closed::caused(&closed_write));
        assert!(Closed::caused(&closed_flush));
        assert!(Closed::caused(&closed_write.context("list procedures")));
        assert!(!Closed::caused(&full_write));
        assert_eq!(full_write.to_string(), "write the output");
    }
}
