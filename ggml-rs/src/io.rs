use crate::context::Context;
pub use ggml_io::{model_io, ModelIO};
use std::io::Error as IoError;

pub enum ModelIOError {
    IoError(IoError),
}

pub trait ModelIO: Sized {
    fn read(ctx: &Context, path: &str) -> Result<Self, ()>;
    fn write(&self, path: &str) -> Result<(), ModelIOError>;
}
