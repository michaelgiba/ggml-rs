use crate::context::Context;
pub use ggml_io::{model_io, ModelIO};
use std::io::Error as IoError;
use crate::tensor::Tensor;

pub enum ModelIOError {
    IoError(IoError),
}

pub trait ModelIO: Sized {
    fn read<R: std::io::Read>(ctx: &Context, reader: &mut R) -> Result<Self, ()>;
    fn read_to_tensor<R: std::io::Read>(ctx: &Context, reader: &mut R) -> Result<Tensor, ()>;
    fn write(&self, path: &str) -> Result<(), ModelIOError>;
}
