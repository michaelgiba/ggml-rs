use crate::context::Context;
use crate::tensor::Tensor;
pub use ggml_io::{model_io, ModelIO};
use std::io::Error as IoError;

pub enum ModelIOError {
    IoError(IoError),
}

pub trait ModelIO: Sized {
    fn read<R: std::io::Read>(ctx: &Context, reader: &mut R) -> Result<Self, ()>;
    fn read_to_tensor<R: std::io::Read>(ctx: &Context, reader: &mut R) -> Result<Tensor, ()>;
    fn write(&self, path: &str) -> Result<(), ModelIOError>;
}
