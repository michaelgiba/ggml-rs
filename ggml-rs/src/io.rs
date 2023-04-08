use crate::context::Context;
use crate::tensor::{Dimension, Tensor};
pub use ggml_io::{static_tensor, ModelIO};
use std::io::Error as IoError;

pub enum ModelIOError {
    IoError(IoError),
}

pub trait ModelIO: Sized {
    fn read<R: std::io::Read>(
        ctx: &Context,
        reader: &mut R,
    ) -> Result<Self, bincode::error::DecodeError>;
    fn to_tensor(
        self,
        ctx: &Context,
        dim: Dimension,
        shape: Vec<Option<usize>>,
    ) -> Result<Tensor, ()>;
    fn read_to_tensor<R: std::io::Read>(
        ctx: &Context,
        reader: &mut R,
        dim: Dimension,
        shape: Vec<Option<usize>>,
    ) -> Result<Tensor, ()>;
    fn write(&self, path: &str) -> Result<(), ModelIOError>;
}
