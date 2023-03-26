use bincode::Error as BincodeError;
use std::io::Error as IoError;

pub enum ModelIOError {
    IoError(IoError),
    BincodeError(BincodeError),
}

pub trait ModelIO: Sized {
    fn read_from_disk(path: &str) -> Result<Self, ModelIOError>;
    fn write_to_disk(&self, path: &str) -> Result<(), ModelIOError>;
}
