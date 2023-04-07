pub fn main() {}

#[cfg(test)]
mod tests {

    use ggml_rs::io::{model_io, ModelIO};
    use ggml_rs::Context;
    use ggml_rs::Dimension;
    use std::fs::File;

    #[model_io]
    struct FourU8Parameter {
        k: [u8; 4],
    }

    #[model_io]
    struct LongU8Parameter {
        k: [[u8; 8]; 8],
    }

    const MEMORY_SIZE: usize = 1024 * 512;

    #[repr(align(16))]
    struct ManagedMemory([u8; MEMORY_SIZE]);

    macro_rules! test_file {
        ($fname:expr) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/", $fname) // assumes Linux ('/')!
        };
    }

    #[test]
    fn test_reader_1d() {
        let mut buffer: ManagedMemory = ManagedMemory([0; MEMORY_SIZE]);
        let ctx = Context::init_managed(&mut buffer.0);

        let test_file_path = test_file!("resources/model.bin");
        let mut reader = File::open(&test_file_path).expect("Failed to open file");

        for _ in 0..4 {
            let read_result =
                FourU8Parameter::read_to_tensor(&ctx, &mut reader, Dimension::D1, vec![None]);
            assert!(read_result.is_ok());
            let tensor = read_result.unwrap();
            assert_eq!(tensor.nbytes(), 4);
        }
        assert!(
            FourU8Parameter::read_to_tensor(&ctx, &mut reader, Dimension::D2, vec!(None)).is_err()
        );
    }
}
