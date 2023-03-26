pub fn main() {}

#[cfg(test)]
mod tests {

    use ggml_rs::io::{model_io, ModelIO};
    use ggml_rs::Context;
    use std::fs::File;

    #[model_io]
    struct AttentionWeights {
        k: [u16; 4],
    }

    #[model_io]
    struct ClipWeights {
        attention1: AttentionWeights,
        attention2: AttentionWeights,
        attention3: AttentionWeights,
        attention4: AttentionWeights,
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
    fn test_reader() {
        let mut buffer: ManagedMemory = ManagedMemory([0; MEMORY_SIZE]);
        let ctx = Context::init_managed(&mut buffer.0);

        let test_file_path = test_file!("resources/model.bin");

        println!("{:?}", test_file_path);

        let mut reader = File::open(&test_file_path).expect("Failed to open file");

        match ClipWeights::read_to_tensor(&ctx, &mut reader) {
            Ok(tensor) => {
                print!("{:?}", tensor);
                assert!(true)
            }
            Err(_) => (assert!(false)),
        }
    }
}
