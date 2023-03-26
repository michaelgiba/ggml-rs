pub fn main() {}

#[cfg(test)]
mod tests {

    use ggml_rs::io::{model_io, ModelIO};
    use ggml_rs::Context;

    #[model_io]
    struct AttentionLayer {
        k: [u32; 64],
    }

    #[model_io]
    struct ClipModel {
        attention1: AttentionLayer,
        attention2: AttentionLayer,
        attention3: AttentionLayer,
    }

    const MEMORY_SIZE: usize = 1024 * 512;

    #[repr(align(16))]
    struct ManagedMemory([u8; MEMORY_SIZE]);

    #[test]
    fn test_managed_memory() {
        let mut buffer: ManagedMemory = ManagedMemory([0; MEMORY_SIZE]);
        let ctx = Context::init_managed(&mut buffer.0);

        match ClipModel::read(&ctx, "test") {
            Ok(model) => assert!(model.write("output.txt").is_ok()),
            Err(_) => (assert!(false)),
        }
    }
}
