pub fn main() {}

#[cfg(test)]
mod tests {

    use ggml_rs::io::{static_tensor, ModelIO};
    use ggml_rs::Context;
    use ggml_rs::Dimension;
    use std::fs::File;
    use std::io::Seek;

    #[static_tensor]
    struct EightBitParam(i8);

    #[static_tensor(ggml_datatype = i32, ggml_dim = D1)]
    struct FourByteParam(i32);

    #[static_tensor(ggml_datatype = i8, ggml_dim = D2)]
    struct RectU8Param([[i8; 4]; 8]);

    #[static_tensor(ggml_datatype = i8, ggml_dim = D3)]
    struct CubeU8Param([[[i8; 2]; 2]; 2]);

    // #[static_layer]
    struct BasicStaticLayer {
        a: EightBitParam,
        b: FourByteParam,
        c: RectU8Param,
        d: CubeU8Param,
    }

    const MEMORY_SIZE: usize = 1024 * 512;

    #[repr(align(16))]
    struct ManagedMemory([u8; MEMORY_SIZE]);

    macro_rules! test_file {
        ($fname:expr) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/", $fname)
        };
    }

    #[test]
    fn test_reader_1d() {
        let mut buffer: ManagedMemory = ManagedMemory([0; MEMORY_SIZE]);
        let ctx = Context::init_managed(&mut buffer.0);

        let test_file_path = test_file!("resources/model.bin");
        let mut reader = File::open(&test_file_path).expect("Failed to open file");

        for i in 0..16 {
            assert_eq!(reader.stream_position().unwrap(), i);

            let read_result =
                EightBitParam::read_to_tensor(&ctx, &mut reader, Dimension::D1, vec![None]);
            assert!(read_result.is_ok());
            let tensor = read_result.unwrap();
            assert_eq!(tensor.nbytes(), 1);
        }
        assert!(
            EightBitParam::read_to_tensor(&ctx, &mut reader, Dimension::D1, vec!(None)).is_err()
        );
    }

    #[test]
    fn test_reader_1d_2() {
        let mut buffer: ManagedMemory = ManagedMemory([0; MEMORY_SIZE]);
        let ctx = Context::init_managed(&mut buffer.0);

        let test_file_path = test_file!("resources/model.bin");
        let mut reader = File::open(&test_file_path).expect("Failed to open file");

        for i in 0..4 {
            assert_eq!(reader.stream_position().unwrap(), i * 4);

            let read_result =
                FourByteParam::read_to_tensor(&ctx, &mut reader, Dimension::D1, vec![Some(1)]);

            assert_eq!(read_result.unwrap().nbytes(), 4);
        }
        assert!(
            FourByteParam::read_to_tensor(&ctx, &mut reader, Dimension::D1, vec!(None)).is_err()
        );
    }

    #[test]
    fn test_reader_2d() {
        let mut buffer: ManagedMemory = ManagedMemory([0; MEMORY_SIZE]);
        let ctx = Context::init_managed(&mut buffer.0);

        let test_file_path = test_file!("resources/model64bytes.bin");
        let mut reader = File::open(&test_file_path).expect("Failed to open file");

        for _ in 0..2 {
            let read_result = RectU8Param::read_to_tensor(
                &ctx,
                &mut reader,
                Dimension::D2,
                vec![Some(16), Some(2)],
            );
            assert!(read_result.is_ok());
            let tensor = read_result.unwrap();
            assert_eq!(tensor.nbytes(), 32);
        }
        assert!(
            RectU8Param::read_to_tensor(&ctx, &mut reader, Dimension::D2, vec!(None, None))
                .is_err()
        );
    }

    #[test]
    fn test_reader_3d() {
        let mut buffer: ManagedMemory = ManagedMemory([0; MEMORY_SIZE]);
        let ctx = Context::init_managed(&mut buffer.0);

        let test_file_path = test_file!("resources/model64bytes.bin");
        let mut reader = File::open(&test_file_path).expect("Failed to open file");

        for _ in 0..8 {
            let read_result = CubeU8Param::read_to_tensor(
                &ctx,
                &mut reader,
                Dimension::D3,
                vec![Some(2), Some(2), Some(2)],
            );
            assert!(read_result.is_ok());
            let tensor = read_result.unwrap();
            assert_eq!(tensor.nbytes(), 8);
        }
        assert!(CubeU8Param::read_to_tensor(
            &ctx,
            &mut reader,
            Dimension::D3,
            vec!(None, None, None)
        )
        .is_err());
    }

    #[test]
    fn test_read_layer() {
        // let ctx = Context::init(MEMORY_SIZE);
        // let test_file_path = test_file!("resources/model64bytes.bin");
        // let mut reader = File::open(&test_file_path).expect("Failed to open file");

        // for _ in 0..8 {
        //     let read_result = BasicStaticLayer::read_layer(
        //         &ctx,
        //         &mut reader,
        //         Dimension::D3,
        //         vec![Some(2), Some(2), Some(2)],
        //     );
        //     assert!(read_result.is_ok());
        //     let tensor = read_result.unwrap();
        //     assert_eq!(tensor.nbytes(), 8);
        // }
        // assert!(BasicStaticLayer::read_to_tensor(
        //     &ctx,
        //     &mut reader,
        //     Dimension::D3,
        //     vec!(None, None, None)
        // )
        // .is_err());
    }
}
