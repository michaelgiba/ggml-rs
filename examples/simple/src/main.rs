pub fn main() {}

#[cfg(test)]
mod tests {

    use ggml_rs::*;

    const MEMORY_SIZE: usize = 4096;

    #[repr(align(16))]
    #[derive(Debug)]
    struct ManagedMemory([u8; MEMORY_SIZE]);

    fn test_i32_value_setting(tensor: &Tensor) {
        tensor.set_i32(0);
        assert_eq!(tensor.read_data::<i8>().unwrap(), vec![0; 5]);
        assert!(tensor.set_i32_1d(0, 1).is_ok());
        assert!(tensor.set_i32_1d(1, 2).is_ok());
        assert!(tensor.set_i32_1d(2, 3).is_ok());
        assert!(tensor.set_i32_1d(3, 5).is_ok());
        assert!(tensor.set_i32_1d(4, 7).is_ok());
        assert!(tensor.set_i32_1d(5, 7).is_err());
        assert_eq!(tensor.read_data::<i8>().unwrap(), vec![1, 2, 3, 5, 7]);
        assert_eq!(tensor.get_i32_1d(0), 1);
        assert_eq!(tensor.get_i32_1d(1), 2);
        assert_eq!(tensor.get_i32_1d(2), 3);
        assert_eq!(tensor.get_i32_1d(3), 5);
        assert_eq!(tensor.get_i32_1d(4), 7);
    }

    #[cfg(test)]
    fn test_f32_value_setting(tensor: &Tensor) {
        tensor.set_f32(0.0);

        assert_eq!(tensor.read_data::<f32>().unwrap(), vec![0.0; 5]);
        assert!(tensor.set_f32_1d(0, 1 as i8).is_ok());
        assert!(tensor.set_f32_1d(1, 2 as i16).is_ok());
        assert!(tensor.set_f32_1d(2, 3 as i16).is_ok());
        assert!(tensor.set_f32_1d(3, 5 as f32).is_ok());
        assert!(tensor.set_f32_1d(4, 7 as f32).is_ok());
        assert!(tensor.set_f32_1d(5, 7 as f32).is_err());
        assert_eq!(
            tensor.read_data::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 5.0, 7.0]
        );
        assert_eq!(tensor.get_f32_1d(0), 1.0);
        assert_eq!(tensor.get_f32_1d(1), 2.0);
        assert_eq!(tensor.get_f32_1d(2), 3.0);
        assert_eq!(tensor.get_f32_1d(3), 5.0);
        assert_eq!(tensor.get_f32_1d(4), 7.0);
    }

    #[test]
    fn test_managed_memory() {
        let mut buffer: ManagedMemory = ManagedMemory([0; MEMORY_SIZE]);
        let ctx = Context::init_managed(&mut buffer.0);

        let tensor_a = ctx.new_tensor_1d(DataType::I8, 5);
        test_i32_value_setting(&tensor_a);

        let tensor_b = ctx.new_tensor_1d(DataType::F32, 5);
        test_f32_value_setting(&tensor_b);

        println!("{:?}", tensor_a);
        println!("{:?}", tensor_b);
    }

    #[test]
    fn test_internally_managed_memory() {
        let ctx = Context::init(MEMORY_SIZE);
        let tensor_a = ctx.new_tensor_1d(DataType::I8, 5);
        test_i32_value_setting(&tensor_a);
        let tensor_b = ctx.new_tensor_1d(DataType::F32, 5);
        test_f32_value_setting(&tensor_b);
    }
}
