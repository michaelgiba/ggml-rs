use ggml_rs::*;

#[repr(align(16))]
#[derive(Debug)]
struct ManagedMemory([u8; 512]);


pub fn main() {
    let mut buffer: ManagedMemory = ManagedMemory([0;512]); 
    let ctx = Context::init_managed(&mut buffer.0);
    let tensor_a = ctx.new_tensor_1d(TYPE_I8, 16);
    let tensor_b = ctx.new_tensor_1d(TYPE_I8, 16);

    tensor_a.set_i32(13);
    tensor_b.set_f32(6.0);

    match tensor_a.set_i32_1d(7, 15) {
        Ok(_) => {
            println!("Properly set.")
        },
        Err(_) => {
            println!("Unable to set.")
        }
    }

    println!("{:?}", buffer);
}