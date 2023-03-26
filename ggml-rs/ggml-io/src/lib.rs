// src/lib.rs

extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(ModelIO)]
pub fn derive_model_io(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let gen = quote! {
        impl ggml_rs::io::ModelIO for #name {
            fn read_to_tensor<R: std::io::Read>(ctx: &Context, reader: &mut R) -> Result<ggml_rs::Tensor, ()> {
                // let new_tensor = ctx.new_tensor_1d(ggml_rs::DataType::I8, 2);
                // let mut local_buf = [0u8; 2];
                let raw_data_type: Result<Self, bincode::error::DecodeError> = bincode::decode_from_std_read(
                    &mut reader,     bincode::config::standard()
                );
                println!("{:?}", raw_data_type);
                // reader.read_exact(&mut local_buf);
                // new_tensor.write_bytes(&local_buf);
                Err(())
            }


            fn read<R: std::io::Read>(ctx: &Context, reader: &mut R) -> Result<Self, ()> {
                Err(())
            }

            fn write(&self, path: &str) -> Result<(), ggml_rs::io::ModelIOError> {
                Ok(())
            }
        }
    };
    gen.into()
}

#[proc_macro_attribute]
pub fn model_io(
    _metadata: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let output = quote! {
        #[derive(bincode::Decode, bincode::Encode, ggml_rs::io::ModelIO)]
        #input
    };
    output.into()
}
