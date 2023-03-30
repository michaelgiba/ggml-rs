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

        macro_rules! model_io_bincode_config {
            () => { bincode::config::standard().skip_fixed_array_length() };
        }

        impl ggml_rs::io::ModelIO for #name {
            fn read_to_tensor<R: std::io::Read>(ctx: &Context, reader: &mut R) -> Result<ggml_rs::Tensor, ()> {
                match Self::read(ctx, reader) {
                    Ok(serialized) => serialized.to_tensor(ctx),
                    Err(_) => {
                        Err(())
                    }
                }
            }


            fn to_tensor(self, ctx: &Context) -> Result<ggml_rs::Tensor, ()> {
                let config = model_io_bincode_config!();
                let mut buf: Vec<u8> = bincode::encode_to_vec(self, config).unwrap();
                let new_tensor = ctx.new_tensor_1d(ggml_rs::DataType::I8, buf.len() as i32);
                if new_tensor.write_bytes(&buf).is_ok() {
                    Ok(new_tensor)
                } else {
                    Err(())
                }

            }

            fn read<R: std::io::Read>(ctx: &Context, reader: &mut R) -> Result<Self, ()> {
                let config = model_io_bincode_config!();
                let raw_data_type: Result<Self, bincode::error::DecodeError> = bincode::decode_from_std_read(
                    reader, config
                );

                match raw_data_type {
                    Ok(data) => Ok(data),
                    Err(_) => Err(()),
                }

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
        #[derive(Debug, bincode::Decode, bincode::Encode, ggml_rs::io::ModelIO)]
        #input
    };
    output.into()
}
