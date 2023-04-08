// src/lib.rs

extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{quote};
use syn::Lit::Str;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(ModelIO, attributes(tensor_params))]
pub fn derive_model_io(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let attr: Vec<_> = input
        .attrs
        .iter()
        .filter(|attr| attr.path.is_ident("tensor_params"))
        .collect();

    let datatype_name_value: syn::MetaNameValue = attr[0].parse_args().unwrap();

    let tensor_type_string = match datatype_name_value.lit {
        Str(string_val) => string_val.value(),
        _ => panic!("Unknown value provided to datatype in macro."),
    };

    let ggml_dtype = match tensor_type_string.as_str() {
        "i8" => quote! { ggml_rs::DataType::I8 },
        "i16" => quote! { ggml_rs::DataType::I16 },
        "i32" => quote! { ggml_rs::DataType::I32 },
        "f16" => quote! { ggml_rs::DataType::F16 },
        "f32" => quote! { ggml_rs::DataType::F32 },
        "count" => quote! { ggml_rs::DataType::COUNT },
        _ => panic!("Invalid datatype provided."),
    };

    let gen = quote! {

        macro_rules! model_io_bincode_config {
            () => { bincode::config::standard().skip_fixed_array_length().with_fixed_int_encoding() };
        }

        impl ggml_rs::io::ModelIO for #name {
            fn read_to_tensor<R: std::io::Read>(
                ctx: &Context,
                reader: &mut R,
                dim: Dimension,
                shape: Vec<Option<usize>>
            ) -> Result<ggml_rs::Tensor, ()> {
                match Self::read(ctx, reader) {
                    Ok(serialized) => {
                        serialized.to_tensor(ctx, dim, shape)
                    },
                    Err(_) => {
                        Err(())
                    }
                }
            }


            fn to_tensor(
                self,
                ctx: &Context,
                dim: Dimension,
                shape: Vec<Option<usize>>
            ) -> Result<ggml_rs::Tensor, ()> {
                let config = model_io_bincode_config!();
                let mut buf: Vec<u8> = bincode::encode_to_vec(self, config).unwrap();
                let new_tensor = match dim {
                    ggml_rs::Dimension::Scalar => ctx.new_f32(0.0),
                    ggml_rs::Dimension::D1 => ctx.new_tensor_1d(
                        #ggml_dtype,
                        shape[0].unwrap_or(buf.len())
                    ),
                    ggml_rs::Dimension::D2 => ctx.new_tensor_2d(
                        #ggml_dtype,
                        shape[0].unwrap_or(buf.len()),
                        shape[1].unwrap_or(1), // WRONG - properly infer shapes
                    ),
                    ggml_rs::Dimension::D3 => ctx.new_tensor_3d(
                        #ggml_dtype,
                        shape[0].unwrap_or(buf.len()),
                        shape[1].unwrap_or(1), // WRONG - properly infer shapes
                        shape[2].unwrap_or(1), // WRONG - properly infer shapes
                    ),
                };

                if new_tensor.write_bytes(&buf).is_ok() {
                    Ok(new_tensor)
                } else {
                    Err(())
                }

            }

            fn read<R: std::io::Read>(
                ctx: &Context,
                reader: &mut R
            ) -> Result<Self, bincode::error::DecodeError> {
                let config = model_io_bincode_config!();
                bincode::decode_from_std_read(reader, config)
            }

            fn write(&self, path: &str) -> Result<(), ggml_rs::io::ModelIOError> {
                Ok(())
            }
        }
    };
    gen.into()
}

fn fetch_static_tensor_datatype(attr_terms: Vec<String>) -> String {
    match attr_terms.len() {
        0 => String::from("i8"),
        3 => {
            if attr_terms[0] == "ggml_datatype" && attr_terms[1] == "=" {
                String::from(attr_terms[2].clone())
            } else {
                panic!("Invalid attribute provided to macro.")
            }
        }
        _ => panic!("Invalid attribute provided to macro."),
    }
}

#[proc_macro_attribute]
pub fn static_tensor(
    metadata: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let attr_terms: Vec<String> = metadata.into_iter().map(|x| x.to_string()).collect();
    let datatype = fetch_static_tensor_datatype(attr_terms);

    let output = quote! {
        #[derive(Debug, bincode::Decode, bincode::Encode, ggml_rs::io::ModelIO)]
        #[tensor_params(datatype=#datatype)]
        #input
    };
    output.into()
}
