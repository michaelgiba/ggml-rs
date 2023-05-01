// src/lib.rs

extern crate proc_macro;
use proc_macro2;
use quote::quote;
use syn::{parse2, parse_macro_input, Attribute, DeriveInput, Lit, MetaNameValue};

fn filter_tensor_params_attributes(attrs: &[Attribute]) -> Vec<&Attribute> {
    attrs
        .iter()
        .filter(|attr| attr.path.is_ident("tensor_params"))
        .collect()
}

fn parse_meta_name_value(attr: &Attribute) -> MetaNameValue {
    attr.parse_args().unwrap()
}

fn get_string_value(lit: &Lit) -> String {
    match lit {
        Lit::Str(string_val) => string_val.value().into(),
        _ => panic!("Unknown value provided in macro."),
    }
}

fn get_ggml_dtype(attr: &Attribute) -> proc_macro2::TokenStream {
    let datatype_name_value = parse_meta_name_value(&attr);
    let tensor_type_string = get_string_value(&datatype_name_value.lit);

    match tensor_type_string.as_str() {
        "i8" => quote! { ggml_rs::DataType::I8 },
        "i16" => quote! { ggml_rs::DataType::I16 },
        "i32" => quote! { ggml_rs::DataType::I32 },
        "f16" => quote! { ggml_rs::DataType::F16 },
        "f32" => quote! { ggml_rs::DataType::F32 },
        "count" => quote! { ggml_rs::DataType::COUNT },
        _ => panic!("Invalid datatype provided."),
    }
}

fn get_ggml_dim(attr: &Attribute) -> proc_macro2::TokenStream {
    let dim_name_value = parse_meta_name_value(attr);
    let tensor_dim_string = get_string_value(&dim_name_value.lit);

    match tensor_dim_string.as_str() {
        "D1" => quote! { ggml_rs::Dimension::D1 },
        "D2" => quote! { ggml_rs::Dimension::D2 },
        "D3" => quote! { ggml_rs::Dimension::D3 },
        _ => panic!("Invalid dim provided."),
    }
}

fn derive_model_io_impl(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input: DeriveInput = parse2(input).unwrap();
    let name = &input.ident;

    let attr = filter_tensor_params_attributes(&input.attrs);
    let ggml_dtype = get_ggml_dtype(&attr[0]);
    let ggml_dim = get_ggml_dim(&attr[1]);

    quote! {
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
                let new_tensor = match #ggml_dim {
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
    }
}

#[proc_macro_derive(ModelIO, attributes(tensor_params))]
pub fn derive_model_io(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_model_io_impl(input.into()).into()
}

fn fetch_static_key_value_pair(attr_terms: &Vec<String>, key: &str) -> Option<String> {
    let pos = attr_terms
        .windows(3)
        .position(|window| window.len() == 3 && window[0] == key && window[1] == "=");

    pos.map(|index| attr_terms[index + 2].clone())
}

fn fetch_static_tensor_dim(attr_terms: &Vec<String>) -> String {
    fetch_static_key_value_pair(attr_terms, "ggml_dim").unwrap_or("D1".into())
}

fn fetch_static_tensor_datatype(attr_terms: &Vec<String>) -> String {
    fetch_static_key_value_pair(attr_terms, "ggml_datatype").unwrap_or("i8".into())
}

#[proc_macro_attribute]
pub fn static_tensor(
    metadata: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let attr_terms: Vec<String> = metadata.into_iter().map(|x| x.to_string()).collect();
    let datatype = fetch_static_tensor_datatype(&attr_terms);
    let dim = fetch_static_tensor_dim(&attr_terms);

    let output = quote! {
        #[derive(Debug, bincode::Decode, bincode::Encode, ggml_rs::io::ModelIO)]
        #[tensor_params(datatype=#datatype)]
        #[tensor_params(dim=#dim)]
        #input
    };
    output.into()
}
