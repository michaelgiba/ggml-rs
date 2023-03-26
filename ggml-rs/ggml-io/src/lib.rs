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
            fn read(ctx: &Context, path: &str) -> Result<Self, ()> {
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
pub fn model_io(_metadata: proc_macro::TokenStream, input: proc_macro::TokenStream)
                 -> proc_macro::TokenStream {

    let input = parse_macro_input!(input as DeriveInput);

    let output = quote! {


        #[derive(bincode::Decode, bincode::Encode, ggml_rs::io::ModelIO)]
        #input
    };
    output.into()
}