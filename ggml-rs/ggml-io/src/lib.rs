// src/lib.rs

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_attribute]
pub fn model_io(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

    let struct_name = &input.ident;
    let attributes = quote! {
        #[derive(serde::Serialize, serde::Deserialize)]
    };

    let read_write_impl = quote! {
        impl #struct_name {
            // Add your read_from_disk and write_to_disk implementation here
        }
    };

    let output = quote! {
        #attributes
        #input
        #read_write_impl
    };

    output.into()
}
