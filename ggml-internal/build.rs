extern crate bindgen;

use std::env;
use std::path::PathBuf;

pub fn main() {
    let dst = cmake::Config::new("ggml").build();

    let bindings = bindgen::Builder::default()
        .header("ggml/include/ggml/ggml.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!(
        "cargo:rustc-link-search=native={}/lib/static/",
        dst.display()
    );
    println!("cargo:rustc-link-lib=static=ggml");
}
