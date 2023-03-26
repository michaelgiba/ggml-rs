




struct HeaderHyperparameters {
    n_layers: i32,
    n_embd: i32,
    n_head: i32,
    temp: f32,
}



ctx_size, model, params = !read_model_header(HeaderHyperparameters);


for i in 1...{params.n_layers} {
    layer = read_layer!(
        format!("layers.{i}.feed_forward.w3.weight"), 
        POPLayer
    );


}

