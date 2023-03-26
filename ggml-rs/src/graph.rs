use crate::tensor::Tensor;
pub struct ComputationGraph {
    pub(crate) inner: ggml_internal::ggml_cgraph,
}

impl ComputationGraph {
    pub fn new(n_threads: i32) -> Self {
        Self {
            inner: ggml_internal::ggml_cgraph {
                n_threads,
                // SAFETY: This should be safe to zero. The original C++ impl
                // just leaves it uninitialized
                ..unsafe { std::mem::zeroed::<ggml_internal::ggml_cgraph>() }
            },
        }
    }

    pub fn build_forward_expand(&mut self, tensor: &Tensor) {
        unsafe { ggml_internal::ggml_build_forward_expand(&mut self.inner, tensor.ptr.as_ptr()) }
    }
}
