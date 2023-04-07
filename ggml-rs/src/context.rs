use crate::graph::ComputationGraph;
use crate::tensor::{DataType, Tensor};
use crate::Dimension;
use std::{ffi::c_void, ptr::NonNull, sync::Arc};

/// Acts as a RAII-guard over a `ggml_internal::ggml_context`, allocating via
/// ggml_init and dropping via ggml_free
pub struct Context {
    /// An `Arc` is used to model the relation between the context and the
    /// allocated tensors. Tensors are owned by the object, so a [`GgmlTensor`]
    /// contains a `Weak` reference underneath and doesn't let you do anything
    /// with it if the underlying context has been deallocated.
    ptr: Arc<NonNull<ggml_internal::ggml_context>>,
}
impl Context {
    pub fn init(mem_size: usize) -> Self {
        let raw = unsafe {
            ggml_internal::ggml_init(ggml_internal::ggml_init_params {
                mem_size,
                mem_buffer: std::ptr::null_mut(),
            })
        };
        Self {
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }

    pub fn init_managed(mem: &mut [u8]) -> Self {
        let raw = unsafe {
            ggml_internal::ggml_init(ggml_internal::ggml_init_params {
                mem_size: mem.len() * std::mem::size_of::<u8>(),
                mem_buffer: mem.as_mut_ptr() as *mut c_void,
            })
        };
        Self {
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }

    fn new_tensor_raw(
        &self,
        raw: *mut ggml_internal::ggml_tensor,
        dim: Dimension,
        shape: [usize; 4],
    ) -> Tensor {
        Tensor {
            ptr: NonNull::new(raw).expect("Should not be null"),
            ctx: Arc::downgrade(&self.ptr),
            dim: dim,
            shape: shape,
        }
    }

    pub fn new_tensor_1d(&self, dtype: DataType, ne0: usize) -> Tensor {
        let raw = unsafe {
            ggml_internal::ggml_new_tensor_1d(self.ptr.as_ptr(), dtype.into(), ne0 as i32)
        };
        self.new_tensor_raw(raw, Dimension::D1, [ne0, 1, 1, 1])
    }

    pub fn new_tensor_2d(&self, dtype: DataType, ne0: usize, ne1: usize) -> Tensor {
        let raw = unsafe {
            ggml_internal::ggml_new_tensor_2d(
                self.ptr.as_ptr(),
                dtype.into(),
                ne0 as i32,
                ne1 as i32,
            )
        };
        self.new_tensor_raw(raw, Dimension::D2, [ne0, ne1, 1, 1])
    }

    pub fn new_tensor_3d(&self, dtype: DataType, ne0: usize, ne1: usize, ne2: usize) -> Tensor {
        let raw = unsafe {
            ggml_internal::ggml_new_tensor_3d(
                self.ptr.as_ptr(),
                dtype.into(),
                ne0 as i32,
                ne1 as i32,
                ne2 as i32,
            )
        };
        self.new_tensor_raw(raw, Dimension::D3, [ne0, ne1, ne2, 1])
    }

    pub fn new_f32(&self, x: f32) -> Tensor {
        let raw = unsafe { ggml_internal::ggml_new_f32(self.ptr.as_ptr(), x) };
        self.new_tensor_raw(raw, Dimension::Scalar, [1, 1, 1, 1])
    }

    pub fn op_get_rows(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe {
            ggml_internal::ggml_get_rows(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr())
        };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_norm(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_internal::ggml_norm(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_mul(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_repeat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe {
            ggml_internal::ggml_repeat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr())
        };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_mul_mat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe {
            ggml_internal::ggml_mul_mat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr())
        };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_add(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_scale(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_scale(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_diag_mask_inf(&self, a: &Tensor, n_past: i32) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_diag_mask_inf(self.ptr.as_ptr(), a.ptr.as_ptr(), n_past) };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_soft_max(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_internal::ggml_soft_max(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_view_1d(&self, a: &Tensor, ne0: i32, offset: usize) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_view_1d(self.ptr.as_ptr(), a.ptr.as_ptr(), ne0, offset) };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_cpy(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_cpy(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_permute(&self, a: &Tensor, axis0: i32, axis1: i32, axis2: i32, axis3: i32) -> Tensor {
        let tensor = unsafe {
            ggml_internal::ggml_permute(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                axis0,
                axis1,
                axis2,
                axis3,
            )
        };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }
    pub fn op_reshape_3d(&self, a: &Tensor, ne0: i32, ne1: i32, ne2: i32) -> Tensor {
        let tensor = unsafe {
            ggml_internal::ggml_reshape_3d(self.ptr.as_ptr(), a.ptr.as_ptr(), ne0, ne1, ne2)
        };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn op_rope(&self, a: &Tensor, npast: i32, ndims: i32, mode: i32) -> Tensor {
        let tensor = unsafe {
            ggml_internal::ggml_rope(self.ptr.as_ptr(), a.ptr.as_ptr(), npast, ndims, mode)
        };
        self.new_tensor_raw(tensor, a.dim.clone(), a.shape) // WARNING: wrong.
    }

    pub fn graph_compute(&self, graph: &mut ComputationGraph) {
        unsafe {
            ggml_internal::ggml_graph_compute(self.ptr.as_ptr(), &mut graph.inner);
        }
    }

    pub fn used_mem(&self) -> usize {
        unsafe { ggml_internal::ggml_used_mem(self.ptr.as_ptr()) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: The only non-weak copy of ptr is no longer accessible after
        // this drop call.
        unsafe {
            ggml_internal::ggml_free(self.ptr.as_ptr());
        }
    }
}
