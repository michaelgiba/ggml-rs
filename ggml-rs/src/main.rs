
use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Weak},
};

pub use ggml_internal::ggml_type as Type;

// pub const TYPE_Q4_0: ggml_internal::ggml_type = ggml_internal::ggml_type_GGML_TYPE_Q4_0;
// pub const TYPE_Q4_1: ggml_internal::ggml_type = ggml_internal::ggml_type_GGML_TYPE_Q4_1;
pub const TYPE_I8: ggml_internal::ggml_type = ggml_internal::ggml_type_GGML_TYPE_I8;
pub const TYPE_I16: ggml_internal::ggml_type = ggml_internal::ggml_type_GGML_TYPE_I16;
pub const TYPE_I32: ggml_internal::ggml_type = ggml_internal::ggml_type_GGML_TYPE_I32;
pub const TYPE_F16: ggml_internal::ggml_type = ggml_internal::ggml_type_GGML_TYPE_F16;
pub const TYPE_F32: ggml_internal::ggml_type = ggml_internal::ggml_type_GGML_TYPE_F32;
pub const TYPE_COUNT: ggml_internal::ggml_type = ggml_internal::ggml_type_GGML_TYPE_COUNT;

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
                // Null here means we want ggml to own this memory. We don't
                // support passing an owned buffer from the Rust side.
                mem_buffer: std::ptr::null_mut(),
            })
        };
        Self {
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }

    fn new_tensor_raw(&self, raw: *mut ggml_internal::ggml_tensor) -> Tensor {
        Tensor {
            ptr: NonNull::new(raw).expect("Should not be null"),
            ctx: Arc::downgrade(&self.ptr),
        }
    }

    pub fn new_tensor_1d(&self, typ: ggml_internal::ggml_type, ne0: i32) -> Tensor {
        let raw = unsafe { ggml_internal::ggml_new_tensor_1d(self.ptr.as_ptr(), typ, ne0) };
        self.new_tensor_raw(raw)
    }

    pub fn new_tensor_2d(&self, typ: ggml_internal::ggml_type, ne0: i32, ne1: i32) -> Tensor {
        let raw = unsafe { ggml_internal::ggml_new_tensor_2d(self.ptr.as_ptr(), typ, ne0, ne1) };
        self.new_tensor_raw(raw)
    }

    pub fn new_tensor_3d(&self, typ: ggml_internal::ggml_type, ne0: i32, ne1: i32, ne2: i32) -> Tensor {
        let raw = unsafe { ggml_internal::ggml_new_tensor_3d(self.ptr.as_ptr(), typ, ne0, ne1, ne2) };
        self.new_tensor_raw(raw)
    }

    pub fn new_f32(&self, x: f32) -> Tensor {
        let raw = unsafe { ggml_internal::ggml_new_f32(self.ptr.as_ptr(), x) };
        self.new_tensor_raw(raw)
    }

    pub fn op_get_rows(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_get_rows(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_norm(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_internal::ggml_norm(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_mul(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_repeat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_repeat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_mul_mat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_mul_mat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_add(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    // pub fn op_silu(&self, a: &Tensor) -> Tensor {
    //     let tensor = unsafe { ggml_internal::ggml_silu(self.ptr.as_ptr(), a.ptr.as_ptr()) };
    //     self.new_tensor_raw(tensor)
    // }

    pub fn op_scale(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_scale(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_diag_mask_inf(&self, a: &Tensor, n_past: i32) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_diag_mask_inf(self.ptr.as_ptr(), a.ptr.as_ptr(), n_past) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_soft_max(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_internal::ggml_soft_max(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_view_1d(&self, a: &Tensor, ne0: i32, offset: usize) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_view_1d(self.ptr.as_ptr(), a.ptr.as_ptr(), ne0, offset) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_cpy(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_cpy(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
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
        self.new_tensor_raw(tensor)
    }
    pub fn op_reshape_3d(&self, a: &Tensor, ne0: i32, ne1: i32, ne2: i32) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_reshape_3d(self.ptr.as_ptr(), a.ptr.as_ptr(), ne0, ne1, ne2) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_rope(&self, a: &Tensor, npast: i32, ndims: i32, mode: i32) -> Tensor {
        let tensor =
            unsafe { ggml_internal::ggml_rope(self.ptr.as_ptr(), a.ptr.as_ptr(), npast, ndims, mode) };
        self.new_tensor_raw(tensor)
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

/// Tensors are owned by the context. A tensor is alive as long as the
/// underlying context it was created with is alive.
pub struct Tensor {
    ptr: NonNull<ggml_internal::ggml_tensor>,
    ctx: Weak<NonNull<ggml_internal::ggml_context>>,
}

impl Tensor {
    /// Creates a shared copy of this tensor pointer.
    pub fn share(&self) -> Self {
        Tensor {
            ptr: self.ptr,
            ctx: Weak::clone(&self.ctx),
        }
    }

    fn with_alive_ctx<U>(&self, f: impl Fn() -> U) -> U {
        if let Some(_ctx) = self.ctx.upgrade() {
            f()
        } else {
            panic!("Using a tensor after the context was dropped")
        }
    }

    pub fn nbytes(&self) -> usize {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_internal::ggml_nbytes(self.ptr.as_ptr()) }
        })
    }

    pub fn data(&self) -> *mut c_void {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { *self.ptr.as_ptr() }.data
        })
    }

    pub fn nelements(&self) -> i32 {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_internal::ggml_nelements(self.ptr.as_ptr()) }
        })
    }

    pub fn get_ne(&self) -> [i32; 4] {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.ne)
    }

    pub fn get_nb(&self) -> [usize; 4] {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.nb)
    }

    pub fn get_type(&self) -> ggml_internal::ggml_type {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.type_)
    }

    pub fn element_size(&self) -> usize {
        self.with_alive_ctx(|| unsafe { ggml_internal::ggml_element_size(self.ptr.as_ptr()) })
    }

    pub unsafe fn write_data(&self, src: &[u8]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.data() as *mut u8, src.len())
    }

    pub unsafe fn read_data(&self, offset: usize, dst: &mut [u8]) {
        let data = unsafe { ggml_internal::ggml_get_data(self.ptr.as_ptr()).add(offset) };
        std::ptr::copy_nonoverlapping(data, dst as *mut _ as _, dst.len())
    }
}

pub struct ComputationGraph {
    inner: ggml_internal::ggml_cgraph,
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


pub fn main() {
    let ctx = Context::init(1024 * 1024);
}