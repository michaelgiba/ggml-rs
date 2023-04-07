use std::{ffi::c_void, ptr::NonNull, sync::Weak};

#[repr(u32)]
pub enum DataType {
    I8 = ggml_internal::ggml_type_GGML_TYPE_I8,
    I16 = ggml_internal::ggml_type_GGML_TYPE_I16,
    I32 = ggml_internal::ggml_type_GGML_TYPE_I32,
    F16 = ggml_internal::ggml_type_GGML_TYPE_F16,
    F32 = ggml_internal::ggml_type_GGML_TYPE_F32,
    COUNT = ggml_internal::ggml_type_GGML_TYPE_COUNT,
}

impl Into<u32> for DataType {
    fn into(self) -> u32 {
        self as u32
    }
}

impl From<u32> for DataType {
    fn from(value: u32) -> Self {
        match value {
            ggml_internal::ggml_type_GGML_TYPE_I8 => DataType::I8,
            ggml_internal::ggml_type_GGML_TYPE_I16 => DataType::I16,
            ggml_internal::ggml_type_GGML_TYPE_I32 => DataType::I32,
            ggml_internal::ggml_type_GGML_TYPE_F16 => DataType::F16,
            ggml_internal::ggml_type_GGML_TYPE_F32 => DataType::F32,
            ggml_internal::ggml_type_GGML_TYPE_COUNT => DataType::COUNT,
            _ => panic!("Invalid ggml type value {}.", value),
        }
    }
}

#[derive(Clone)]
pub enum Dimension {
    Scalar,
    D1,
    D2,
    D3,
}

/// Tensors are owned by the context. A tensor is alive as long as the
/// underlying context it was created with is alive.
pub struct Tensor {
    pub(crate) ptr: NonNull<ggml_internal::ggml_tensor>,
    pub(crate) ctx: Weak<NonNull<ggml_internal::ggml_context>>,
    pub(crate) dim: Dimension,
    pub(crate) shape: [usize; 4],
}

impl Tensor {
    /// Creates a shared copy of this tensor pointer.
    pub fn share(&self) -> Self {
        Tensor {
            ptr: self.ptr,
            ctx: Weak::clone(&self.ctx),
            dim: self.dim.clone(),
            shape: self.shape,
        }
    }

    pub fn raw_weak(&self) -> Weak<NonNull<ggml_internal::ggml_tensor>> {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_cazll guarantees the context is alive
            unsafe { Weak::from_raw(&self.ptr) }
        })
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

    pub fn nelements(&self) -> usize {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            (unsafe { ggml_internal::ggml_nelements(self.ptr.as_ptr()) }) as usize
        })
    }

    pub fn set_i32<T: Into<i32> + Copy>(&self, value: T) {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_internal::ggml_set_i32(self.ptr.as_ptr(), value.into()) }
        });
    }

    pub fn set_f32<T: Into<f32> + Copy>(&self, value: T) {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_internal::ggml_set_f32(self.ptr.as_ptr(), value.into()) }
        });
    }

    pub fn set_i32_1d<T: Into<i32> + Copy>(&self, idx: usize, value: T) -> Result<(), ()> {
        self.with_alive_ctx(|| {
            if self.nelements() <= idx {
                Err(())
            } else {
                // SAFETY: The with_alive_call guarantees the context is alive
                unsafe {
                    ggml_internal::ggml_set_i32_1d(self.ptr.as_ptr(), idx as i32, value.into())
                };
                Ok(())
            }
        })
    }

    pub fn set_f32_1d<T: Into<f32> + Copy>(&self, idx: usize, value: T) -> Result<(), ()> {
        self.with_alive_ctx(|| {
            if self.nelements() <= idx {
                Err(())
            } else {
                // SAFETY: The with_alive_call guarantees the context is alive
                unsafe {
                    ggml_internal::ggml_set_f32_1d(self.ptr.as_ptr(), idx as i32, value.into())
                };
                Ok(())
            }
        })
    }

    pub fn get_i32_1d(&self, i: i32) -> i32 {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_internal::ggml_get_i32_1d(self.ptr.as_ptr(), i) }
        })
    }

    pub fn get_f32_1d(&self, i: i32) -> f32 {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_internal::ggml_get_f32_1d(self.ptr.as_ptr(), i) }
        })
    }

    pub fn get_ne(&self) -> [i32; 4] {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.ne)
    }

    pub fn get_nb(&self) -> [usize; 4] {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.nb)
    }

    pub fn get_type(&self) -> DataType {
        self.with_alive_ctx(|| unsafe { (*self.ptr.as_ptr()).type_.into() })
    }

    pub fn element_size(&self) -> usize {
        self.with_alive_ctx(|| unsafe { ggml_internal::ggml_element_size(self.ptr.as_ptr()) })
    }

    /// # Safety
    /// Caller should ensure bounds are checked or use `set_*` functions    
    unsafe fn write_data_raw(&self, src: &[u8]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.data() as *mut u8, src.len())
    }

    /// # Safety
    /// Caller should `dst` length is valid for tensor.
    unsafe fn read_data_raw<T>(&self, offset: usize, count: usize) -> &[T] {
        let data = ggml_internal::ggml_get_data(self.ptr.as_ptr()).add(offset);
        std::slice::from_raw_parts(data as *mut _ as _, count)
    }

    pub fn read_elements<T>(&self, offset: usize, count: usize) -> Result<&[T], ()> {
        let byte_offset = std::mem::size_of::<T>() * offset;
        let num_bytes = std::mem::size_of::<T>() * count;

        if byte_offset + num_bytes > self.nbytes() {
            Err(())
        } else {
            unsafe { Ok(self.read_data_raw::<T>(byte_offset, count)) }
        }
    }

    pub fn read_data<T: Clone>(&self) -> Result<&[T], ()> {
        self.read_elements::<T>(0, self.nelements())
    }

    pub fn write_bytes(&self, src: &[u8]) -> Result<(), ()> {
        if self.nbytes() < src.len() {
            Err(())
        } else {
            unsafe {
                self.write_data_raw(src);
            }
            Ok(())
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Tensor");

        match self.get_type() {
            DataType::I8 => debug_struct.field("data", &self.read_data::<i8>()),
            DataType::I16 => debug_struct.field("data", &self.read_data::<i16>()),
            DataType::I32 => debug_struct.field("data", &self.read_data::<i32>()),
            DataType::F16 => debug_struct.field("data", &self.read_data::<f32>()), // warning, need f16
            DataType::F32 => debug_struct.field("data", &self.read_data::<f32>()),
            DataType::COUNT => debug_struct.field("data", &self.read_data::<usize>()),
        }
        .field("ptr", &self.ptr)
        .field("ctx", &self.ctx)
        .finish()
    }
}
