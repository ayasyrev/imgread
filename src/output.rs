//! Output storage: the Rust API keeps a Vec; the Python JPEG path owns a NumPy array.
#[cfg(feature = "turbojpeg")]
use crate::error::ImgReadError;
use image::DynamicImage;
#[cfg(feature = "turbojpeg")]
use numpy::PyArray3;
#[cfg(feature = "turbojpeg")]
use pyo3::prelude::*;

/// Shared decoding policy with separate ownership of its initialized pixels.
pub(crate) trait RgbOutput: Sized {
    #[cfg(feature = "turbojpeg")]
    type Buffer: RgbBuffer;

    #[cfg(feature = "turbojpeg")]
    fn allocate(width: u32, height: u32, size: usize) -> Result<Self::Buffer, ImgReadError>;

    /// # Safety
    /// Every byte in `buffer`'s checked RGB region must have been initialized.
    #[cfg(feature = "turbojpeg")]
    unsafe fn finish(
        width: u32,
        height: u32,
        size: usize,
        buffer: Self::Buffer,
    ) -> Result<Self, ImgReadError>;

    fn from_image(image: DynamicImage) -> Self;
}

/// Private, exclusively owned storage for an in-progress native decode.
///
/// # Safety
/// `data()` must address at least the requested allocation size in writable
/// storage that remains valid until self is dropped. No other access to those
/// bytes is allowed while decoding, and dropping it must not read its contents.
#[cfg(feature = "turbojpeg")]
pub(crate) unsafe trait RgbBuffer: Send {
    fn data(&mut self) -> *mut u8;
}

// SAFETY: a reserved Vec keeps its allocation until drop; len remains zero
// until the codec initializes the full output. Its ownership is exclusive.
#[cfg(feature = "turbojpeg")]
unsafe impl RgbBuffer for Vec<u8> {
    fn data(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}

impl RgbOutput for DynamicImage {
    #[cfg(feature = "turbojpeg")]
    type Buffer = Vec<u8>;

    #[cfg(feature = "turbojpeg")]
    fn allocate(_width: u32, _height: u32, size: usize) -> Result<Vec<u8>, ImgReadError> {
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(size)
            .map_err(|e| ImgReadError::LimitExceeded(format!("allocation failed: {e}")))?;
        Ok(pixels)
    }

    #[cfg(feature = "turbojpeg")]
    unsafe fn finish(
        width: u32,
        height: u32,
        size: usize,
        mut pixels: Vec<u8>,
    ) -> Result<Self, ImgReadError> {
        // SAFETY: caller guarantees all size bytes were initialized by the codec.
        unsafe { pixels.set_len(size) };
        crate::decoder::rgb_image(width, height, pixels)
    }

    fn from_image(image: DynamicImage) -> Self {
        image
    }
}

#[derive(Debug)]
pub(crate) enum NumpyImage {
    #[cfg(feature = "turbojpeg")]
    Array(Py<PyArray3<u8>>),
    Image(DynamicImage),
}

#[cfg(feature = "turbojpeg")]
pub(crate) struct NumpyBuffer {
    array: Py<PyArray3<u8>>,
    data: std::ptr::NonNull<u8>,
}

// SAFETY: this object exclusively owns an unexposed u8 NumPy array. The strong
// Py reference keeps its data allocation alive across GIL release; only the
// synchronous decoder accesses it. It is deliberately not Sync.
#[cfg(feature = "turbojpeg")]
unsafe impl Send for NumpyBuffer {}

// SAFETY: PyArray_Empty allocated the checked H*W*3 region. No Python caller can
// obtain the array before finish, and dropping an uninitialized u8 array is safe.
#[cfg(feature = "turbojpeg")]
unsafe impl RgbBuffer for NumpyBuffer {
    fn data(&mut self) -> *mut u8 {
        self.data.as_ptr()
    }
}

impl RgbOutput for NumpyImage {
    #[cfg(feature = "turbojpeg")]
    type Buffer = NumpyBuffer;

    #[cfg(feature = "turbojpeg")]
    fn allocate(width: u32, height: u32, size: usize) -> Result<NumpyBuffer, ImgReadError> {
        use crate::limits::arithmetic_limit;
        use numpy::{npyffi, PyArrayMethods, PyUntypedArrayMethods, PY_ARRAY_API};

        let mut dims = [
            npyffi::npy_intp::try_from(height).map_err(|_| arithmetic_limit())?,
            npyffi::npy_intp::try_from(width).map_err(|_| arithmetic_limit())?,
            3,
        ];
        Python::attach(|py| {
            let allocation_error = |err: PyErr| {
                // Match the existing fallible Vec allocation contract: resource
                // errors are terminal and must never trigger a fallback decode.
                ImgReadError::LimitExceeded(format!("output allocation failed: {err}"))
            };
            // SAFETY: GIL held; dims live for the synchronous call; u8 descriptor
            // is a new reference stolen by PyArray_Empty, including on failure.
            // No uninitialized element is read or exposed as a Rust reference.
            let array = unsafe {
                let dtype =
                    PY_ARRAY_API.PyArray_DescrFromType(py, npyffi::NPY_TYPES::NPY_UBYTE as _);
                if dtype.is_null() {
                    return Err(allocation_error(PyErr::fetch(py)));
                }
                let ptr = PY_ARRAY_API.PyArray_Empty(py, 3, dims.as_mut_ptr(), dtype, 0);
                Bound::from_owned_ptr_or_err(py, ptr)
                    .map_err(allocation_error)?
                    .cast_into_unchecked::<PyArray3<u8>>()
            };
            if array.len() != size {
                return Err(arithmetic_limit());
            }
            let data = std::ptr::NonNull::new(array.data()).ok_or_else(arithmetic_limit)?;
            Ok(NumpyBuffer {
                array: array.unbind(),
                data,
            })
        })
    }

    #[cfg(feature = "turbojpeg")]
    unsafe fn finish(
        _width: u32,
        _height: u32,
        _size: usize,
        buffer: NumpyBuffer,
    ) -> Result<Self, ImgReadError> {
        // Caller guarantees initialization before making the array accessible.
        Ok(Self::Array(buffer.array))
    }

    fn from_image(image: DynamicImage) -> Self {
        Self::Image(image)
    }
}
