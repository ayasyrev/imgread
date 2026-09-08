use numpy::{ndarray::Array3, PyArray3};
use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyRuntimeWarning, PyTypeError},
    prelude::*,
    types::{PyAny, PyBytes, PyModule, PyString, PyTuple},
};
use std::path::PathBuf;

pub mod decoder;
pub mod error;
mod input_buffer;
#[cfg(feature = "turbojpeg")]
mod jpeg_reuse;
pub mod limits;
mod loader;
#[cfg(feature = "turbojpeg")]
mod turbo_backend;

use decoder::{DecodeBackend, DecodeOutput};
use error::ImgReadError;
use limits::{allocate_bytes, arithmetic_limit, DecodeLimits};

fn validate_args(color: &str, dtype: &str) -> Result<bool, ImgReadError> {
    let bgr = if color.eq_ignore_ascii_case("rgb") {
        false
    } else if color.eq_ignore_ascii_case("bgr") {
        true
    } else {
        return Err(ImgReadError::InvalidArgument(format!(
            "unsupported color mode: {color}"
        )));
    };
    if !dtype.eq_ignore_ascii_case("uint8") {
        return Err(ImgReadError::InvalidArgument(format!(
            "unsupported dtype: {dtype}"
        )));
    }
    Ok(bgr)
}

fn python_path(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    let path = PyModule::import(py, "os")?.call_method1("fspath", (path,))?;
    if !path.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "path must be str or os.PathLike[str]; bytes paths are not supported",
        ));
    }
    path.extract()
}

fn buffer_bytes(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    limits: DecodeLimits,
) -> PyResult<Vec<u8>> {
    // Immutable bytes can be copied directly with safe Rust after fallible
    // reservation, avoiding a redundant zero-fill of a potentially large input.
    if let Ok(bytes) = data.cast::<PyBytes>() {
        let source = bytes.as_bytes();
        let length = u64::try_from(source.len()).map_err(|_| arithmetic_limit().to_pyerr())?;
        let size = limits.check_input(length).map_err(|err| err.to_pyerr())?;
        let mut owned = Vec::new();
        owned.try_reserve_exact(size).map_err(|err| {
            ImgReadError::LimitExceeded(format!("input allocation failed: {err}")).to_pyerr()
        })?;
        owned.extend_from_slice(source);
        return Ok(owned);
    }
    let buffer = PyBuffer::<u8>::get(data)
        .map_err(|_| PyTypeError::new_err("data must be a uint8-compatible buffer"))?;
    let length = u64::try_from(buffer.len_bytes()).map_err(|_| arithmetic_limit().to_pyerr())?;
    let size = limits.check_input(length).map_err(|err| err.to_pyerr())?;
    let mut bytes = allocate_bytes(size).map_err(|err| err.to_pyerr())?;
    buffer
        .copy_to_slice(py, &mut bytes)
        .map_err(|_| PyTypeError::new_err("data must be a uint8-compatible buffer"))?;
    Ok(bytes)
}

fn to_numpy(py: Python<'_>, result: Result<DecodeOutput, ImgReadError>) -> PyResult<Py<PyAny>> {
    let output = result.map_err(|err| err.to_pyerr())?;
    if let Some(warning) = output.fallback_warning {
        // No Rust warning cache: Python's filters own default/always/ignore/error.
        // A native pyfunction adds no Python frame; stacklevel 1 is its caller.
        PyModule::import(py, "warnings")?.call_method1(
            "warn",
            (warning.message, py.get_type::<PyRuntimeWarning>(), 1),
        )?;
    }
    let height =
        usize::try_from(output.image.height()).map_err(|_| arithmetic_limit().to_pyerr())?;
    let width = usize::try_from(output.image.width()).map_err(|_| arithmetic_limit().to_pyerr())?;
    let raw = match output.image {
        image::DynamicImage::ImageRgb8(image) => image.into_raw(),
        _ => return Err(ImgReadError::Decode("internal non-RGB output".into()).to_pyerr()),
    };
    let array = Array3::from_shape_vec((height, width, 3), raw)
        .map_err(|err| ImgReadError::Decode(err.to_string()).to_pyerr())?;
    Ok(PyArray3::from_owned_array(py, array).into_any().unbind())
}

/// Decode a str/PathLike path to a writable, C-contiguous H×W×3 uint8 array.
/// JPEG, PNG and the first TIFF page are supported; alpha and metadata are ignored.
/// color: rgb/bgr; dtype: uint8; backend: auto/image/turbojpeg.
/// limits: safe (bounded, default) or unlimited (trusted inputs only).
/// Raises TypeError, ValueError, OSError subclasses or RuntimeError; fallback warns.
#[pyfunction(signature = (path, color = "rgb", dtype = "uint8", backend = "auto", *, limits = "safe"))]
fn load_numpy(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    color: &str,
    dtype: &str,
    backend: &str,
    limits: &str,
) -> PyResult<Py<PyAny>> {
    let bgr = validate_args(color, dtype).map_err(|err| err.to_pyerr())?;
    let backend = DecodeBackend::parse(backend).map_err(|err| err.to_pyerr())?;
    let limits = DecodeLimits::parse(limits).map_err(|err| err.to_pyerr())?;
    let path = python_path(py, path)?;
    to_numpy(
        py,
        py.detach(|| decoder::decode_path(&path, backend, bgr, false, limits)),
    )
}

/// Decode a uint8-compatible buffer to a writable C-contiguous H×W×3 uint8 array.
/// Strided buffers are copied in C order before releasing the GIL.
/// Options, exceptions and warnings match load_numpy; limits is keyword-only.
#[pyfunction(signature = (data, color = "rgb", dtype = "uint8", backend = "auto", *, limits = "safe"))]
fn load_numpy_from_bytes(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    color: &str,
    dtype: &str,
    backend: &str,
    limits: &str,
) -> PyResult<Py<PyAny>> {
    let bgr = validate_args(color, dtype).map_err(|err| err.to_pyerr())?;
    let backend = DecodeBackend::parse(backend).map_err(|err| err.to_pyerr())?;
    let limits = DecodeLimits::parse(limits).map_err(|err| err.to_pyerr())?;
    let bytes = buffer_bytes(py, data, limits)?;
    to_numpy(
        py,
        py.detach(|| decoder::decode_bytes(&bytes, None, backend, bgr, false, limits)),
    )
}

/// Decode a JPEG str/PathLike path to an H×W×3 uint8 RGB array.
/// Uses TurboJPEG with image fallback and RuntimeWarning. Non-JPEG raises ValueError.
/// limits: safe (default) or unlimited (trusted inputs only); I/O errors retain errno.
#[pyfunction(signature = (path, *, limits = "safe"))]
fn load_numpy_simple(py: Python<'_>, path: &Bound<'_, PyAny>, limits: &str) -> PyResult<Py<PyAny>> {
    let limits = DecodeLimits::parse(limits).map_err(|err| err.to_pyerr())?;
    let path = python_path(py, path)?;
    to_numpy(
        py,
        py.detach(|| decoder::decode_path(&path, DecodeBackend::TurboJpeg, false, true, limits)),
    )
}

/// Decode a JPEG uint8-compatible buffer to an H×W×3 uint8 RGB array.
/// Uses TurboJPEG with image fallback and RuntimeWarning. Non-JPEG raises ValueError.
/// Strided buffers are accepted; limits is safe (default) or trusted-only unlimited.
#[pyfunction(signature = (data, *, limits = "safe"))]
fn load_numpy_simple_from_bytes(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    limits: &str,
) -> PyResult<Py<PyAny>> {
    let limits = DecodeLimits::parse(limits).map_err(|err| err.to_pyerr())?;
    let bytes = buffer_bytes(py, data, limits)?;
    to_numpy(
        py,
        py.detach(|| {
            decoder::decode_bytes(&bytes, None, DecodeBackend::TurboJpeg, false, true, limits)
        }),
    )
}

/// Return the backend names enabled in this build as a tuple of strings.
#[pyfunction]
fn supported_backends(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, decoder::supported_backends())?
        .into_any()
        .unbind())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let version = env!("CARGO_PKG_VERSION")
        .replace("-beta.", "b")
        .replace("-dev.", ".dev");
    m.add("__version__", version)?;
    m.add_class::<loader::Loader>()?;
    // Private wheel provenance, not per-call diagnostics or a public API.
    m.add(
        "_build_info",
        (
            env!("IMGREAD_BUILD_SHA"),
            env!("IMGREAD_BUILD_DIRTY"),
            env!("IMGREAD_BUILD_PROFILE"),
            env!("IMGREAD_BUILD_RUST_DEBUG"),
            env!("IMGREAD_BUILD_NATIVE_DEBUG"),
        ),
    )?;
    m.add_function(wrap_pyfunction!(load_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(load_numpy_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(load_numpy_simple, m)?)?;
    m.add_function(wrap_pyfunction!(load_numpy_simple_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(supported_backends, m)?)?;
    Ok(())
}
