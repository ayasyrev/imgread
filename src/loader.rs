//! Frozen Python configuration with one nonblocking, process-owned workspace.
use crate::{
    decoder::{DecodeBackend, DecoderWorkspace},
    limits::DecodeLimits,
    python_path, to_numpy, validate_args,
};
#[cfg(feature = "loader-diagnostics")]
use pyo3::types::PyDict;
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyBytes, PyModule, PyString, PyTuple},
};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, MutexGuard, TryLockError,
    },
};

fn integer<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: the GIL is held, object is live, and PyNumber_Index returns a new
    // owned integer or NULL with an exception. It invokes the integer protocol once.
    unsafe { Bound::from_owned_ptr_or_err(object.py(), pyo3::ffi::PyNumber_Index(object.as_ptr())) }
}

struct BufferCap(usize);
impl<'a, 'py> FromPyObject<'a, 'py> for BufferCap {
    type Error = PyErr;
    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let value = integer(&object)?;
        if value.lt(0)? || value.gt(isize::MAX)? {
            return Err(PyValueError::new_err(
                "max_buffer_bytes must be a non-negative addressable integer",
            ));
        }
        Ok(Self(value.extract()?))
    }
}

struct Admission<'a>(&'a AtomicBool);
impl Drop for Admission<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
struct ProcessWorkspace {
    pid: u32,
    decoder: DecoderWorkspace,
}

/// Reuse bounded input and independent native decoding state for individual paths.
/// Snapshot paths once; files are opened lazily, relative to cwd at each load.
#[pyclass(frozen, mapping, module = "imgread")]
pub(crate) struct Loader {
    paths: Option<Vec<PathBuf>>,
    backend: DecodeBackend,
    bgr: bool,
    limits: DecodeLimits,
    cap: usize,
    busy: AtomicBool,
    workspace: Mutex<Option<ProcessWorkspace>>,
}
impl Loader {
    fn admit(&self) -> PyResult<Admission<'_>> {
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| PyRuntimeError::new_err("Loader is busy"))?;
        Ok(Admission(&self.busy))
    }
    fn workspace(&self) -> PyResult<MutexGuard<'_, Option<ProcessWorkspace>>> {
        let mut state = match self.workspace.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Err(PyRuntimeError::new_err("Loader is busy")),
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut guard = poisoned.into_inner();
                *guard = None;
                self.workspace.clear_poison();
                guard
            }
        };
        let pid = std::process::id();
        if state.as_ref().is_none_or(|state| state.pid != pid) {
            // Idle fork gives the child its own heap copy. Destroy that inherited
            // copy before creating new native state; the parent is unaffected.
            *state = None;
            *state = Some(ProcessWorkspace {
                pid,
                decoder: DecoderWorkspace::new(self.cap, true),
            });
        }
        Ok(state)
    }
    fn load(&self, py: Python<'_>, path: &Path) -> PyResult<Py<PyAny>> {
        let result = py.detach(|| {
            let mut state = self.workspace()?;
            Ok::<_, PyErr>(
                state
                    .as_mut()
                    .expect("initialized workspace")
                    .decoder
                    .decode_path(path, self.backend, self.bgr, false, self.limits),
            )
        })?;
        // Neither mutex nor input lease survives into Python callbacks/conversion.
        to_numpy(py, result)
    }
    fn manifest(&self) -> PyResult<&[PathBuf]> {
        self.paths
            .as_deref()
            .ok_or_else(|| PyTypeError::new_err("Loader has no paths snapshot"))
    }
}

#[pymethods]
impl Loader {
    #[new]
    #[pyo3(signature = (paths=None, *, color="rgb", dtype="uint8", backend="auto", limits="safe", max_buffer_bytes=BufferCap(1048576)), text_signature = "(paths=None, *, color='rgb', dtype='uint8', backend='auto', limits='safe', max_buffer_bytes=1048576)")]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        paths: Option<&Bound<'_, PyAny>>,
        color: &str,
        dtype: &str,
        backend: &str,
        limits: &str,
        max_buffer_bytes: BufferCap,
    ) -> PyResult<Self> {
        let bgr = validate_args(color, dtype).map_err(|e| e.to_pyerr())?;
        let backend = DecodeBackend::parse(backend).map_err(|e| e.to_pyerr())?;
        let limits = DecodeLimits::parse(limits).map_err(|e| e.to_pyerr())?;
        let paths = paths
            .map(|paths| {
                if paths.is_instance_of::<PyString>()
                    || paths.is_instance_of::<PyBytes>()
                    || paths.get_type().hasattr("__fspath__")?
                {
                    return Err(PyTypeError::new_err(
                        "paths must be a finite iterable of paths, not a single path",
                    ));
                }
                let mut snapshot = Vec::new();
                for path in paths.try_iter()? {
                    let path = python_path(py, &path?)?;
                    snapshot.try_reserve(1).map_err(|e| {
                        PyValueError::new_err(format!("manifest allocation failed: {e}"))
                    })?;
                    snapshot.push(path);
                }
                Ok(snapshot)
            })
            .transpose()?;
        Ok(Self {
            paths,
            backend,
            bgr,
            limits,
            cap: max_buffer_bytes.0,
            busy: AtomicBool::new(false),
            workspace: Mutex::new(None),
        })
    }
    fn __call__(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let _admission = self.admit()?;
        let path = python_path(py, path)?;
        self.load(py, &path)
    }
    fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let _admission = self.admit()?;
        let paths = self.manifest()?;
        if index.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "Loader indices must be integers, not bool",
            ));
        }
        let mut index = integer(index)?;
        if index.lt(0)? {
            index = index.add(paths.len())?;
        }
        if index.lt(0)? || index.ge(paths.len())? {
            return Err(PyIndexError::new_err("Loader index out of range"));
        }
        self.load(py, &paths[index.extract::<usize>()?])
    }
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.manifest()?.len())
    }
    fn __bool__(&self) -> bool {
        true
    }

    fn __reduce_ex__(&self, py: Python<'_>, _protocol: usize) -> PyResult<Py<PyAny>> {
        let rebuild = PyModule::import(py, "imgread._loader_pickle")?.getattr("rebuild")?;
        let paths = match &self.paths {
            None => py.None(),
            // PathBuf converts through pathlib.Path, which normalizes spelling.
            // OsStr converts losslessly to a filesystem string, preserving even
            // trailing separators and Unix surrogateescaped filename bytes.
            Some(paths) => PyTuple::new(py, paths.iter().map(|path| path.as_os_str()))?
                .into_any()
                .unbind(),
        };
        let backend = match self.backend {
            DecodeBackend::Auto => "auto",
            DecodeBackend::Image => "image",
            DecodeBackend::TurboJpeg => "turbojpeg",
        };
        let color = if self.bgr { "bgr" } else { "rgb" };
        let limits = if self.limits.max_input_bytes.is_some() {
            "safe"
        } else {
            "unlimited"
        };
        Ok(
            (rebuild, (paths, color, "uint8", backend, limits, self.cap))
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )
    }

    #[cfg(feature = "loader-diagnostics")]
    fn _debug_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let _admission = self.admit()?;
        let guard = self.workspace()?;
        let state = guard.as_ref().expect("initialized workspace");
        let (creations, live) = state.decoder.native_counts();
        let manifest_bytes = self.paths.as_ref().map_or(0, |paths| {
            paths.capacity() * std::mem::size_of::<PathBuf>()
                + paths.iter().map(PathBuf::capacity).sum::<usize>()
        });
        let values = [
            ("pid", state.pid as usize),
            ("input_len", 0),
            ("input_capacity", state.decoder.input.capacity()),
            ("input_growths", state.decoder.input.growths),
            ("input_peak_capacity", state.decoder.input.peak_capacity),
            ("manifest_entries", self.paths.as_ref().map_or(0, Vec::len)),
            ("manifest_bytes", manifest_bytes),
            ("native_generation", creations),
            ("native_creations", creations),
            ("native_live", live),
        ];
        drop(guard);
        let result = PyDict::new(py);
        for (key, value) in values {
            result.set_item(key, value)?;
        }
        Ok(result)
    }
}
