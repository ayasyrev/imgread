use imgread::error::ImgReadError;
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::Python;

#[test]
fn error_mapping_not_found_maps_to_file_not_found() {
    Python::attach(|py| {
        let err = ImgReadError::Io {
            path: "missing.jpg".into(),
            source: std::io::Error::from_raw_os_error(2),
        };
        let pyerr = err.to_pyerr();
        assert!(pyerr.is_instance_of::<PyFileNotFoundError>(py));
    });
}

#[cfg(unix)]
#[test]
fn error_mapping_preserves_non_utf8_filename() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt, path::PathBuf};

    use pyo3::types::{PyAnyMethods, PyString};

    let path = PathBuf::from(OsString::from_vec(b"missing-\xff.jpg".to_vec()));
    let expected = path.clone().into_os_string();
    Python::attach(|py| {
        let err = ImgReadError::Io {
            path,
            source: std::io::Error::from_raw_os_error(2),
        };
        let pyerr = err.to_pyerr();
        let filename = pyerr.value(py).getattr("filename").unwrap();

        assert!(pyerr.is_instance_of::<PyFileNotFoundError>(py));
        assert!(filename.is_instance_of::<PyString>());
        assert_eq!(filename.extract::<OsString>().unwrap(), expected);
    });
}

#[test]
fn error_mapping_unsupported_maps_to_value_error() {
    Python::attach(|py| {
        let err = ImgReadError::UnsupportedFormat("txt".to_string());
        let pyerr = err.to_pyerr();
        assert!(pyerr.is_instance_of::<PyValueError>(py));
    });
}

#[test]
fn error_mapping_decode_maps_to_runtime_error() {
    Python::attach(|py| {
        let err = ImgReadError::Decode("bad data".to_string());
        let pyerr = err.to_pyerr();
        assert!(pyerr.is_instance_of::<PyRuntimeError>(py));
    });
}
