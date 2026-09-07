use std::{io, path::PathBuf};

use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImgReadError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("resource limit exceeded: {0}")]
    LimitExceeded(String),
}

impl ImgReadError {
    pub fn to_pyerr(&self) -> PyErr {
        match self {
            Self::Io { path, source } => {
                // OSError's constructor selects the errno-specific Python subclass.
                if let Some(errno) = source.raw_os_error() {
                    PyOSError::new_err((errno, source.to_string(), path.as_os_str().to_os_string()))
                } else {
                    let err: PyErr = io::Error::new(source.kind(), source.to_string()).into();
                    err
                }
            }
            Self::UnsupportedFormat(msg)
            | Self::InvalidArgument(msg)
            | Self::LimitExceeded(msg) => PyValueError::new_err(msg.clone()),
            Self::Decode(msg) => PyRuntimeError::new_err(msg.clone()),
        }
    }
}

impl From<image::ImageError> for ImgReadError {
    fn from(err: image::ImageError) -> Self {
        match err {
            image::ImageError::Limits(_) => Self::LimitExceeded(err.to_string()),
            _ => Self::Decode(err.to_string()),
        }
    }
}
