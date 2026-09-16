//! Bounded descriptor reads. The lease clears input even on error or unwind.
use std::{fs::File, io::Read, ops::Deref, path::Path};

use crate::{
    error::ImgReadError,
    limits::{arithmetic_limit, DecodeLimits},
};

#[derive(Default)]
pub(crate) struct ReusableInput {
    bytes: Vec<u8>,
    cap: usize,
    #[cfg(any(test, feature = "loader-diagnostics"))]
    pub(crate) growths: usize,
    #[cfg(any(test, feature = "loader-diagnostics"))]
    pub(crate) peak_capacity: usize,
}

impl ReusableInput {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            cap,
            ..Self::default()
        }
    }

    pub(crate) fn read<'a>(
        &'a mut self,
        path: &Path,
        limits: DecodeLimits,
    ) -> Result<InputLease<'a>, ImgReadError> {
        let io_error = |source| ImgReadError::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut file = File::open(path).map_err(io_error)?;
        let metadata = file.metadata().map_err(io_error)?;
        let size = limits.check_input(metadata.len())?;
        self.read_from(&mut file, metadata.is_file().then_some(size), path, limits)
    }

    fn read_from<'a>(
        &'a mut self,
        reader: &mut impl Read,
        size: Option<usize>,
        path: &Path,
        limits: DecodeLimits,
    ) -> Result<InputLease<'a>, ImgReadError> {
        let temporary = self.cap == 0 || size.is_some_and(|size| size > self.cap);
        let bytes = if temporary {
            Vec::new()
        } else {
            std::mem::take(&mut self.bytes)
        };
        let mut lease = InputLease {
            owner: self,
            bytes,
            temporary,
        };
        if let Some(size) = size {
            lease.reserve(size)?;
        }
        let mut chunk = [0_u8; 64 * 1024];
        loop {
            let remaining = limits.max_input_bytes.map_or(chunk.len(), |cap| {
                usize::try_from(
                    cap.saturating_sub(lease.bytes.len() as u64)
                        .saturating_add(1),
                )
                .unwrap_or(chunk.len())
                .min(chunk.len())
            });
            let count = match reader.read(&mut chunk[..remaining]) {
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result.map_err(|source| ImgReadError::Io {
                    path: path.to_path_buf(),
                    source,
                })?,
            };
            if count == 0 {
                break;
            }
            let size = lease
                .bytes
                .len()
                .checked_add(count)
                .ok_or_else(arithmetic_limit)?;
            limits.check_input(u64::try_from(size).map_err(|_| arithmetic_limit())?)?;
            lease.reserve(count)?;
            lease.bytes.extend_from_slice(&chunk[..count]);
        }
        Ok(lease)
    }

    #[cfg(any(test, feature = "loader-diagnostics"))]
    pub(crate) fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
}

pub(crate) struct InputLease<'a> {
    owner: &'a mut ReusableInput,
    bytes: Vec<u8>,
    temporary: bool,
}

impl InputLease<'_> {
    fn reserve(&mut self, additional: usize) -> Result<(), ImgReadError> {
        #[cfg(any(test, feature = "loader-diagnostics"))]
        let before = self.bytes.capacity();
        self.bytes
            .try_reserve_exact(additional)
            .map_err(|e| ImgReadError::LimitExceeded(format!("input allocation failed: {e}")))?;
        #[cfg(any(test, feature = "loader-diagnostics"))]
        {
            self.owner.growths += usize::from(self.bytes.capacity() != before);
            self.owner.peak_capacity = self.owner.peak_capacity.max(self.bytes.capacity());
        }
        if self.bytes.capacity() > self.owner.cap {
            self.temporary = true;
        }
        Ok(())
    }

    pub(crate) fn into_vec(mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }
}
impl Deref for InputLease<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}
impl Drop for InputLease<'_> {
    fn drop(&mut self) {
        self.bytes.clear();
        if !self.temporary && self.bytes.capacity() <= self.owner.cap {
            self.owner.bytes = std::mem::take(&mut self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};
    fn read(input: &mut ReusableInput, count: usize, hint: Option<usize>) {
        let lease = input
            .read_from(
                &mut Cursor::new(vec![7; count]),
                hint,
                Path::new("test"),
                DecodeLimits::SAFE,
            )
            .unwrap();
        assert_eq!(lease.len(), count);
    }
    #[test]
    fn warm_small_large_and_zero_cap() {
        let mut input = ReusableInput::new(100);
        read(&mut input, 50, Some(50));
        let ptr = input.bytes.as_ptr();
        let growths = input.growths;
        for _ in 0..5 {
            read(&mut input, 50, Some(50));
        }
        assert_eq!(input.bytes.as_ptr(), ptr);
        assert_eq!(input.growths, growths);
        read(&mut input, 200, Some(200));
        assert_eq!(input.bytes.as_ptr(), ptr);
        assert_eq!(input.bytes.len(), 0);
        read(&mut input, 200, None);
        assert!(input.capacity() <= 100);
        read(&mut input, 50, Some(1));
        let mut zero = ReusableInput::new(0);
        read(&mut zero, 50, Some(50));
        assert_eq!(zero.capacity(), 0);
    }
    #[test]
    fn growth_stops_at_limit_plus_one() {
        let mut input = ReusableInput::new(8);
        for hint in [Some(1), None] {
            let mut reader = Cursor::new(vec![1; 100]);
            let limits = DecodeLimits {
                max_input_bytes: Some(9),
                ..DecodeLimits::SAFE
            };
            assert!(matches!(
                input.read_from(&mut reader, hint, Path::new("test"), limits),
                Err(ImgReadError::LimitExceeded(_))
            ));
            assert_eq!(reader.position(), 10);
            assert!(input.capacity() <= 8);
            assert!(input.bytes.is_empty());
        }
    }
    struct FailingReader(usize);
    impl Read for FailingReader {
        fn read(&mut self, dest: &mut [u8]) -> io::Result<usize> {
            self.0 += 1;
            match self.0 {
                1 => Err(io::ErrorKind::Interrupted.into()),
                2 => {
                    dest[0] = 1;
                    Ok(1)
                }
                _ => Err(io::ErrorKind::PermissionDenied.into()),
            }
        }
    }
    #[test]
    fn failure_and_reservation_overflow_cleanup() {
        let mut input = ReusableInput::new(100);
        assert!(matches!(
            input.read_from(
                &mut FailingReader(0),
                None,
                Path::new("test"),
                DecodeLimits::SAFE
            ),
            Err(ImgReadError::Io { .. })
        ));
        assert!(input.bytes.is_empty());
        assert!(matches!(
            input.read_from(
                &mut Cursor::new([]),
                Some(usize::MAX),
                Path::new("test"),
                DecodeLimits::UNLIMITED
            ),
            Err(ImgReadError::LimitExceeded(_))
        ));
        assert!(input.capacity() <= 100);
    }
}
