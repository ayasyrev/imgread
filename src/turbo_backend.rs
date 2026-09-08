//! A small RAII wrapper over the vendored TurboJPEG 3 API. The high-level crate
//! does not expose MAXMEMORY; using its allocating helper would bypass the budget.
use crate::{
    decoder::rgb_image,
    error::ImgReadError,
    limits::{arithmetic_limit, DecodeLimits},
};
use std::{ffi::CStr, ptr::NonNull};
use turbojpeg::raw;

pub(crate) struct Decompressor(NonNull<std::ffi::c_void>);

// SAFETY: the handle and all native state are exclusively owned. Moving it
// between threads is safe; all access is behind Loader's admission gate/mutex.
// No input/output pointer escapes a synchronous decode. The handle is not Sync.
unsafe impl Send for Decompressor {}

fn is_resource_failure(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    // These are the allocation and MAXMEMORY failures emitted by the vendored
    // TurboJPEG/libjpeg memory paths. Other messages that mention a maximum or
    // memory can describe codec capabilities or malformed input and may fall
    // back to the image backend.
    [
        "memory allocation failure",
        "insufficient memory",
        "memory limit exceeded",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
}

impl Decompressor {
    fn new() -> Result<Self, ImgReadError> {
        let init = i32::try_from(raw::TJINIT_TJINIT_DECOMPRESS).map_err(|_| arithmetic_limit())?;
        // SAFETY: initializes a new, exclusively owned native decompressor.
        NonNull::new(unsafe { raw::tj3Init(init) })
            .map(Self)
            .ok_or_else(|| {
                ImgReadError::LimitExceeded("cannot allocate TurboJPEG decompressor".into())
            })
    }

    fn error(&self) -> ImgReadError {
        // SAFETY: live handle; the native error string is NUL-terminated and copied immediately.
        let message = unsafe { CStr::from_ptr(raw::tj3GetErrorStr(self.0.as_ptr())) }
            .to_string_lossy()
            .into_owned();
        // libjpeg reports resource failures through text, not a separate error code.
        // Treat these as terminal even in unlimited mode; never retry an OOM elsewhere.
        if is_resource_failure(&message) {
            ImgReadError::LimitExceeded(message)
        } else {
            ImgReadError::Decode(message)
        }
    }

    fn set(&mut self, param: raw::TJPARAM, value: i32) -> Result<(), ImgReadError> {
        let param = i32::try_from(param).map_err(|_| arithmetic_limit())?;
        // SAFETY: exclusively owned live handle and documented integer parameter.
        if unsafe { raw::tj3Set(self.0.as_ptr(), param, value) } != 0 {
            return Err(self.error());
        }
        Ok(())
    }

    fn get(&self, param: raw::TJPARAM) -> Result<u32, ImgReadError> {
        let param = i32::try_from(param).map_err(|_| arithmetic_limit())?;
        // SAFETY: live handle after a successful header read.
        let value = unsafe { raw::tj3Get(self.0.as_ptr(), param) };
        u32::try_from(value).map_err(|_| self.error())
    }
}

impl Drop for Decompressor {
    fn drop(&mut self) {
        // SAFETY: this handle was returned by tj3Init and is destroyed exactly once.
        unsafe {
            raw::tj3Destroy(self.0.as_ptr());
        }
    }
}

impl Decompressor {
    pub(crate) fn decode(
        &mut self,
        bytes: &[u8],
        bgr: bool,
        limits: DecodeLimits,
    ) -> Result<image::DynamicImage, ImgReadError> {
        if let Some(budget) = limits.max_decoder_alloc {
            let megabytes =
                i32::try_from(budget / (1024 * 1024)).map_err(|_| arithmetic_limit())?;
            if megabytes == 0 {
                return Err(ImgReadError::LimitExceeded(
                    "TurboJPEG memory budget is below 1 MiB".into(),
                ));
            }
            self.set(raw::TJPARAM_TJPARAM_MAXMEMORY, megabytes)?;
        } else {
            self.set(raw::TJPARAM_TJPARAM_MAXMEMORY, 0)?;
        }
        // Do not save ICC/EXIF markers; metadata does not form part of the pixel contract.
        self.set(raw::TJPARAM_TJPARAM_SAVEMARKERS, 0)?;
        let input_len = bytes.len().try_into().map_err(|_| arithmetic_limit())?;
        // SAFETY: immutable input slice stays alive for the entire call; length matches it.
        if unsafe { raw::tj3DecompressHeader(self.0.as_ptr(), bytes.as_ptr(), input_len) } != 0 {
            return Err(self.error());
        }
        let width = self.get(raw::TJPARAM_TJPARAM_JPEGWIDTH)?;
        let height = self.get(raw::TJPARAM_TJPARAM_JPEGHEIGHT)?;
        let size = limits.check_dimensions(u64::from(width), u64::from(height))?;
        let pitch = i32::try_from(width.checked_mul(3).ok_or_else(arithmetic_limit)?)
            .map_err(|_| arithmetic_limit())?;
        if width == 0 || height == 0 {
            return Err(ImgReadError::Decode("empty JPEG dimensions".into()));
        }
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(size)
            .map_err(|e| ImgReadError::LimitExceeded(format!("allocation failed: {e}")))?;
        let output = pixels.spare_capacity_mut()[..size]
            .as_mut_ptr()
            .cast::<u8>();
        let format = if bgr {
            raw::TJPF_TJPF_BGR
        } else {
            raw::TJPF_TJPF_RGB
        };
        // SAFETY: header and decode use the same immutable input and the same unscaled
        // handle. `output` points to `size` writable bytes in the spare capacity of
        // `pixels`; pitch is width * 3, and RGB/BGR write every byte in that region.
        // No pointers escape. The vector length remains zero unless decoding succeeds.
        if unsafe {
            raw::tj3Decompress8(
                self.0.as_ptr(),
                bytes.as_ptr(),
                input_len,
                output,
                pitch,
                format,
            )
        } != 0
        {
            return Err(self.error());
        }
        // SAFETY: a successful full-image tj3Decompress8 call initialized all `size`
        // bytes described above. The allocation has capacity for at least that region.
        unsafe {
            pixels.set_len(size);
        }
        rgb_image(width, height, pixels)
    }
}

/// Keeps only successful, table-independent decoder state.
#[derive(Default)]
pub(crate) struct NativeSlot {
    decoder: Option<Decompressor>,
    #[cfg(any(test, feature = "loader-diagnostics"))]
    pub(crate) creations: usize,
}
impl NativeSlot {
    pub(crate) fn decode(
        &mut self,
        bytes: &[u8],
        bgr: bool,
        limits: DecodeLimits,
        reuse: bool,
    ) -> Result<image::DynamicImage, ImgReadError> {
        let certified = reuse && crate::jpeg_reuse::is_self_contained(bytes);
        // Uncertified inputs must never see tables left by an earlier image.
        // Drop retained state first, keeping at most one native allocation alive.
        if !certified {
            self.decoder = None;
        }
        if self.decoder.is_none() {
            self.decoder = Some(Decompressor::new()?);
            #[cfg(any(test, feature = "loader-diagnostics"))]
            {
                self.creations += 1;
            }
        }
        let result = self
            .decoder
            .as_mut()
            .expect("initialized decoder")
            .decode(bytes, bgr, limits);
        if result.is_err() || !certified {
            self.decoder = None;
        }
        result
    }

    #[cfg(any(test, feature = "loader-diagnostics"))]
    pub(crate) fn live(&self) -> usize {
        usize::from(self.decoder.is_some())
    }
}

#[cfg(test)]
fn decode(
    bytes: &[u8],
    bgr: bool,
    limits: DecodeLimits,
) -> Result<image::DynamicImage, ImgReadError> {
    NativeSlot::default().decode(bytes, bgr, limits, false)
}

#[cfg(test)]
mod tests {
    use super::{decode, is_resource_failure, raw, Decompressor};
    use crate::{error::ImgReadError, limits::DecodeLimits};
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
    use std::io::Cursor;

    #[test]
    fn native_identity_limits_and_fresh_table_independence() {
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::new(16, 16))
            .write_to(&mut encoded, ImageFormat::Jpeg)
            .unwrap();
        let bytes = encoded.into_inner();
        let mut slot = super::NativeSlot::default();
        for bgr in [false, true, false] {
            assert_eq!(
                slot.decode(&bytes, bgr, DecodeLimits::SAFE, true)
                    .unwrap()
                    .as_bytes(),
                decode(&bytes, bgr, DecodeLimits::SAFE).unwrap().as_bytes()
            );
            assert_eq!(slot.creations, 1);
            assert_eq!(slot.live(), 1);
        }
        let limits = DecodeLimits {
            max_width: Some(1),
            ..DecodeLimits::SAFE
        };
        assert!(matches!(
            slot.decode(&bytes, false, limits, true),
            Err(ImgReadError::LimitExceeded(_))
        ));
        assert_eq!(slot.live(), 0);
        slot.decode(&bytes, false, DecodeLimits::UNLIMITED, true)
            .unwrap();
        assert_eq!(slot.creations, 2);
        for marker in [0xdb, 0xc4] {
            let mut missing = bytes.clone();
            let at = missing
                .windows(2)
                .position(|value| value == [0xff, marker])
                .unwrap();
            let length = usize::from(u16::from_be_bytes([missing[at + 2], missing[at + 3]]));
            missing.drain(at..at + 2 + length);
            let fresh = decode(&missing, false, DecodeLimits::SAFE)
                .map(|image| image.into_bytes())
                .map_err(|e| e.to_string());
            let reused = slot
                .decode(&missing, false, DecodeLimits::SAFE, true)
                .map(|image| image.into_bytes())
                .map_err(|e| e.to_string());
            assert_eq!(fresh, reused);
            assert_eq!(slot.live(), 0);
            slot.decode(&bytes, false, DecodeLimits::SAFE, true)
                .unwrap();
        }
        // Reset MAXMEMORY when switching from bounded to unlimited internally.
        let mut decoder = Decompressor::new().unwrap();
        decoder.decode(&bytes, false, DecodeLimits::SAFE).unwrap();
        assert_eq!(decoder.get(raw::TJPARAM_TJPARAM_MAXMEMORY).unwrap(), 512);
        decoder
            .decode(&bytes, false, DecodeLimits::UNLIMITED)
            .unwrap();
        assert_eq!(decoder.get(raw::TJPARAM_TJPARAM_MAXMEMORY).unwrap(), 0);
    }

    #[test]
    fn only_allocation_and_maxmemory_errors_are_resource_failures() {
        for message in [
            "tj3Decompress8(): Memory allocation failure",
            "Insufficient memory (case 4)",
            "Memory limit exceeded",
        ] {
            assert!(is_resource_failure(message), "{message}");
        }
        for message in [
            "Maximum supported image dimension is 65500 pixels",
            "Invalid memory pool code 3",
            "MAX_ALLOC_CHUNK is wrong, please fix",
            "Sampling factors too large for interleaved scan",
            "Image is too large",
        ] {
            assert!(!is_resource_failure(message), "{message}");
        }
    }

    #[test]
    fn entropy_decode_error_does_not_expose_partial_output() {
        let image = RgbImage::from_fn(32, 32, |x, y| {
            Rgb([
                x.wrapping_mul(7) as u8,
                y.wrapping_mul(11) as u8,
                (x ^ y) as u8,
            ])
        });
        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut cursor, ImageFormat::Jpeg)
            .expect("failed to encode JPEG fixture");
        let mut bytes = cursor.into_inner();
        let scan = bytes
            .windows(2)
            .position(|marker| marker == [0xFF, 0xDA])
            .expect("JPEG fixture has no start-of-scan marker");
        let segment_length = usize::from(u16::from_be_bytes([bytes[scan + 2], bytes[scan + 3]]));
        let entropy_start = scan + 2 + segment_length;
        let entropy_end = bytes
            .windows(2)
            .rposition(|marker| marker == [0xFF, 0xD9])
            .expect("JPEG fixture has no end-of-image marker");
        let corruption = entropy_start + (entropy_end - entropy_start) / 2;
        bytes[corruption] = 0xFF;
        bytes[corruption + 1] = 0xC4;

        let header_decoder = Decompressor::new().expect("failed to create header decoder");
        let input_len = bytes.len().try_into().expect("fixture length does not fit");
        // SAFETY: the fixture slice remains alive and its exact length is supplied.
        let header_status = unsafe {
            raw::tj3DecompressHeader(header_decoder.0.as_ptr(), bytes.as_ptr(), input_len)
        };
        assert_eq!(
            header_status, 0,
            "corruption must occur after header parsing"
        );

        assert!(matches!(
            decode(&bytes, false, DecodeLimits::UNLIMITED),
            Err(ImgReadError::Decode(_))
        ));
    }
}
