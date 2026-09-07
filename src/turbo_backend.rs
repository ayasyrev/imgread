//! A small RAII wrapper over the vendored TurboJPEG 3 API. The high-level crate
//! does not expose MAXMEMORY; using its allocating helper would bypass the budget.
use crate::{
    decoder::rgb_image,
    error::ImgReadError,
    limits::{arithmetic_limit, DecodeLimits},
};
use std::{ffi::CStr, ptr::NonNull};
use turbojpeg::raw;

struct Decompressor(NonNull<std::ffi::c_void>);

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

pub(crate) fn decode(
    bytes: &[u8],
    bgr: bool,
    limits: DecodeLimits,
) -> Result<image::DynamicImage, ImgReadError> {
    let mut decoder = Decompressor::new()?;
    if let Some(budget) = limits.max_decoder_alloc {
        let megabytes = i32::try_from(budget / (1024 * 1024)).map_err(|_| arithmetic_limit())?;
        if megabytes == 0 {
            return Err(ImgReadError::LimitExceeded(
                "TurboJPEG memory budget is below 1 MiB".into(),
            ));
        }
        decoder.set(raw::TJPARAM_TJPARAM_MAXMEMORY, megabytes)?;
    }
    // Do not save ICC/EXIF markers; metadata does not form part of the pixel contract.
    decoder.set(raw::TJPARAM_TJPARAM_SAVEMARKERS, 0)?;
    let input_len = bytes.len().try_into().map_err(|_| arithmetic_limit())?;
    // SAFETY: immutable input slice stays alive for the entire call; length matches it.
    if unsafe { raw::tj3DecompressHeader(decoder.0.as_ptr(), bytes.as_ptr(), input_len) } != 0 {
        return Err(decoder.error());
    }
    let width = decoder.get(raw::TJPARAM_TJPARAM_JPEGWIDTH)?;
    let height = decoder.get(raw::TJPARAM_TJPARAM_JPEGHEIGHT)?;
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
            decoder.0.as_ptr(),
            bytes.as_ptr(),
            input_len,
            output,
            pitch,
            format,
        )
    } != 0
    {
        return Err(decoder.error());
    }
    // SAFETY: a successful full-image tj3Decompress8 call initialized all `size`
    // bytes described above. The allocation has capacity for at least that region.
    unsafe {
        pixels.set_len(size);
    }
    rgb_image(width, height, pixels)
}

#[cfg(test)]
mod tests {
    use super::{decode, is_resource_failure, raw, Decompressor};
    use crate::{error::ImgReadError, limits::DecodeLimits};
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
    use std::io::Cursor;

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
