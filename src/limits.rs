//! Per-image policy budgets. Arithmetic and address-space checks are unconditional.
use crate::error::ImgReadError;

#[derive(Clone, Copy, Debug)]
pub struct DecodeLimits {
    pub max_input_bytes: Option<u64>,
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub max_pixels: Option<u64>,
    pub max_output_bytes: Option<u64>,
    pub max_decoder_alloc: Option<u64>,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self::SAFE
    }
}

impl DecodeLimits {
    pub const SAFE: Self = Self {
        max_input_bytes: Some(256 * 1024 * 1024),
        max_width: Some(32_768),
        max_height: Some(32_768),
        max_pixels: Some(100_000_000),
        max_output_bytes: Some(512 * 1024 * 1024),
        max_decoder_alloc: Some(512 * 1024 * 1024),
    };
    pub const UNLIMITED: Self = Self {
        max_input_bytes: None,
        max_width: None,
        max_height: None,
        max_pixels: None,
        max_output_bytes: None,
        max_decoder_alloc: None,
    };

    pub fn parse(value: &str) -> Result<Self, ImgReadError> {
        if value.eq_ignore_ascii_case("safe") {
            Ok(Self::SAFE)
        } else if value.eq_ignore_ascii_case("unlimited") {
            Ok(Self::UNLIMITED)
        } else {
            Err(ImgReadError::InvalidArgument(format!(
                "unsupported limits profile: {value}"
            )))
        }
    }

    pub fn check_input(self, bytes: u64) -> Result<usize, ImgReadError> {
        check_cap("max_input_bytes", bytes, self.max_input_bytes)?;
        addressable(bytes)
    }

    pub fn check_dimensions(self, width: u64, height: u64) -> Result<usize, ImgReadError> {
        check_cap("max_width", width, self.max_width.map(u64::from))?;
        check_cap("max_height", height, self.max_height.map(u64::from))?;
        let pixels = width.checked_mul(height).ok_or_else(arithmetic_limit)?;
        check_cap("max_pixels", pixels, self.max_pixels)?;
        let bytes = pixels.checked_mul(3).ok_or_else(arithmetic_limit)?;
        check_cap("max_output_bytes", bytes, self.max_output_bytes)?;
        addressable(bytes)
    }

    pub fn check_decoder_bytes(self, bytes: u64) -> Result<usize, ImgReadError> {
        check_cap("max_decoder_alloc", bytes, self.max_decoder_alloc)?;
        addressable(bytes)
    }

    pub(crate) fn check_jpeg_working_set(
        self,
        width: u64,
        height: u64,
        decoded_bytes: u64,
    ) -> Result<(), ImgReadError> {
        // image 0.25.10 does not forward max_alloc to zune-jpeg 0.5.15.
        // Budget conservatively for every JPEG, including progressive and
        // baseline multi-scan images: four full i16 coefficient planes, plus
        // the decoded output and row/upsampling storage. Output color alone
        // cannot tell us the input component count (e.g. CMYK becomes RGB).
        //
        // Sampling factors are at most four, so MCU padding adds at most 31
        // pixels per axis (vertical factor three is supported too). The row
        // allowance bounds all four components' raw coefficients, upsampling
        // buffers and scratch, including transient replacement allocations;
        // 64 KiB covers the small fixed tables. Re-audit on decoder upgrades.
        // Encoded input/metadata and allocator overhead are separate, as with
        // the other decoder budgets; this is not a process memory hard cap.
        let padded_width = width.checked_add(31).ok_or_else(arithmetic_limit)?;
        let padded_height = height.checked_add(31).ok_or_else(arithmetic_limit)?;
        let coefficients = padded_width
            .checked_mul(padded_height)
            .and_then(|size| size.checked_mul(4 * 2))
            .ok_or_else(arithmetic_limit)?;
        let working_set = padded_width
            .checked_mul(2048)
            .and_then(|size| size.checked_add(64 * 1024))
            .and_then(|size| size.checked_add(coefficients))
            .and_then(|size| size.checked_add(decoded_bytes))
            .ok_or_else(arithmetic_limit)?;
        self.check_decoder_bytes(working_set)?;
        Ok(())
    }

    pub fn image_limits(self) -> image::Limits {
        let mut limits = image::Limits::no_limits();
        limits.max_image_width = self.max_width;
        limits.max_image_height = self.max_height;
        limits.max_alloc = self.max_decoder_alloc;
        limits
    }
}

fn check_cap(name: &str, value: u64, cap: Option<u64>) -> Result<(), ImgReadError> {
    if let Some(cap) = cap {
        if value > cap {
            return Err(ImgReadError::LimitExceeded(format!(
                "{name}: {value} exceeds {cap}"
            )));
        }
    }
    Ok(())
}

pub fn arithmetic_limit() -> ImgReadError {
    ImgReadError::LimitExceeded("image exceeds arithmetic or address-space capacity".into())
}

pub fn addressable(bytes: u64) -> Result<usize, ImgReadError> {
    let size = usize::try_from(bytes).map_err(|_| arithmetic_limit())?;
    isize::try_from(size).map_err(|_| arithmetic_limit())?;
    Ok(size)
}

pub fn allocate_bytes(size: usize) -> Result<Vec<u8>, ImgReadError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|e| ImgReadError::LimitExceeded(format!("allocation failed: {e}")))?;
    bytes.resize(size, 0);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_boundaries() {
        let limits = DecodeLimits::SAFE;
        assert!(limits.check_input(256 * 1024 * 1024).is_ok());
        assert!(limits.check_input(256 * 1024 * 1024 + 1).is_err());
        assert!(limits.check_dimensions(32_768, 1).is_ok());
        assert!(limits.check_dimensions(32_769, 1).is_err());
        assert!(limits.check_dimensions(1, 32_769).is_err());
        assert!(limits.check_dimensions(10_000, 10_000).is_ok());
        assert!(limits.check_dimensions(10_001, 10_000).is_err());
        assert!(limits.check_decoder_bytes(512 * 1024 * 1024).is_ok());
        assert!(limits.check_decoder_bytes(512 * 1024 * 1024 + 1).is_err());
        let output_only = DecodeLimits {
            max_output_bytes: Some(12),
            ..DecodeLimits::UNLIMITED
        };
        assert!(output_only.check_dimensions(2, 2).is_ok());
        assert!(output_only.check_dimensions(3, 2).is_err());
    }

    #[test]
    fn unlimited_keeps_overflow_and_address_space_guards() {
        assert!(DecodeLimits::UNLIMITED
            .check_dimensions(u64::MAX, 2)
            .is_err());
        assert!(DecodeLimits::UNLIMITED
            .check_dimensions(u64::MAX / 2, 1)
            .is_err());
        assert!(DecodeLimits::UNLIMITED.check_input(u64::MAX).is_err());
        assert!(allocate_bytes(usize::MAX).is_err());
    }

    #[test]
    fn jpeg_working_set_includes_coefficients_padding_and_scratch() {
        let limits = DecodeLimits::SAFE;
        // The reported 10,000-square progressive JPEG passes pixel/output
        // checks, but its coefficient planes exceed the decoder budget.
        assert!(limits.check_dimensions(10_000, 10_000).is_ok());
        assert!(limits.check_decoder_bytes(300_000_000).is_ok());
        assert!(limits
            .check_jpeg_working_set(10_000, 10_000, 300_000_000)
            .is_err());
        assert!(DecodeLimits::UNLIMITED
            .check_jpeg_working_set(10_000, 10_000, 300_000_000)
            .is_ok());

        let (width, height, decoded) = (17, 25, 17 * 25 * 3);
        let bound = 8 * (width + 31) * (height + 31) + 2048 * (width + 31) + 64 * 1024 + decoded;
        for (cap, accepted) in [(bound, true), (bound - 1, false)] {
            let limits = DecodeLimits {
                max_decoder_alloc: Some(cap),
                ..DecodeLimits::UNLIMITED
            };
            assert_eq!(
                limits
                    .check_jpeg_working_set(width, height, decoded)
                    .is_ok(),
                accepted
            );
        }
        for (width, height, decoded) in [(u64::MAX, 1, 3), (1, u64::MAX, 3), (1, 1, u64::MAX)] {
            assert!(DecodeLimits::UNLIMITED
                .check_jpeg_working_set(width, height, decoded)
                .is_err());
        }
    }
}
