use std::{
    fs::File,
    io::{Cursor, Read},
    path::Path,
};

use image::{ColorType, DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage};

use crate::{
    error::ImgReadError,
    limits::{allocate_bytes, arithmetic_limit, DecodeLimits},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeBackend {
    Auto,
    Image,
    TurboJpeg,
}

impl DecodeBackend {
    pub fn parse(value: &str) -> Result<Self, ImgReadError> {
        if value.eq_ignore_ascii_case("auto") {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case("image") {
            Ok(Self::Image)
        } else if ["turbojpeg", "turbo_jpeg", "turbo-jpeg"]
            .iter()
            .any(|name| value.eq_ignore_ascii_case(name))
        {
            Ok(Self::TurboJpeg)
        } else {
            Err(ImgReadError::InvalidArgument(format!(
                "unsupported backend: {value}"
            )))
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DecodeFallbackWarning {
    pub message: String,
}

#[derive(Debug)]
pub struct DecodeOutput {
    pub image: DynamicImage,
    pub fallback_warning: Option<DecodeFallbackWarning>,
    pub is_bgr: bool,
}

pub fn read_image_bytes(path: &Path, limits: DecodeLimits) -> Result<Vec<u8>, ImgReadError> {
    let io_error = |source| ImgReadError::Io {
        path: path.to_path_buf(),
        source,
    };
    let mut file = File::open(path).map_err(io_error)?;
    // Metadata comes from the same open descriptor. Reads remain bounded if the file grows.
    let metadata = file.metadata().map_err(io_error)?;
    limits.check_input(metadata.len())?;
    let mut bytes = Vec::new();
    if metadata.is_file() {
        let size = limits.check_input(metadata.len())?;
        bytes
            .try_reserve_exact(size)
            .map_err(|e| ImgReadError::LimitExceeded(format!("input allocation failed: {e}")))?;
    }
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        // Read at most one byte over the input cap, including special/unknown-size files.
        let remaining = limits.max_input_bytes.map_or(chunk.len(), |cap| {
            usize::try_from(
                cap.saturating_sub(u64::try_from(bytes.len()).unwrap_or(u64::MAX))
                    .saturating_add(1),
            )
            .unwrap_or(chunk.len())
            .min(chunk.len())
        });
        let count = match file.read(&mut chunk[..remaining]) {
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result.map_err(io_error)?,
        };
        if count == 0 {
            break;
        }
        let size = bytes
            .len()
            .checked_add(count)
            .ok_or_else(arithmetic_limit)?;
        limits.check_input(u64::try_from(size).map_err(|_| arithmetic_limit())?)?;
        bytes
            .try_reserve_exact(count)
            .map_err(|e| ImgReadError::LimitExceeded(format!("input allocation failed: {e}")))?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    Ok(bytes)
}

fn detect_format(bytes: &[u8], path: Option<&Path>) -> Result<ImageFormat, ImgReadError> {
    let format = image::guess_format(bytes)
        .ok()
        .or_else(|| {
            path.and_then(Path::extension)
                .and_then(|ext| ext.to_str())
                .and_then(ImageFormat::from_extension)
        })
        .ok_or_else(|| ImgReadError::UnsupportedFormat("unknown image format".into()))?;
    if matches!(
        format,
        ImageFormat::Jpeg | ImageFormat::Png | ImageFormat::Tiff
    ) {
        Ok(format)
    } else {
        Err(ImgReadError::UnsupportedFormat(format!("{format:?}")))
    }
}

fn decode_with_image(
    bytes: &[u8],
    format: ImageFormat,
    limits: DecodeLimits,
) -> Result<DynamicImage, ImgReadError> {
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    reader.limits(limits.image_limits());
    let decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    let output_size = limits.check_dimensions(u64::from(width), u64::from(height))?;
    let decode_size = limits.check_decoder_bytes(decoder.total_bytes())?;
    if format == ImageFormat::Jpeg {
        limits.check_jpeg_working_set(
            u64::from(width),
            u64::from(height),
            decoder.total_bytes(),
        )?;
    }
    let color = decoder.color_type();
    let mut raw = allocate_bytes(decode_size)?;
    decoder.read_image(&mut raw)?;
    // Conversion uses a fallible allocation too; Image::into_rgb8() does not.
    limits.check_dimensions(u64::from(width), u64::from(height))?;
    let rgb = convert_rgb(raw, color, output_size)?;
    rgb_image(width, height, rgb)
}

fn convert_rgb(
    raw: Vec<u8>,
    color: ColorType,
    output_size: usize,
) -> Result<Vec<u8>, ImgReadError> {
    if color == ColorType::Rgb8 {
        return Ok(raw);
    }
    let mut output = allocate_bytes(output_size)?;
    let channels = usize::from(color.channel_count());
    let pixel_bytes = usize::from(color.bytes_per_pixel());
    let sample_bytes = pixel_bytes / channels;
    for (pixel, rgb) in raw
        .chunks_exact(pixel_bytes)
        .zip(output.as_chunks_mut::<3>().0)
    {
        for (channel, destination) in rgb.iter_mut().enumerate() {
            let source = if channels <= 2 {
                0
            } else {
                channel * sample_bytes
            };
            *destination = match sample_bytes {
                1 => pixel[source],
                2 => {
                    let value = u16::from_ne_bytes([pixel[source], pixel[source + 1]]);
                    u8::try_from((u32::from(value) + 128) / 257).map_err(|_| arithmetic_limit())?
                }
                4 => {
                    let value = f32::from_ne_bytes(
                        pixel[source..source + 4]
                            .try_into()
                            .map_err(|_| arithmetic_limit())?,
                    );
                    // Saturating float-to-byte conversion: NaN maps to zero, as in image-rs.
                    (value.clamp(0.0, 1.0) * 255.0).round() as u8
                }
                _ => {
                    return Err(ImgReadError::Decode(format!(
                        "unsupported color type: {color:?}"
                    )))
                }
            };
        }
    }
    Ok(output)
}

pub(crate) fn rgb_image(
    width: u32,
    height: u32,
    pixels: Vec<u8>,
) -> Result<DynamicImage, ImgReadError> {
    RgbImage::from_raw(width, height, pixels)
        .map(DynamicImage::ImageRgb8)
        .ok_or_else(|| ImgReadError::Decode("invalid RGB buffer shape".into()))
}

pub fn decode_bytes(
    bytes: &[u8],
    path: Option<&Path>,
    backend: DecodeBackend,
    bgr: bool,
    simple: bool,
    limits: DecodeLimits,
) -> Result<DecodeOutput, ImgReadError> {
    limits.check_input(u64::try_from(bytes.len()).map_err(|_| arithmetic_limit())?)?;
    if simple && !bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Err(ImgReadError::UnsupportedFormat(
            "simple JPEG API only supports JPEG input".into(),
        ));
    }
    let format = detect_format(bytes, path)?;
    let backend = if simple {
        DecodeBackend::TurboJpeg
    } else {
        backend
    };
    let mut fallback = None;
    if backend != DecodeBackend::Image {
        if format == ImageFormat::Jpeg {
            #[cfg(feature = "turbojpeg")]
            match crate::turbo_backend::decode(bytes, bgr, limits) {
                Ok(image) => {
                    return Ok(DecodeOutput {
                        image,
                        fallback_warning: None,
                        is_bgr: bgr,
                    })
                }
                Err(err @ ImgReadError::LimitExceeded(_)) => return Err(err),
                Err(err) => {
                    fallback = Some(format!(
                        "turbojpeg decode failed: {err}; fell back to 'image'"
                    ))
                }
            }
            #[cfg(not(feature = "turbojpeg"))]
            if backend == DecodeBackend::TurboJpeg {
                fallback = Some(
                    "backend 'turbojpeg' is not enabled in this build; fell back to 'image'".into(),
                );
            }
        } else if backend == DecodeBackend::TurboJpeg {
            fallback = Some(format!(
                "backend 'turbojpeg' only supports JPEG, got {format:?}; fell back to 'image'"
            ));
        }
    }
    let mut image = decode_with_image(bytes, format, limits)?;
    if bgr {
        if let Some(rgb) = image.as_mut_rgb8() {
            for pixel in rgb.as_mut().as_chunks_mut::<3>().0 {
                pixel.swap(0, 2);
            }
        }
    }
    Ok(DecodeOutput {
        image,
        fallback_warning: fallback.map(|message| DecodeFallbackWarning { message }),
        is_bgr: bgr,
    })
}

pub fn decode_path(
    path: &Path,
    backend: DecodeBackend,
    bgr: bool,
    simple: bool,
    limits: DecodeLimits,
) -> Result<DecodeOutput, ImgReadError> {
    let bytes = read_image_bytes(path, limits)?;
    decode_bytes(&bytes, Some(path), backend, bgr, simple, limits)
}

// The unpublished Rust API keeps its convenience entry points, all safe by default.
pub fn decode_image_with_backend_and_color(
    path: &Path,
    backend: DecodeBackend,
    bgr: bool,
) -> Result<DecodeOutput, ImgReadError> {
    decode_path(path, backend, bgr, false, DecodeLimits::SAFE)
}
pub fn decode_image_from_bytes_with_backend_and_color(
    bytes: &[u8],
    backend: DecodeBackend,
    bgr: bool,
) -> Result<DecodeOutput, ImgReadError> {
    decode_bytes(bytes, None, backend, bgr, false, DecodeLimits::SAFE)
}
pub fn decode_image_with_backend(
    path: &Path,
    backend: DecodeBackend,
) -> Result<DecodeOutput, ImgReadError> {
    decode_image_with_backend_and_color(path, backend, false)
}
pub fn decode_image_from_bytes_with_backend(
    bytes: &[u8],
    backend: DecodeBackend,
) -> Result<DecodeOutput, ImgReadError> {
    decode_image_from_bytes_with_backend_and_color(bytes, backend, false)
}
pub fn decode_image(path: &Path) -> Result<DynamicImage, ImgReadError> {
    decode_image_with_backend(path, DecodeBackend::Auto).map(|output| output.image)
}
pub fn decode_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage, ImgReadError> {
    decode_image_from_bytes_with_backend(bytes, DecodeBackend::Auto).map(|output| output.image)
}
pub fn decode_simple_jpeg_from_bytes(bytes: &[u8]) -> Result<DecodeOutput, ImgReadError> {
    decode_bytes(
        bytes,
        None,
        DecodeBackend::TurboJpeg,
        false,
        true,
        DecodeLimits::SAFE,
    )
}
pub fn decode_simple_jpeg(path: &Path) -> Result<DecodeOutput, ImgReadError> {
    decode_path(
        path,
        DecodeBackend::TurboJpeg,
        false,
        true,
        DecodeLimits::SAFE,
    )
}
pub fn supported_backends() -> Vec<&'static str> {
    if cfg!(feature = "turbojpeg") {
        vec!["auto", "image", "turbojpeg"]
    } else {
        vec!["auto", "image"]
    }
}
