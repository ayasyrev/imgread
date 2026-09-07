use imgread::decoder::{
    decode_bytes, decode_image, decode_image_from_bytes, decode_image_from_bytes_with_backend,
    decode_image_with_backend, decode_simple_jpeg, decode_simple_jpeg_from_bytes, DecodeBackend,
};
use imgread::error::ImgReadError;
use imgread::limits::DecodeLimits;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("imgread_{nanos}_{name}"))
}

fn write_valid_image(path: &PathBuf, format: image::ImageFormat) {
    let img = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
    img.save_with_format(path, format)
        .expect("failed to write image fixture");
}

fn valid_image_bytes(format: image::ImageFormat) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
    let mut cursor = std::io::Cursor::new(Vec::new());
    img.write_to(&mut cursor, format)
        .expect("failed to encode image fixture");
    cursor.into_inner()
}

fn maximum_dimension_jpeg_bytes() -> Vec<u8> {
    let image = image::RgbImage::from_pixel(u32::from(u16::MAX), 1, image::Rgb([10, 20, 30]));
    let mut cursor = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut cursor, image::ImageFormat::Jpeg)
        .expect("failed to encode maximum-dimension JPEG fixture");
    cursor.into_inner()
}

#[test]
fn decode_valid_jpeg() {
    let path = temp_path("ok.jpg");
    write_valid_image(&path, image::ImageFormat::Jpeg);
    let decoded = decode_image(&path).expect("expected jpeg decode success");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
    let _ = fs::remove_file(path);
}

#[test]
fn decode_valid_png() {
    let path = temp_path("ok.png");
    write_valid_image(&path, image::ImageFormat::Png);
    let decoded = decode_image(&path).expect("expected png decode success");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
    let _ = fs::remove_file(path);
}

#[test]
fn decode_valid_tiff() {
    let path = temp_path("ok.tiff");
    write_valid_image(&path, image::ImageFormat::Tiff);
    let decoded = decode_image(&path).expect("expected tiff decode success");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
    let _ = fs::remove_file(path);
}

#[test]
fn decode_valid_jpeg_from_bytes() {
    let bytes = valid_image_bytes(image::ImageFormat::Jpeg);
    let decoded = decode_image_from_bytes(&bytes).expect("expected jpeg bytes decode success");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
}

#[test]
fn decode_valid_png_from_bytes() {
    let bytes = valid_image_bytes(image::ImageFormat::Png);
    let decoded = decode_image_from_bytes(&bytes).expect("expected png bytes decode success");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
}

#[test]
fn decode_valid_tiff_from_bytes() {
    let bytes = valid_image_bytes(image::ImageFormat::Tiff);
    let decoded = decode_image_from_bytes(&bytes).expect("expected tiff bytes decode success");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
}

#[test]
fn decode_valid_png_without_extension() {
    let path = temp_path("ok_no_ext");
    write_valid_image(&path, image::ImageFormat::Png);
    let decoded = decode_image(&path).expect("expected extensionless png decode success");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
    let _ = fs::remove_file(path);
}

#[test]
fn decode_valid_jpeg_without_extension() {
    let path = temp_path("ok_no_ext_jpeg");
    write_valid_image(&path, image::ImageFormat::Jpeg);
    let decoded = decode_image(&path).expect("expected extensionless jpeg decode success");
    assert_eq!(decoded.width(), 2);
    assert_eq!(decoded.height(), 2);
    let _ = fs::remove_file(path);
}

#[test]
fn decode_corrupt_returns_decode_error() {
    let path = temp_path("corrupt.jpg");
    fs::write(&path, b"not a real jpeg").expect("failed to write corrupt fixture");
    let result = decode_image(&path);
    assert!(matches!(result, Err(ImgReadError::Decode(_))));
    let _ = fs::remove_file(path);
}

#[test]
fn decode_unsupported_returns_unsupported_format_error() {
    let path = temp_path("not-image.txt");
    fs::write(&path, b"hello").expect("failed to write unsupported fixture");
    let result = decode_image(&path);
    assert!(matches!(result, Err(ImgReadError::UnsupportedFormat(_))));
    let _ = fs::remove_file(path);
}

#[test]
fn decode_unsupported_bytes_returns_unsupported_format_error() {
    let result = decode_image_from_bytes(b"hello");
    assert!(matches!(result, Err(ImgReadError::UnsupportedFormat(_))));
}

#[test]
fn decode_corrupt_jpeg_bytes_returns_decode_error() {
    let result = decode_image_from_bytes(b"\xFF\xD8\xFF not a real jpeg payload");
    assert!(matches!(result, Err(ImgReadError::Decode(_))));
}

#[test]
fn decode_directory_path_returns_io_error() {
    let path = temp_path("directory_input");
    fs::create_dir_all(&path).expect("failed to create directory fixture");

    let result = decode_image(&path);
    assert!(matches!(result, Err(ImgReadError::Io { .. })));

    let _ = fs::remove_dir(path);
}

#[test]
fn decode_turbojpeg_backend_falls_back_for_non_jpeg() {
    let path = temp_path("ok_png_for_turbo_backend.png");
    write_valid_image(&path, image::ImageFormat::Png);

    let decoded = decode_image_with_backend(&path, DecodeBackend::TurboJpeg)
        .expect("expected png decode success through fallback");
    assert_eq!(decoded.image.width(), 2);
    assert_eq!(decoded.image.height(), 2);
    assert!(decoded.fallback_warning.is_some());
    let _ = fs::remove_file(path);
}

#[test]
fn decode_turbojpeg_backend_jpeg_has_expected_warning_behavior() {
    let path = temp_path("ok_jpeg_for_turbo_backend.jpg");
    write_valid_image(&path, image::ImageFormat::Jpeg);

    let decoded = decode_image_with_backend(&path, DecodeBackend::TurboJpeg)
        .expect("expected jpeg decode success");
    assert_eq!(decoded.image.width(), 2);
    assert_eq!(decoded.image.height(), 2);

    #[cfg(feature = "turbojpeg")]
    assert!(decoded.fallback_warning.is_none());
    #[cfg(not(feature = "turbojpeg"))]
    assert!(decoded.fallback_warning.is_some());

    let _ = fs::remove_file(path);
}

#[test]
fn decode_turbojpeg_backend_falls_back_for_non_jpeg_bytes() {
    let bytes = valid_image_bytes(image::ImageFormat::Png);

    let decoded = decode_image_from_bytes_with_backend(&bytes, DecodeBackend::TurboJpeg)
        .expect("expected png bytes decode success through fallback");
    assert_eq!(decoded.image.width(), 2);
    assert_eq!(decoded.image.height(), 2);
    assert!(decoded.fallback_warning.is_some());
}

#[test]
fn codec_dimension_capability_error_falls_back_to_image_backend() {
    let bytes = maximum_dimension_jpeg_bytes();
    for backend in [
        DecodeBackend::Image,
        DecodeBackend::Auto,
        DecodeBackend::TurboJpeg,
    ] {
        let decoded = decode_bytes(&bytes, None, backend, false, false, DecodeLimits::UNLIMITED)
            .expect("expected maximum-dimension JPEG decode success");
        assert_eq!(decoded.image.width(), u32::from(u16::MAX));
        assert_eq!(decoded.image.height(), 1);
        assert_eq!(
            decoded.fallback_warning.is_some(),
            backend == DecodeBackend::TurboJpeg
                || (backend == DecodeBackend::Auto && cfg!(feature = "turbojpeg"))
        );
        if cfg!(feature = "turbojpeg") && backend != DecodeBackend::Image {
            assert!(decoded
                .fallback_warning
                .as_ref()
                .is_some_and(|warning| warning
                    .message
                    .contains("Maximum supported image dimension")));
        }
    }
}

#[test]
fn simple_jpeg_decodes_valid_jpeg_bytes_to_raw_rgb() {
    let bytes = valid_image_bytes(image::ImageFormat::Jpeg);
    let result = decode_simple_jpeg_from_bytes(&bytes);

    let decoded = result.expect("expected simple JPEG decode or fallback");
    assert_eq!(decoded.image.width(), 2);
    assert_eq!(decoded.image.height(), 2);
    assert_eq!(
        decoded.fallback_warning.is_some(),
        !cfg!(feature = "turbojpeg")
    );
}

#[test]
fn simple_jpeg_decodes_valid_jpeg_path_to_raw_rgb() {
    let path = temp_path("simple_ok.jpg");
    write_valid_image(&path, image::ImageFormat::Jpeg);
    let result = decode_simple_jpeg(&path);

    let decoded = result.expect("expected simple JPEG decode or fallback");
    assert_eq!(decoded.image.width(), 2);
    assert_eq!(decoded.image.height(), 2);
    assert_eq!(
        decoded.fallback_warning.is_some(),
        !cfg!(feature = "turbojpeg")
    );

    let _ = fs::remove_file(path);
}

#[test]
fn simple_jpeg_rejects_non_jpeg_bytes_as_unsupported_format() {
    let bytes = valid_image_bytes(image::ImageFormat::Png);
    let result = decode_simple_jpeg_from_bytes(&bytes);

    assert!(matches!(result, Err(ImgReadError::UnsupportedFormat(_))));
}

#[test]
fn simple_jpeg_corrupt_jpeg_bytes_return_decode_error() {
    let result = decode_simple_jpeg_from_bytes(b"\xFF\xD8\xFF not a real jpeg payload");

    assert!(matches!(result, Err(ImgReadError::Decode(_))));
}

#[test]
fn simple_jpeg_missing_path_returns_not_found_error() {
    let path = temp_path("missing_simple.jpg");
    let result = decode_simple_jpeg(&path);

    assert!(matches!(result, Err(ImgReadError::Io { .. })));
}
