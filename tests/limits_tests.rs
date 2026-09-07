use imgread::{
    decoder::{decode_bytes, DecodeBackend},
    error::ImgReadError,
    limits::DecodeLimits,
};
use std::io::Cursor;

fn jpeg_with_frame(progressive: bool, separate_scan: bool) -> Vec<u8> {
    let mut buffer = Cursor::new(Vec::new());
    image::RgbImage::new(2, 2)
        .write_to(&mut buffer, image::ImageFormat::Jpeg)
        .unwrap();
    let mut bytes = buffer.into_inner();
    let frame = bytes
        .windows(2)
        .position(|pair| pair == [0xff, 0xc0])
        .unwrap();
    bytes[frame + 1] = if progressive { 0xc2 } else { 0xc0 };
    bytes[frame + 5..frame + 9].copy_from_slice(&[2, 0, 2, 0]); // 512 x 512
    let scan = bytes
        .windows(2)
        .position(|pair| pair == [0xff, 0xda])
        .unwrap();
    // A tiny, deliberately incomplete stream: header parsing succeeds, but
    // working memory must be rejected before entropy decoding is attempted.
    bytes.truncate(scan);
    if separate_scan {
        bytes.extend_from_slice(&[0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0]);
    } else {
        bytes.extend_from_slice(&[
            0xff,
            0xda,
            0,
            12,
            3,
            1,
            0,
            2,
            0x11,
            3,
            0x11,
            0,
            if progressive { 0 } else { 63 },
            0,
        ]);
    }
    bytes.extend_from_slice(&[0xff, 0xd9]);
    bytes
}

#[test]
fn image_jpeg_working_memory_is_checked_before_entropy_decode() {
    let limits = DecodeLimits {
        max_decoder_alloc: Some(1024 * 1024),
        ..DecodeLimits::SAFE
    };
    assert!(limits.check_decoder_bytes(512 * 512 * 3).is_ok());
    for (progressive, separate_scan) in [(true, false), (false, true), (false, false)] {
        let bytes = jpeg_with_frame(progressive, separate_scan);
        let err =
            decode_bytes(&bytes, None, DecodeBackend::Image, false, false, limits).unwrap_err();
        assert!(
            matches!(&err, ImgReadError::LimitExceeded(message) if message.contains("max_decoder_alloc")),
            "{err}"
        );
    }
}

#[cfg(not(feature = "turbojpeg"))]
#[test]
fn image_jpeg_working_memory_covers_unavailable_backend_fallbacks() {
    let bytes = jpeg_with_frame(true, false);
    let limits = DecodeLimits {
        max_decoder_alloc: Some(1024 * 1024),
        ..DecodeLimits::SAFE
    };
    for (backend, simple) in [
        (DecodeBackend::Auto, false),
        (DecodeBackend::TurboJpeg, false),
        (DecodeBackend::TurboJpeg, true),
    ] {
        let err = decode_bytes(&bytes, None, backend, false, simple, limits).unwrap_err();
        assert!(
            matches!(&err, ImgReadError::LimitExceeded(message) if message.contains("max_decoder_alloc")),
            "{err}"
        );
    }
}

#[cfg(feature = "turbojpeg")]
#[test]
fn image_jpeg_working_memory_covers_native_decoder_failure_fallbacks() {
    let mut bytes = jpeg_with_frame(false, false);
    let frame = bytes
        .windows(2)
        .position(|pair| pair == [0xff, 0xc0])
        .unwrap();
    // zune accepts vertical sampling 3/2/1, but libjpeg-turbo rejects the
    // fractional ratio with a non-resource error. It must not bypass the
    // image backend's memory preflight when taking that fallback.
    for (component, sampling) in [0x13, 0x12, 0x11].into_iter().enumerate() {
        bytes[frame + 11 + component * 3] = sampling;
    }
    let limits = DecodeLimits {
        max_decoder_alloc: Some(1024 * 1024),
        ..DecodeLimits::SAFE
    };
    for (backend, simple) in [
        (DecodeBackend::Auto, false),
        (DecodeBackend::TurboJpeg, false),
        (DecodeBackend::TurboJpeg, true),
    ] {
        let err = decode_bytes(&bytes, None, backend, false, simple, limits).unwrap_err();
        assert!(
            matches!(&err, ImgReadError::LimitExceeded(message) if message.contains("max_decoder_alloc")),
            "{err}"
        );
    }
}

#[test]
fn decoder_and_output_budgets_are_terminal() {
    let mut buffer = Cursor::new(Vec::new());
    image::RgbImage::new(3, 2)
        .write_to(&mut buffer, image::ImageFormat::Png)
        .unwrap();
    for backend in [
        DecodeBackend::Image,
        DecodeBackend::Auto,
        DecodeBackend::TurboJpeg,
    ] {
        for limits in [
            DecodeLimits {
                max_output_bytes: Some(17),
                ..DecodeLimits::SAFE
            },
            DecodeLimits {
                max_decoder_alloc: Some(17),
                ..DecodeLimits::SAFE
            },
        ] {
            let err =
                decode_bytes(buffer.get_ref(), None, backend, false, false, limits).unwrap_err();
            assert!(matches!(err, ImgReadError::LimitExceeded(_)));
        }
    }
}

#[cfg(feature = "turbojpeg")]
#[test]
fn progressive_native_memory_limit_does_not_fallback() {
    let mut compressor = turbojpeg::Compressor::new().unwrap();
    compressor.set_quality(90).unwrap();
    compressor.set_subsamp(turbojpeg::Subsamp::None).unwrap();
    compressor.set_progressive(true).unwrap();
    let image = turbojpeg::Image {
        pixels: vec![64_u8; 512 * 512 * 3],
        width: 512,
        height: 512,
        pitch: 512 * 3,
        format: turbojpeg::PixelFormat::RGB,
    };
    let encoded = compressor.compress_to_vec(image.as_deref()).unwrap();
    let limits = DecodeLimits {
        max_decoder_alloc: Some(1024 * 1024),
        ..DecodeLimits::SAFE
    };
    for backend in [DecodeBackend::Auto, DecodeBackend::TurboJpeg] {
        let err = decode_bytes(&encoded, None, backend, false, false, limits).unwrap_err();
        assert!(matches!(err, ImgReadError::LimitExceeded(_)), "{err}");
    }
}
