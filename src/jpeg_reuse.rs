//! Conservative certificate that a JPEG cannot consume a previous image's tables.
//! Failure selects a fresh decoder; this parser never decides format acceptance.
pub(crate) fn is_self_contained(bytes: &[u8]) -> bool {
    certificate(bytes).unwrap_or(false)
}

fn certificate(bytes: &[u8]) -> Option<bool> {
    if bytes.get(..2)? != [0xff, 0xd8] {
        return Some(false);
    }
    let mut at = 2usize;
    let mut quant = [false; 4];
    let mut huffman = [[false; 4]; 2];
    let mut components = [(0u8, 0usize); 4];
    let mut component_count = 0;
    let mut progressive = false;
    let mut scanned = false;
    let mut entropy = false;
    loop {
        if entropy {
            loop {
                let byte = *bytes.get(at)?;
                at = at.checked_add(1)?;
                if byte != 0xff {
                    continue;
                }
                let start = at - 1;
                while *bytes.get(at)? == 0xff {
                    at = at.checked_add(1)?;
                }
                let marker = *bytes.get(at)?;
                if marker == 0 || (0xd0..=0xd7).contains(&marker) {
                    at += 1;
                    continue;
                }
                at = start;
                break;
            }
            entropy = false;
        }
        if *bytes.get(at)? != 0xff {
            return Some(false);
        }
        while *bytes.get(at)? == 0xff {
            at = at.checked_add(1)?;
        }
        let marker = *bytes.get(at)?;
        at = at.checked_add(1)?;
        if marker == 0xd9 {
            return Some(scanned);
        }
        let length = usize::from(u16::from_be_bytes([
            *bytes.get(at)?,
            *bytes.get(at.checked_add(1)?)?,
        ]));
        if length < 2 {
            return Some(false);
        }
        let end = at.checked_add(length)?;
        let payload = bytes.get(at.checked_add(2)?..end)?;
        at = end;
        match marker {
            0xdb => {
                let mut offset = 0usize;
                while offset < payload.len() {
                    let info = *payload.get(offset)?;
                    let id = usize::from(info & 15);
                    if id >= 4 || info >> 4 > 1 {
                        return Some(false);
                    }
                    offset = offset.checked_add(1 + 64 * (usize::from(info >> 4) + 1))?;
                    if offset > payload.len() {
                        return Some(false);
                    }
                    quant[id] = true;
                }
                if offset == 0 {
                    return Some(false);
                }
            }
            0xc4 => {
                let mut offset = 0usize;
                while offset < payload.len() {
                    let info = *payload.get(offset)?;
                    let id = usize::from(info & 15);
                    let class = usize::from(info >> 4);
                    if id >= 4 || class >= 2 {
                        return Some(false);
                    }
                    let counts = payload.get(offset + 1..offset + 17)?;
                    let count: usize = counts.iter().map(|&value| usize::from(value)).sum();
                    if count == 0 || count > 256 {
                        return Some(false);
                    }
                    offset = offset.checked_add(17 + count)?;
                    if offset > payload.len() {
                        return Some(false);
                    }
                    huffman[class][id] = true;
                }
                if offset == 0 {
                    return Some(false);
                }
            }
            0xc0..=0xc2 => {
                if component_count != 0 || *payload.first()? != 8 {
                    return Some(false);
                }
                component_count = usize::from(*payload.get(5)?);
                if !(1..=4).contains(&component_count) || payload.len() != 6 + 3 * component_count {
                    return Some(false);
                }
                progressive = marker == 0xc2;
                for index in 0..component_count {
                    let id = payload[6 + index * 3];
                    let sampling = payload[7 + index * 3];
                    let table = usize::from(payload[8 + index * 3]);
                    if components[..index].iter().any(|&(other, _)| id == other)
                        || table >= 4
                        || !(1..=4).contains(&(sampling & 15))
                        || !(1..=4).contains(&(sampling >> 4))
                    {
                        return Some(false);
                    }
                    components[index] = (id, table);
                }
            }
            0xda => {
                let count = usize::from(*payload.first()?);
                if component_count == 0
                    || count == 0
                    || count > component_count
                    || payload.len() != 4 + 2 * count
                {
                    return Some(false);
                }
                // Be conservative even about components first scanned later:
                // no frame component may see an inherited quantization table.
                if components[..component_count]
                    .iter()
                    .any(|&(_, table)| !quant[table])
                {
                    return Some(false);
                }
                let ss = payload[1 + 2 * count];
                let se = payload[2 + 2 * count];
                let approx = payload[3 + 2 * count];
                if (!progressive && (ss != 0 || se != 63 || approx != 0))
                    || (progressive
                        && (ss > se
                            || se > 63
                            || (ss == 0 && se != 0)
                            || (ss > 0 && count != 1)
                            || approx >> 4 > 13
                            || approx & 15 > 13))
                {
                    return Some(false);
                }
                let mut seen = [false; 4];
                for index in 0..count {
                    let id = payload[1 + index * 2];
                    let selector = payload[2 + index * 2];
                    let dc = usize::from(selector >> 4);
                    let ac = usize::from(selector & 15);
                    let component = components[..component_count]
                        .iter()
                        .position(|&(value, _)| value == id)?;
                    if seen[component] || dc >= 4 || ac >= 4 || !quant[components[component].1] {
                        return Some(false);
                    }
                    seen[component] = true;
                    if (ss == 0 && !huffman[0][dc]) || (se > 0 && !huffman[1][ac]) {
                        return Some(false);
                    }
                }
                scanned = true;
                entropy = true;
            }
            0xdd => {
                if payload.len() != 2 {
                    return Some(false);
                }
            }
            0xe0..=0xef | 0xfe => {}
            _ => return Some(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat, RgbImage};
    use std::io::Cursor;

    pub(crate) fn jpeg() -> Vec<u8> {
        let mut output = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::new(16, 16))
            .write_to(&mut output, ImageFormat::Jpeg)
            .unwrap();
        output.into_inner()
    }
    #[test]
    fn baseline_and_every_truncation() {
        let jpeg = jpeg();
        assert!(is_self_contained(&jpeg));
        for end in 0..jpeg.len() {
            assert!(!is_self_contained(&jpeg[..end]), "{end}");
        }
    }
    #[test]
    fn removed_tables_unknown_frame_and_embedded_markers() {
        let jpeg = jpeg();
        for marker in [0xdb, 0xc4] {
            let mut missing = jpeg.clone();
            let start = missing
                .windows(2)
                .position(|m| m == [0xff, marker])
                .unwrap();
            let length = usize::from(u16::from_be_bytes([missing[start + 2], missing[start + 3]]));
            missing.drain(start..start + 2 + length);
            assert!(!is_self_contained(&missing));
        }
        let mut unknown = jpeg.clone();
        let start = unknown.windows(2).position(|m| m == [0xff, 0xc0]).unwrap();
        unknown[start + 1] = 0xc3;
        assert!(!is_self_contained(&unknown));
        let mut comment = jpeg.clone();
        comment.splice(2..2, [0xff, 0xfe, 0, 6, 0xff, 0xd9, 0xff, 0xda]);
        assert!(is_self_contained(&comment));
        for length in [0, 1, 255] {
            let mut bad = jpeg.clone();
            bad[4] = length;
            bad[5] = length;
            assert!(!is_self_contained(&bad));
        }
    }
}
