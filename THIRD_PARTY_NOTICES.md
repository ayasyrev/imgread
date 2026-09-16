# Third-party notices

imgread includes code from the locked Rust dependency graph for Linux x86_64
and macOS x86_64/arm64. Full texts and per-license crate/version attribution,
including build dependencies, are in `licenses/rust/THIRD_PARTY_LICENSES.txt`.
After changing Cargo.lock, regenerate the Rust notices with
`cargo about generate --locked --all-features about.hbs > licenses/rust/THIRD_PARTY_LICENSES.txt`.

The `turbojpeg-sys` 1.2.0 crate vendors **libjpeg-turbo 3.1.0**,
which is statically linked in official wheels. The Rust wrapper's MIT/Unlicense
choice does not replace the native IJG, BSD-3-Clause and zlib license requirements.
The exact vendored license files are preserved in `licenses/libjpeg-turbo-3.1.0/`.
Upstream: https://github.com/libjpeg-turbo/libjpeg-turbo/tree/3.1.0

This software is based in part on the work of the Independent JPEG Group.

NumPy is an independently installed Python dependency and carries its own license
notices. It is not bundled in imgread. No additional non-system native shared
libraries are permitted in release wheels without a new inventory and notices.
