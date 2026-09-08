//! Bind locally built study wheels to their source commit without embedding paths.
use std::{env, path::Path, process::Command};

fn git(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("--no-optional-locks")
        .args(arguments)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let root = env::var("CARGO_MANIFEST_DIR").expect("Cargo manifest directory");
    let in_checkout = git(&["rev-parse", "--show-toplevel"])
        .is_some_and(|path| Path::new(&path) == Path::new(&root));
    let sha = if in_checkout {
        git(&["rev-parse", "HEAD"]).unwrap_or_default()
    } else {
        String::new()
    };
    let dirty = !in_checkout
        || git(&["status", "--porcelain", "--untracked-files=all"])
            .is_none_or(|status| !status.is_empty());
    if in_checkout {
        if let Some(files) = git(&["ls-files"]) {
            for file in files.lines() {
                println!("cargo:rerun-if-changed={file}");
            }
        }
        for arguments in [
            &["rev-parse", "--git-path", "HEAD"][..],
            &["rev-parse", "--git-path", "index"][..],
        ] {
            if let Some(path) = git(arguments) {
                println!("cargo:rerun-if-changed={path}");
            }
        }
        if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
            if let Some(path) = git(&["rev-parse", "--git-path", &reference]) {
                println!("cargo:rerun-if-changed={path}");
            }
        }
    }
    println!("cargo:rerun-if-env-changed=CFLAGS");
    println!("cargo:rustc-env=IMGREAD_BUILD_SHA={sha}");
    println!("cargo:rustc-env=IMGREAD_BUILD_DIRTY={dirty}");
    println!(
        "cargo:rustc-env=IMGREAD_BUILD_PROFILE={}",
        env::var("PROFILE").unwrap_or_default()
    );
    println!(
        "cargo:rustc-env=IMGREAD_BUILD_RUST_DEBUG={}",
        env::var("DEBUG").unwrap_or_default()
    );
    let native_debug = env::var("CFLAGS")
        .unwrap_or_default()
        .split_whitespace()
        .any(|flag| flag == "-g" || flag == "-g3" || flag == "-g2");
    println!("cargo:rustc-env=IMGREAD_BUILD_NATIVE_DEBUG={native_debug}");
}
