//! Generates an `obs.lib` import library from the installed `obs.dll`.
//!
//! OBS doesn't ship an import library with the normal Windows installer, and
//! requiring a full obs-studio source checkout just to link a plugin is a rough
//! ask. Instead we read the export table straight out of `obs.dll`, emit a `.def`,
//! and let MSVC's `lib.exe` build the import library for us.
//!
//! Override the OBS location with `OBS_INSTALL_DIR` if it isn't in Program Files.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_OBS_DIRS: &[&str] = &[
    r"C:\Program Files\obs-studio",
    r"C:\Program Files (x86)\obs-studio",
];

fn main() {
    println!("cargo:rerun-if-env-changed=OBS_INSTALL_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        // On Linux/macOS libobs is a normal shared library on the system path.
        println!("cargo:rustc-link-lib=dylib=obs");
        return;
    }

    let obs_dir = find_obs_dir().unwrap_or_else(|| {
        panic!(
            "Could not find an OBS Studio installation. Looked in {:?}.\n\
             Set OBS_INSTALL_DIR to your OBS folder (the one containing bin\\64bit\\obs.dll).",
            DEFAULT_OBS_DIRS
        )
    });

    let dll = obs_dir.join("bin").join("64bit").join("obs.dll");
    if !dll.is_file() {
        panic!("Found OBS at {} but {} is missing.", obs_dir.display(), dll.display());
    }
    println!("cargo:rerun-if-changed={}", dll.display());

    let exports = read_pe_exports(&dll)
        .unwrap_or_else(|e| panic!("Failed to read exports from {}: {e}", dll.display()));
    if exports.is_empty() {
        panic!("{} reported no exports; is it really libobs?", dll.display());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let def_path = out_dir.join("obs.def");
    let mut def = String::from("EXPORTS\n");
    for name in &exports {
        def.push_str(name);
        def.push('\n');
    }
    fs::write(&def_path, def).expect("failed to write obs.def");

    let lib_path = out_dir.join("obs.lib");
    let target = env::var("TARGET").unwrap();
    let mut cmd = cc::windows_registry::find(&target, "lib.exe").unwrap_or_else(|| {
        panic!(
            "lib.exe not found for target {target}. Install the Visual Studio Build Tools \
             (Desktop development with C++)."
        )
    });
    let status = cmd
        .arg(format!("/def:{}", def_path.display()))
        .arg("/machine:x64")
        .arg("/nologo")
        .arg(format!("/out:{}", lib_path.display()))
        .status()
        .expect("failed to run lib.exe");
    if !status.success() {
        panic!("lib.exe failed with {status} while building {}", lib_path.display());
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=obs");
    // Surfaced so `cargo build` output tells the user where to install the plugin.
    println!("cargo:warning=Linked against {} ({} exports)", dll.display(), exports.len());
}

fn find_obs_dir() -> Option<PathBuf> {
    if let Ok(dir) = env::var("OBS_INSTALL_DIR") {
        let p = PathBuf::from(dir);
        if p.is_dir() {
            return Some(p);
        }
    }
    DEFAULT_OBS_DIRS
        .iter()
        .map(PathBuf::from)
        .find(|p| p.join("bin").join("64bit").join("obs.dll").is_file())
}

// --- Minimal PE export-table reader -----------------------------------------
//
// Only enough of PE32+ to walk DataDirectory[0] and pull the exported names.

fn u16le(b: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(off..off + 2)?.try_into().ok()?))
}

fn u32le(b: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(off..off + 4)?.try_into().ok()?))
}

fn read_pe_exports(path: &Path) -> Result<Vec<String>, String> {
    let buf = fs::read(path).map_err(|e| e.to_string())?;
    let err = |m: &str| m.to_string();

    if u16le(&buf, 0).ok_or_else(|| err("truncated"))? != 0x5A4D {
        return Err(err("not an MZ image"));
    }
    let pe = u32le(&buf, 0x3C).ok_or_else(|| err("truncated"))? as usize;
    if u32le(&buf, pe).ok_or_else(|| err("truncated"))? != 0x0000_4550 {
        return Err(err("missing PE signature"));
    }

    let coff = pe + 4;
    let num_sections = u16le(&buf, coff + 2).ok_or_else(|| err("truncated"))? as usize;
    let opt_size = u16le(&buf, coff + 16).ok_or_else(|| err("truncated"))? as usize;
    let opt = coff + 20;

    // DataDirectory sits at +0x70 in PE32+ and +0x60 in PE32.
    let magic = u16le(&buf, opt).ok_or_else(|| err("truncated"))?;
    let dd = match magic {
        0x20B => opt + 0x70,
        0x10B => opt + 0x60,
        m => return Err(format!("unsupported optional header magic {m:#x}")),
    };
    let export_rva = u32le(&buf, dd).ok_or_else(|| err("truncated"))?;
    if export_rva == 0 {
        return Ok(Vec::new());
    }

    // Section table, used to map RVAs back to file offsets.
    let sections: Vec<(u32, u32, u32)> = (0..num_sections)
        .filter_map(|i| {
            let s = opt + opt_size + i * 40;
            Some((u32le(&buf, s + 0x0C)?, u32le(&buf, s + 0x08)?, u32le(&buf, s + 0x14)?))
        })
        .collect();

    let to_offset = |rva: u32| -> Option<usize> {
        sections
            .iter()
            .find(|&&(va, vsize, _)| rva >= va && rva < va.saturating_add(vsize.max(1)))
            .map(|&(va, _, raw)| (rva - va + raw) as usize)
    };

    let exp = to_offset(export_rva).ok_or_else(|| err("export RVA outside all sections"))?;
    let num_names = u32le(&buf, exp + 0x18).ok_or_else(|| err("truncated export dir"))? as usize;
    let names_rva = u32le(&buf, exp + 0x20).ok_or_else(|| err("truncated export dir"))?;
    let names_off = to_offset(names_rva).ok_or_else(|| err("name table outside all sections"))?;

    let mut out = Vec::with_capacity(num_names);
    for i in 0..num_names {
        let rva = match u32le(&buf, names_off + i * 4) {
            Some(r) => r,
            None => break,
        };
        let Some(start) = to_offset(rva) else { continue };
        let end = buf[start..].iter().position(|&c| c == 0).map(|n| start + n);
        if let Some(end) = end {
            if let Ok(s) = std::str::from_utf8(&buf[start..end]) {
                out.push(s.to_owned());
            }
        }
    }
    Ok(out)
}
