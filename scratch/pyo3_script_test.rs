use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let interpreter = args
        .next()
        .expect("usage: pyo3_script_test <python-interpreter>");

    const SCRIPT: &str = r#"
# Allow the script to run on Python 2, so that nicer error can be printed later.
from __future__ import print_function

import os.path
import platform
import struct
import sys
from sysconfig import get_config_var, get_platform

PYPY = platform.python_implementation() == "PyPy"
GRAALPY = platform.python_implementation() == "GraalVM"

if GRAALPY:
    graalpy_ver = map(int, __graalpython__.get_graalvm_version().split('.'));
    print("graalpy_major", next(graalpy_ver))
    print("graalpy_minor", next(graalpy_ver))

# sys.base_prefix is missing on Python versions older than 3.3; this allows the script to continue
# so that the version mismatch can be reported in a nicer way later.
base_prefix = getattr(sys, "base_prefix", None)

if base_prefix:
    # Anaconda based python distributions have a static python executable, but include
    # the shared library. Use the shared library for embedding to avoid rust trying to
    # LTO the static library (and failing with newer gcc's, because it is old).
    ANACONDA = os.path.exists(os.path.join(base_prefix, "conda-meta"))
else:
    ANACONDA = False

def print_if_set(varname, value):
    if value is not None:
        print(varname, value)

# Windows always uses shared linking
WINDOWS = platform.system() == "Windows"

# macOS framework packages use shared linking
FRAMEWORK = bool(get_config_var("PYTHONFRAMEWORK"))

# unix-style shared library enabled
SHARED = bool(get_config_var("Py_ENABLE_SHARED"))

print("implementation", platform.python_implementation())
print("version_major", sys.version_info[0])
print("version_minor", sys.version_info[1])
print("shared", PYPY or GRAALPY or ANACONDA or WINDOWS or FRAMEWORK or SHARED)
print_if_set("ld_version", get_config_var("LDVERSION"))
print_if_set("libdir", get_config_var("LIBDIR"))
print_if_set("base_prefix", base_prefix)
print("executable", sys.executable)
print("calcsize_pointer", struct.calcsize("P"))
print("mingw", get_platform().startswith("mingw"))
print("ext_suffix", get_config_var("EXT_SUFFIX"))
"#;

    let mut child = Command::new(&interpreter)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    child
        .stdin
        .as_mut()
        .expect("piped stdin")
        .write_all(SCRIPT.as_bytes())?;

    let output = child.wait_with_output()?;

    println!("status: {}", output.status);
    println!("--- stdout ---\n{}\n--- end stdout ---", String::from_utf8_lossy(&output.stdout));

    Ok(())
}
