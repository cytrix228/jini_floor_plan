import os, platform, struct, sys, sysconfig
PYPY = platform.python_implementation() == "PyPy"
GRAALPY = platform.python_implementation() == "GraalVM"
base_prefix = getattr(sys, "base_prefix", None)
if base_prefix:
    anaconda = os.path.exists(os.path.join(base_prefix, "conda-meta"))
else:
    anaconda = False
WINDOWS = platform.system() == "Windows"
FRAMEWORK = bool(sysconfig.get_config_var("PYTHONFRAMEWORK"))
SHARED = bool(sysconfig.get_config_var("Py_ENABLE_SHARED"))
shared = PYPY or GRAALPY or anaconda or WINDOWS or FRAMEWORK or SHARED
ld_version = sysconfig.get_config_var("LDVERSION")
version = sysconfig.get_config_var("VERSION")
implementation = platform.python_implementation()
if ld_version:
    lib_name = f"python{ld_version}"
else:
    major, minor = sys.version_info[:2]
    if minor > 7:
        lib_name = f"python{major}.{minor}"
    else:
        lib_name = f"python{major}.{minor}m"
libdir = sysconfig.get_config_var("LIBDIR")
pointer_width = struct.calcsize("P") * 8
flags = []
for flag in ("Py_DEBUG", "Py_REF_DEBUG", "Py_TRACE_REFS", "COUNT_ALLOCS"):
    if sysconfig.get_config_var(flag):
        flags.append(flag)
if "Py_DEBUG" in flags and "Py_REF_DEBUG" not in flags:
    flags.append("Py_REF_DEBUG")
build_flags = ",".join(sorted(set(flags)))
config_lines = [
    f"implementation={implementation}",
    f"version={version}",
    f"shared={str(shared)}",
    "abi3=False",
    f"lib_name={lib_name}",
    f"lib_dir={libdir}",
    f"executable={sys.executable}",
    f"pointer_width={pointer_width}",
    f"build_flags={build_flags}",
    "suppress_build_script_link_lines=False",
]
config_path = os.path.join(os.path.dirname(__file__), "pyo3_houdini.cfg")
with open(config_path, "w", encoding="utf-8") as fh:
    for line in config_lines:
        fh.write(line + "\n")
print(config_path)
