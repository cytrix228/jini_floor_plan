PYO3_CONFIG_FILE="$(pwd)/scratch/pyo3_houdini.cfg" \
PYO3_PYTHON="$(pwd)/.venv_houdini/bin/python" \
maturin develop --release --features python-bindings
