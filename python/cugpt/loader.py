import os
from pathlib import Path

from cuda.pathfinder import load_nvidia_dynamic_lib

_cuda_dll_dirs = []


def setup_cuda():
    libs = ("cudart", "cublas", "cublasLt")

    directories = set()

    for name in libs:
        lib = load_nvidia_dynamic_lib(name)
        directories.add(Path(lib.abs_path).parent)

    if os.name == "nt":
        for directory in directories:
            _cuda_dll_dirs.append(os.add_dll_directory(str(directory)))
