from __future__ import annotations

import numpy as np
from setuptools import Extension, setup


common_compile_args = [
    "-O3",
    "-std=c++17",
    "-march=native",
    "-fopenmp-simd",
    "-ffast-math",
    "-fno-math-errno",
]

etar_ext = Extension(
    "etar._etar_cpp",
    sources=["cpp/etar_cpp.cpp"],
    include_dirs=[np.get_include()],
    language="c++",
    extra_compile_args=common_compile_args,
)

storage_ext = Extension(
    "etar._etar_storage_cpp",
    sources=["cpp/etar_storage_cpp.cpp"],
    include_dirs=[np.get_include()],
    language="c++",
    extra_compile_args=common_compile_args,
)

setup(ext_modules=[etar_ext, storage_ext])
