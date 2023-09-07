
from setuptools import setup
from Cython.Build import cythonize
import numpy as np  # add this line

setup(
    ext_modules = cythonize(["const_decoder.pyx","hpsg_decoder.pyx"]),
    include_dirs=[np.get_include()]  # add this line
)
