import glob
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_kwargs = {'extra_compile_args': ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', '-O1']}
ext_mods = [Extension(fp.replace('/', '.')[:-4], sources=[fp], **ext_kwargs) for fp in glob.glob('flashrl/envs/*.pyx')]
setup(name='flashrl', ext_modules=cythonize(ext_mods), packages=['flashrl'], include_dirs=[numpy.get_include()])
