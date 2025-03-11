import numpy
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize

kwargs = {'extra_compile_args': ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', '-O1']}
mods = [Extension(f'flashrl.envs.{Path(fp.stem)}', sources=[fp], **kwargs) for fp in Path('flashrl/envs').glob('*.pyx')]
setup(ext_modules=cythonize(mods), packages=['flashrl', 'flashrl.envs'], include_dirs=[numpy.get_include()],
      include_package_data=True)
