''' setup
'''

import re
import io
from distutils.command.build_ext import build_ext as build_ext_orig
from setuptools import setup, find_packages, Extension

# source: https://stackoverflow.com/a/39671214
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('pyckmeans/__init__.py', encoding='utf_8_sig').read()
).group(1)

# ==== ctypes extensions
class CTypesExtension(Extension):
    '''CTypesExtension'''

class build_ext(build_ext_orig):
    '''build_ext'''
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)

distance_module = CTypesExtension(
    'pyckmeans.distance.lib.distance',
    sources=['pyckmeans/distance/src/distance.cpp'],
    language='c++',
)

nucencode_module = CTypesExtension(
    'pyckmeans.io.lib.nucencode',
    sources=['pyckmeans/io/src/nucencode.cpp'],
    language='c++',
)

ext_modules = [
    distance_module,
    nucencode_module,
]

install_requires = [
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'tqdm',
]

# ====
description = 'A consensus K-Means implementation.'

long_description_content_type = 'text/x-rst'
long_description = 'A consensus K-Means implementation.'
# ====
setup(
    name='pyckmeans',
    version=__version__,
    packages=find_packages(),
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author='Tankred Ott',
    platforms=['any'],
    python_requires='>=3.6',
    install_requires=install_requires,
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    url='https://github.com/TankredO/pyckmeans',
)
