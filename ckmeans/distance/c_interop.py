import ctypes
import pathlib

# path of the shared library
libfile = pathlib.Path(__file__).parent / 'lib' / 'distance.so'
print(libfile)

lib = ctypes.CDLL(str(libfile))
