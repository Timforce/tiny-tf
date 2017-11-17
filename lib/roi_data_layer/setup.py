import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize

#setup(ext_modules = cythonize('s_img_rndcrop5_c.pyx'))

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

class custom_build_ext(build_ext):
    def build_extensions(self):
        build_ext.build_extensions(self)

setup(
    name='tiny_SR',
    ext_modules=[    
        Extension('minibatch',
        sources=['minibatch.pyx'],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
	    include_dirs = [numpy_include])
        ],
    cmdclass={'build_ext': build_ext}
)
