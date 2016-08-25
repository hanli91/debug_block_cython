from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize([
    Extension("debugblocker_cython", sources=["debugblocker_cython.pyx", "TopPair.cpp", "PrefixEvent.cpp", "ReuseInfo.cpp"],
              language="c++", libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp", "-std=c++11"],
              extra_link_args=['-fopenmp']),
    Extension("new_topk_sim_join", sources=["new_topk_sim_join.pyx", "TopPair.cpp", "PrefixEvent.cpp", "ReuseInfo.cpp"],
              language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-std=c++11"],),
    Extension("original_topk_sim_join", sources=["original_topk_sim_join.pyx", "TopPair.cpp", "PrefixEvent.cpp", "ReuseInfo.cpp"],
              language="c++",
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-std=c++11"],),
 ]))
