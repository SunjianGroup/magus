import os, sys
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from magus import __version__

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(DIR, "pybind11"))
from pybind11.setup_helpers import Pybind11Extension
del sys.path[-1]

# Avoid a gcc warning below:
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid
# for C/ObjC but not for C++
class BuildExt(build_ext):
    def build_extensions(self):
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        if '-Wsign-compare' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wsign-compare')
            self.compiler.compiler_so.append('-Wno-sign-compare')
        super().build_extensions()


#generatenew
module_GenerateNew = Pybind11Extension('magus.generators.GenerateNew',
                    sources = ['generatenew/src/main.cpp'],
                    extra_compile_args=['-std=c++11'],
                    )

#nepdes
module_nepdes = Pybind11Extension('magus.fingerprints.nepdes',
                    sources = ['nepdes/src/nep.cpp'],
                    extra_compile_args=['-std=c++11'],
                    )

with open('README.md') as f:
    long_description = f.read()

setup(
    name="magus-test",
    version=__version__,
    author="Gao Hao, Wang Junjie, Han Yu, DC, Sun Jian",
    author_email="141120108@smail.nju.edu",
    url="https://git.nju.edu.cn/gaaooh/magus",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        #"dscribe",
        "numpy<1.22.0",     # numba not support numpy >= 1.22.0
        "ase>=3.18",
        "pyyaml>=6.0",
        "networkx",
        "scipy",
        "scikit-learn",
        "spglib",
        "pandas",
        "prettytable",
        "packaging",
    ],
    extras_require={
        "recommend": [
            "BeautifulReport", 
            "plotly==5.6.0"],
        # "torchml": ["torch>=1.0"],
        },
    license="MIT",
    description="Magus: Machine learning And Graph theory assisted Universal structure Searcher",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[module_GenerateNew, module_nepdes], 
    cmdclass={'build_ext': BuildExt},
    entry_points={"console_scripts": ["magus = magus.entrypoints.main:main"]},
)
