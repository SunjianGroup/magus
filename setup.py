import os, io, sysconfig
from setuptools import setup, find_packages
from distutils.core import Extension
from magus import __version__


try:
    include_dirs = os.getenv('MAGUS_INCLUDE_PATH').split(':') 
except:
    include_dirs = [
        sysconfig.get_config_var('INCLUDEDIR'),
        sysconfig.get_config_var('INCLUDEPY'),
        ]
# python_ld_lib = os.getenv('MAGUS_PY_LIB') or sysconfig.get_config_var('INCLUDEPY').split('/')[-1]
libraries = ['boost_python', 'boost_numpy']
# libraries.append(python_ld_lib)
try:
    library_dirs = os.getenv('MAGUS_LD_LIBRARY_PATH').split(':')
except:
    library_dirs = [sysconfig.get_config_var('LIBDIR')]

#generatenew
module_GenerateNew = Extension('magus.generators.GenerateNew',
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    sources = ['generatenew/main.cpp'],
                    extra_compile_args=['-std=c++11'],
                    )

#lrpot
module_lrpot = Extension('magus.lrpot',
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    sources = ['lrpot/lrpot.cpp'],
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
        "numpy",
        "ase>=3.18",
        "pyyaml",
        "networkx",
        "scipy",
        "scikit-learn",
        "spglib",
        "pandas",
        "pyyaml"
    ],
    extras_require={
        "torchml": ["torch>=1.0"],
        "test": ["BeautifulReport"]
        },
    #license="MIT",
    description="Magus: Machine learning And Graph theory assisted Universal structure Searcher",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[module_GenerateNew, module_lrpot], 
    entry_points={"console_scripts": ["magus = magus.entrypoints.main:main"]},
)
