import os
import io
from setuptools import setup, find_packages
from distutils.core import Extension
import yaml

# for installation on NJU machine
include_dirs = [
    '/fs00/software/anaconda/3-5.0.1/include',
    '/fs00/software/anaconda/3-5.0.1/include/python3.6m']
libraries = ['boost_python', 'boost_numpy', 'python3.6m']
library_dirs = ['/fs00/software/anaconda/3-5.0.1/lib']

# for installaion at other places
#paths = yaml.load(open('paths.yaml'))
#include_dirs = paths['include_dirs']
#libraries = paths['libraries']
#library_dirs = paths['library_dirs']

#generatenew
module_GenerateNew = Extension('magus.GenerateNew',
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
    name="magus",
    version="1.0.0",
    author="Gao Hao, Wang Junjie, Han Yu, DC, Sun Jian",
    email="141120108@smail.nju.edu",
    url="https://git.nju.edu.cn/gaaooh/magus",
    packages=find_packages(),
    #scripts=[
    #    "tools/magus-clean","tools/magus-search","tools/magus-prepare","tools/magus-summary",
    #],
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
    ],
    extras_require={"torchml": ["torch>=1.0"]},
    #license="MIT",
    description="Magus: Machine learning And Graph theory assisted Universal structure Searcher",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[module_GenerateNew, module_lrpot], 
    entry_points={"console_scripts": ["magus = magus.entrypoints.main:main"]},
)
