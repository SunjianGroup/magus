import os
import io
from setuptools import setup, find_packages
from distutils.core import Extension


include_dirs = [
    '/fs00/software/anaconda/3-5.0.1/include',
    '/fs00/software/anaconda/3-5.0.1/include/python3.6m']
libraries = ['boost_python', 'boost_numpy', 'python3.6m']
library_dirs = ['/fs00/software/anaconda/3-5.0.1/lib']

#generatenew
module_GenerateNew = Extension('GenerateNew',
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    sources = ['generatenew/main.cpp'],
                    extra_compile_args=['-std=c++11'],
                    )

#lrpot
module_lrpot = Extension('lrpot',
                    include_dirs = include_dirs,
                    libraries = libraries,
                    library_dirs = library_dirs,
                    sources = ['lrpot/lrpot.cpp'],
                    extra_compile_args=['-std=c++11'],
                    )


with open('README.md') as f:
    long_description = f.read()


setup(
    name="mugus",
    version="0.0.1",
    author="Gao Hao, Wang Junjie, Han Yu, DC, Sun Jian",
    email="gaaooh@126.com",
    url="https://git.nju.edu.cn/gaaooh/magus",
    packages=find_packages(),
    scripts=[
        #"tools/csp-clean","tools/csp-search","tools/csp-prepare","tools/csp-summary",
        "tools/magus-clean","tools/magus-search","tools/magus-prepare","tools/magus-summary",
    ],
    python_requires=">=3.6",
    install_requires=[
        #"torch>=1.1",
        "numpy",
        "ase>=3.18",
        "pyyaml",
        "networkx==2.1",
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
)
