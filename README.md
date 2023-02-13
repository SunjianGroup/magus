# **Magus: Machine learning And Graph theory assisted Universal structure Searcher**

Magus is an open-source Python package designed to predict crystal structures. 

# Current Features
* Generation of atomic structures for a given symmetry , support cluster, 2D, 3D including molecules
* Geometry optimization a large amount of structures with active learning
* Multi-target search for structures with fixed or variationally component

# Documentation
* An overview of code documentation and tutorials for getting started with Magus can be found [here](https://gitlab.com/bigd4/magus/-/blob/master/doc/manual.pdf)


# Requirements

|  Package    |    version   |
|  -------    |    -------   |
| [python](https://www.python.org/)                          |     |
| [gcc](https://gcc.gnu.org/)                                |   |
| [numpy](https://docs.scipy.org/doc/numpy/reference/)       | <  1.22.0 |
| [scipy](https://docs.scipy.org/doc/scipy/reference/)       | >= 1.1    |
| [scikit-learn](https://scikit-learn.org/stable/index.html) |           |
| [ase](https://wiki.fysik.dtu.dk/ase/index.html)            | >= 3.18.0 |
| [pyyaml](https://pyyaml.org/)                              | >= 6.0    |
| [spglib](https://spglib.github.io/spglib/)                 |           |
| [pandas](https://pandas.pydata.org/)                       |           |
| [prettytable](https://github.com/jazzband/prettytable)     |           |
| [packaging](https://packaging.pypa.io/en/stable/)          |           |

\* The requirements will be installed automatically when using [pip](#using_pip)  
   
And the following packages are optional: 

|  Package    |    function   |
|  ----       |     ----      |
|[beautifulreport](https://github.com/mocobk/BeautifulReport) |Generate html report for `magus test`|
|[plotly](https://plotly.com/python/)                         |Generate html phasediagram for varcomp search|
|[dscribe](https://singroup.github.io/dscribe/latest/)        |Use fingerprint such as soap|
|[networkx](https://networkx.org/)                            |Use graph module|
|[pymatgen](https://pymatgen.org/)                            |Use reconstruct and cluster module|



# Installation
<span id= "using_pip"> </span>
## Use pip 
```shell
$ pip install git+https://gitlab.com/bigd4/magus.git
```
or 
```shell
$ pip install git+ssh://git@gitlab.com/bigd4/magus.git
```
Your may need to add `--user` if you do not have the root permission.

## From Source
First use clone or download to get the source code:
```shell
$ git clone --recursive https://gitlab.com/bigd4/magus.git
```
It must be careful that the downloaded zip will not include the submodules (nepdes, pybind11, gensym), you may need download them manually.  
You can install the requirements by:
```shell
$ pip install -r requirement.txt
```
or by yourself. 

Then build the [gensym](https://gitlab.com/bigd4/gensym) and put it in `magus/generator`:
```shell
$ cd gensym
$ mkdir build && cd build
$ cmake .. && make
$ cp gensym.so ../../magus/generators
```
Then build the [nepdes](https://gitlab.com/bigd4/nepdes) and put it in `magus/fingerprints`:
```shell
$ cd nepdes
$ mkdir build && cd build
$ cmake .. && make
$ cp nepdes.so ../../magus/fingerprints
```

Add `magus` (the folder you download, not the inner `magus`) to your [`PYTHONPATH`](https://wiki.fysik.dtu.dk/ase/install.html#envvar-PYTHONPATH) environment variable in your `~/.bashrc` file. 

```shell
$ export PYTHONPATH=<path-to-magus-package>:$PYTHONPATH
```
## Check
You can use 
```shell
$ magus -v
``` 
to check if you have installed successfully
and 
```shell
$ magus checkpack
```
to see what features you can use.

# Citations
|  Reference    |    cite for what   |
|  ----  | ----  |
|[1, 2]|for any work that used `Magus`|
|[3, 4]|Graph theory|
|[5]|Surface reconstruction|
|[6]|Structure searching in confined space|

# Reference

[1] Junjie Wang, et al. "MAGUS: machine learning and graph theory assisted universal structure searcher". (under review)

[2] Kang Xia, Hao Gao, Cong Liu, Jianan Yuan, Jian Sun, Hui-Tian Wang, Dingyu Xing, “A novel superhard tungsten nitride predicted by machine-learning accelerated crystal structure search”, Sci. Bull. 63, 817 (2018).


[3] Hao Gao, Junjie Wang, Yu Han, Jian Sun, “Enhancing Crystal Structure Prediction by Decomposition and Evolution Schemes Based on Graph Theory”, Fundamental Research 1, 466 (2021).

[4] Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, “Determining dimensionalities and multiplicities of crystal nets” npj Comput. Mater. 6, 143 (2020).

[5] Yu Han, Junjie Wang, Chi Ding, Hao Gao, Shuning Pan, Qiuhan Jia, Jian Sun, arXiv:2212.11549.

[6] Chi Ding, Junjie Wang, Yu Han, Jianan Yuan, Hao Gao, and Jian Sun, “High Energy Density Polymeric Nitrogen Nanotubes inside Carbon Nanotubes”, Chin. Phys. Lett. 39, 036101 (2022). (Express Letter)
