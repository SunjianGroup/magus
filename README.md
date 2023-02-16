# **MAGUS: Machine learning And Graph theory assisted Universal structure Searcher**

`MAGUS` is a Python package designed to predict crystal structures, which is free for non-commercial academic use.

# Current Features
* Generation of atomic structures for a given symmetry , support cluster, 2D, 3D including molecules
* Geometry optimization a large amount of structures with active learning
* Multi-target search for structures with fixed or variationally component

# Documentation
* An overview of code documentation and tutorials for getting started with `MAGUS` can be found [here](https://gitlab.com/bigd4/magus/-/blob/master/doc/manual.pdf)


# Requirements

`MAGUS` need [Python](https://www.python.org/) (3.6 or newer) and [gcc](https://gcc.gnu.org/) to build some module. Besides, the following python package are required:
| Package                                                    | version   |
| ---------------------------------------------------------- | --------- |
| [numpy](https://docs.scipy.org/doc/numpy/reference/)       |           |
| [scipy](https://docs.scipy.org/doc/scipy/reference/)       | >= 1.1    |
| [scikit-learn](https://scikit-learn.org/stable/index.html) |           |
| [ase](https://wiki.fysik.dtu.dk/ase/index.html)            | >= 3.18.0 |
| [pyyaml](https://pyyaml.org/)                              | >= 6.0    |
| [spglib](https://spglib.github.io/spglib/)                 |           |
| [pandas](https://pandas.pydata.org/)                       |           |
| [prettytable](https://github.com/jazzband/prettytable)     |           |
| [packaging](https://packaging.pypa.io/en/stable/)          |           |

\* These requirements will be installed automatically when using [pip](#using_pip), so you don't need to install them manually.  


And the following packages are optional: 

| Package                                                      | function                                      |
| ------------------------------------------------------------ | --------------------------------------------- |
| [beautifulreport](https://github.com/mocobk/BeautifulReport) | Generate html report for `magus test`         |
| [plotly](https://plotly.com/python/)                         | Generate html phasediagram for varcomp search |
| [dscribe](https://singroup.github.io/dscribe/latest/)        | Use fingerprint such as soap                  |
| [networkx](https://networkx.org/)                            | Use graph module                              |
| [pymatgen](https://pymatgen.org/)                            | Use reconstruct and cluster module            |



# Installation
<span id= "using_pip"> </span>
## Use pip 
You can use https:
```shell
$ pip install git+https://gitlab.com/bigd4/magus.git
```
or use [ssh](https://docs.gitlab.com/ee/user/ssh.html)
```shell
$ pip install git+ssh://git@gitlab.com/bigd4/magus.git
```
Your may need to add `--user` if you do not have the root permission.  
Or use `--force-reinstall` if you already  have `MAGUS` (add `--no-dependencies` if you do not want to reinstall the dependencies).

## From Source
1. Use clone or download to get the source code:
```shell
$ git clone --recursive https://gitlab.com/bigd4/magus.git
```
It must be careful that the downloaded zip will not include the submodules (nepdes, pybind11, gensym), you may need download them manually.  
2. Install the requirements by:
```shell
$ pip install -r requirements.txt
```
or by yourself. 

3. build the [gensym](https://gitlab.com/bigd4/gensym) and put it in `magus/generator`:
```shell
$ cd gensym
$ mkdir build && cd build
$ cmake .. && make
$ cp gensym.so ../../magus/generators
```
4. build the [nepdes](https://gitlab.com/bigd4/nepdes) and put it in `magus/fingerprints`:
```shell
$ cd nepdes
$ mkdir build && cd build
$ cmake .. && make
$ cp nepdes.so ../../magus/fingerprints
```

5. Add `magus` (the folder you download, not the inner `magus`) to your [`PYTHONPATH`](https://wiki.fysik.dtu.dk/ase/install.html#envvar-PYTHONPATH) environment variable in your `~/.bashrc` file. 

```shell
$ export PYTHONPATH=<path-to-magus-package>:$PYTHONPATH
```

6. (Optional) Create a new file named `magus` with following content and put it in your `PATH`.
```python
#!your-path-to-python
# -*- coding: utf-8 -*-
import re
import sys 
from magus.entrypoints.main import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```
This step can be skipped, but you should use 
```shell
$ python -m magus.main ...
```
to use magus instead of
```shell
$ magus
```
## Offline package
We provide an offline package [here](). You can also use [conda-build](https://docs.conda.io/projects/conda-build/en/latest/) and [constructor](https://conda.github.io/constructor/) to make it by yourself as described in [here](https://gitlab.com/bigd4/magus/-/tree/master/conda).  
After get the package,
```shell
$ chmod +x magus-***-Linux-x86_64.sh
$ ./magus-***-Linux-x86_64.sh
```
and follow the guide.
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
## Update
If you installed by pip, use:
```shell
$ magus update
```
If you installed from source, use:
```shell
$ cd <path-to-magus-package>
$ git pull origin master
```

# Environment variables
## Job management system
Add
```shell
$ export JOB_SYSTEM=LSF/SLURM/PBS
```
in your `~/.bashrc` according to your job management system.

## Auto completion
Put [`auto_complete.sh`](https://gitlab.com/bigd4/magus/-/blob/master/magus/auto_complete.sh) in your `PATH` like:
```shell
export PATH=$PATH:<your-path-to-auto_complete.sh>
```

# Interface
`MAGUS` now support the following packages to calculate the energy of structures, some of them are commercial or need registration to get the permission to use.

- [VASP](https://www.vasp.at/)
- [CASTEP](http://www.castep.org/)
- [ORCA](https://orcaforum.kofo.mpg.de/app.php/portal)
- [MTP](https://mlip.skoltech.ru/)
- [NEP](https://gpumd.zheyongfan.org/index.php/Main_Page)
- [DeepMD](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html) 
- [gulp](https://gulp.curtin.edu.au/gulp/) 
- [lammps](https://www.lammps.org/)
- [XTB](https://xtb-docs.readthedocs.io/en/latest/contents.html)  

You can also write interfaces to connect `MAGUS` and other codes by add them in the `/magus/calculators` directory.  
## VASP
For now we use the VASP calculator provided by [ase](https://wiki.fysik.dtu.dk/ase/index.html), so you need to do some preparations like this:  
1. create a new file `run_vasp.py`:
```python
import subprocess
exitcode = subprocess.call("mpiexec.hydra vasp_std", shell=True)
```
2. A directory containing the pseudopotential directories potpaw (LDA XC) potpaw_GGA (PW91 XC) and potpaw_PBE (PBE XC) is also needed, you can use symbolic link like:
```shell
$ ln -s <your-path-to-PBE-5.4> mypps/potpaw_PBE
```

3. Set both environment variables in your `~/.bashrc`:
```shell
$ export VASP_SCRIPT=<your-path-to-run_vasp.py>
$ export VASP_PP_PATH=<your-path-to-mypps>
```
More details can be seen [here](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#module-ase.calculators.vasp).

## Castep
We use the Castep calculator provided by [ase](https://wiki.fysik.dtu.dk/ase/index.html). Unlike vasp, we don't have to set up env variables, but write them directly to `input.yaml`. For more infomation, check the castep example under `examples` folder.

# Contributors
MAGUS is developed by Prof. Jian Sun's group at the School of Physics at Nanjing University. 

The current main developers are:  
- Hao Gao
- Junjie Wang
- Yu Han
- Shuning Pan
- Qiuhan Jia
- Yong Wang
- Chi Ding
- Bin Li

# Citations
| Reference | cite for what                         |
| --------- | ------------------------------------- |
| [1, 2]    | for any work that used `MAGUS`        |
| [3, 4]    | Graph theory                          |
| [5]       | Surface reconstruction                |
| [6]       | Structure searching in confined space |

# Reference

[1] Junjie Wang, et al. "MAGUS: machine learning and graph theory assisted universal structure searcher". (under review)

[2] Kang Xia, Hao Gao, Cong Liu, Jianan Yuan, Jian Sun, Hui-Tian Wang, Dingyu Xing, “A novel superhard tungsten nitride predicted by machine-learning accelerated crystal structure search”, Sci. Bull. 63, 817 (2018).


[3] Hao Gao, Junjie Wang, Yu Han, Jian Sun, “Enhancing Crystal Structure Prediction by Decomposition and Evolution Schemes Based on Graph Theory”, Fundamental Research 1, 466 (2021).

[4] Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, “Determining dimensionalities and multiplicities of crystal nets” npj Comput. Mater. 6, 143 (2020).

[5] Yu Han, Junjie Wang, Chi Ding, Hao Gao, Shuning Pan, Qiuhan Jia, Jian Sun, arXiv:2212.11549.

[6] Chi Ding, Junjie Wang, Yu Han, Jianan Yuan, Hao Gao, and Jian Sun, “High Energy Density Polymeric Nitrogen Nanotubes inside Carbon Nanotubes”, Chin. Phys. Lett. 39, 036101 (2022). (Express Letter)
