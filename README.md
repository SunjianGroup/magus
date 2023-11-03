# **MAGUS: Machine learning And Graph theory assisted Universal structure Searcher**

[[_TOC_]]

# Introduction
`MAGUS` is a machine learning and graph theory assisted crystal structure prediction method developed by Prof. Jian Sun's group at the School of Physics at Nanjing University. The programming languages are mainly Python and C++ and it is built as a pip installable package. Users can use just a few commands to install the package. MAGUS has also the advantage of high modularity and extensibility. All source codes are transparent to users after installation, and users can modify particular parts according to their needs.

MAGUS has been used to study many systems. Several designed new materials have been synthesized experimentally, and a number of high-profile academic papers have been published. ([Publications using MAGUS](https://gitlab.com/bigd4/magus/-/wikis/home/Publications))

# Current Features
* Generation of atomic structures for a given symmetry, support cluster, surface, 2D and 3D crystals including molecules, confined systems, etc;
* Geometry optimization of a large number of structures with DFT or active learning machine learning potential;
* Multi-target search for structures with fixed or variationally component;
* API for VASP, CASTEP, Quantum ESPRESSO, ORCA, MTP, NEP, DeepMD, gulp, lammps, XTB, ASE, etc. Easy for extension.

# Documentation
* An overview of code documentation and tutorials for getting started with `MAGUS` can be found in doc folder. (press [here](https://gitlab.com/bigd4/magus/-/raw/master/doc/MAGUS_manual.pdf) to download)


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
Your may need to add `--user` if you do not have the root permission. Or use `--force-reinstall` if you already  have `MAGUS` (add `--no-dependencies` if you do not want to reinstall the dependencies).

**Notice**: you should make sure there is gcc in your environment. If you use https protocol, you should also make sure your git version is not too low.

## From Source
1. Use git clone to get the source code:
```shell
$ git clone --recursive https://gitlab.com/bigd4/magus.git
```
If the command works properly, all the submodules (nepdes, pybind11, gensym) will be downloaded automatically. But if your network connection has problem and you fail to download some of the submodules, you need download them manually and replace the corresponding empty folders in the source code.  

Alternatively, you can download the source code from website. Notice that the package you download does not include any submodules, so you should download them at the same time.

2. Go into the directory and install with pip:
```shell
$ pip install -e .
```
pip will read **setup.py** in your current directory and install. The `-e` option means python will directly import the module from the current path, but not copy the codes to the default lib path and import the module there, which is convenient for modifying in the future. If you do not have the need, you can remove the option.

## Offline package

We provide an offline package in the [release](https://gitlab.com/bigd4/magus/-/releases). You can also use [conda-build](https://docs.conda.io/projects/conda-build/en/latest/) and [constructor](https://conda.github.io/constructor/) to make it by yourself as described [here](https://gitlab.com/bigd4/magus/-/tree/master/conda).  
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
in your `~/.bashrc` according to your job management system (choose one of them).  

## Auto completion
Put [`auto_complete.sh`](https://gitlab.com/bigd4/magus/-/blob/master/magus/auto_complete.sh) in your `PATH` like:
```shell
source <your-path-to>/auto_complete.sh
```

# Interface
`MAGUS` now support the following packages to calculate the energy of structures, some of them are commercial or need registration to get the permission to use.

- [VASP](https://www.vasp.at/)
- [CASTEP](http://www.castep.org/)
- [Quantum ESPRESSO](https://www.quantum-espresso.org/)
- [ORCA](https://orcaforum.kofo.mpg.de/app.php/portal)
- [ASE built-in EMT & LJ](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#module-ase.calculators)
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
$ export VASP_SCRIPT=<your-path-to>/run_vasp.py
$ export VASP_PP_PATH=<your-path-to-mypps>
```
More details can be seen [here](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#module-ase.calculators.vasp).

## Castep
We use the Castep calculator provided by [ase](https://wiki.fysik.dtu.dk/ase/index.html). Unlike vasp, we don't have to set up env variables, but write them directly to `input.yaml`. For more infomation, check the castep example under `examples` folder.

# Contributors
MAGUS is developed by Prof. Jian Sun's group at the School of Physics at Nanjing University. Please contact us with email (magus@nju.edu.cn) if you have any questions concerning MAGUS. 

The main contributors are:
- Jian Sun
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

[1] Junjie Wang, Hao Gao, Yu Han, Chi Ding, Shuning Pan, Yong Wang, Qiuhan Jia, Hui-Tian Wang, Dingyu Xing, and Jian Sun, “MAGUS: machine learning and graph theory assisted universal structure searcher”, Natl. Sci. Rev. 10, nwad128, (2023). (https://doi.org/10.1093/nsr/nwad128)

[2] Kang Xia, Hao Gao, Cong Liu, Jianan Yuan, Jian Sun, Hui-Tian Wang, Dingyu Xing, “A novel superhard tungsten nitride predicted by machine-learning accelerated crystal structure search”, Sci. Bull. 63, 817 (2018). (https://doi.org/10.1016/j.scib.2018.05.027)


[3] Hao Gao, Junjie Wang, Yu Han, Jian Sun, “Enhancing Crystal Structure Prediction by Decomposition and Evolution Schemes Based on Graph Theory”, Fundamental Research 1, 466 (2021). (https://doi.org/10.1016/j.fmre.2021.06.005)

[4] Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, “Determining dimensionalities and multiplicities of crystal nets” npj Comput. Mater. 6, 143 (2020). (https://doi.org/10.1038/s41524-020-00409-0)

[5] Yu Han, Junjie Wang, Chi Ding, Hao Gao, Shuning Pan, Qiuhan Jia, and Jian Sun, “Prediction of surface reconstructions using MAGUS”, J. Chem. Phys. 158, 174109 (2023). (https://doi.org/10.1063/5.0142281)

[6] Chi Ding, Junjie Wang, Yu Han, Jianan Yuan, Hao Gao, and Jian Sun, “High Energy Density Polymeric Nitrogen Nanotubes inside Carbon Nanotubes”, Chin. Phys. Lett. 39, 036101 (2022). (Express Letter) (https://doi.org/10.1088/0256-307X/39/3/036101)
