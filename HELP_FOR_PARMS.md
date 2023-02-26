# **HELP FOR INPUTS**
In magus we use inputs including command lines and parameter input file (in yaml format) to control programs. 

If you are a new user of MAGUS (AND SINCERELY THANKS VERY MUCH FOR USING OUR PROGRAM! ), before you read the full description about all inputs below, we recommand first taking a look at examples which are easier to follow. 

# Command lines
You can simply type
```shell
$ magus -h
```
to see which commands are supported for magus. You will see
```shell
usage: magus [-h] [-v]
             {search,summary,clean,prepare,calculate,generate,checkpack,test,update,getslabtool,mutate,parmhelp}
             ...

Magus: Machine learning And Graph theory assisted Universal structure Searcher

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         print version

Valid subcommands:
  {search,summary,clean,prepare,calculate,generate,checkpack,test,update,getslabtool,mutate,parmhelp}
    search              search structures
    summary             summary the results
    clean               clean the path
    prepare             generate InputFold etc to prepare for the search
    calculate           calculate many structures
    generate            generate many structures
    checkpack           check full
    test                do unit test of magus
    update              update magus
    getslabtool         tools to getslab in surface search mode
    mutate              mutation test
    parmhelp            help with parameters. Output all default and required
                        parameters.
```
which prints valid subcommands. You can also helps for each command line:  
**search** : Run a GA search.
```shell
$ magus search -h
usage: magus search [-h] [-ll {DEBUG,INFO,WARNING,ERROR}] [-lp LOG_PATH]
                    [-i INPUT_FILE] [-m] [-r]

optional arguments:
  -h, --help            show this help message and exit
  -ll {DEBUG,INFO,WARNING,ERROR}, --log-level {DEBUG,INFO,WARNING,ERROR}
                        set verbosity level by strings: ERROR, WARNING, INFO
                        and DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk (default:
                        log.txt)
  -i INPUT_FILE, --input-file INPUT_FILE
                        the input parameter file in yaml format (default:
                        input.yaml)
  -m, --use-ml          use ml to accelerate(?) the search (default: False)
  -r, --restart         Restart the searching. (default: False)
```
**summary**: (After search) Analyze the result structures.
```shell
$ magus summary -h
usage: magus summary [-h] [-p PREC] [-r] [-s] [--need-sort] [-o OUTDIR]
                     [-n SHOW_NUMBER] [-sb SORTED_BY [SORTED_BY ...]]
                     [-rm REMOVE_FEATURES [REMOVE_FEATURES ...]]
                     [-a ADD_FEATURES [ADD_FEATURES ...]] [-v]
                     [-b BOUNDARY [BOUNDARY ...]] [-t {bulk,cluster}]
                     filenames [filenames ...]

positional arguments:
  filenames             file (or files) to summary

optional arguments:
  -h, --help            show this help message and exit
  -p PREC, --prec PREC  tolerance for symmetry finding (default: 0.1)
  -r, --reverse         whether to reverse sort (default: False)
  -s, --save            whether to save POSCARS (default: False)
  --need-sort           whether to sort (default: False)
  -o OUTDIR, --outdir OUTDIR
                        where to save POSCARS (default: .)
  -n SHOW_NUMBER, --show-number SHOW_NUMBER
                        number of show in screen (default: 100)
  -sb SORTED_BY [SORTED_BY ...], --sorted-by SORTED_BY [SORTED_BY ...]
                        sorted by which arg (default: Default)
  -rm REMOVE_FEATURES [REMOVE_FEATURES ...], --remove-features REMOVE_FEATURES [REMOVE_FEATURES ...]
                        the features to be removed from the show features
                        (default: [])
  -a ADD_FEATURES [ADD_FEATURES ...], --add-features ADD_FEATURES [ADD_FEATURES ...]
                        the features to be added to the show features
                        (default: [])
  -v, --var             use variable composition mode (default: False)
  -b BOUNDARY [BOUNDARY ...], --boundary BOUNDARY [BOUNDARY ...]
                        in variable composition mode: add boundary (default:
                        [])
  -t {bulk,cluster}, --atoms-type {bulk,cluster}
```
**clean**: clean all former log files, result dictionary etc. in the path
```shell
$ magus clean -h
usage: magus clean [-h] [-f]

optional arguments:
  -h, --help   show this help message and exit
  -f, --force  rua!!!! (default: False)
```
**prepare**: prepare input files in the path
```shell
$ magus prepare -h
usage: magus prepare [-h] [-v] [-m]

optional arguments:
  -h, --help  show this help message and exit
  -v, --var   variable composition search (default: False)
  -m, --mol   molecule crystal search (default: False)
```
**calculate**: run local relaxation for input structures.
```shell
$ magus calculate -h
usage: magus calculate [-h] [-ll {DEBUG,INFO,WARNING,ERROR}] [-lp LOG_PATH]
                       [-m {scf,relax}] [-i INPUT_FILE] [-o OUTPUT_FILE]
                       [-p PRESSURE]
                       filename

positional arguments:
  filename              structures to relax

optional arguments:
  -h, --help            show this help message and exit
  -ll {DEBUG,INFO,WARNING,ERROR}, --log-level {DEBUG,INFO,WARNING,ERROR}
                        set verbosity level by strings: ERROR, WARNING, INFO
                        and DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk (default:
                        log.txt)
  -m {scf,relax}, --mode {scf,relax}
                        scf or relax (default: relax)
  -i INPUT_FILE, --input-file INPUT_FILE
                        the input parameter file in yaml format (default:
                        input.yaml)
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        output traj file (default: out.traj)
  -p PRESSURE, --pressure PRESSURE
                        add pressure (default: None)
```
**generate**: generate random structures
```shell                        
$ magus generate -h
usage: magus generate [-h] [-ll {DEBUG,INFO,WARNING,ERROR}] [-lp LOG_PATH]
                      [-i INPUT_FILE] [-o OUTPUT_FILE] [-n NUMBER]

optional arguments:
  -h, --help            show this help message and exit
  -ll {DEBUG,INFO,WARNING,ERROR}, --log-level {DEBUG,INFO,WARNING,ERROR}
                        set verbosity level by strings: ERROR, WARNING, INFO
                        and DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk (default:
                        log.txt)
  -i INPUT_FILE, --input-file INPUT_FILE
                        the input parameter file in yaml format (default:
                        input.yaml)
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        where to save generated traj (default: gen.traj)
  -n NUMBER, --number NUMBER
                        generate number (default: 10)
```
**checkpack**: check if you have installed FULL version MAGUS
```shell
$ magus checkpack -h
usage: magus checkpack [-h] [-ll {DEBUG,INFO,WARNING,ERROR}] [-lp LOG_PATH]
                       [{all,calculators,comparators,fingerprints}]

positional arguments:
  {all,calculators,comparators,fingerprints}
                        the package to check (default: all)

optional arguments:
  -h, --help            show this help message and exit
  -ll {DEBUG,INFO,WARNING,ERROR}, --log-level {DEBUG,INFO,WARNING,ERROR}
                        set verbosity level by strings: ERROR, WARNING, INFO
                        and DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk (default:
                        log.txt)
```
**test**: run tests
```shell
$ magus test -h
usage: magus test [-h] [totest]

positional arguments:
  totest      the package to test (default: *)

optional arguments:
  -h, --help  show this help message and exit
```
**update**: update MAGUS to latest version
```shell
$ magus update -h
usage: magus update [-h] [-u] [-f]

optional arguments:
  -h, --help   show this help message and exit
  -u, --user   add --user to pip install (default: False)
  -f, --force  add --force-reinstall to pip install (default: False)
```
**getslabtool**: (Surface mode) get slab model.
```shell
$ magus getslabtool -h
usage: magus getslabtool [-h] [-f FILENAME] [-s SLABFILE]

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        defaults is './Ref/layerslices.traj' of slab model and
                        'results' for analyze results. (default: )
  -s SLABFILE, --slabfile SLABFILE
                        slab file (default: slab.vasp)
```
**mutate**: do mutations and crossovers with input structures
```shell
$ magus mutate -h
usage: magus mutate [-h] [-i INPUT_FILE] [-s SEED_FILE] [-o OUTPUT_FILE]
                    [--cutandsplice] [--replaceball] [--soft] [--perm]
                    [--lattice] [--ripple] [--slip] [--rotate] [--rattle]
                    [--formula] [--lyrslip] [--shell] [--lyrsym] [--clusym]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        input_file (default: input.yaml)
  -s SEED_FILE, --seed_file SEED_FILE
                        seed_file (default: seed.traj)
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output_file (default: result)
  --cutandsplice        add option to use operation! (default: False)
  --replaceball         add option to use operation! (default: False)
  --soft                add option to use operation! (default: False)
  --perm                add option to use operation! (default: False)
  --lattice             add option to use operation! (default: False)
  --ripple              add option to use operation! (default: False)
  --slip                add option to use operation! (default: False)
  --rotate              add option to use operation! (default: False)
  --rattle              add option to use operation! (default: False)
  --formula             add option to use operation! (default: False)
  --lyrslip             add option to use operation! (default: False)
  --shell               add option to use operation! (default: False)
  --lyrsym              add option to use operation! (default: False)
  --clusym              add option to use operation! (default: False)
```  
**parmhelp**: show help information for all parameters.
```shell
$ magus parmhelp -h
usage: magus parmhelp [-h] [-s] [-a] [--parameters] [--generators]
                      [--individuals] [--populations] [--operations]
                      [--calculators] [-t TYPE] [-i INPUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -s, --show_avail      show names of all available types (default: False)
  -a, --all             output help for all! (default: False)
  --parameters          add option to see help for this module! (default:False)                 
  --generators          add option to see help for this module! (default:False)                 
  --individuals         add option to see help for this module! (default:False)                
  --populations         add option to see help for this module! (default:False)                
  --operations          add option to see help for this module! (default:False)                 
  --calculators         add option to see help for this module! (default:False)                  
  -t TYPE, --type TYPE  select a type from available types. (default: non-set)
  -i INPUT_FILE, --input-file INPUT_FILE
                        the input parameter file in yaml format. It is not
                        required to print help for parameters, but if you give
                        an input file we will also show you the real time
                        value in program by your input file. (default: None)
```

# parameter input files

A yaml format parameter file is necessary and important. By default is 'input.yaml' and again we strongly recommend a read of 'examples' first to know how to set parameters for different purposes briefly. And if you want to know more, we provide command line "magus parmhelp" to show more supported settings in our program.   

So the first thing is our program includes several modules namely, **generators** to generate random structures (mainly for the first generation), **individuals** (i.e. structures), **populations** (generations), **operations** (mutations and heredity) and **calculators** (for local relaxation). And each module includes different **type** of instances, which are 
```shell
$ magus parmhelp --show_avail
+ Available types for module **generators** include: 
	++ Molecule generators
	++ Layer generators
	++ Cluster generators
	++ Bulk generators
	++ Surface generators
+ Available types for module **individuals** include: 
	++ Bulk individuals
	++ Layer individuals
	++ Surface individuals
	++ Cluster individuals
+ Available types for module **populations** include: 
	++ Fix populations
	++ Var populations
+ Available types for module **operations** include: 
	++ cutandsplice operations
	++ perm operations
	++ lattice operations
	++ ripple operations
	++ slip operations
	++ rotate operations
	++ rattle operations
	++ formula operations
	++ lyrslip operations
	++ shell operations
	++ lyrsym operations
	++ clusym operations
+ Available types for module **calculators** include: 
	++ emt calculators
	++ lammps calculators
	++ mtp-noselect calculators
	++ mtp calculators
	++ mtp-lammps calculators
	++ quip calculators
	++ nep calculators
	++ vasp calculators
	++ castep calculators
	++ lj calculators
	++ gulp calculators
	++ naive calculators
	++ share-trainset calculators
```  
All above modules support user-defined parameters in the parameter.yaml file. 

Secondly there are two types of parameters in our program: required parameters and non-necessary ones. The former ones ("requirement parameters") are required to be set in parameters.yaml or else the program will raise an error and stops. But the latter ones ("default parameters") have pre-set default values by us, and they will be used if not set by user. You can check those parameters by (for example I want to check parameters for **type** Bulk in **generator** module)
```shell
$ magus parmhelp --generator --type Bulk
parameter information for <class 'magus.generators.random.SPGGenerator'>
+++++	default parameters	+++++
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
dimension      : dimension
                  default value: 3
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
full_ele       : only generate structures with full elements
                  default value: True
max_attempts   : max attempts to generate a random structure
                  default value: 50
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_n_formula  : maximum formula
                  default value: None
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
max_volume     : max volume
                  default value: -1
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_n_formula  : minimum formula
                  default value: None
min_volume     : min volume
                  default value: -1
n_split        : split cell into n_split parts
                  default value: [1]
p_pri          : probability of generate primitive cell
                  default value: 0.0
spacegroup     : spacegroup to generate random structures
                  default value: 2~231
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
+++++	requirement parameters	+++++
formula        : formula
formula_type   : type of formula, choose from fix or var
max_n_atoms    : maximum number of atoms per unit cell
min_n_atoms    : minimum number of atoms per unit cell
symbols        : atom symbols
----------------------------------------------------------------
```

Finally here is an additional function for parmhelp command. If there is an parameters.yaml, you can see the real time value set by this file. Take examples/08--1-graphene/input.yaml for example, check parameters of type Layer in module generator by:
```shell
$ magus parmhelp -i input.yaml --generator -t Layer
parameter information for LayerSPGGenerator
-------------------
formula_type   : fix
symbols        : ['C']
min_n_atoms    : 4
max_n_atoms    : 12
-------------------

+++++	default parameters	+++++
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
                  real-time value: 0.6
dimension      : dimension
                  default value: 2
                  real-time value: 2
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
                  real-time value: None
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
                  real-time value: 0
full_ele       : only generate structures with full elements
                  default value: True
                  real-time value: True
max_attempts   : max attempts to generate a random structure
                  default value: 50
                  real-time value: 50
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
                  real-time value: [-1, -1, -1, -1, -1, -1]
max_n_formula  : maximum formula
                  default value: None
                  real-time value: None
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
                  real-time value: 1000
max_volume     : max volume
                  default value: -1
                  real-time value: -1
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
                  real-time value: [-1, -1, -1, -1, -1, -1]
min_n_formula  : minimum formula
                  default value: None
                  real-time value: None
min_volume     : min volume
                  default value: -1
                  real-time value: -1
n_split        : split cell into n_split parts
                  default value: [1]
                  real-time value: [1]
p_pri          : probability of generate primitive cell
                  default value: 0.0
                  real-time value: 0.0
spacegroup     : spacegroup to generate random structures
                  default value: For planegroups use 1~17; for layergroups use 1~80
                  real-time value: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
spg_type       : choose from layer and plane to decide if use layergroup/planegroup
                  default value: layer
                  real-time value: plane
symprec        : tolerance for symmetry finding for molucule
                  default value: 0.1
                  real-time value: 0.1
threshold_mol  : distance between each pair of two molecules in the structure is 
                 not less than (mol_radius1+mol_radius2)*threshold_mol
                  default value: 1.0
                  real-time value: 1.0
vacuum_thickness: vacuum_thickness
                  default value: 10
                  real-time value: 15
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
                  real-time value: 3
+++++	requirement parameters	+++++
formula        : formula
                  real-time value: [[1]]
formula_type   : type of formula, choose from fix or var
                  real-time value: fix
max_n_atoms    : maximum number of atoms per unit cell
                  real-time value: 12
max_thickness  : maximum thickness
                  real-time value: 2
min_n_atoms    : minimum number of atoms per unit cell
                  real-time value: 4
min_thickness  : minimum thickness
                  real-time value: 1
symbols        : atom symbols
                  real-time value: ['C']
----------------------------------------------------------------
```
Show help for all parameters:
```shell
$ magus parmhelp --all
parameter information for <class 'magus.parameters.magusParameters'>
+++++	default parameters	+++++
DFTRelax       : DFTRelax
                  default value: False
addSym         : whether to add symmetry before crossover and mutation
                  default value: True
autoOpRatio    : automantic GA operation ratio
                  default value: False
autoRandomRatio: automantic random structure generation ratio
                  default value: False
bondRatio      : limitation to detect clusters
                  default value: 1.15
chkMol         : use mol dectector
                  default value: False
chkSeed        : check seeds
                  default value: True
comparator     : comparator, type magus checkpack to see which comparators you have.
                  default value: nepdes
dRatio         : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 0.7
diffE          : energy difference to determin structure duplicates
                  default value: 0.01
diffV          : volume difference to determin structure duplicates
                  default value: 0.05
eleSize        : used in variable composition mode, control how many boundary structures are generated
                  default value: 0
formulaType    : type of formula, choose from fix or var
                  default value: fix
fp_calc        : fingerprints, type magus checkpack to see which fingerprint method you have.
                  default value: nepdes
goodSize       : number of good indivials per generation
                  default value: =popSize
initSize       : size of first population
                  default value: =popSize
mlRelax        : use Machine learning relaxation
                  default value: False
molDetector    : methods to detect mol, choose from 1 and 2. See
                 Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, "Determining dimensionalities and multiplicities
                 of crystal nets" npj Comput. Mater. 6, 143 (2020) [doi.org/10.1016/j.fmre.2021.06.005]
                 for more details.
                  default value: 0
molMode        : search molecule clusters
                  default value: False
n_cluster      : number of good individuals per generation
                  default value: =saveGood
randRatio      : ratio of new generated random structures in next generation
                  default value: 0.2
spacegroup     : spacegroup to generate random structures
                  default value: [1-230]
structureType  : structure type, choose from bulk, layer, confined_bulk, cluster, surface
                  default value: bulk
symprec        : tolerance for symmetry finding
                  default value: 0.1
volRatio       : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 2
----------------------------------------------------------------
parameter information for <class 'magus.generators.random.MoleculeSPGGenerator'>
+++++	default parameters	+++++
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
dimension      : dimension
                  default value: 3
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
full_ele       : only generate structures with full elements
                  default value: True
max_attempts   : max attempts to generate a random structure
                  default value: 50
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_n_formula  : maximum formula
                  default value: None
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
max_volume     : max volume
                  default value: -1
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_n_formula  : minimum formula
                  default value: None
min_volume     : min volume
                  default value: -1
n_split        : split cell into n_split parts
                  default value: [1]
p_pri          : probability of generate primitive cell
                  default value: 0.0
spacegroup     : spacegroup to generate random structures
                  default value: 2~231
symprec        : tolerance for symmetry finding for molucule
                  default value: 0.1
threshold_mol  : distance between each pair of two molecules in the structure is 
                 not less than (mol_radius1+mol_radius2)*threshold_mol
                  default value: 1.0
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
+++++	requirement parameters	+++++
formula        : formula
formula_type   : type of formula, choose from fix or var
input_mols     : input molecules
max_n_atoms    : maximum number of atoms per unit cell
min_n_atoms    : minimum number of atoms per unit cell
symbols        : atom symbols
----------------------------------------------------------------
parameter information for <class 'magus.generators.random.LayerSPGGenerator'>
+++++	default parameters	+++++
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
dimension      : dimension
                  default value: 2
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
full_ele       : only generate structures with full elements
                  default value: True
max_attempts   : max attempts to generate a random structure
                  default value: 50
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_n_formula  : maximum formula
                  default value: None
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
max_volume     : max volume
                  default value: -1
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_n_formula  : minimum formula
                  default value: None
min_volume     : min volume
                  default value: -1
n_split        : split cell into n_split parts
                  default value: [1]
p_pri          : probability of generate primitive cell
                  default value: 0.0
spacegroup     : spacegroup to generate random structures
                  default value: For planegroups use 1~17; for layergroups use 1~80
spg_type       : choose from layer and plane to decide if use layergroup/planegroup
                  default value: layer
symprec        : tolerance for symmetry finding for molucule
                  default value: 0.1
threshold_mol  : distance between each pair of two molecules in the structure is 
                 not less than (mol_radius1+mol_radius2)*threshold_mol
                  default value: 1.0
vacuum_thickness: vacuum_thickness
                  default value: 10
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
+++++	requirement parameters	+++++
formula        : formula
formula_type   : type of formula, choose from fix or var
max_n_atoms    : maximum number of atoms per unit cell
max_thickness  : maximum thickness
min_n_atoms    : minimum number of atoms per unit cell
min_thickness  : minimum thickness
symbols        : atom symbols
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.generator.ClusterSPGGenerator'>
+++++	default parameters	+++++
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
dimension      : dimension
                  default value: 3
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
full_ele       : only generate structures with full elements
                  default value: True
max_attempts   : max attempts to generate a random structure
                  default value: 50
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_n_formula  : maximum formula
                  default value: None
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
max_volume     : max volume
                  default value: -1
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_n_formula  : minimum formula
                  default value: None
min_volume     : min volume
                  default value: -1
n_split        : split cell into n_split parts
                  default value: [1]
p_pri          : probability of generate primitive cell
                  default value: 0.0
spacegroup     : spacegroup to generate random structures
                  default value: 2~231
vacuum_thickness: vacuum thickness
                  default value: 10
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
+++++	requirement parameters	+++++
formula        : formula
formula_type   : type of formula, choose from fix or var
max_n_atoms    : maximum number of atoms per unit cell
min_n_atoms    : minimum number of atoms per unit cell
symbols        : atom symbols
----------------------------------------------------------------
parameter information for <class 'magus.generators.random.SPGGenerator'>
+++++	default parameters	+++++
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
dimension      : dimension
                  default value: 3
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
full_ele       : only generate structures with full elements
                  default value: True
max_attempts   : max attempts to generate a random structure
                  default value: 50
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_n_formula  : maximum formula
                  default value: None
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
max_volume     : max volume
                  default value: -1
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_n_formula  : minimum formula
                  default value: None
min_volume     : min volume
                  default value: -1
n_split        : split cell into n_split parts
                  default value: [1]
p_pri          : probability of generate primitive cell
                  default value: 0.0
spacegroup     : spacegroup to generate random structures
                  default value: 2~231
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
+++++	requirement parameters	+++++
formula        : formula
formula_type   : type of formula, choose from fix or var
max_n_atoms    : maximum number of atoms per unit cell
min_n_atoms    : minimum number of atoms per unit cell
symbols        : atom symbols
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.generator.SurfaceGenerator'>
+++++	default parameters	+++++
buffer         : use buffer layer
                  default value: True
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
dimension      : dimension
                  default value: 3
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
full_ele       : only generate structures with full elements
                  default value: True
max_attempts   : max attempts to generate a random structure
                  default value: 50
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_n_formula  : maximum formula
                  default value: None
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
max_volume     : max volume
                  default value: -1
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_n_formula  : minimum formula
                  default value: None
min_volume     : min volume
                  default value: -1
n_split        : split cell into n_split parts
                  default value: [1]
p_pri          : probability of generate primitive cell
                  default value: 0.0
randwalk_range : maximum range of random walk
                  default value: 0.5
randwalk_ratio : ratio of random walk atoms
                  default value: 0.3
rcs_formula    : formula of surface region
                  default value: None
rcs_x          : size[x] of reconstruction
                  default value: [1]
rcs_y          : size[y] of reconstruction
                  default value: [1]
spacegroup     : spacegroup to generate random structures
                  default value: 2~231
spg_type       : generate with planegroup/layergroup
                  default value: plane
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
+++++	requirement parameters	+++++
formula        : formula
formula_type   : type of formula, choose from fix or var
max_n_atoms    : maximum number of atoms per unit cell
min_n_atoms    : minimum number of atoms per unit cell
symbols        : atom symbols
+++++	default_slabinfo parameters	+++++
addH           : passivate bottom surface with H
                  default value: False
buffer_layernum: number of atom layers in buffer region
                  default value: 3
bulk_file      : file of bulk structure
                  default value: None
bulk_layernum  : number of atom layers in substrate region
                  default value: 3
cutslices      : bulk_file contains how many atom layers
                  default value: 2
direction      : Miller indices of surface direction, i.e.[1,0,0]
                  default value: None
matrix         : matrix notation
                  default value: None
pcell          : use primitive cell
                  default value: True
rcs_layernum   : number of atom layers in top surface region
                  default value: 2
rotate         : R
                  default value: 0
+++++	default_modification parameters	+++++
adsorb         : adsorb atoms to cleaved surface
                  default value: {}
clean          : clean cleaved surface
                  default value: {}
defect         : add defect to cleaved surface
                  default value: {}
----------------------------------------------------------------
parameter information for <class 'magus.populations.individuals.Bulk'>
+++++	default parameters	+++++
check_seed     : if check seeds
                  default value: False
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
full_ele       : full_ele
                  default value: True
max_attempts   : maximum attempts
                  default value: 50
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
mol_detector   : methods to detect mol, choose from 1 and 2. See
                 Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, "Determining dimensionalities and multiplicities
                 of crystal nets" npj Comput. Mater. 6, 143 (2020) [doi.org/10.1016/j.fmre.2021.06.005]
                 for more details.
                  default value: 0
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
radius         : radius
                  default value: None
+++++	requirement parameters	+++++
symprec        : tolerance for symmetry finding
----------------------------------------------------------------
parameter information for <class 'magus.populations.individuals.Layer'>
+++++	default parameters	+++++
bond_ratio     : bond_ratio
                  default value: 1.1
check_seed     : if check seeds
                  default value: False
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
full_ele       : full_ele
                  default value: True
max_attempts   : maximum attempts
                  default value: 50
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
radius         : radius
                  default value: None
vacuum_thickness: vacuum_thickness
                  default value: 10
+++++	requirement parameters	+++++
symprec        : tolerance for symmetry finding
----------------------------------------------------------------
parameter information for <class 'magus.populations.individuals.ConfinedBulk'>
+++++	default parameters	+++++
check_seed     : if check seeds
                  default value: False
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
full_ele       : full_ele
                  default value: True
max_attempts   : maximum attempts
                  default value: 50
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
radius         : radius
                  default value: None
vacuum_thickness: vacuum thickness
                  default value: 10
+++++	requirement parameters	+++++
symprec        : tolerance for symmetry finding
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.individuals.Surface'>
+++++	default parameters	+++++
buffer         : use buffer region
                  default value: True
check_seed     : if check seeds
                  default value: False
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
fixbulk        : fix atom positions in substrate
                  default value: True
full_ele       : full_ele
                  default value: True
max_attempts   : maximum attempts
                  default value: 50
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
radius         : radius
                  default value: None
slices_file    : file name for slices_file
                  default value: Ref/layerslices.traj
vacuum_thickness: vacuum thickness
                  default value: 10
+++++	requirement parameters	+++++
symprec        : tolerance for symmetry finding
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.individuals.Cluster'>
+++++	default parameters	+++++
check_seed     : if check seeds
                  default value: False
cutoff         : two atoms are "connected" if their distance < cutoff*radius.
                  default value: 1.0
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
full_ele       : full_ele
                  default value: True
max_attempts   : maximum attempts
                  default value: 50
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
radius         : radius
                  default value: None
vacuum_thickness: vacuum thickness surrounding cluster to break pbc when runing calculation
                  default value: 10
weighten       : use weighten atoms when appending or removing atoms
                  default value: True
+++++	requirement parameters	+++++
symprec        : tolerance for symmetry finding
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.individuals.AdClus'>
+++++	default parameters	+++++
check_seed     : if check seeds
                  default value: False
cutoff         : two atoms are "connected" if their distance < cutoff*radius.
                  default value: 1.0
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
dist_clus2surface: distance from cluster to surface
                  default value: 2
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
full_ele       : full_ele
                  default value: True
max_attempts   : maximum attempts
                  default value: 50
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
radius         : radius
                  default value: None
size           : size
                  default value: [1, 1]
substrate      : substrate file name
                  default value: substrate.vasp
vacuum_thickness: vacuum thickness surrounding cluster to break pbc when runing calculation
                  default value: 10
weighten       : use weighten atoms when appending or removing atoms
                  default value: True
+++++	requirement parameters	+++++
symprec        : tolerance for symmetry finding
----------------------------------------------------------------
parameter information for <class 'magus.populations.populations.FixPopulation'>
+++++	default parameters	+++++
check_seed     : if check seed is turned on, we will check your seeds and delete those donot meet requirements
                  default value: False
+++++	requirement parameters	+++++
formula        : formula
pop_size       : population size
results_dir    : path for results
symbols        : symbols
----------------------------------------------------------------
parameter information for <class 'magus.populations.populations.VarPopulation'>
+++++	default parameters	+++++
check_seed     : if check seed is turned on, we will check your seeds and delete those donot meet requirements
                  default value: False
ele_size       : ele_size
                  default value: 0
+++++	requirement parameters	+++++
formula        : formula
pop_size       : population size
results_dir    : path for results
symbols        : symbols
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.individuals.RcsPopulation'>
+++++	default parameters	+++++
check_seed     : if check seed is turned on, we will check your seeds and delete those donot meet requirements
                  default value: False
+++++	requirement parameters	+++++
formula        : formula
pop_size       : population size
results_dir    : path for results
symbols        : symbols
----------------------------------------------------------------
parameter information for <class 'magus.operations.crossovers.CutAndSplicePairing'>
+++++	default parameters	+++++
best_match     : choose best match
                  default value: False
cut_disp       : cut displacement
                  default value: 0
tryNum         : try attempts
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.crossovers.ReplaceBallPairing'>
+++++	default parameters	+++++
cut_range      : cut range
                  default value: [1, 2]
tryNum         : try attempts
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.SoftMutation'>
+++++	default parameters	+++++
bounds         : bounds
                  default value: [0.5, 2.0]
tryNum         : try attempts
                  default value: 10
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.PermMutation'>
+++++	default parameters	+++++
frac_swaps     : possibility to swap
                  default value: 0.5
tryNum         : try attempts
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.LatticeMutation'>
+++++	default parameters	+++++
cell_cut       : coefficient of gauss distribution in cell mutation
                  default value: 1
keep_volume    : whether to keep the volume unchange
                  default value: True
sigma          : Gauss distribution standard deviation
                  default value: 0.1
tryNum         : try attempts
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.RippleMutation'>
+++++	default parameters	+++++
eta            : eta
                  default value: 1
mu             : mu
                  default value: 2
rho            : rho
                  default value: 0.3
tryNum         : try attempts
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.SlipMutation'>
+++++	default parameters	+++++
cut            : cut position
                  default value: 0.5
randRange      : range of movement
                  default value: [0.5, 2]
tryNum         : try attempts
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.RotateMutation'>
+++++	default parameters	+++++
p              : possibility
                  default value: 1
tryNum         : try attempts
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.RattleMutation'>
+++++	default parameters	+++++
d_ratio        : d_ratio
                  default value: 0.7
keep_sym       : if keeps symmetry when rattles
                  default value: None
p              : possibility
                  default value: 0.25
rattle_range   : range of rattle
                  default value: 4
symprec        : tolerance for symmetry finding
                  default value: 0.1
tryNum         : try attempts
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.FormulaMutation'>
+++++	default parameters	+++++
n_candidate    : number of candidates
                  default value: 5
tryNum         : try attempts
                  default value: 10
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.ga.LyrSlipMutation'>
+++++	default parameters	+++++
cut            : cut
                  default value: 0.2
randRange      : randRange
                  default value: [0, 1]
tryNum         : tryNum
                  default value: 10
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.ga.ShellMutation'>
+++++	default parameters	+++++
d              : d
                  default value: 0.23
tryNum         : tryNum
                  default value: 10
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.ga.LyrSymMutation'>
+++++	default parameters	+++++
symprec        : symprec
                  default value: 0.0001
tryNum         : tryNum
                  default value: 10
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.ga.CluSymMutation'>
+++++	default parameters	+++++
symprec        : symprec
                  default value: 0.0001
tryNum         : tryNum
                  default value: 10
----------------------------------------------------------------
parameter information for <class 'magus.calculators.emt.EMTCalculator'>
+++++	default parameters	+++++
eps            : convergence energy
                  default value: 0.05
max_move       : max range of movement
                  default value: 0.1
max_step       : maximum number of relax steps
                  default value: 100
optimizer      : optimizer method, choose from bfgs, fire, lbfgs
                  default value: bfgs
pressure       : pressure
                  default value: 0.0
relax_lattice  : if to relax lattice
                  default value: True
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
work_dir       : work dictionary
----------------------------------------------------------------
parameter information for <class 'magus.calculators.lammps.LammpsCalculator'>
+++++	default parameters	+++++
atomStyle      : atomStyle
                  default value: atomic
exe_cmd        : command line to run lammps
                  default value: 
job_prefix     : job_prefix
                  default value: Lammps
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
save_traj      : save_traj
                  default value: False
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
symbols        : symbols
work_dir       : work dictionary
+++++	requirement_parallel parameters	+++++
num_core       : num_core
queue_name     : quene name
+++++	default_parallel parameters	+++++
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
verbose        : verbose
                  default value: False
wait_time      : wait_time
                  default value: 200
----------------------------------------------------------------
parameter information for <class 'magus.calculators.mtp.MTPNoSelectCalculator'>
+++++	default parameters	+++++
force_tolerance: force_tolerance
                  default value: 0.05
job_prefix     : job_prefix
                  default value: MTP
min_dist       : min_dist
                  default value: 0.5
mode           : choose from parallel or serial
                  default value: parallel
n_epoch        : n_epoch
                  default value: 200
pressure       : pressure
                  default value: 0.0
stress_tolerance: stress_tolerance
                  default value: 1.0
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
symbols        : symbols
work_dir       : work dictionary
+++++	requirement_parallel parameters	+++++
num_core       : num_core
queue_name     : quene name
+++++	default_parallel parameters	+++++
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
verbose        : verbose
                  default value: False
wait_time      : wait_time
                  default value: 200
----------------------------------------------------------------
parameter information for <class 'magus.calculators.mtp.MTPSelectCalculator'>
+++++	default parameters	+++++
force_tolerance: force_tolerance
                  default value: 0.05
ignore_weights : ignore_weights
                  default value: True
job_prefix     : job_prefix
                  default value: MTP
min_dist       : min_dist
                  default value: 0.5
mode           : choose from parallel or serial
                  default value: parallel
n_epoch        : n_epoch
                  default value: 200
n_fail         : n_fail
                  default value: 0
pressure       : pressure
                  default value: 0.0
scaled_by_force: scaled_by_force
                  default value: 0.0
stress_tolerance: stress_tolerance
                  default value: 1.0
weights        : weights
                  default value: [1.0, 0.01, 0.001]
xc             : xc
                  default value: PBE
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
query_calculator: query_calculator
symbols        : symbols
work_dir       : work dictionary
+++++	requirement_parallel parameters	+++++
num_core       : num_core
queue_name     : quene name
+++++	default_parallel parameters	+++++
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
verbose        : verbose
                  default value: False
wait_time      : wait_time
                  default value: 200
----------------------------------------------------------------
parameter information for <class 'magus.calculators.mtp.MTPLammpsCalculator'>
+++++	default parameters	+++++
force_tolerance: force_tolerance
                  default value: 0.05
ignore_weights : ignore_weights
                  default value: True
job_prefix     : job_prefix
                  default value: MTP
min_dist       : min_dist
                  default value: 0.5
mode           : choose from parallel or serial
                  default value: parallel
n_epoch        : n_epoch
                  default value: 200
n_fail         : n_fail
                  default value: 0
pressure       : pressure
                  default value: 0.0
scaled_by_force: scaled_by_force
                  default value: 0.0
stress_tolerance: stress_tolerance
                  default value: 1.0
weights        : weights
                  default value: [1.0, 0.01, 0.001]
xc             : xc
                  default value: PBE
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
query_calculator: query_calculator
symbols        : symbols
work_dir       : work dictionary
+++++	requirement_parallel parameters	+++++
num_core       : num_core
queue_name     : quene name
+++++	default_parallel parameters	+++++
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
verbose        : verbose
                  default value: False
wait_time      : wait_time
                  default value: 200
----------------------------------------------------------------
parameter information for <class 'magus.calculators.quip.QUIPCalculator'>
+++++	default parameters	+++++
eps            : convergence energy
                  default value: 0.05
max_move       : max range of movement
                  default value: 0.1
max_step       : maximum number of relax steps
                  default value: 100
optimizer      : optimizer method, choose from bfgs, fire, lbfgs
                  default value: bfgs
pressure       : pressure
                  default value: 0.0
relax_lattice  : if to relax lattice
                  default value: True
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
work_dir       : work dictionary
----------------------------------------------------------------
parameter information for <class 'magus.calculators.nep.PyNEPCalculator'>
+++++	default parameters	+++++
cutoff         : cutoff
                  default value: [5, 5]
eps            : convergence energy
                  default value: 0.05
generation     : generation
                  default value: 1000
job_prefix     : job_prefix
                  default value: NEP
max_move       : max range of movement
                  default value: 0.1
max_step       : maximum number of relax steps
                  default value: 100
neuron         : neuron
                  default value: 30
optimizer      : optimizer method, choose from bfgs, fire, lbfgs
                  default value: bfgs
pressure       : pressure
                  default value: 0.0
relax_lattice  : if to relax lattice
                  default value: True
version        : version
                  default value: 4
xc             : xc
                  default value: PBE
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
query_calculator: query_calculator
symbols        : symbols
work_dir       : work dictionary
----------------------------------------------------------------
parameter information for <class 'magus.calculators.vasp.VaspCalculator'>
+++++	default parameters	+++++
job_prefix     : job_prefix
                  default value: Vasp
mode           : choose from parallel or serial
                  default value: parallel
pp_label       : pp_label
                  default value: None
pressure       : pressure
                  default value: 0.0
xc             : xc
                  default value: PBE
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
symbols        : symbols
work_dir       : work dictionary
+++++	requirement_parallel parameters	+++++
num_core       : num_core
queue_name     : quene name
+++++	default_parallel parameters	+++++
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
verbose        : verbose
                  default value: False
wait_time      : wait_time
                  default value: 200
----------------------------------------------------------------
parameter information for <class 'magus.calculators.castep.CastepCalculator'>
+++++	default parameters	+++++
castep_command : castep_command
                  default value: castep
castep_pp_path : castep_pp_path
                  default value: None
job_prefix     : job_prefix
                  default value: Castep
kpts           : kpts
                  default value: {'density': 10, 'gamma': True, 'even': False}
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
pspot          : pspot
                  default value: 00PBE
suffix         : suffix
                  default value: usp
xc_functional  : xc_functional
                  default value: PBE
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
symbols        : symbols
work_dir       : work dictionary
+++++	requirement_parallel parameters	+++++
num_core       : num_core
queue_name     : quene name
+++++	default_parallel parameters	+++++
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
verbose        : verbose
                  default value: False
wait_time      : wait_time
                  default value: 200
----------------------------------------------------------------
parameter information for <class 'magus.calculators.lj.LJCalculator'>
+++++	default parameters	+++++
eps            : convergence energy
                  default value: 0.05
max_move       : max range of movement
                  default value: 0.1
max_step       : maximum number of relax steps
                  default value: 100
optimizer      : optimizer method, choose from bfgs, fire, lbfgs
                  default value: bfgs
pressure       : pressure
                  default value: 0.0
relax_lattice  : if to relax lattice
                  default value: True
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
work_dir       : work dictionary
----------------------------------------------------------------
parameter information for <class 'magus.calculators.gulp.GulpCalculator'>
+++++	default parameters	+++++
exe_cmd        : command line to run gulp
                  default value: gulp < input > output
job_prefix     : job_prefix
                  default value: Gulp
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
work_dir       : work dictionary
+++++	requirement_parallel parameters	+++++
num_core       : num_core
queue_name     : quene name
+++++	default_parallel parameters	+++++
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
verbose        : verbose
                  default value: False
wait_time      : wait_time
                  default value: 200
----------------------------------------------------------------
parameter information for <class 'magus.calculators.base.AdjointCalculator'>
+++++	default parameters	+++++
pressure       : pressure
                  default value: 0.0
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
work_dir       : work dictionary
----------------------------------------------------------------
parameter information for <class 'magus.calculators.mtp.TwoShareMTPCalculator'>
+++++	default parameters	+++++
pressure       : pressure
                  default value: 0.0
+++++	requirement parameters	+++++
job_prefix     : calculation dictionary
work_dir       : work dictionary
----------------------------------------------------------------
```
