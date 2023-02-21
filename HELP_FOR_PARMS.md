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

```
which prints valid subcommands. You can also helps for each command line:
**search**
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
**summary**
```shell
$ usage: magus summary [-h] [-p PREC] [-r] [-s] [--need-sort] [-o OUTDIR]
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
**clean**
```shell
$ magus clean -h
usage: magus clean [-h] [-f]

optional arguments:
  -h, --help   show this help message and exit
  -f, --force  rua!!!! (default: False)
```
**prepare**
```shell
$ magus prepare -h
usage: magus prepare [-h] [-v] [-m]

optional arguments:
  -h, --help  show this help message and exit
  -v, --var   variable composition search (default: False)
  -m, --mol   molecule crystal search (default: False)
```
**calculate**
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
**generate**
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
**checkpack**
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
**test**
```shell
$ magus test -h
usage: magus test [-h] [totest]

positional arguments:
  totest      the package to test (default: *)

optional arguments:
  -h, --help  show this help message and exit
```
**update**
```shell
Magus update -h
usage: magus update [-h] [-u] [-f]

optional arguments:
  -h, --help   show this help message and exit
  -u, --user   add --user to pip install (default: False)
  -f, --force  add --force-reinstall to pip install (default: False)
```
**getslabtool**
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
**mutate**
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

# parameter input files

A yaml format parameter file is also necessary. By default is 'input.yaml'. IN THE FUTURE we will add command line to easily export notes and default values by
```shell
$ magus parmhelp
```
to see help for which parameters you can set in 'input.yaml'. For current version they are:
```shell
parameter information for <class 'magus.parameters.magusParameters'>
+++++	Default parameters	+++++
formulaType    : type of formula, choose from fix or var
                  default value: fix
structureType  : structure type, choose from bulk, layer, confined_bulk, cluster, surface
                  default value: bulk
spacegroup     : spacegroup to generate random structures
                  default value: [1-230]
DFTRelax       : DFTRelax
                  default value: False
initSize       : size of first population
                  default value: =popSize
goodSize       : number of good indivials per generation
                  default value: =popSize
molMode        : search molecule clusters
                  default value: False
mlRelax        : use Machine learning relaxation
                  default value: False
symprec        : tolerance for symmetry finding
                  default value: 0.1
bondRatio      : limitation to detect clusters
                  default value: 1.15
eleSize        : used in variable composition mode, control how many boundary structures are generated
                  default value: 0
volRatio       : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 2
dRatio         : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 0.7
molDetector    : methods to detect mol, choose from 1 and 2. See
                 Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, "Determining dimensionalities and multiplicities
                 of crystal nets" npj Comput. Mater. 6, 143 (2020) [doi.org/10.1016/j.fmre.2021.06.005]
                 for more details.
                  default value: 0
addSym         : whether to add symmetry before crossover and mutation
                  default value: True
randRatio      : ratio of new generated random structures in next generation
                  default value: 0.2
chkMol         : use mol dectector
                  default value: False
chkSeed        : check seeds
                  default value: True
diffE          : energy difference to determin structure duplicates
                  default value: 0.01
diffV          : volume difference to determin structure duplicates
                  default value: 0.05
comparator     : comparator, type magus checkpack to see which comparators you have.
                  default value: nepdes
fp_calc        : fingerprints, type magus checkpack to see which fingerprint method you have.
                  default value: nepdes
n_cluster      : number of good individuals per generation
                  default value: =saveGood
autoOpRatio    : automantic GA operation ratio
                  default value: False
autoRandomRatio: automantic random structure generation ratio
                  default value: False
----------------------------------------------------------------
parameter information for <class 'magus.generators.random.MoleculeSPGGenerator'>
+++++	Requirement parameters	+++++
input_mols     : input molecules
formula_type   : type of formula, choose from fix or var
symbols        : atom symbols
formula        : formula
min_n_atoms    : minimum number of atoms per unit cell
max_n_atoms    : maximum number of atoms per unit cell
+++++	Default parameters	+++++
symprec        : tolerance for symmetry finding for molucule
                  default value: 0.1
threshold_mol  : distance between each pair of two molecules in the structure is 
                 not less than (mol_radius1+mol_radius2)*threshold_mol
                  default value: 1.0
max_attempts   : max attempts to generate a random structure
                  default value: 50
p_pri          : probability of generate primitive cell
                  default value: 0.0
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
n_split        : split cell into n_split parts
                  default value: [1]
dimension      : dimension
                  default value: 3
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_volume     : min volume
                  default value: -1
max_volume     : max volume
                  default value: -1
min_n_formula  : minimum formula
                  default value: None
max_n_formula  : maximum formula
                  default value: None
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
spacegroup     : spacegroup to generate random structures
                  default value: [1-230]
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
full_ele       : only generate structures with full elements
                  default value: True
----------------------------------------------------------------
parameter information for <class 'magus.generators.random.LayerSPGGenerator'>
+++++	Requirement parameters	+++++
min_thickness  : minimum thickness
max_thickness  : maximum thickness
formula_type   : type of formula, choose from fix or var
symbols        : atom symbols
formula        : formula
min_n_atoms    : minimum number of atoms per unit cell
max_n_atoms    : maximum number of atoms per unit cell
+++++	Default parameters	+++++
symprec        : symprec
                  default value: 0.1
threshold_mol  : threshold_mol
                  default value: 1.0
spg_type       : spg_type
                  default value: layer
vacuum_thickness: vacuum_thickness
                  default value: 10
max_attempts   : max attempts to generate a random structure
                  default value: 50
p_pri          : probability of generate primitive cell
                  default value: 0.0
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
n_split        : split cell into n_split parts
                  default value: [1]
dimension      : dimension
                  default value: 3
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_volume     : min volume
                  default value: -1
max_volume     : max volume
                  default value: -1
min_n_formula  : minimum formula
                  default value: None
max_n_formula  : maximum formula
                  default value: None
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
spacegroup     : spacegroup to generate random structures
                  default value: [1-230]
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
full_ele       : only generate structures with full elements
                  default value: True
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.generator.ClusterSPGGenerator'>
+++++	Requirement parameters	+++++
formula_type   : type of formula, choose from fix or var
symbols        : atom symbols
formula        : formula
min_n_atoms    : minimum number of atoms per unit cell
max_n_atoms    : maximum number of atoms per unit cell
+++++	Default parameters	+++++
vacuum_thickness: vacuum thickness
                  default value: 10
----------------------------------------------------------------
parameter information for <class 'magus.generators.random.SPGGenerator'>
+++++	Requirement parameters	+++++
formula_type   : type of formula, choose from fix or var
symbols        : atom symbols
formula        : formula
min_n_atoms    : minimum number of atoms per unit cell
max_n_atoms    : maximum number of atoms per unit cell
+++++	Default parameters	+++++
max_attempts   : max attempts to generate a random structure
                  default value: 50
p_pri          : probability of generate primitive cell
                  default value: 0.0
volume_ratio   : cell_volume/SUM(atom_ball_volume) when generating structures (around this number)
                  default value: 1.5
n_split        : split cell into n_split parts
                  default value: [1]
dimension      : dimension
                  default value: 3
ele_size       : number of single compontent structures to
                 generate to decide hull boundarys in variable composition mode
                  default value: 0
min_lattice    : min lattice
                  default value: [-1, -1, -1, -1, -1, -1]
max_lattice    : max lattice
                  default value: [-1, -1, -1, -1, -1, -1]
min_volume     : min volume
                  default value: -1
max_volume     : max volume
                  default value: -1
min_n_formula  : minimum formula
                  default value: None
max_n_formula  : maximum formula
                  default value: None
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
spacegroup     : spacegroup to generate random structures
                  default value: [1-230]
max_ratio      :  max formula ratio in variable composition mode, for example set 10 and Zn11(OH) is not allowed
                  default value: 1000
full_ele       : only generate structures with full elements
                  default value: True
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.generator.SurfaceGenerator'>
+++++	Requirement parameters	+++++
formula_type   : type of formula, choose from fix or var
symbols        : atom symbols
formula        : formula
min_n_atoms    : minimum number of atoms per unit cell
max_n_atoms    : maximum number of atoms per unit cell
+++++	Default parameters	+++++
randwalk_range : maximum range of random walk
                  default value: 0.5
randwalk_ratio : ratio of random walk atoms
                  default value: 0.3
rcs_x          : size[x] of reconstruction
                  default value: [1]
rcs_y          : size[y] of reconstruction
                  default value: [1]
buffer         : use buffer layer
                  default value: True
rcs_formula    : formula of surface region
                  default value: None
spg_type       : generate with planegroup/layergroup
                  default value: plane
+++++	slabinfo parameters	+++++
bulk_file      : file of bulk structure
                  default value: None
cutslices      : bulk_file contains how many atom layers
                  default value: 2
bulk_layernum  : number of atom layers in substrate region
                  default value: 3
buffer_layernum: number of atom layers in buffer region
                  default value: 3
rcs_layernum   : number of atom layers in top surface region
                  default value: 2
direction      : Miller indices of surface direction, i.e.[1,0,0]
                  default value: None
rotate         : R
                  default value: 0
matrix         : matrix notation
                  default value: None
addH           : passivate bottom surface with H
                  default value: False
pcell          : use primitive cell
                  default value: True
+++++	modification parameters	+++++
adsorb         : adsorb atoms to cleaved surface
                  default value: {}
clean          : clean cleaved surface
                  default value: {}
defect         : add defect to cleaved surface
                  default value: {}
----------------------------------------------------------------
parameter information for <class 'magus.populations.individuals.Bulk'>
+++++	Requirement parameters	+++++
symprec        : tolerance for symmetry finding
+++++	Default parameters	+++++
mol_detector   : methods to detect mol, choose from 1 and 2. See
                 Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, "Determining dimensionalities and multiplicities
                 of crystal nets" npj Comput. Mater. 6, 143 (2020) [doi.org/10.1016/j.fmre.2021.06.005]
                 for more details.
                  default value: 0
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
max_attempts   : maximum attempts
                  default value: 50
check_seed     : if check seeds
                  default value: False
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
radius         : radius
                  default value: None
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
full_ele       : full_ele
                  default value: True
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
----------------------------------------------------------------
parameter information for <class 'magus.populations.individuals.Layer'>
+++++	Requirement parameters	+++++
symprec        : tolerance for symmetry finding
+++++	Default parameters	+++++
vacuum_thickness: vacuum_thickness
                  default value: 10
bond_ratio     : bond_ratio
                  default value: 1.1
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
max_attempts   : maximum attempts
                  default value: 50
check_seed     : if check seeds
                  default value: False
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
radius         : radius
                  default value: None
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
full_ele       : full_ele
                  default value: True
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
----------------------------------------------------------------
parameter information for <class 'magus.populations.individuals.ConfinedBulk'>
+++++	Requirement parameters	+++++
symprec        : tolerance for symmetry finding
+++++	Default parameters	+++++
vacuum_thickness: vacuum thickness
                  default value: 10
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
max_attempts   : maximum attempts
                  default value: 50
check_seed     : if check seeds
                  default value: False
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
radius         : radius
                  default value: None
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
full_ele       : full_ele
                  default value: True
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.individuals.Surface'>
+++++	Requirement parameters	+++++
symprec        : tolerance for symmetry finding
+++++	Default parameters	+++++
vacuum_thickness: vacuum thickness
                  default value: 10
buffer         : use buffer region
                  default value: True
fixbulk        : fix atom positions in substrate
                  default value: True
slices_file    : file name for slices_file
                  default value: Ref/layerslices.traj
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
max_attempts   : maximum attempts
                  default value: 50
check_seed     : if check seeds
                  default value: False
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
radius         : radius
                  default value: None
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
full_ele       : full_ele
                  default value: True
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.individuals.Cluster'>
+++++	Requirement parameters	+++++
symprec        : tolerance for symmetry finding
+++++	Default parameters	+++++
vacuum_thickness: vacuum thickness surrounding cluster to break pbc when runing calculation
                  default value: 10
cutoff         : two atoms are "connected" if their distance < cutoff*radius.
                  default value: 1.0
weighten       : use weighten atoms when appending or removing atoms
                  default value: True
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
max_attempts   : maximum attempts
                  default value: 50
check_seed     : if check seeds
                  default value: False
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
radius         : radius
                  default value: None
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
full_ele       : full_ele
                  default value: True
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.individuals.AdClus'>
+++++	Requirement parameters	+++++
symprec        : tolerance for symmetry finding
+++++	Default parameters	+++++
substrate      : substrate file name
                  default value: substrate.vasp
dist_clus2surface: distance from cluster to surface
                  default value: 2
size           : size
                  default value: [1, 1]
vacuum_thickness: vacuum thickness surrounding cluster to break pbc when runing calculation
                  default value: 10
cutoff         : two atoms are "connected" if their distance < cutoff*radius.
                  default value: 1.0
weighten       : use weighten atoms when appending or removing atoms
                  default value: True
n_repair_try   : attempts to repair structures when doing GA operation
                  default value: 5
max_attempts   : maximum attempts
                  default value: 50
check_seed     : if check seeds
                  default value: False
min_lattice    : min_lattice
                  default value: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0]
max_lattice    : max_lattice
                  default value: [99, 99, 99, 135, 135, 135]
d_ratio        : distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*d_ratio
                  default value: 1.0
distance_matrix: distance between each pair of two atoms in the structure is
                 not less than (radius1+radius2)*distance_matrix[1][2]
                  default value: None
radius         : radius
                  default value: None
max_forces     : if forces of a structure larger is than this number it will be deleted.
                  default value: 50.0
max_enthalpy   : if enthalpy of a structure is larger than this number it will be deleted.
                  default value: 100.0
full_ele       : full_ele
                  default value: True
max_length_ratio: if max-cell-length/min-cell-length of a structure is larger than this number it will be deleted.
                  default value: 8
----------------------------------------------------------------
parameter information for <class 'magus.populations.populations.FixPopulation'>
+++++	Requirement parameters	+++++
results_dir    : path for results
pop_size       : population size
symbols        : symbols
formula        : formula
+++++	Default parameters	+++++
check_seed     : if check seed is turned on, we will check your seeds and delete those donot meet requirements
                  default value: False
----------------------------------------------------------------
parameter information for <class 'magus.populations.populations.VarPopulation'>
+++++	Requirement parameters	+++++
results_dir    : path for results
pop_size       : population size
symbols        : symbols
formula        : formula
+++++	Default parameters	+++++
check_seed     : if check seed is turned on, we will check your seeds and delete those donot meet requirements
                  default value: False
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.individuals.RcsPopulation'>
+++++	Requirement parameters	+++++
results_dir    : path for results
pop_size       : population size
symbols        : symbols
formula        : formula
+++++	Default parameters	+++++
check_seed     : if check seed is turned on, we will check your seeds and delete those donot meet requirements
                  default value: False
----------------------------------------------------------------
parameter information for <class 'magus.operations.crossovers.CutAndSplicePairing'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 50
cut_disp       : cut displacement
                  default value: 0
best_match     : choose best match
                  default value: False
----------------------------------------------------------------
parameter information for <class 'magus.operations.crossovers.ReplaceBallPairing'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 50
cut_range      : cut range
                  default value: [1, 2]
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.SoftMutation'>
+++++	Default parameters	+++++
tryNum         : tryNum
                  default value: 50
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.PermMutation'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 50
frac_swaps     : possibility to swap
                  default value: 0.5
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.LatticeMutation'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 50
sigma          : Gauss distribution standard deviation
                  default value: 0.1
cell_cut       : coefficient of gauss distribution in cell mutation
                  default value: 1
keep_volume    : whether to keep the volume unchange
                  default value: True
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.RippleMutation'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 50
rho            : rho
                  default value: 0.3
mu             : mu
                  default value: 2
eta            : eta
                  default value: 1
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.SlipMutation'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 50
cut            : cut position
                  default value: 0.5
randRange      : range of movement
                  default value: [0.5, 2]
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.RotateMutation'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 50
p              : possibility
                  default value: 1
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.RattleMutation'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 50
p              : possibility
                  default value: 0.25
rattle_range   : range of rattle
                  default value: 4
d_ratio        : d_ratio
                  default value: 0.7
keep_sym       : if keeps symmetry when rattles
                  default value: None
symprec        : tolerance for symmetry finding
                  default value: 0.1
----------------------------------------------------------------
parameter information for <class 'magus.operations.mutations.FormulaMutation'>
+++++	Default parameters	+++++
tryNum         : try attempts
                  default value: 10
n_candidate    : number of candidates
                  default value: 5
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.ga.LyrSlipMutation'>
+++++	Default parameters	+++++
tryNum         : tryNum
                  default value: 10
cut            : cut
                  default value: 0.2
randRange      : randRange
                  default value: [0, 1]
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.ga.ShellMutation'>
+++++	Default parameters	+++++
tryNum         : tryNum
                  default value: 10
d              : d
                  default value: 0.23
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.ga.LyrSymMutation'>
+++++	Default parameters	+++++
tryNum         : tryNum
                  default value: 10
symprec        : symprec
                  default value: 0.0001
----------------------------------------------------------------
parameter information for <class 'magus.reconstruct.ga.CluSymMutation'>
+++++	Default parameters	+++++
tryNum         : tryNum
                  default value: 10
symprec        : symprec
                  default value: 0.0001
----------------------------------------------------------------
parameter information for <class 'magus.calculators.emt.EMTCalculator'>
+++++	Requirement parameters	+++++
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default parameters	+++++
eps            : convergence energy
                  default value: 0.05
max_step       : maximum number of relax steps
                  default value: 100
optimizer      : optimizer method, choose from bfgs, fire, lbfgs
                  default value: bfgs
max_move       : max range of movement
                  default value: 0.1
relax_lattice  : if to relax lattice
                  default value: True
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.lammps.LammpsCalculator'>
+++++	Default parameters	+++++
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
exe_cmd        : command line to run lammps
                  default value: 
save_traj      : save_traj
                  default value: False
atomStyle      : atomStyle
                  default value: atomic
job_prefix     : job_prefix
                  default value: Lammps
+++++	Requirement_parallel parameters	+++++
queue_name     : quene name
num_core       : num_core
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default_parallel parameters	+++++
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
wait_time      : wait_time
                  default value: 200
verbose        : verbose
                  default value: False
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.mtp.MTPNoSelectCalculator'>
+++++	Default parameters	+++++
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
force_tolerance: force_tolerance
                  default value: 0.05
stress_tolerance: stress_tolerance
                  default value: 1.0
min_dist       : min_dist
                  default value: 0.5
n_epoch        : n_epoch
                  default value: 200
job_prefix     : job_prefix
                  default value: MTP
+++++	Requirement_parallel parameters	+++++
queue_name     : quene name
num_core       : num_core
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default_parallel parameters	+++++
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
wait_time      : wait_time
                  default value: 200
verbose        : verbose
                  default value: False
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.mtp.MTPSelectCalculator'>
+++++	Default parameters	+++++
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
xc             : xc
                  default value: PBE
weights        : weights
                  default value: [1.0, 0.01, 0.001]
scaled_by_force: scaled_by_force
                  default value: 0.0
force_tolerance: force_tolerance
                  default value: 0.05
stress_tolerance: stress_tolerance
                  default value: 1.0
min_dist       : min_dist
                  default value: 0.5
n_epoch        : n_epoch
                  default value: 200
ignore_weights : ignore_weights
                  default value: True
job_prefix     : job_prefix
                  default value: MTP
n_fail         : n_fail
                  default value: 0
+++++	Requirement_parallel parameters	+++++
queue_name     : quene name
num_core       : num_core
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default_parallel parameters	+++++
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
wait_time      : wait_time
                  default value: 200
verbose        : verbose
                  default value: False
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.mtp.MTPLammpsCalculator'>
+++++	Default parameters	+++++
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
xc             : xc
                  default value: PBE
weights        : weights
                  default value: [1.0, 0.01, 0.001]
scaled_by_force: scaled_by_force
                  default value: 0.0
force_tolerance: force_tolerance
                  default value: 0.05
stress_tolerance: stress_tolerance
                  default value: 1.0
min_dist       : min_dist
                  default value: 0.5
n_epoch        : n_epoch
                  default value: 200
ignore_weights : ignore_weights
                  default value: True
job_prefix     : job_prefix
                  default value: MTP
n_fail         : n_fail
                  default value: 0
+++++	Requirement_parallel parameters	+++++
queue_name     : quene name
num_core       : num_core
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default_parallel parameters	+++++
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
wait_time      : wait_time
                  default value: 200
verbose        : verbose
                  default value: False
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.quip.QUIPCalculator'>
+++++	Requirement parameters	+++++
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default parameters	+++++
eps            : convergence energy
                  default value: 0.05
max_step       : maximum number of relax steps
                  default value: 100
optimizer      : optimizer method, choose from bfgs, fire, lbfgs
                  default value: bfgs
max_move       : max range of movement
                  default value: 0.1
relax_lattice  : if to relax lattice
                  default value: True
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.vasp.VaspCalculator'>
+++++	Default parameters	+++++
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
xc             : xc
                  default value: PBE
pp_label       : pp_label
                  default value: None
job_prefix     : job_prefix
                  default value: Vasp
+++++	Requirement_parallel parameters	+++++
queue_name     : quene name
num_core       : num_core
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default_parallel parameters	+++++
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
wait_time      : wait_time
                  default value: 200
verbose        : verbose
                  default value: False
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.castep.CastepCalculator'>
+++++	Default parameters	+++++
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
xc_functional  : xc_functional
                  default value: PBE
pspot          : pspot
                  default value: 00PBE
suffix         : suffix
                  default value: usp
job_prefix     : job_prefix
                  default value: Castep
kpts           : kpts
                  default value: {'density': 10, 'gamma': True, 'even': False}
castep_command : castep_command
                  default value: castep
castep_pp_path : castep_pp_path
                  default value: None
+++++	Requirement_parallel parameters	+++++
queue_name     : quene name
num_core       : num_core
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default_parallel parameters	+++++
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
wait_time      : wait_time
                  default value: 200
verbose        : verbose
                  default value: False
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.lj.LJCalculator'>
+++++	Requirement parameters	+++++
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default parameters	+++++
eps            : convergence energy
                  default value: 0.05
max_step       : maximum number of relax steps
                  default value: 100
optimizer      : optimizer method, choose from bfgs, fire, lbfgs
                  default value: bfgs
max_move       : max range of movement
                  default value: 0.1
relax_lattice  : if to relax lattice
                  default value: True
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.gulp.GulpCalculator'>
+++++	Default parameters	+++++
mode           : choose from parallel or serial
                  default value: parallel
pressure       : pressure
                  default value: 0.0
exe_cmd        : command line to run gulp
                  default value: gulp < input > output
job_prefix     : job_prefix
                  default value: Gulp
+++++	Requirement_parallel parameters	+++++
queue_name     : quene name
num_core       : num_core
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default_parallel parameters	+++++
pre_processing : serves to add any sentence you wish when submiting the job to change system variables, load modules etc.
                  default value: 
wait_time      : wait_time
                  default value: 200
verbose        : verbose
                  default value: False
kill_time      : kill_time
                  default value: 100000
num_parallel   : num_parallel
                  default value: 1
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.base.AdjointCalculator'>
+++++	Requirement parameters	+++++
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default parameters	+++++
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------
parameter information for <class 'magus.calculators.mtp.TwoShareMTPCalculator'>
+++++	Requirement parameters	+++++
work_dir       : work dictionary
job_prefix     : calculation dictionary
+++++	Default parameters	+++++
pressure       : pressure
                  default value: 0.0
----------------------------------------------------------------

