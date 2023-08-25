Example 3.4  
GAsearch of fixed composition MgAlO under high pressure by GULP  
=====================================================  
```shell
$ ls
 inputFold/  input.yaml
```  
Consistent with former examples, "input.yaml" sets all parameters and most of them work similarly.  
```shell  
$ cat input.yaml  
 #GAsearch of fixed composition MgAlO.
 formulaType: fix        
 structureType: bulk
 pressure: 100
 initSize: 40        # number of structures of 1st generation
 popSize: 40         # number of structures of per generation
 numGen: 40          # number of total generation
 saveGood: 4         # number of good structures kept to the next generation
 #structure parameters
 symbols: ['Mg','Al','O']
 formula: [4,8,16]         
 min_n_atoms: 28              # minimum number of atoms per unit cell
 max_n_atoms: 28              # maximum number of atoms per unit cell
 spacegroup: [2-230] 
 volume_ratio: 1.7
 d_ratio: 0.7
 #GA parameters
 rand_ratio: 0.3               # fraction of random structures per generation (except 1st gen.)
 choice_func: 'exp'            # The probability of being selected as a parent is related to 
 k: 0.25                       # exp^-(k*its_enthalpy_domination).            
 history_punish: 0.9           # Avoid an identical structure selected as a parent for too many times
 autoOpRatio: True             # Auto adjust the probability of GA operations
 auto_random_ratio: True       # Auto adjust rand_ratio since 3rd generation
 OffspringCreator:
  rattle:
   prob: -1                    # Auto set by 1.0 - Sum(others) = 1.0-0.14*5 = 0.3
  cutandsplice:
   prob: 0.14
  lattice:
   prob: 0.14
  perm:
   prob: 0.14
  ripple:
   prob: 0.14
  slip:
   prob: 0.14
#main calculator settings
 MainCalculator:
  calculator: 'gulp'
  jobPrefix: ['Gulp1', 'Gulp2', 'Gulp3', 'Gulp4']
  #gulp settings
  exeCmd: gulp < input > output           #command to run gulp in your system
  #parallel settings
  numParallel: 8              # number of parallel jobs
  numCore: 4                # number of cores
  preProcessing: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4    
  queueName: e52692v2ib! 
```  
Submit search job and summary the results:  
```shell
$ magus search -i input.yaml -ll DEBUG  
$ magus summary results/good.traj -s  
```
We obtained best structure Pnma (62) with enthalpy -23.395043eV/atom:
```shell
$ cat POSCAR_1.vasp  
 O Mg Al 
 1.0000000000000000
     2.6883670000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    9.3461569999999998    0.0000000000000000
     0.0000000000000000    0.0000000000000000    8.0021810000000002
 O   Mg  Al 
  16   4   8
 Direct
  0.5000000000000000  0.0000000000000000  0.0000000000000000
  0.0000000000000000  0.5000000000000000  0.4723620000000000
  0.5000000000000000  0.2889740000000000  0.4053020000000000
  1.0000000000000000  0.1972810000000000  0.8614310000000001
  0.5000000000000000  0.5447910000000000  0.1996100000000000
  0.5000000000000000  0.2348880000000000  0.1109310000000000
  1.0000000000000000  0.7889740000000000  0.0670610000000000
  0.0000000000000000  0.7348880000000000  0.3614310000000001
  1.0000000000000000  0.4321690000000000  0.9723619999999999
  0.0000000000000000  0.0447910000000000  0.2727530000000001
  0.5000000000000000  0.6431950000000001  0.9053020000000001
  0.5000000000000000  0.3873780000000001  0.6996100000000000
  0.5000000000000000  0.9321690000000000  0.5000000000000000
  1.0000000000000000  0.1431950000000000  0.5670610000000001
  1.0000000000000000  0.8873780000000001  0.7727530000000001
  0.5000000000000000  0.6972810000000000  0.6109310000000000
  0.5000000000000000  0.8608890000000001  0.2379180000000000
  0.0000000000000000  0.3608890000000000  0.2344450000000000
  0.5000000000000000  0.0712800000000000  0.7379180000000000
  1.0000000000000000  0.5712800000000001  0.7344450000000000
  0.5000000000000000  0.1002680000000000  0.4093420000000000
  0.0000000000000000  0.8295510000000000  0.5643190000000000
  0.5000000000000000  0.3295510000000000  0.9080440000000001
  0.5000000000000000  0.6026180000000000  0.4080440000000001
  0.5000000000000000  0.8319010000000001  0.9093420000000000
  1.0000000000000000  0.1026180000000000  0.0643190000000000
  1.0000000000000000  0.3319010000000000  0.5630200000000000
  0.0000000000000000  0.6002680000000001  0.0630200000000000
```