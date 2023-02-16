Example 3.4  
GAsearch of fixed composition MgAlO under high pressure by GULP  
=====================================================  
```shell
$ ls  
  
```  
Consistent with former examples, "input.yaml" sets all parameters and most of them work similarly.  
```shell  
$ cat input.yaml  
```
 #GAsearch of fixed composition MgAlO.  
 formulaType: fix  
 structureType: bulk  
 pressure: 100  
 initSize: 40        # number of structures of 1st generation  
 popSize: 40         # number of structures of per generation  
 numGen: 60          # number of total generation  
 saveGood: 2         # number of good structures kept to the next generation  
 #structure parameters  
 symbols: ['Mg','Al','O']  
 formula: [4,8,16]  
 min_n_atoms: 28              # minimun number of atoms per unit cell  
 max_n_atoms: 28              # maxium number of atoms per unit cell  
 spacegroup: [2-230]  
 d_ratio: 0.5  
 volume_ratio: 3  
 #GA parameters  
 rand_ratio: 0.3               # fraction of random structures per generation (except 1st gen.)  
 add_sym: False               # add symmetry to each structure during evolution  
 #main calculator settings  
 MainCalculator:  
  calculator: 'gulp'  
  jobPrefix: ['Gulp1', 'Gulp2', 'Gulp3', 'Gulp4']  
  #gulp settings  
  exeCmd: gulp < input > output   #command to run gulp in your system  
  #parallel settings  
  numParallel: 8              # number of parallel jobs  
  numCore: 4                # number of cores  
  preProcessing: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4  
  queueName: name  
```  
Submit search job and summary the results:  
```shell
$ magus search -i input.yaml -ll DEBUG  
$ magus summary results/good.traj -s  
```