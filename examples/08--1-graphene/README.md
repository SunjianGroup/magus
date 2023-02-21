Example 8.1  
GAsearch of 2D carbon (graphene) by VASP  
=========================================
```shell  
$ ls  
 inputFold/  input.yaml  
```  
Consistent with former examples, "input.yaml" sets all parameters and most of them work similarly.  
```shell  
$ cat input.yaml  
 #GAsearch of 2D carbon (graphene) by VASP.
 formulaType: fix        
 structureType: layer
 dimension: 2
 initSize: 40        # number of structures of 1st generation
 popSize: 40         # number of structures of per generation
 numGen: 20          # number of total generation
 saveGood: 5         # number of good structures kept to the next generation
 #structure parameters
 symbols: ["C"]
 formula: [1]                
 min_n_atoms: 4              # minimum number of atoms per unit cell
 max_n_atoms: 12              # maximum number of atoms per unit cell
 spacegroup: [2-17]            # use plane group no. 2-17
 min_thickness: 1                # minimum cell thickness
 max_thickness: 2              # maximum cell thickness
 spg_type: plane                  # Note-1
 vacuum_thickness: 15
 d_ratio: 0.6
 volume_ratio: 3
 #GA parameters
 rand_ratio: 0.4               # fraction of random structures per generation (except 1st gen.)
 add_sym: True               # add symmetry to each structure during evolution
 #main calculator settings
 MainCalculator:
  calculator: 'vasp'
  jobPrefix: ['VASP1','VASP2', 'VASP3', 'VASP4']
  #vasp settings
  xc: PBE
  ppLabel: ['']
  #parallel settings
  numParallel: 8              # number of parallel jobs
  numCore: 8                # number of cores
  queueName: name  
```  
**#Note-1: In this example we used plane group to generate random structures. In this situation we are most likely to get single-layer structures (if not folded during local relaxation). If you want to generate and search multi-layers, use spg_type: layer to use layergroup. (no.1-80)**  
Submit search job:  
```shell
$ magus search -i input.yaml -ll DEBUG  
```  
Several vasp calculations will be carried and summary the result by:  
```shell
$ magus summary results/good.traj -a energy -s  
   symmetry  enthalpy formula priFormula
 1    P6/mmm (191) -9.228931      C8         C2
 2       Pmmm (47) -8.926389     C10        C10
 3       Cmmm (65) -8.910854     C12         C6
 4     P-6m2 (187) -8.819556     C10        C10
 5     P-62m (189) -8.761912     C12        C12
 6    P4/mmm (123) -8.713044      C4         C4
 7    P6/mmm (191) -8.583558     C12        C12
 8     P-62m (189) -8.577817     C10        C10
 9     P-6m2 (187) -8.538271      C4         C4
 10      Pmma (51) -8.498755     C12        C12

```  
We obtained best structure P6/mmm (191) with enthalpy -9.228931eV:  
```shell
$ cat POSCAR_1.vasp  
 C 
 1.0000000000000000
 4.9114209632398831    0.0000000000000000    0.0000000000000000
 -2.4557104816199415    4.2534153228451759    0.0000000000000000
 0.0000000000000000    0.0000000000000000   15.0000000000000160
 C  
 8
 Direct
 0.1665455102971287  0.2767662376654696  0.5000000100000008
 0.3334544897028642  0.1102207273683338  0.5000000100000008
 0.6665455102971286  0.2762658446043058  0.5000000100000008
 0.8334544897028713  0.1097203343071771  0.5000000100000008
 0.1665455102971287  0.7767662376654696  0.5000000100000008
 0.3334544897028641  0.6102207273683337  0.5000000100000008
 0.6665455102971286  0.7762658446043058  0.5000000100000008
 0.8334544897028713  0.6097203343071770  0.5000000100000008
```  
You can also find our several best results in result_ref dir.