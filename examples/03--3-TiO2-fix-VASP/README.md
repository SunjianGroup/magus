Example 3.3  
GAsearch of fixed composition TiO2 (12 atoms per cell)  
=========================================
```shell  
$ ls  
 inputFold/  input.yaml  
```  
Consistent with former examples, "input.yaml" sets all parameters and most of them work similarly.  
```shell  
$ cat input.yaml  
 #GAsearch of fixed composition TiO2 (12 atoms per cell).  
 formulaType: fix  
 structureType: bulk  
 pressure: 0  
 initSize: 20        # number of structures of 1st generation  
 popSize: 20         # number of structures of per generation  
 numGen: 10          # number of total generation  
 saveGood: 3         # number of good structures kept to the next generation  
 #structure parameters  
 symbols: ["Ti", "O"]  
 formula: [1, 2]  
 min_n_atoms: 12              # minimum number of atoms per unit cell  
 max_n_atoms: 12              # maximum number of atoms per unit cell  
 spacegroup: [2-230]  
 d_ratio: 0.6  
 volume_ratio: 3  
 #GA parameters  
 rand_ratio: 0.3               # fraction of random structures per generation (except 1st gen.)  
 add_sym: True               # add symmetry to each structure during evolution  
 #main calculator settings  
 MainCalculator:  
  calculator: 'vasp'  
  jobPrefix: ['VASP1', 'VASP2', 'VASP3', 'VASP4']  
  #vasp settings  
  xc: PBE  
  ppLabel: ['','_s']  
  #parallel settings  
  numParallel: 10              # number of parallel jobs  
  numCore: 24                # number of cores  
  queueName: name  
```  
Submit search job:  
```shell
$ magus search -i input.yaml -ll DEBUG  
```  
Several vasp calculations will be carried and summary the result by:  
```shell
$ magus summary results/good.traj -a energy -s  
         symmetry  enthalpy  formula priFormula      energy  
 1   I4_1/amd (141) -8.831266  (O2Ti)4    (O2Ti)2 -105.975189  
 2        Pbcn (60) -8.806694  (O2Ti)4    (O2Ti)4 -105.680332  
 3   P4_2/mnm (136) -8.804505  (O2Ti)4    (O2Ti)2 -105.654062  
 4        P2/m (10) -8.803832  (O2Ti)4    (O2Ti)4 -105.645980  
 5        Pmmn (59) -8.803583  (O2Ti)4    (O2Ti)4 -105.642999  
 6     P4_2nm (102) -8.802507  (O2Ti)4    (O2Ti)2 -105.630084  
 7           Cm (8) -8.784565  (O2Ti)4    (O2Ti)4 -105.414778  
 8      P2_1/c (14) -8.780317  (O2Ti)4    (O2Ti)4 -105.363805  
 9   I4_1/amd (141) -8.778052  (O2Ti)4    (O2Ti)2 -105.336620  
 10       Pnma (62) -8.777406  (O2Ti)4    (O2Ti)4 -105.328870  
```  
We obtained best structure I4_1/amd (141) with energy -105.975189eV:  
```shell
$ cat POSCAR_1.vasp  
 O Ti  
 1.0000000000000000  
    5.4388119991605031   -1.1690237938155603    0.0000000000000000  
    -1.5748525106095808    5.1610417074312629    0.0000000000000000  
    0.0000000000000000    0.0000000000000000    5.3964132138366159  
 O   Ti  
 8   4  
 Direct  
 0.2824158127588741  0.1498623907420390  0.8761454280668156  
 0.6077295948491369  0.0627030148600110  0.6263891269527470  
 0.7832305363925544  0.1527092675484613  0.1263891269527470  
 0.1085443184828100  0.0655498916664333  0.3761454280668156  
 0.1077295948491365  0.5627030148600110  0.8736108730472530  
 0.7824158127588741  0.6498623907420394  0.6238545719331844  
 0.6085443184828099  0.5655498916664328  0.1238545719331844  
 0.2832305363925547  0.6527092675484613  0.3736108730472530  
 0.1954938525606264  0.3577474059038677  0.1244270324423253  
 0.6954938525606263  0.8577474059038674  0.3755729675576748  
 0.6954662786810647  0.3576648765046047  0.8755729675576750  
 0.1954662786810646  0.8576648765046049  0.6244270324423250  
```