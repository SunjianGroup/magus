Example 3.6  
GAsearch of variable composition Znx(OH)y  
========================================  
```shell
$ ls  
 inputFold/  input.yaml  
```  
Set parameters in "input.yaml":  
```shell  
$ cat input.yaml  
 #GAsearch of variable composition Znx(OH)y.  
 formulaType: var  
 structureType: bulk  
 pressure: 0  
 initSize: 150        # number of structures of 1st generation  
 popSize: 100         # number of structures of per generation  
 numGen: 40          # number of total generation  
 saveGood: 8         # number of good structures kept to the next generation  
 #structure parameters  
 symbols: ['Zn','O','H']  
 formula: [[1,0,0],[0,1,1]]         #Zn: (OH) = (0~1):1  
 min_n_atoms: 8              # minimum number of atoms per unit cell  
 max_n_atoms: 16              # maximum number of atoms per unit cell  
 full_ele: True                #structure must contain all elements  
 spacegroup: [2-230]  
 d_ratio: 0.5  
 volume_ratio: 10  
 #GA parameters  
 rand_ratio: 0.3               # fraction of random structures per generation (except 1st gen.)  
 add_sym: True               # add symmetry to each structure during evolution  
 #main calculator settings  
 MainCalculator:  
  calculator: 'gulp'  
  jobPrefix: ['Gulp1', 'Gulp2', 'Gulp3']  
  #gulp settings  
  exeCmd: gulp < input > output      #Note-1  
  #parallel settings  
  numParallel: 10              # number of parallel jobs  
  numCore: 4                # number of cores  
  preProcessing: export OMP_NUM_THREADS=4; export MKL_NUM_THREADS=4  
  queueName: name  
```  
#Note-1: exeCmd is command to run gulp in your system.  
  
Submit search job:  
```shell
$ magus search -i input.yaml -ll DEBUG  
```  
Several gulp calculations will be carried and summary the result by:  
```shell
$ magus summary results/good.traj -sb ehull -a ehull  
     symmetry  enthalpy   formula priFormula     energy     ehull  
 1       C2 (5)    -1.700  Zn13(HO)   Zn13(HO) -25.493731  0.000000  
 2       P1 (1)    -3.681   Zn(HO)6    Zn(HO)6 -47.854825  0.000000  
 3       P1 (1)    -3.673  Zn3(HO)4   Zn3(HO)4 -40.398082  0.000000  
 4       P1 (1)    -1.670  Zn14(HO)   Zn14(HO) -26.719714  0.000000  
 5       P1 (1)    -3.305  Zn4(HO)3   Zn4(HO)3 -33.051814  0.000000  
 6       P1 (1)    -3.749  Zn2(HO)6   Zn2(HO)6 -52.492753  0.000000  
 7       P1 (1)    -3.774  Zn2(HO)4   Zn2(HO)4 -37.742902  0.000000  
 8       P1 (1)    -3.772  Zn2(HO)5   Zn2(HO)5 -45.258533  0.000000  
 9       P1 (1)    -3.668   Zn(HO)7    Zn(HO)7 -55.014814  0.000000  
 10      P1 (1)    -3.528  Zn4(HO)4   Zn4(HO)4 -42.334411  0.000000  
 11      P1 (1)    -1.698  Zn13(HO)   Zn13(HO) -25.463444  0.002000  
 12      C2 (5)    -3.747  Zn2(HO)6   Zn2(HO)6 -52.460941  0.002000  
 13      P1 (1)    -3.746  Zn2(HO)6    Zn(HO)3 -52.441497  0.003000  
 14      Cc (9)    -3.746  Zn2(HO)6   Zn2(HO)6 -52.438716  0.003000  
 15      Cm (8)    -3.745  Zn2(HO)6   Zn2(HO)6 -52.427101  0.004000  
 16   Fdd2 (43)    -3.769  Zn2(HO)4   Zn2(HO)4 -37.686376  0.005000  
 17      P1 (1)    -3.743  Zn2(HO)6   Zn2(HO)6 -52.399117  0.006000  
 18      Cm (8)    -3.698  Zn2(HO)3   Zn2(HO)3 -29.584924  0.006562  
 19      P1 (1)    -1.726  Zn12(HO)   Zn12(HO) -24.163123  0.006755  
 20      P1 (1)    -3.764  Zn2(HO)5   Zn2(HO)5 -45.168564  0.008000  
```  
We use a convex hull to check whether a component will decompose into a mixed phase of neighboring stable components.  
Only structures with ehull = 0 are stable (or to say, they are the most stable among structures investigated by our program.)    
So the best structures we obtained are No.1-10.  
But also notice we only searched formula Znx(OH)y while there are also components like ZnxOyHz so best structures above are not granted stable compared to ZnxOyHz structures.    
You can find our result for this example in result_ref dir.