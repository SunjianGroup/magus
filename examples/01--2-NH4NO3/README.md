Example 1.2  
Generate 3d periodic crystal structures of molecule crystal with 8 NH4 and NO3 molecules per unit cell by symmetry Pccn (56)  
=====================================================================
```shell  
$ ls  
 input.yaml NH4.xyz NO3.xyz  
```  
"NH4.xyz" and "NO3.xyz" are .xyz format molecule files.  
```shell  
$ cat NH4.xyz  
 5  
 H32 N16 O24  
 H    4.511281    4.375470    3.210227  
 H    3.584655    4.486488    1.796710  
 H    4.670180    3.191076    2.019142  
 H    3.246077    3.272899    2.937012  
 N    4.000271    3.837356    2.488938    
$ cat NO3.xyz  
 4  
 H32 N16 O24  
 N    2.012707    2.014563    4.870574  
 O    1.714319    0.953807    5.478185  
 O    2.311095    3.075319    5.478185  
 O    2.012707    2.014563    3.582428  
```  
Consistent with the former example, "input.yaml" sets all parameters:  
```shell  
$ cat input.yaml  
 #Generate 3d periodic crystal structures of molecule crystal with 8 NH4 and NO3 molecules per unit cell by symmetry.  
 formulaType: fix  
 structureType: bulk  
 symbols: ['H', 'N', 'O']  
 molMode: True           #use molecule crystal  
 inputMols: ['NH4.xyz', 'NO3.xyz']  
 formula: [1,1]              #formula: ((NH4)1(NO3)1)n  
 min_n_atoms: 72  
 max_n_atoms: 72  
 spacegroup: [56]     #symmetry spacegroup  
 d_ratio: 0.5  
 volume_ratio: 15  
 threshold_mol: 1.0      #Note-1  
```  
#Note-1: "threshold_mol": distance between each pair of two molecules in the structure is not less than (mol_radius1+mol_radius2)*threshold_mol  
  
Submit job to generate 10 structures and output to “gen.traj”:  
```shell
$ magus generate -i input.yaml -o gen.traj -n 10  
```  
Summary the result file by:  
```shell
$ magus summary gen.traj -s  
      symmetry enthalpy    formula priFormula  
 1   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 2   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 3   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 4   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 5   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 6   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 7   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 8   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 9   Pccn (56)     None  (H4N2O3)8  (H4N2O3)8  
 10  Pccn (56)     None  (H4N2O3)8  (H4N2O3)8
```