Example 1.1  
Generate 3d periodic crystal structures of boron with 12 atoms per unit cell by symmetry  
=====================================================================  
```shell  
$ ls  
 input.yaml  
```  
"input.yaml" sets all parameters used in our program.    
```shell  
$ cat input.yaml  
 #Generate 3d periodic crystal structures of boron with 12 atoms per unit cell by symmetry.  
 formulaType: fix  
 structureType: bulk  
 #structure parameters  
 symbols: ['B']  
 formula: [1]  
 min_n_atoms: 12            # minimum number of atoms per unit cell  
 max_n_atoms: 12             # maximum number of atoms per unit cell  
 spacegroup: [1-230]     #symmetry spacegroup  
 d_ratio: 0.5                  #Note-1  
 volume_ratio: 8            #Note-2  
```  
#Note-1: "d_ratio": distance between each pair of two atoms in the structure is not less than (radius1+radius2)*d_ratio  
#Note-2: "volume_ratio": In our program, volume-ratio of each structure is calculated by cell_volume / SUM(atom_ball_volume).  
When generating random structures, we set the lower limit and the upper limit of volume-ratio to 0.5 and 1.5 times this number to avoid too loose or dense arrangement of atom positions.  
  
Submit job to generate 10 structures and output to “gen.traj”:  
```shell  
$ magus generate -i input.yaml -o gen.traj -n 10  
```  
Summary the result file by:  
```shell  
$ magus summary gen.traj -s  
    symmetry enthalpy formula priFormula  
 1   I4_1md (109)     None     B12         B6  
 2      Pbam (55)     None     B12        B12  
 3      P2/c (13)     None     B12        B12  
 4      Ccc2 (37)     None     B12         B6  
 5    C222_1 (20)     None     B12         B6  
 6    Pca2_1 (29)     None     B12        B12  
 7     P6_3 (173)     None     B12        B12  
 8     P6_1 (169)     None     B12        B12  
 9    Fm-3m (225)     None     B12         B3  
 10  P4/mmm (123)     None     B12         B6  
```    
"-s" option means each structure is saved to the current dictionary named POSCAR_1.vasp, POSCAR_2.vasp ... POSCAR_10.vasp.  
```shell  
$ cat POSCAR_1.vasp  
 B  
 1.0000000000000000  
 -3.1932700070323072    0.0000000000000000    0.0000000000000000  
 0.0000000000000000   -3.1932700070323072    0.0000000000000000  
 0.0000000000000000    0.0000000000000000   11.6816672382506361  
 B  
 12  
 Direct  
 -0.0000000000000000 -0.0000000000000000  0.8759497082214569  
 0.4999999999999999  0.4999999999999999  0.3759497082214570  
 0.4999999999999999 -0.0000000000000000  0.1259497082214569  
 -0.0000000000000000  0.4999999999999999  0.6259497082214569  
 -0.0000000000000000 -0.0000000000000000  0.0566660156737389  
 0.4999999999999999  0.4999999999999999  0.5566660156737389  
 0.4999999999999999 -0.0000000000000000  0.3066660156737389  
 -0.0000000000000000  0.4999999999999999  0.8066660156737389  
 -0.0000000000000000 -0.0000000000000000  0.4356586697677424  
 0.4999999999999999  0.4999999999999999  0.9356586697677424  
 0.4999999999999999 -0.0000000000000000  0.6856586697677425  
 -0.0000000000000000  0.4999999999999999  0.1856586697677423  
```