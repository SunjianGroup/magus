Example 6.1  
Surface reconstruction of diamond (100)-2×1  
========================================  
```shell
$ ls  
 diamond.vasp  inputFold/  input.yaml  
$ cat diamond.vasp  
 C  
 1.0000000000000000  
     3.5737100000000002    0.0000000000000000    0.0000000000000000  
     0.0000000000000000    3.5737100000000002    0.0000000000000000  
     0.0000000000000000    0.0000000000000000    3.5737100000000002  
 C  
 8  
 Cartesian  
 0.8934275000000000  2.6802825000000001  0.8934275000000000  
 0.0000000000000000  0.0000000000000000  1.7868550000000001  
 0.8934275000000000  0.8934275000000000  2.6802825000000001  
 0.0000000000000000  1.7868550000000001  0.0000000000000000  
 2.6802825000000001  2.6802825000000001  2.6802825000000001  
 1.7868550000000001  0.0000000000000000  0.0000000000000000  
 2.6802825000000001  0.8934275000000000  0.8934275000000000  
 1.7868550000000001  1.7868550000000001  1.7868550000000001  
```  
An input file describing the bulk structure of target system which can be read by ASE is necessary.  
Consistent with former examples, "input.yaml" sets all parameters and most of them work similarly.  
Unique parameters for surface searching include:  
```shell  
$ cat input.yaml  
 ...  
 structureType: surface  
 vacuum_thickness: 10    #vacuum thickness of both sides of slab model  
 rcs_x: [2]                      #size-x  
 rcs_y: [1]                        #size-y  
 #bulk structure in /bulk_file/ contains /cutslices/ atom layers.  
 #a slab model contains /bulk_layernum/ layers in bulk region, /buffer_layernum/ layers in buffer region and  
 #/rcs_layernum/ layers on the top will be built.  
 slabinfo:  
  bulk_file: "dimond.vasp"  
  cutslices: 2  
  bulk_layernum: 2  
  buffer_layernum: 1  
  rcs_layernum: 1  
  pcell: True                 #Use primitive cell  
  direction: [1,0,0]  
 ...  
```  
matrix notation to build the slab is also supported:  
```shell
 slabinfo:  
  matrix: [[1,0], [0,1]]  
```  
For testing purposes, low precision calculations are used (see 'inputFold/vasp1/INCAR').  
  
use the following command to check the slab model (optional):  
```shell
$ magus rcstool --getslab  
```
'slab.vasp' is generated and you can adjust some parameters when build the slab.  
(!) Please REMEMBER TO DELETE 'Ref/' if slab parms are changed.  
  
Submit search job:  
```shell
$ magus search -i input.yaml -ll DEBUG  
```  
Several vasp calculations will be carried and summary the result by:  
```shell
$ magus summary results/good.traj -s  
     symmetry    origin                          Eo      energy  
 1  Pmm2 (25)    rand.randmove  -2.815324 -122.330600  
 2     Pm (6)    rand.randmove  -1.598609 -121.113885  
 3     Pm (6)    RattleMutation   0.318038 -119.197238  
 4     P1 (1)    RattleMutation   2.013119 -117.502158  
```  
  
"Eo" stands for surface energy in fixed composition mode. In this mode, our program firstly calculates  
    1) E_bulk = energy of bulk_file  
    2) E_slab = energy of slab model    
Eo = E_ind - E_slab - SUM(num_of_adsorption_atoms*E_bulk/num_atoms_in_bulk)  
  
After above steps several candidate surface structures are given.  
To obtain more accurate surface energy, expand layer numbers in the substrate and use higher precision calculations for candidates.  
Here we get best candidate:  
```shell  
$ cat POSCAR_1.vasp  
 C  H  C  H  C  
 1.0000000000000000  
     0.0000000000000000   -3.5737100000000002    3.5737100000000002  
     0.0000000000000000   -1.7868550000000001   -1.7868550000000001  
 28.6474200000000003    0.0000000000000000    0.0000000000000000  
 C   H   C   H   C  
 8   1   3   1   3  
 Selective dynamics  
 Direct  
 0.0170393648648477  0.4988875135661464  0.5670087057559872   T   T   T  
 0.1128397026490967  0.0000822846237379  0.5903941376783096   T   T   T  
 0.4844762596904407  0.4993156764955805  0.5669245238147956   T   T   T  
 0.3880829310339974 -0.0016922997574677  0.5902401998900512   T   T   T  
 0.2503294217009658 -0.0010175913729076  0.5011621622958197   T   T   T  
 0.2507140005784200  0.4995737249111734  0.5304614308808483   T   T   T  
 0.7506753766686631 -0.0014461327011420  0.5061512277008781   T   T   T  
 0.7509883512774344  0.4983652479009783  0.5393004065217253   T   T   T  
 0.2500000000000000 -0.0000000000000000  0.3790111081556375   F   F   F  
 -0.0000000000000000  0.5000000000000000  0.4413851369512543   F   F   F  
 0.2500000000000000  0.5000000000000000  0.4101981225534460   F   F   F  
 -0.0000000000000000 -0.0000000000000000  0.4725721513490555   F   F   F  
 0.7500000000000000 -0.0000000000000000  0.3790111081556375   F   F   F  
 0.5000000000000000  0.5000000000000000  0.4413851369512543   F   F   F  
 0.7500000000000000  0.5000000000000000  0.4101981225534460   F   F   F  
 0.5000000000000000 -0.0000000000000000  0.4725721513490555   F   F   F  
```
