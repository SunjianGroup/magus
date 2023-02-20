Example 3.2  
GAsearch of fixed composition Al (4 atoms per cell) by VASP  
================================================  
```shell  
$ ls  
 inputFold/  input.yaml  
```  
Set parameters in "input.yaml":  
```shell  
$ cat input.yaml  
 #GAsearch of fixed composition Al (4 atoms per cell) by VASP  
 ...  
 #main calculator settings  
 MainCalculator:  
  calculator: 'vasp'  
  jobPrefix: ['VASP1', 'VASP2', 'VASP3', 'VASP4']    #Note-1  
  #vasp settings  
  xc: PBE  
  ppLabel: ['']  
  #parallel settings  
  numParallel: 10              # number of parallel jobs  
  numCore: 24                # number of cores  
  queueName: name  
  ...  
```  
#Note-1: Since initial structures are often disordered, usually multiple INCARs are used for stepwise optimization.  
Firstly optimize atomic positions and lattice shapes with fixed volume (ISIF=4) (See inputFold/VASP1/INCAR for more details),  
followed by free optimization of atomic positions and lattices (ISIF=3) (See inputFold/VASP2/INCAR and inputFold/VASP3/INCAR for more details),  
and finally calculate high precision self-consistent single point energy. (NSW=0) (See inputFold/VASP4/INCAR and inputFold/VASP3/INCAR for more details).  
  
Submit search job:  
```shell  
$ magus search -i input.yaml -ll DEBUG  
```  
Several vasp calculations will be carried and summary the result by:  
```shell  
$ magus summary results/good.traj -a energy -s  
      symmetry  enthalpy formula priFormula     energy  
 1   I4/mmm (139) -3.760592     Al4         Al -15.042367  
 2    Fm-3m (225) -3.746651     Al4         Al -14.986604  
 3      C2/m (12) -3.730817     Al4         Al -14.923266  
 4        P-1 (2) -3.726298     Al4         Al -14.905191  
 5     R-3m (166) -3.726182     Al4         Al -14.904726  
 6      Immm (71) -3.697638     Al4         Al -14.790552  
 7        P-1 (2) -3.696733     Al4         Al -14.786933  
 8      Cmcm (63) -3.690156     Al4        Al2 -14.760622  
 9      C2/c (15) -3.687762     Al4        Al2 -14.751047  
 10     C2/m (12) -3.681332     Al4         Al -14.725329  
```  
We obtained best structure I4/mmm (139) with energy -15.042367eV:  
```shell  
$ cat POSCAR_1.vasp  
 Al  
 1.0000000000000000  
 4.1434177141810924    0.0000000000000000    0.0000000000000000  
 0.0000000000000000    4.0016506124448048    0.0000000000000000  
 0.0000000000000000    0.0000000000000000    4.0005917472665296  
 Al  
 4  
 Direct  
 0.0000000000000000  0.5000053175929308  0.0000000000000000  
 0.0000000000000000  0.0000053175929312  0.5000000000000000  
 0.5000000000000000  0.4999946824070688  0.5000000000000000  
 0.5000000000000000 -0.0000053175929310  0.0000000000000000  
```  
Here we also provide input file and results for fixed composition Al (12 atoms per cell):  
```shell  
$ magus search -i 12at.yaml -ll DEBUG  
$ magus summary results/good.traj -s  
        symmetry  enthalpy formula priFormula     energy  
 1      I4/mmm (139) -3.757072    Al12         Al -45.084863  
 2         Immm (71) -3.749610    Al12         Al -44.995325  
 3       Fm-3m (225) -3.746010    Al12         Al -44.952117  
 4         C2/m (12) -3.738640    Al12         Al -44.863676  
 5        R-3m (166) -3.734951    Al12         Al -44.819411  
 6    P6_3/mmc (194) -3.733351    Al12        Al4 -44.800214  
 7         C2/m (12) -3.728369    Al12         Al -44.740422  
 8           P-1 (2) -3.727234    Al12         Al -44.726805  
 9         Immm (71) -3.726891    Al12         Al -44.722694  
 10        Cmcm (63) -3.725921    Al12        Al4 -44.711057  
$ cat POSCAR_1.vasp  
 Al  
 1.0000000000000000  
    11.9973322173354298    0.0000000000000000   -0.0000000000000000  
    0.0000000000000000    4.0710326351310133    0.0000000000000000  
    -0.0000000000000000    0.0000000000000000    4.0710326351310133  
 Al  
 12  
 Direct  
 0.1666666666666643  0.5000000000000000  0.0000000000000000  
 0.1666666666666643  0.0000000000000000  0.5000000000000000  
 0.0000000000000000  0.5000000000000000  0.5000000000000000  
 0.0000000000000000  0.0000000000000000  0.0000000000000000  
 0.5000000000000000  0.5000000000000000  0.0000000000000000  
 0.5000000000000000  0.0000000000000000  0.5000000000000000  
 0.3333333333333357  0.5000000000000000  0.5000000000000000  
 0.3333333333333357  0.0000000000000000  0.0000000000000000  
 0.8333333333333357  0.5000000000000000  0.0000000000000000  
 0.8333333333333357  0.0000000000000000  0.5000000000000000  
 0.6666666666666643  0.5000000000000000  0.5000000000000000  
 0.6666666666666643  0.0000000000000000  0.0000000000000000  
 ```