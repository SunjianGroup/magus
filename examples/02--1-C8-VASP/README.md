Example 2.1  
Structure relaxation of diamond by vasp interface  
=========================================  
```shell
$ ls  
 inputFold/  input.yaml diamond.vasp  
```  
Set parameters in "input.yaml":  
```shell
$ cat input.yaml  
 symbols: ['C']  
 #main calculator settings  
 MainCalculator:  
  calculator: 'vasp'  
  jobPrefix: ['VASP']  
  #vasp settings  
  xc: PBE  
  ppLabel: ['']  
  #parallel settings  
  preProcessing: export PATH=$PATH  
  numParallel: 20  
  numCore: 24  
  queueName: name  
```  
Parallel settings set parameters to submit VASP calculation jobs to LSF systems.  
The above parameters submit jobs run in 24 cores each, up to 20 jobs at the same time (if more than one structures are to calculate) to queueName.  
preProcessing string serves to add any sentence you wish when submiting the job to change system variables, load modules etc.  
i.e., our program AUTOMATICALLY generates job files like:  
```shell
$ cat relax.sh  
 #BSUB -q name  
 #BSUB -n 24  
 #BSUB -o relax-out  
 #BSUB -e relax-err  
 #BSUB -J VASP_r_0  
 #BSUB -W 1666  
 export PATH=$PATH  
 python -m magus.calculators.vasp vaspSetup.yaml initPop.traj optPop.traj  
 [[ $? -eq 0 ]] && touch DONE || touch ERROR(base)  
```
And submits to LSF system and AUTOMATICALLY collects results.  
  
A user needs to prepare vasp INCAR file in inputFold/$jobPrefix:  
```shell
$ cat inputFold/VASP/INCAR  
 SYSTEM = C  
 PREC = Accurate  
 EDIFF = 1e-4  
 IBRION = 2  
 ISIF = 3  
 NSW = 40  
 ISMEAR = 0  
 SIGMA = 0.050  
 POTIM = 0.250  
 ISTART = 0  
 LCHARG = FALSE  
 LWAVE = FALSE  
 EDIFFG = 1e-3  
 KSPACING = 0.314  
 NCORE= 4  
```  
Prepare structure to relax:  
```shell  
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
Submit job:  
```shell
$ magus calculate diamond.vasp -o out.traj  
```  
After vasp calculations are finished, summary result:  
```shell
$ magus summary out.traj -s  
 symmetry  enthalpy formula priFormula  
 1  Fd-3m (227) -9.097279      C8         C2  
```  
We get the relaxed structure of symmetry Fd-3m (227), energy of -9.097279eV/atom:  
```shell
$ cat POSCAR_1.vasp  
 C  
 1.0000000000000000  
 3.5624724060878035    0.0000000000000000    0.0000000000000000  
 0.0000000000000000    3.5624724060878035   -0.0000000000000000  
 0.0000000000000000   -0.0000000000000000    3.5624724060878035  
 C  
 8  
 Direct  
 0.2500000000000000  0.7500000000000000  0.2500000000000000  
 0.0000000000000000  0.0000000000000000  0.5000000000000000  
 0.2500000000000000  0.2500000000000000  0.7500000000000000  
 0.0000000000000000  0.5000000000000000  0.0000000000000000  
 0.7500000000000000  0.7500000000000000  0.7500000000000000  
 0.5000000000000000  0.0000000000000000  0.0000000000000000  
 0.7500000000000000  0.2500000000000000  0.2500000000000000  
 0.5000000000000000  0.5000000000000000  0.5000000000000000  
```