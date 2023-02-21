Example 4  
#GAsearch of molecule crystal CH4 with 4 molecules per unit cell  
=======================================================  
```shell$ ls  
 inputFold/  input.yaml CH4.xyz  
```  
Set parameters in "input.yaml":  
```shell  
$ cat input.yaml  
 formulaType: fix  
 structureType: bulk  
 pressure: 50  
 initSize: 20        # number of structures of 1st generation  
 popSize: 20         # number of structures of per generation  
 numGen: 40          # number of total generation  
 saveGood: 3         # number of good structures kept to the next generation  
 #structure parameters  
 symbols: ['C', 'H']  
 molMode: True           #mode of molecule crystal search  
 inputMols: ['CH4.xyz']  
 formula: [4]  
 min_n_atoms: 20              # minimum number of atoms per unit cell  
 max_n_atoms: 20              # maximum number of atoms per unit cell  
 spacegroup: [2-230]  
 d_ratio: 0.8  
 volume_ratio: 5  
 #GA parameters  
 rand_ratio: 0.4               # fraction of random structures per generation (except 1st gen.)  
 add_sym: True               # add symmetry to each structure during evolution  
 chkMol: True                    #Note-1  
 molDetector: 1                  #Note-1  
 #main calculator settings  
 MainCalculator:  
  calculator: 'vasp'  
  jobPrefix: ['Vasp1', 'Vasp2']  
  #vasp settings  
  xc: PBE  
  ppLabel: ['','']  
  #parallel settings  
  numParallel: 4              # number of parallel jobs  
  numCore: 6                # number of cores  
  queueName: name  
```  
#Note-1: if chkMol is turned on, our program will detect molecules or clusters inside periodic crystals using quotient graphs.  
Two ways are provided, namely molDetector 1 and 2, see   
    Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, "Determining dimensionalities and multiplicities of crystal nets" npj Comput. Mater. 6, 143 (2020) [doi.org/10.1016/j.fmre.2021.06.005]  
for more details.  
  
Prepare input molecule file $inputMols:  
```shell
$ cat CH4.xyz   
 5  
 C  H   
 C    2.260984    1.227715    2.255654  
 H    2.597307    0.217093    2.238728  
 H    1.194544    1.227505    2.236584  
 H    2.611534    1.725297    1.379429  
 H    2.590593    1.698611    3.156207  
```  
Submit search job:  
```shell
$ magus search -i input.yaml -ll DEBUG  
```  
Several vasp calculations will be carried and summary the result by:  
```shell
$ magus summary results/good.traj -s -rm priFormula  
         symmetry  enthalpy  formula     energy  
 1       P2_1/m (11) -3.287447  (C4H16) -88.480202  
 2           P-1 (2) -3.285680  (C4H16) -88.485286  
 3       P2_1/c (14) -3.285622  (C4H16) -88.484906  
 4         Cmcm (63) -3.285494  (C4H16) -88.488090  
 5   P2_12_12_1 (19) -3.285427  (C4H16) -88.619558  
 6          P2_1 (4) -3.285114  (C4H16) -88.621070  
 7           P-1 (2) -3.283942  (C4H16) -88.462630  
 8            P1 (1) -3.283371  (C4H16) -88.483374  
 9         Pbcm (57) -3.283040  (C4H16) -88.550971  
 10      Pmn2_1 (31) -3.282654  (C4H16) -88.548796  
```  
So the best structure we obtained is P2_1/m (11) with energy -88.480202eV.  
```shell
$ cat cat POSCAR_1.vasp   
 H  C   
 1.0000000000000000  
     -6.1045784273275121    0.0000000000000000    0.6951502283368540  
     0.0000000000000000    3.8063975737848823    0.0000000000000000  
     -3.4851990466988325    0.0000000000000000   -2.7378152859577871  
 H   C  
 16   4  
 Direct  
 0.1855108949262191  0.7524811448448626  0.7887625385632374  
 0.2871291211990609  0.7499868588013624  0.1379533594474318  
 0.4792953505415852  0.0137484683967097  0.3404972509646488  
 0.0207046494584153  0.5137484683967171  0.6595027490353512  
 0.5207046494584160  0.9862515316032900  0.6595027490353512  
 0.9792953505415838  0.4862515316032901  0.3404972509646488  
 0.3144891050737810  0.2524811448448642  0.2112374614367629  
 0.6855108949262194  0.7475188551551375  0.7887625385632374  
 0.2128708788009394  0.2499868588013631  0.8620466405525701  
 0.4811326916756208  0.4827936340789404  0.3421989630915355  
 1.0188673083243798  0.9827936340789407  0.6578010369084645  
 0.5188673083243868  0.5172063659210595  0.6578010369084645  
 -0.0188673083243864  0.0172063659210597  0.3421989630915352  
 0.7128708788009402  0.2500131411986372  0.8620466405525701  
 0.7871291211990596  0.7500131411986376  0.1379533594474317  
 0.8144891050737805  0.2475188551551432  0.2112374614367629  
 1.0026172844154677  0.7495494643163968  0.8069241714185612  
 -0.0026172844154666  0.2504505356836036  0.1930758285814321  
 0.5026172844154669  0.7504505356836033  0.8069241714185683  
 0.4973827155845337  0.2495494643163965  0.1930758285814322  
```