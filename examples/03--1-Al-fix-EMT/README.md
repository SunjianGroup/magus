Example 3.1
GAsearch of fixed composition Al (12 atoms per cell) by EMT.

$ ls
    inputFold/  input.yaml 

Consistent with former examples, "input.yaml" sets all parameters and most of them work similarly.
Unique parameters for EMT search include:

$ cat input.yaml
    ...
    #main calculator settings
    MainCalculator:
     calculator: 'emt'
     jobPrefix: ['EMT'] 
     #emt relax settings
     eps: 0.05                      # convergence energy < 0.05
     maxStep: 30                # maximum number of relax steps
     maxMove: 0.1                # maximum relax step length
     optimizer: bfgs            # use bfgs as optimizer

Submit search job:
$ magus search -i input.yaml -ll DEBUG

Several EMT calculations will be carried and summary the result by:
$ magus summary results/good.traj -s
           symmetry  enthalpy formula priFormula
1    P6_3/mmc (194) -0.008824    Al12        Al2
2         Cmcm (63) -0.007991    Al12        Al2
3       P2_1/m (11) -0.007648    Al12        Al6
4            Pm (6) -0.007403    Al12        Al6
5        R-3m (166) -0.005918    Al12        Al4
6        R-3m (166) -0.005508    Al12        Al6
7       Pca2_1 (29) -0.005325    Al12       Al12
8       P2_1/m (11) -0.005191    Al12        Al2
9    P6_3/mmc (194) -0.005041    Al12        Al6
10      Fm-3m (225) -0.004883    Al12         Al

We obtained best structure with enthalpy -0.008824 per atom with symmetry P6_3/mmc (194):
$ cat POSCAR_1.vasp 
Al 
 1.0000000000000000
     8.4641407318216721    0.0000729888981805    0.0000000000012648
    -2.8213381037806213    4.8879656864421017   -0.0000000000003647
     0.0000000000006679    0.0000000000000512    4.4714281595647325
 Al 
  12
Direct
  0.3125109738750370  0.0696798750464143  0.7499999999999812
  0.2014203348899876  0.2363972238022523  0.2499999999999864
  0.3125109738750100  0.5696798750463858  0.7499999999999841
  0.2014203348899946  0.7363972238022585  0.2499999999999920
  0.6458443072083505  0.0696798750463894  0.7499999999999807
  0.5347536682233250  0.2363972238022533  0.2499999999999847
  0.6458443072083709  0.5696798750464176  0.7499999999999728
  0.5347536682233274  0.7363972238022533  0.2499999999999848
  0.9791776405417000  0.0696798750463926  0.7499999999999741
  0.8680870015566671  0.2363972238022491  0.2499999999999847
  0.9791776405416995  0.5696798750464102  0.7499999999999793
  0.8680870015566652  0.7363972238022437  0.2499999999999812

Note: We used EMT potential in the ASE package in order to have a quick demonstration. 
For more accurate searches, we also run this example with VASP in Example 3.1. You may notice these two examples didn't give same results.
FYI, we also calculated structures we got in Example 3.1 by VASP with EMT:
$ magus calculate 3.1/results/good.traj -o 3_1.traj
$ magus summary 3_1.traj 
            symmetry  enthalpy formula priFormula  
    1   P6_3/mmc (194) -0.008823    Al12        Al2 
    2   P6_3/mmc (194) -0.008805    Al12        Al2 
    3   P6_3/mmc (194) -0.008793    Al12        Al2 
    4   P6_3/mmc (194) -0.008789    Al12        Al2 
    5   P6_3/mmc (194) -0.008766    Al12        Al2 
    6   P6_3/mmc (194) -0.008650    Al12        Al2 
    7       R-3m (166) -0.006510    Al12        Al3 
    8       R-3m (166) -0.006452    Al12        Al3 
    9        C2/m (12) -0.005858    Al12        Al3 
    10       C2/m (12) -0.005643    Al12        Al4 
So we got the same ground state structure by two methods.
