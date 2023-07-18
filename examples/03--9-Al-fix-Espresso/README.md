Example 3.9
GAsearch of fixed composition Al (4 atoms per cell) by Espresso
================================================  
```shell  
$ ls
 inputFold/  input.yaml
```
Set parameters in "input.yaml":
```shell  
$ cat input.yaml
#GAsearch of fixed composition Al (4 atoms per cell) by Espresso
...  
#main calculator settings 
MainCalculator:
 calculator: 'espresso'
 jobPrefix: ['PWscf1']
 ppLabel: ['Al_ONCV_PBE_sr.upf']
...

# please specify the Pseudopotential file names in file input.yaml
# for "Al" with a Norm_conserving pseudopotential, it is "ppLabel: ['Al_ONCV_PBE_sr.upf']"
```
Notes:
```shell
# please specify the directory of the Pseudopotential files in file inputFold/PWscf1/pw.relax
# see example 3.2 to find more informations about the GAsearch
```

