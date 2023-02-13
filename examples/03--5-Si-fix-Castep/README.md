# Castep Interface

```yaml
MainCalculator:
  jobPrefix: ["first"]
  calculator: castep
  mode: parallel
  # castep settings
  xc_functional: PBE
  kpts: { "density": 10, "gamma": True, "even": False }
  castep_command: castep
  castep_pp_path: /fs08/home/js_pansn/apps/CASTEP-22.11/Test/Pseudopotentials
  pspot: 00PBE
  suffix: usp
```

This interface use `ASE`'s castep interface. Click [here](https://wiki.fysik.dtu.dk/ase/) for more information.
