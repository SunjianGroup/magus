# NEP search

NEP is a MLFF. Go to its [website](https://gpumd.org/) for more information.

You need to install `pyNEP`, which is available on github:  
https://github.com/bigd4/PyNEP

A typical input file:

```yaml
MLCalculator:
  jobPrefix: NEP
  numCore: 1
  pre_processing: |
    #BSUB -gpu "num=1" 
    module purge
    module load gcc/7.4.0 ips/2018u4 cuda/11.2.0
  queueName: 83a100ib
  version: 3
  generation: 1000
  neuron: 30
  cutoff: [5, 5]
```

## Key words

- `version`: NEP version.
- `generation`: How many training iters each generation. You need to check whether it is enough for training to converge. 1000 is small, just for showcase.
- `neuron`: How many neurons in NEP model. 30 is enough.
- `cutoff`: The cutoff value of descriptor.

Check NEP manual for more information.
