import os
from magus.calculators.base import ClusterCalculator
from magus.utils import *


class LammpsCalculator(ClusterCalculator):
    def __init__(self, parameters, prefix='Lammps'):
        super().__init__(parameters, prefix)
        Requirement = ['symbols']
        Default = {'exeCmd':'', 'jobPrefix': 'Lammps'}
        checkParameters(self.p,parameters, Requirement, Default)

    def lammps_job(self, calcDic, jobName):
        with open('lammpsSetup.yaml', 'w') as f:
            f.write(yaml.dump(calcDic))
        with open('parallel.sh', 'w') as f:
            f.write(
                "#BSUB -q {0}\n"
                "#BSUB -n {1}\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J {2}\n"
                "{3}\n"
                "python -m magus.runscripts.runlammps lammpsSetup.yaml"
                "".format(self.p.queueName, self.p.numCore, jobName, self.p.Preprocessing))
        self.J.bsub('bsub < parallel.sh',jobName)

    def scfjob(self):
        calcDic = {
            'calcNum': 0,
            'pressure': self.p.pressure,
            'exeCmd': self.p.exeCmd,
            'inputDir': "{}/inputFold".format(self.p.workDir),
            'numCore': self.p.numCore,
            'symbol_to_type': {j: i for i, j in enumerate(self.p.symbols)},
            'type_to_symbol': {i: j for i, j in enumerate(self.p.symbols)},
        }
        index = os.getcwd().split('/')[-1]
        jobName = self.p.jobPrefix + '_s_' + str(index)
        self.lammps_job(calcDic, jobName)

    def relaxjob(self):
        calcDic = {
            'calcNum': self.p.calcNum,
            'pressure': self.p.pressure,
            'exeCmd': 'mpirun -np {} {}'.format(self.p.numCore, self.p.exeCmd),
            'inputDir': "{}/inputFold".format(self.p.workDir),
            'symbol_to_type': {j: i for i, j in enumerate(self.p.symbols)},
            'type_to_symbol': {i: j for i, j in enumerate(self.p.symbols)},
        }
        index = os.getcwd().split('/')[-1]
        jobName = self.p.jobPrefix + '_r_' + str(index)
        self.lammps_job(calcDic, jobName)
