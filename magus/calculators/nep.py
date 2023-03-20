import logging
import os
import shutil

import numpy as np
from pynep.calculate import NEP
from pynep.io import load_nep, dump_nep
from pynep.select import FarthestPointSample


from magus.calculators.base import ASECalculator, ClusterCalculator
from magus.utils import CALCULATOR_PLUGIN, check_parameters
from magus.populations.populations import Population

log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('nep-noselect')
class PyNEPCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.relax_calc = NEP("{}/nep.txt".format(self.input_dir))
        self.scf_calc = NEP("{}/nep.txt".format(self.input_dir))


@CALCULATOR_PLUGIN.register('nep')
class PyNEPCalculator(ASECalculator, ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['query_calculator', 'symbols']
        Default = {
            'xc': 'PBE',
            'job_prefix': 'NEP',
            'version': 4,
            'generation': 1000,
            'neuron': 30,
            'cutoff': [5, 5]
        }
        check_parameters(self, parameters, Requirement, Default)

        # copy files
        self.ml_dir = "{}/mlFold/{}".format(self.work_dir, self.job_prefix)
        if not os.path.exists(self.ml_dir):
            os.makedirs(self.ml_dir)
        if not os.path.exists('{}/nep.txt'.format(self.ml_dir)):
            try:
                shutil.copy('{}/nep.txt'.format(self.input_dir),
                            '{}/nep.txt'.format(self.ml_dir))
                self.relax_calc = NEP("{}/nep.txt".format(self.ml_dir))
                self.scf_calc = NEP("{}/nep.txt".format(self.ml_dir))
            except:
                log.warning("No initial nep.txt.")
        if not os.path.exists('{}/train.xyz'.format(self.ml_dir)):
            try:
                shutil.copy('{}/train.xyz'.format(self.input_dir),
                            '{}/train.xyz'.format(self.ml_dir))
                # same training and test set
                shutil.copy('{}/train.xyz'.format(self.ml_dir),
                            '{}/test.xyz'.format(self.ml_dir))
            except:
                log.info("No inital train.xyz")
                os.system("touch {}/train.xyz'.format(self.ml_dir)")
                os.system("touch {}/test.xyz'.format(self.ml_dir)")


    def train(self):
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        # write an input
        lines = []
        lines.append("version "+str(self.version))
        lines.append("generation "+str(self.generation))
        lines.append("neuron "+str(self.neuron))
        type = ' '.join([str(len(self.symbols))] + self.symbols)
        lines.append("type "+type)
        lines.append(f"cutoff {str(self.cutoff[0])} {str(self.cutoff[1])}")
        with open("nep.in", "w") as f:
            f.write("\n".join(lines))
        self.J.sub("nep < nep.in", name='train', file='train.sh',
                   out='train-out', err='train-err')
        self.J.wait_jobs_done(self.wait_time)
        self.J.clear()
        # read trained pot in
        self.relax_calc = NEP("{}/nep.txt".format(self.ml_dir))
        self.scf_calc = NEP("{}/nep.txt".format(self.ml_dir))
        os.chdir(nowpath)

    def select(self, pop):
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        if not os.path.exists("nep.txt"):
            log.warning("No nep.txt, so we have to select all the pop.")
            return pop
        pot = NEP("nep.txt")
        des_current = np.array(
            [np.mean(pot.get_property('descriptor', atoms), axis=0) for atoms in self.trainset])
        des_new = np.array(
            [np.mean(pot.get_property('descriptor', atoms), axis=0) for atoms in pop])
        sampler = FarthestPointSample(min_distance=0.05)
        ret = [pop[i] for i in sampler.select(des_current, des_new)]
        os.chdir(nowpath)
        if isinstance(pop, Population):
            return pop.__class__(ret)
        return ret

    def get_loss(self, frames):
        nep_result = self.calc_efs(frames)
        nep_e = np.array([atoms.info['energy'] / len(atoms)
                         for atoms in nep_result])
        dft_e = np.array([atoms.info['energy'] / len(atoms)
                         for atoms in frames])
        mae = np.abs(nep_e-dft_e).mean()
        return [mae]

    def calc_efs(self, frames):
        if isinstance(frames, Atoms):
            frames = [frames]
        # need to copy, prevent losing dft info
        return self.scf_([atoms.copy() for atoms in frames])

    @property
    def trainset(self):
        try:
            return load_nep('{}/train.xyz'.format(self.ml_dir), ftype="exyz")
        except:
            log.warning("No training set now.")
            return []

    def updatedataset(self, frames):
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        dump_nep('tmp.xyz', frames, ftype='exyz')
        os.system("cat tmp.xyz >> train.xyz")
        os.system("cat tmp.xyz >> test.xyz")
        os.chdir(nowpath)
