import logging, yaml
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN
from ase.calculators.lj import LennardJones
from quippy.potential import Potential as QUIP


@CALCULATOR_PLUGIN.register('quip')
class QUIPCalculator(ASECalculator):
    def set_calc(self):
        with open("{}/quip_relax.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.relax_calc = QUIP(**params)
        with open("{}/quip_scf.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.scf_scf = QUIP(**params)
