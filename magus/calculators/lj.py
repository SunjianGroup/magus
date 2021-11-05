from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN
from ase.calculators.lj import LennardJones


@CALCULATOR_PLUGIN.register('lj')
class LJCalculator(ASECalculator):
    def set_calc(self):
        self.relax_calc = LennardJones()
        self.scf_calc = LennardJones()
