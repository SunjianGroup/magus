import logging
from magus.calculators.base import ASECalculator
from ase.calculators.emt import EMT
from magus.utils import CALCULATOR_PLUGIN


log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('emt')
class EMTCalculator(ASECalculator):
    def set_calc(self):
        self.relax_calc = EMT()
        self.scf_calc = EMT()
