import logging
from magus.calculators.base import ASECalculator
from pynep.calculate import NEP
from magus.utils import CALCULATOR_PLUGIN


log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('nep')
class PyNEPCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.relax_calc = NEP("{}/nep.txt".format(self.input_dir))
        self.scf_calc = NEP("{}/nep.txt".format(self.input_dir))
