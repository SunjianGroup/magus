import logging
from mace.calculators import mace_mp
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN

log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('mace')
class MACECalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.relax_calc = mace_mp()
        self.scf_calc = mace_mp()
