import logging

from chgnet.model.dynamics import CHGNetCalculator
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN

log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('chgnet')
class CHGNETCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.relax_calc = CHGNetCalculator()
        self.scf_calc = CHGNetCalculator()
