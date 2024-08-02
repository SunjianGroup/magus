import logging
import matgl
from matgl.ext.ase import PESCalculator
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN

log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('m3gnet')
class M3GNETCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        self.relax_calc = PESCalculator(potential)
        self.scf_calc = PESCalculator(potential)
