from ase.calculators.kim import KIM
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN

@CALCULATOR_PLUGIN.register('KIM')
class KIMCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        model = parameters['model']
        self.relax_calc = KIM(model)
        self.scf_calc = KIM(model)
