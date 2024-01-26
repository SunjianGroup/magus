# Model for Stillinger-Weber with original parameters for Si (Z=14)

from ase.calculators.kim import KIM
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN

@CALCULATOR_PLUGIN.register('KIM')
class KIMCalculator(ASECalculator):
# A module defining a module needs to define only one variable,
# named `calculator`, which should be an instance of the ase.calculator.Calculator,
# a subclass of this, or a compatible class implementing the calculator interface.

    def __init__(self, **parameters):
        super().__init__(**parameters)
        model = parameters['model']
        #A modified Stillinger-Weber potential for modelling silicon surfaces
        #https://www.sciencedirect.com/science/article/abs/pii/0039602896008011

        self.relax_calc = KIM(model)
        self.scf_calc = KIM(model)
