from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN
from ase.calculators.lj import LennardJones
import yaml

@CALCULATOR_PLUGIN.register('lj')
class LJCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        with open("{}/lj.yaml".format(self.input_dir)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.relax_calc = LennardJones(**params)
            self.scf_calc = LennardJones(**params)

