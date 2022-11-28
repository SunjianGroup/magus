import logging, yaml
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN
from quippy.potential import Potential


@CALCULATOR_PLUGIN.register('quip')
class QUIPCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        with open("{}/quip.yaml".format(self.input_dir)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.relax_calc = Potential(**params)
            self.scf_calc = Potential(**params)
