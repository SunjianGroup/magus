import yaml
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN
from tblite.ase import TBLite


@CALCULATOR_PLUGIN.register('tblite')
class TBLiteCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        with open("{}/tblite.yaml".format(self.input_dir)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.relax_calc = TBLite(**params)
        self.scf_calc = TBLite(**params)
