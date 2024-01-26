import yaml
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN
from xtb.ase.calculator import XTB


@CALCULATOR_PLUGIN.register('xtb')
class XTBCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        with open("{}/xtb.yaml".format(self.input_dir)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.relax_calc = XTB(**params)
        self.scf_calc = XTB(**params)
#        with open("{}/xtb_relax.yaml".format(self.input_dir)) as f:
#            params = yaml.load(f)
#            self.relax_calc = XTB(**params)
#        with open("{}/xtb_scf.yaml".format(self.input_dir)) as f:
#            params = yaml.load(f)
#            self.scf_calc = XTB(**params)
