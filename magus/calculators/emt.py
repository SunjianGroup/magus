import logging
from magus.calculators.base import ASECalculator
from ase.calculators.emt import EMT
from magus.utils import CALCULATOR_PLUGIN
import yaml



log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('emt')
class EMTCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.relax_calc = EMT()
        self.scf_calc = EMT()
        with open("{}/emt.yaml".format(self.input_dir)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.update_parameters(params)
