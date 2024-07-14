import logging
import yaml
from mace.calculators import mace_mp, mace_off, mace_anicc
from mace.calculators import MACECalculator as MACEAseCalculator
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN

log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('mace')
class MACECalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        with open("{}/mace.yaml".format(self.input_dir)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        if params['modelType'] == 'mp':
            self.relax_calc = mace_mp(**params)
            self.scf_calc = mace_mp(**params)
        elif params['modelType'] == 'off':
            self.relax_calc = mace_off(**params)
            self.scf_calc = mace_off(**params)
        elif params['modelType'] == 'anicc':
            self.relax_calc = mace_anicc(**params)
            self.scf_calc = mace_anicc(**params)
        else:
            self.relax_calc = MACEAseCalculator(**params)
            self.scf_calc = MACEAseCalculator(**params)

