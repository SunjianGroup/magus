import yaml
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN
from ase.calculators.dftb import Dftb


@CALCULATOR_PLUGIN.register('dftb')
class DFTBCalculator(ASECalculator):
    # Noto: Only use this calculator to perform static calculations.
    # Relaxations should be conducted by ASE's driver
    # Set the environment varaible 'ASE_DFTB_COMMAND' in ~/.bashrc or in the submission script. For parallel task, set this conmmand in 'pre_processing' in input.yaml
    # The parameter 'kpts' of Dftb can be set as a dict to fix k-point density for varaible configurations. See https://wiki.fysik.dtu.dk/ase/ase/calculators/dftb.html
    # When 'kpts' is a dict, the structre should also be provided. So here Dftb type and parameters will be attahed. And the calculator instance will be created in base.py
    def __init__(self, **parameters):
        super().__init__(**parameters)
        with open("{}/dftb.yaml".format(self.input_dir)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.relax_calc = params
        self.scf_calc = params
        self.ase_calc_type = Dftb
