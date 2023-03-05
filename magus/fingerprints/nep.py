from .nepdes import NEPDes
import numpy as np
from magus.utils import FINGERPRINT_PLUGIN, check_parameters
from .base import FingerprintCalculator


@FINGERPRINT_PLUGIN.register('nepdes')
class NEPFp(FingerprintCalculator):
    def __init__(self, **parameters) -> None:
        Requirement = ['symbols']
        Default={'r_cut': 5., 'a_cut': 3, 'r_nmax': 10, 'a_nmax': 8, 'r_basis_size': 10,
                 'a_basis_size': 8, 'l_max': 4, 'l4_max': 2, 'l5_max': 1}
        check_parameters(self, parameters, Requirement, Default)
        self.calc = NEPDes(len(self.symbols), self.symbols, self.r_cut, self.a_cut, self.r_nmax, self.a_nmax, 
                           self.r_basis_size, self.a_basis_size, self.l_max, self.l4_max, self.l5_max)
        self.type_dict = {e: i for i, e in enumerate(self.symbols)}

    def get_all_fingerprints(self, atoms):
        symbols = atoms.get_chemical_symbols()
        _type = [self.type_dict[k] for k in symbols]
        _box = atoms.cell.transpose(1, 0).reshape(-1).tolist()
        _position = atoms.get_positions().transpose(1, 0).reshape(-1).tolist()
        descriptor = np.array(self.calc.find_descriptor(_type, _box, _position))
        descriptor = descriptor.reshape(-1, len(atoms)).transpose(1, 0)
        eFps = np.mean(descriptor, axis=0)
        return eFps, eFps, eFps
