from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
import numpy as np
from magus.utils import FINGERPRINT_PLUGIN, check_parameters
from .base import FingerprintCalculator


@FINGERPRINT_PLUGIN.register('soap')
class SoapFp(FingerprintCalculator):
    def __init__(self, **parameters):
        Requirement = ['symbols']
        Default={'r_cut': 5, 'nmax': 6, 'lmax': 4, 'periodic': False}
        check_parameters(self, parameters, Requirement, Default)
        self.soap = SOAP(species=self.symbols, periodic=self.periodic, r_cut=self.r_cut, 
                         n_max=self.nmax, l_max=self.lmax)

    def get_all_fingerprints(self, atoms):
        eFps = np.mean(self.soap.create(atoms), axis=0)
        return eFps, eFps, eFps
