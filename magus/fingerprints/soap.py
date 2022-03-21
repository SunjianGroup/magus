from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
import numpy as np
from magus.utils import FINGERPRINT_PLUGIN
from .base import FingerprintCalculator


@FINGERPRINT_PLUGIN.register('soap')
class SoapFp(FingerprintCalculator):
    def __init__(self, symbols, rcut=4.0, nmax=6, lmax=4, periodic=True, **kwargs):
        self.soap = SOAP(species=symbols, periodic=periodic, rcut=rcut, nmax=nmax, lmax=lmax)

    def get_all_fingerprints(self, atoms):
        eFps = np.sum(self.soap.create(atoms), axis=0)
        return eFps, eFps, eFps
