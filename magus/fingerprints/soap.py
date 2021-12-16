from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
import numpy as np
from magus.utils import FINGERPRINT_PLUGIN
from .base import FingerprintCalculator


@FINGERPRINT_PLUGIN.register('soap')
class SoapFp(FingerprintCalculator):
    def __init__(self, symbols, rcut=6.0, nmax=8, lmax=6, periodic=True, **kwargs):
        self.soap = SOAP(species=symbols, periodic=periodic, rcut=rcut, nmax=nmax, lmax=lmax)

    def get_all_fingerprints(self, atoms):
        eFps = normalize(self.soap.create(atoms))
        return eFps, eFps , eFps
