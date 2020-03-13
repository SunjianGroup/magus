from scipy.spatial.distance import cdist
import numpy as np
class Comparator:
    def __init__(self,dE=0.01,dV=0.05,topri=False):
        self.dE = dE
        self.dV = dV
        self.topri = topri

    def looks_like(self,a,b):
        a,b = a.atoms,b.atoms
        if a.get_chemical_formula() != b.get_chemical_formula():
            return False
        if abs(1-a.get_volume()/b.get_volume()) > self.dV:
            return False
        if 'enthalpy' in a.info and 'enthalpy' in b.info:
            if a.info['enthalpy'] - b.info['enthalpy'] > self.dE:
                return False
        return True

class FingerprintComparator(Comparator):
    def __init__(self, dE=0.01, dV=0.05, dD=0.1, topri=False):
        super().__init__(dE=dE, dV=dV, topri=topri)
        self.dD = dD

    def looks_like(self, a1, a2):
        distance = np.linalg.norm(a1.fingerprint - a2.fingerprint)
        if distance > self.dD:
            return False
        return super().looks_like(a1,a2)
