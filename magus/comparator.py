from scipy.spatial.distance import cdist
import numpy as np
from collections import Counter
import spglib

class Comparator:
    def __init__(self,dE=0.01,dV=0.05,topri=False):
        self.dE = dE
        self.dV = dV
        self.topri = topri

    def looks_like(self,aInd,bInd):
        for ind in [aInd, bInd]:
            if 'spg' not in ind.atoms.info:
                ind.find_spg()
        a,b = aInd.atoms,bInd.atoms
        # if a.get_chemical_formula() != b.get_chemical_formula():
        if Counter(a.info['priNum']) != Counter(b.info['priNum']):
            return False
        if a.info['spg'] != b.info['spg']:
            return False
        if abs(1-a.info['priVol']/b.info['priVol']) > self.dV:
            return False
        if 'enthalpy' in a.info and 'enthalpy' in b.info:
            if abs(a.info['enthalpy'] - b.info['enthalpy']) > self.dE:
                return False
        return True

class FingerprintComparator(Comparator):
    def __init__(self, dE=0.01, dV=0.05, dD=0.05, topri=False):
        super().__init__(dE=dE, dV=dV, topri=topri)
        self.dD = dD

    def looks_like(self, a1, a2):
        x,y = a1.fingerprint , a2.fingerprint
        dist = 1 - np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
        if dist > self.dD:
            return False
        return super().looks_like(a1,a2)
