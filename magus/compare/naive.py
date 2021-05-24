from scipy.spatial.distance import cdist
import numpy as np
from collections import Counter
import spglib

class NaiveComparator:
    def __init__(self,dE=0.01,dV=0.05):
        self.dE = dE
        self.dV = dV

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
        #if 'enthalpy' in a.info and 'enthalpy' in b.info:
        #    if abs(a.info['enthalpy'] - b.info['enthalpy']) > self.dE:
        if 'energy' in a.info and 'energy' in b.info:
            if abs(a.info['energy']/len(a) - b.info['energy']/len(b)) > self.dE:
                return False
        return True

