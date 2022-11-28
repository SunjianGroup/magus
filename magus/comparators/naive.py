from collections import Counter
from ase import Atoms
from magus.utils import COMPARATOR_PLUGIN


@COMPARATOR_PLUGIN.register('naive')
class NaiveComparator:
    def __init__(self,dE=0.01, dV=0.05, **kwargs):
        self.dE = dE
        self.dV = dV

    def looks_like(self, ind1, ind2):
        if 'spg' not in ind1.info:
            ind1.find_spg()
        if isinstance(ind2, Atoms):
            ind2 = [ind2]
        for ind in ind2:
            if 'spg' not in ind.info:
                ind.find_spg()
            if ind1.info['spg'] != ind.info['spg']:
                continue
            if abs(1 - ind1.info['priVol'] / ind.info['priVol']) > self.dV:
                continue
            if 'energy' in ind1.info and 'energy' in ind.info:
                if abs(ind1.info['energy'] / len(ind1) - ind.info['energy'] / len(ind)) > self.dE:
                    continue
            return True
        return False
