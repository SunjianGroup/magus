from collections import Counter
from magus.utils import COMPARATOR_PLUGIN


@COMPARATOR_PLUGIN.register('naive')
class NaiveComparator:
    def __init__(self,dE=0.01, dV=0.05, **kwargs):
        self.dE = dE
        self.dV = dV

    def looks_like(self, ind1, ind2):
        for ind in [ind1, ind2]:
            if 'spg' not in ind.info:
                ind.find_spg()
        if Counter(ind1.info['priNum']) != Counter(ind2.info['priNum']):
            return False
        if ind1.info['spg'] != ind1.info['spg']:
            return False
        if abs(1 - ind1.info['priVol'] / ind2.info['priVol']) > self.dV:
            return False
        if 'energy' in ind1.info and 'energy' in ind2.info:
            if abs(ind1.info['energy'] / len(ind1) - ind2.info['energy'] / len(ind2)) > self.dE:
                return False
        return True
