# TODO
# molecule cell split
import numpy as np
import logging
from types import MethodType


log = logging.getLogger(__name__)


class CellSplitFilter:
    """
    a filter to split a large cell into subcells and splice them
    """
    def __init__(self, generator, n_split=4):
        self.generator = generator
        self.generator.__getattr__ = MethodType(CellSplitFilter.filterget, self.generator)
        self.n_split = n_split

    def __getattr__(self, name):
        if not hasattr(self.generator, name):
            raise Exception("CellSplitFilter not has attribute: {}".format(name))
        return getattr(self.generator, name)

    def Generate_ind(self, spg, numlist):
        n_split = self.n_split
        numlist = np.ceil(numlist / n_split).astype(np.int)
        label, atoms = self.generator.Generate_ind(spg, numlist)
        if label:
            while n_split > 1:
                i = 2
                while i < np.sqrt(n_split):
                    if n_split % i == 0:
                        break
                    i += 1
                to_expand = np.argmin(atoms.get_cell_lengths_and_angles()[:3])
                expand_matrix = [1, 1, 1]
                expand_matrix[to_expand] = i
                atoms = atoms * expand_matrix
                n_split /= i
            return label, atoms
        else:
            return label, None
