import numpy as np
from scipy.spatial.distance import cdist, pdist
import itertools
class Population:
    """
    a class of atoms population
    """
    def __init__(self,parameters):
        self.pop = []
        self.parameters = parameters
        self.threshold = self.parameters.threshold
        self.minLen = self.parameters.minLen
        self.maxLen = self.parameters.maxLen

    def del_duplicate(self):
        pass

    def check_dist(self):
        checkPop = []
        for ind in self.pop:
            if check_dist_individual(ind, self.threshold, self.minLen, self.maxLen):
                checkPop.append(ind)
        self.pop = checkPop

    def symmetrize_pop(self):
        pass

    def cluster(self):
        pass

class Individual:
    def __init__(self,atoms,parameters):
        self.atoms = atoms
        self.parameters = parameters
        self.info = dict()
    
    def new(self,atoms):
        newatoms = atoms.copy()
        parameters = self.parameters()
        return Individual(newatoms,parameters)

    def check_cellpar(self,atoms=None):
        """
        check if cellpar reasonable
        """
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()

        minLen = self.parameters.minLen
        maxLen = self.parameters.maxLen
        cellPar = a.get_cell_lengths_and_angles()
        if (minLen < cellPar).all() and (cellPar < maxLen).all():
            return True
        else:
            return False

    def check_distance(self,atoms=None):
        """
        The distance between the atoms should be larger than
        threshold * sumR(the sum of the covalent radii of the two
        corresponding atoms).
        """
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()

        threshold = self.parameters.threshold  
        cell = a.get_cell()
        nums = a.get_atomic_numbers()
        unique_types = sorted(list(set(nums)))
        pos = a.get_positions()

        for nx, ny, nz in itertools.product([-1,0,1],repeat=3):
            displacement = np.array([nx, ny, nz])@cell
            pos_new = pos + displacement
            distances = cdist(pos, pos_new)

            if nx == 0 and ny == 0 and nz == 0:
                distances += 1e2 * np.identity(len(a))

            iterator = itertools.combinations_with_replacement(unique_types, 2)
            for type1, type2 in iterator:
                x1 = np.where(nums == type1)
                x2 = np.where(nums == type2)
                if np.min(distances[x1].T[x2]) < (covalent_radii[type1]+covalent_radii[type2])*threshold:
                    return False
        return True

    def check(self,atoms=None):
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        return self.check_cellpar(a) and self.check_distance(a)