"""
Base module for all operators that create offspring.
steal from ase.ga
"""
import numpy as np
from random import random
from ase import Atoms

class OffspringCreator:
    """Base class for all procreation operators

    Parameters:

    verbose: Be verbose and print some stuff

    """

    def __init__(self, verbose=False, num_muts=1):
        self.descriptor = 'OffspringCreator'
        self.verbose = verbose
        self.min_inputs = 0
        self.num_muts = num_muts

    def get_min_inputs(self):
        """Returns the number of inputs required for a mutation,
        this is to know how many candidates should be selected
        from the population."""
        return self.min_inputs

    def get_new_individual(self, parents):
        """Function that returns a new individual.
        Overwrite in subclass."""
        raise NotImplementedError

    def finalize_individual(self, indi):
        #Call this function just before returning the new individual
        
        #indi.info['key_value_pairs']['origin'] = self.descriptor

        return indi
        


    @classmethod
    def initialize_individual(cls, parent, indi=None):
        #TODO
        if indi is None:
            indi = Atoms(pbc=parent.get_pbc(), cell=parent.get_cell())
        else:
            indi = indi.copy()
        return indi
        """Initializes a new individual that inherits some parameters
        from the parent, and initializes the info dictionary.
        If the new individual already has more structure it can be
        supplied in the parameter indi."""
        """
        if indi is None:
            indi = Atoms(pbc=parent.get_pbc(), cell=parent.get_cell())
        else:
            indi = indi.copy()
        # key_value_pairs for numbers and strings
        indi.info['key_value_pairs'] = {'extinct': 0}
        # data for lists and the like
        indi.info['data'] = {}

        return indi
        """

class OperationSelector(object):
    """Class used to randomly select a procreation operation
    from a list of operations.

    Parameters:

    probabilities: A list of probabilities with which the different
        mutations should be selected. The norm of this list
        does not need to be 1.

    oplist: The list of operations to select from.
    """

    def __init__(self, probabilities, oplist):
        assert len(probabilities) == len(oplist)
        self.oplist = oplist
        self.rho = np.cumsum(probabilities)

    def __get_index__(self):
        v = random() * self.rho[-1]
        for i in range(len(self.rho)):
            if self.rho[i] > v:
                return i

    def get_new_individual(self, candidate_list):
        """Choose operator and use it on the candidate. """
        to_use = self.__get_index__()
        return self.oplist[to_use].get_new_individual(candidate_list)

    def get_operator(self):
        """Choose operator and return it."""
        to_use = self.__get_index__()
        return self.oplist[to_use]

class CombinationMutation(OffspringCreator):
    """Combine two or more mutations into one operation.

    Parameters:

    mutations: Operator instances
        Supply two or more mutations that will applied one after the other
        as one mutation operation. The order of the supplied mutations prevail
        when applying the mutations.

    """

    def __init__(self, *mutations, verbose=False):
        super(CombinationMutation, self).__init__(verbose=verbose)
        self.descriptor = 'CombinationMutation'

        # Check that a combination mutation makes sense
        msg = "Too few operators supplied to a CombinationMutation"
        assert len(mutations) > 1, msg

        self.operators = mutations

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: {}'.format(self.descriptor)

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return (self.finalize_individual(indi),
                'mutation: {}'.format(self.descriptor))

    def mutate(self, atoms):
        """Perform the mutations one at a time."""
        for op in self.operators:
            if atoms is not None:
                atoms = op.mutate(atoms)
        return atoms

class SoftMutation:
    """
    Mutates the structure by displacing it along the lowest (nonzero)
    frequency modes found by vibrational analysis, as in:

    * `Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632`__

      __ https://dx.doi.org/10.1016/j.cpc.2010.06.007

    As in the reference above, the next-lowest mode is used if the
    structure has already been softmutated along the current-lowest
    mode.

    Parameters:

    bounds: list
            Lower and upper limits (in Angstrom) for the largest
            atomic displacement in the structure. For a given mode,
            the algorithm starts at zero amplitude and increases
            it until either blmin is violated or the largest
            displacement exceeds the provided upper bound).
            If the largest displacement in the resulting structure
            is lower than the provided lower bound, the mutant is
            considered too similar to the parent and None is
            returned.   
    """

    def __init__(self, calculator, bounds=[0.5, 2.0], verbose=False):
        self.bounds = bounds
        self.calc = calculator
        self.descriptor = 'SoftMutation'

    def _get_hessian(self, atoms, dx):
        """
        Returns the Hessian matrix d2E/dxi/dxj using a first-order
        central difference scheme with displacements dx.
        """
        N = len(atoms)
        pos = atoms.get_positions()
        hessian = np.zeros((3 * N, 3 * N))

        for i in range(3 * N):
            row = np.zeros(3 * N)
            for direction in [-1, 1]:
                disp = np.zeros(3)
                disp[i % 3] = direction * dx
                pos_disp = np.copy(pos)
                pos_disp[i // 3] += disp
                atoms.set_positions(pos_disp)
                f = atoms.get_forces()
                row += -1 * direction * f.flatten()

            row /= (2. * dx)
            hessian[i] = row

        hessian += np.copy(hessian).T
        hessian *= 0.5
        atoms.set_positions(pos)

        return hessian

    def _calculate_normal_modes(self, atoms, dx=0.02, massweighing=False):
        """Performs the vibrational analysis."""
        hessian = self._get_hessian(atoms, dx)
        if massweighing:
            m = np.array([np.repeat(atoms.get_masses()**-0.5, 3)])
            hessian *= (m * m.T)

        eigvals, eigvecs = np.linalg.eigh(hessian)
        modes = {eigval: eigvecs[:, i] for i, eigval in enumerate(eigvals)}
        return modes

    def mutate(self, atoms):
        """ Does the actual mutation. """
        a = atoms.copy()
        a.set_calculator(self.calc)

        pos = a.get_positions()
        modes = self._calculate_normal_modes(a)

        # Select the mode along which we want to move the atoms;
        # The first 3 translational modes as well as previously
        # applied modes are discarded.

        keys = np.array(sorted(modes))
        index = 3

        key = keys[index]
        mode = modes[key].reshape(np.shape(pos))

        # Find a suitable amplitude for translation along the mode;
        # at every trial amplitude both positive and negative
        # directions are tried.

        mutant = atoms.copy()
        amplitude = 0.
        increment = 0.1
        direction = 1
        largest_norm = np.max(np.apply_along_axis(np.linalg.norm, 1, mode))

        while amplitude * largest_norm < self.bounds[1]:
            pos_new = pos + direction * amplitude * mode
            mutant.set_positions(pos_new)
            mutant.wrap()
            too_close = check_dist_individual(mutant,threshold,minLen,maxLen)
            if too_close:
                amplitude -= increment
                pos_new = pos + direction * amplitude * mode
                mutant.set_positions(pos_new)
                mutant.wrap()
                break

            if direction == 1:
                direction = -1
            else:
                direction = 1
                amplitude += increment

        if amplitude * largest_norm < self.bounds[0]:
            mutant = None

        return mutant

    def get_new_individual(self, ind):

        newind = self.mutate(ind)
        if newind is None:
            return ind, 'mutation: soft'

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return self.finalize_individual(indi), 'mutation: soft'