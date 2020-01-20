"""
Base module for all operators that create offspring.
steal from ase.ga
"""
import numpy as np
from ase import Atoms
from .renew import match_lattice
from ase.geometry import cell_to_cellpar,cellpar_to_cell,get_duplicate_atoms

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

    def mutate(self, ind):
        """ Does the actual mutation. """
        a = ind.atoms.copy()
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

        mutant = ind.atoms.copy()
        amplitude = 0.
        increment = 0.1
        direction = 1
        largest_norm = np.max(np.apply_along_axis(np.linalg.norm, 1, mode))

        while amplitude * largest_norm < self.bounds[1]:
            pos_new = pos + direction * amplitude * mode
            mutant.set_positions(pos_new)
            mutant.wrap()
            too_close = ind.check_distance(mutant)
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
            return None

        return ind.new(mutant)

    def get_new_individual(self, ind):
        newind = self.mutate(ind)
        if newind is None:
            return ind

        indi.info['parents'] = [ind.info['confid']]

        return indi

class CutAndSplicePairing:
    """ A cut and splice operator for bulk structures.

    For more information, see also:

    * `Glass, Oganov, Hansen, Comp. Phys. Comm. 175 (2006) 713-720`__

      __ https://doi.org/10.1016/j.cpc.2006.07.020

    * `Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387`__

      __ https://doi.org/10.1016/j.cpc.2010.07.048
    """
    def __init__(self, verbose=False):
        self.descriptor = 'CutAndSplicePairing'

    def get_new_individual(self,parents):
        """ The method called by the user that
        returns the paired structure. """
        f, m = parents

        indi = self.cross(f, m)

        if indi is None:
            return indi
        indi.info['parents'] = [f.info['confid'],m.info['confid']]

        return indi

    def cross(self, ind1, ind2,tryNum = 10):
        #TODO standardize_pop
        for _ in range(tryNum):
            indi = self.cut_cell(ind1,ind2)
            indi.merge_atoms()
            if indi.repair_atoms():
                break

    def cut_cell(self, ind1, ind2, cutDisp=0):
        """cut two cells to get a new cell
        
        Arguments:
            atoms1 {atoms} -- atoms1 to be cut
            atoms2 {atoms} -- atoms2 to be cut
        
        Keyword Arguments:
            cutDisp {int} -- dispalacement in cut (default: {0})
        
        Raises:
            RuntimeError: no atoms in new cell
        
        Returns:
            atoms -- generated atoms
        """
        atoms1,atoms2,ratio1,ratio2 = match_lattice(ind1.atoms,ind2.atoms)
        cutCell = (atoms1.get_cell()+atoms2.get_cell())*0.5
        cutCell[2] = (atoms1.get_cell()[2]/ratio1+atoms2.get_cell()[2]/ratio2)*0.5
        cutVol = (atoms1.get_volume()/ratio1+atoms2.get_volume()/ratio2)*0.5
        cutCellPar = cell_to_cellpar(cutCell)
        ratio = cutVol/abs(np.linalg.det(cutCell))
        if ratio > 1:
            cutCellPar[:3] = [length*ratio**(1/3) for length in cutCellPar[:3]]

        cutAtoms = Atoms(cell=cutCellPar,pbc = True,)
        scaled_positions = []
        cutPos = 0.5+cutDisp*np.random.uniform(-0.5, 0.5)
        for atom in atoms1:
            if 0 <= atom.c < cutPos/ratio1:
                cutAtoms.append(atom)
                scaled_positions.append([atom.a,atom.b,atom.c])
        for atom in atoms2:
            if 0 <= atom.c < (1-cutPos)/ratio2:
                cutAtoms.append(atom)
                scaled_positions.append([atom.a,atom.b,atom.c+cutPos/ratio1])

        cutAtoms.set_scaled_positions(scaled_positions)
        if len(cutAtoms) == 0:
            raise RuntimeError('No atoms in the new cell')
        cutInd = ind1.new(cutAtoms)
        cutInd.parents = [ind1 ,ind2]
        return cutInd



if __name__ == '__main__':
    from population import Individual
    from readparm import read_parameters
    from util import EmptyClass
    parameters = read_parameters('../test/input.yaml')
    p = EmptyClass()
    for key, val in parameters.items():
        setattr(p, key, val)