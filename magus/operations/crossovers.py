import math
import numpy as np
import logging, copy
from ase import Atoms, Atom 
from ase.constraints import voigt_6_to_full_3x3_strain as v2f
from ase.geometry import cell_to_cellpar,cellpar_to_cell,get_duplicate_atoms
from ase.neighborlist import NeighborList
from ase.data import covalent_radii,chemical_symbols
from .population import Population
from .molecule import Molfilter
import ase.io
from .utils import *
from spglib import get_symmetry_dataset
from collections import Counter


log = logging.getLogger(__name__)


class CutAndSplicePairing(Crossover):
    """ 
    A cut and splice operator for bulk structures.

    For more information, see also:

    * `Glass, Oganov, Hansen, Comp. Phys. Comm. 175 (2006) 713-720`__

      __ https://doi.org/10.1016/j.cpc.2006.07.020

    * `Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387`__

      __ https://doi.org/10.1016/j.cpc.2010.07.048
    """
    Default = {'tryNum': 50, 'cut_disp': 0, 'best_match': False}

    def cross(self, ind1, ind2):
        if self.best_match:
            M1, M2 = match_lattice(ind1, ind2)
            axis = 2
        else:
            axis = np.random.choice([0, 1, 2])
            atoms1 = ind1.for_mutate
            atoms2 = ind2.for_mutate

        atoms1.set_scaled_positions(atoms1.get_scaled_positions() + np.random.rand(3))
        atoms2.set_scaled_positions(atoms2.get_scaled_positions() + np.random.rand(3))
 
        cut_cell   = 0.5 * (atoms1.get_cell()   + atoms2.get_cell())
        cut_volume = 0.5 * (atoms1.get_volume() + atoms2.get_volume())
        cut_cellpar = cell_to_cellpar(cut_cell)
        ratio = cut_volume / abs(np.linalg.det(cut_cell))
        cut_cellpar[:3] = [length * ratio ** (1/3) for length in cut_cellpar[:3]]

        cut_atoms = atoms1.__class__(cell=cut_cellpar, pbc=True,)

        scaled_positions = []
        cut_position = [0, 0.5 + self.cut_disp * np.random.uniform(-0.5, 0.5), 1]

        for n, atoms in enumerate([atoms1, atoms2]):
            spositions = atoms.get_scaled_positions()
            for i, atom in enumerate(atoms):
                if cut_position[n] <= spositions[i, axis] < cut_position[n+1]:
                    cut_atoms.append(atom)
                    scaled_positions.append(spositions[i])
        if len(scaled_positions) == 0:
            return None

        cut_atoms.set_scaled_positions(scaled_positions)

        return ind1.__class__(cut_atoms)


class ReplaceBallPairing(Crossover):
    """
    replace some atoms in a ball
    """
    parm = {'tryNum': 50, 'cut_range': [1, 2]}

    def cross(self, ind1, ind2):
        """
        replace some atoms in a ball
        """
        cut_radius = np.random.uniform(*self.cutrange)
        atoms1, atoms2 = ind1.for_mutate, ind2.for_mutate
        center_i, center_j = np.random.randint(len(atoms1)), np.random.randint(len(atoms2))
        newatoms = atoms1.__class__(pbc=atoms1.pbc, cell=atoms1.cell)
        positions1, positions2 = atoms1.get_positions(), atoms2.get_positions()
        atoms2.positions += atoms1.positions[center_i] - atoms2.positions[center_j]
        
        nl = NewPrimitiveNeighborList(cutoffs=[cut_radius / 2] * len(atoms1), bothways=True)
        nl.update(pbc=atoms1.pbc, cell=atoms1.cell, positions=positions1)
        neighbor_i = nl.get_neighbors(center_i)[0]
        for i, atom in enumerate(atoms1):
            if i not in neighbor_i:
                newatoms.append(atom)

        nl = NewPrimitiveNeighborList(cutoffs=[cut_radius / 2] * len(atoms2), bothways=True)
        nl.update(pbc=atoms2.pbc, cell=atoms2.cell, positions=positions2)
        neighbor_j = nl.get_neighbors(center_j)[0]
        neighbor_j.append(center_j)
        newatoms.extend(atoms2[neighbor_j])

        return ind1.__class__(newatoms)
