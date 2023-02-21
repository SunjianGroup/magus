import numpy as np
from ase import Atoms
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.neighborlist import NewPrimitiveNeighborList
from magus.utils import *
from .base import Crossover

__all__ = ['CutAndSplicePairing', 'ReplaceBallPairing',]


class CutAndSplicePairing(Crossover):

    Default = {'tryNum': 50, 'cut_disp': 0, 'best_match': False}

    @staticmethod
    def match_lattice(ind1, ind2):
        """
        transform ind1 and ind2 for best match lattice
        """
        raise Exception("Best lattice match temporary removed")

    @staticmethod
    def cut_and_splice(atoms1, atoms2, axis, cut_disp):
        atoms1.set_scaled_positions(atoms1.get_scaled_positions() + np.random.rand(3))
        atoms2.set_scaled_positions(atoms2.get_scaled_positions() + np.random.rand(3))
 
        cut_cell   = 0.5 * (atoms1.get_cell()   + atoms2.get_cell())
        cut_volume = 0.5 * (atoms1.get_volume() + atoms2.get_volume())
        cut_cellpar = cell_to_cellpar(cut_cell)
        ratio = cut_volume / abs(np.linalg.det(cut_cell))
        cut_cellpar[:3] = [length * ratio ** (1/3) for length in cut_cellpar[:3]]

        cut_atoms = atoms1.__class__(Atoms(cell=cut_cellpar, pbc=True,))

        scaled_positions = []
        cut_position = [0, 0.5 + cut_disp * np.random.uniform(-0.5, 0.5), 1]

        for n, atoms in enumerate([atoms1, atoms2]):
            spositions = atoms.get_scaled_positions()
            for i, atom in enumerate(atoms):
                if cut_position[n] <= spositions[i, axis] < cut_position[n+1]:
                    cut_atoms.append(atom)
                    scaled_positions.append(spositions[i])
        if len(scaled_positions) == 0:
            return None
        cut_atoms.set_scaled_positions(scaled_positions)
        return cut_atoms

    def cross_bulk(self, ind1, ind2):
        if self.best_match:
            M1, M2 = self.match_lattice(ind1, ind2)
            axis = 2
        else:
            axis = np.random.choice([0, 1, 2])
            atoms1 = ind1.for_heredity()
            atoms2 = ind2.for_heredity()
        cut_atoms = self.cut_and_splice(atoms1, atoms2, axis, self.cut_disp)
        if cut_atoms is None:
            return None
        else:
            return ind1.__class__(cut_atoms)

    def cross_layer(self, ind1, ind2):
        axis = np.random.choice([0, 1])
        atoms1 = ind1.for_heredity()
        atoms2 = ind2.for_heredity()
        cut_atoms = self.cut_and_splice(atoms1, atoms2, axis, self.cut_disp)
        if cut_atoms is None:
            return None
        else:
            cut_atoms =  ind1.add_vacuum(cut_atoms, ind1.vacuum_thickness)
            return ind1.__class__(cut_atoms)


class ReplaceBallPairing(Crossover):
    """
    replace some atoms in a ball
    """
    Default = {'tryNum': 50, 'cut_range': [1, 2]}

    def cross_bulk(self, ind1, ind2):
        """
        replace some atoms in a bal\][]
        """
        cut_radius = np.random.uniform(*self.cut_range)
        atoms1, atoms2 = ind1.for_heredity(), ind2.for_heredity()
        # random choose replace center
        center_i, center_j = np.random.randint(len(atoms1)), np.random.randint(len(atoms2))
        newatoms = atoms1.__class__(Atoms(pbc=atoms1.pbc, cell=atoms1.cell))
        positions1, positions2 = atoms1.get_positions(), atoms2.get_positions()
        # translate of atoms so that center i and center j are coincided
        atoms2.positions += atoms1.positions[center_i] - atoms2.positions[center_j]
        
        # for atoms 1, we choose the atoms outside of the ball
        nl = NewPrimitiveNeighborList(cutoffs=[cut_radius / 2] * len(atoms1), bothways=True)
        nl.update(pbc=atoms1.pbc, cell=atoms1.cell, positions=positions1)
        neighbor_i = nl.get_neighbors(center_i)[0]
        for i, atom in enumerate(atoms1):
            if i not in neighbor_i:
                newatoms.append(atom)
        # for atoms 2, we choose the atoms inside of the ball
        nl = NewPrimitiveNeighborList(cutoffs=[cut_radius / 2] * len(atoms2), bothways=True)
        nl.update(pbc=atoms2.pbc, cell=atoms2.cell, positions=positions2)
        neighbor_j = list(nl.get_neighbors(center_j)[0])
        neighbor_j.append(center_j)          # append j because j is not in its neighbor
        newatoms.extend(atoms2[neighbor_j])

        return ind1.__class__(newatoms)
