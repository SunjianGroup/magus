import numbers, copy
from math import cos, sin
import numpy as np
from ase.atoms import Atoms
from ase.data import atomic_numbers,covalent_radii,atomic_masses
from ase.cell import Cell
from .crystgraph import atoms_to_mol_1, atoms_to_mol_2


class Atomset:
    def __init__(self,positions,symbols):
        self.symbols = symbols
        self.position = np.mean(positions,axis=0)
        self.relative_positions = positions - self.position

    def __len__(self):
        return len(self.symbols)

    def to_atoms(self):
        return Atoms(symbols=self.symbols,positions=self.positions)

    def rotate(self, phi, theta, psi):
        rot1 = np.array([[cos(phi), -sin(phi), 0.], [sin(phi), cos(phi), 0.], [0., 0., 1.]])
        rot2 = np.array([[1., 0., 0.], [0., cos(theta), -sin(theta)], [0., sin(theta), cos(theta)]])
        rot3 = np.array([[cos(psi), -sin(psi), 0.], [sin(psi), cos(psi), 0.], [0., 0., 1.]])
        self.relative_positions = self.relative_positions @ rot1 @ rot2 @ rot3

    @property
    def positions(self):
        return self.position + self.relative_positions

    @positions.setter
    def positions(self, pos):
        assert len(pos) == len(self.relative_positions)
        self.position = np.mean(pos,axis=0)
        self.relative_positions = pos - self.position

    @property
    def symbol(self):
        s = []
        unique_symbols = sorted(np.unique(self.symbols))
        for symbol in unique_symbols:
            s.append(symbol)
            n = self.symbols.count(symbol)
            if n > 1:
                s.append(str(n))
        s = ''.join(s)
        return s

    @property
    def mass(self):
        return sum([atomic_masses[atomic_numbers[symbol]] for symbol in self.symbols])

    @property
    def number(self):
        numbers = [atomic_numbers[symbol] for symbol in self.symbols]
        radius = [covalent_radii[number] for number in numbers]
        return numbers[np.argmax(radius)]


class Molfilter:
    def __init__(self, atoms, detector=1, coef=1.1):
        self.pbc = atoms.pbc
        self.cell = atoms.cell
        self.mols = []
        if detector == 1:
            tags, offsets = atoms_to_mol_1(atoms, coef)
        elif detector == 2:
            tags, offsets = atoms_to_mol_2(atoms, coef)

        # add offsets
        positions = atoms.get_positions()
        positions += np.dot(offsets, self.cell)
        symbols = atoms.get_chemical_symbols()

        for tag in np.unique(tags):
            indices = np.where(tags == tag)[0]
            pos = [positions[i] for i in indices]
            sym = [symbols[i] for i in indices]
            self.mols.append(Atomset(pos, sym))

    def __len__(self):
        return len(self.mols)

    def __iter__(self):
        for mol in self.mols:
            yield mol

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            return self.mols[i]
        else:
            newmol = self.copy()
            if isinstance(i, slice):
                newmol.mols = newmol.mols[i]
            else:
                indices = np.array(i)
                if indices.dtype == bool:
                    try:
                        indices = np.arange(len(self))[indices]
                    except IndexError:
                        raise IndexError('length of item mask '
                                        'mismatches that of {0} '
                                        'object'.format(self.__class__.__name__))
                newmol.mols = [newmol.mols[i] for i in indices]
            return newmol

    def copy(self):
        return copy.deepcopy(self)

    def get_positions(self):
        return np.array([mol.position for mol in self.mols])

    def set_positions(self, positions):
        for i, mol in enumerate(self.mols):
            mol.position = positions[i]

    @property
    def positions(self):
        return self.get_positions()

    @positions.setter
    def positions(self, positions):
        self.set_positions(positions)

    def get_scaled_positions(self):
        return self.cell.scaled_positions(self.get_positions())

    def set_scaled_positions(self, scaled_positions):
        positions = np.dot(scaled_positions, self.cell)
        self.set_positions(positions)

    def get_volume(self):
        return self.cell.volume

    def get_cell(self):
        return self.cell.copy()

    def set_cell(self, cell, scale_atoms=False, keep_mol=True):
        """
        Set the cell
        scale_atoms: scale centor of molecules or not
        keep_mol: keep the relative positions in mol or not
        """
        cell = Cell.new(cell)
        if scale_atoms:
            for mol in self.mols:
                mol.position = self.cell.scaled_positions(mol.position) @ cell
                if not keep_mol:
                    mol.positions = self.cell.scaled_positions(mol.positions) @ cell
        self.cell = cell

    def append(self, mol):
        self.mols.append(mol)

    def extend(self, mols):
        self.mols.extend(mols)

    def to_atoms(self):
        positions = []
        symbols = []
        for mol in self.mols:
            symbols.extend(mol.symbols)
            positions.extend(mol.positions)
        atoms = Atoms(symbols=symbols, positions=positions, pbc=self.pbc, cell=self.cell)
        return atoms
