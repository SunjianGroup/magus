from ase.atoms import Atoms
from ase.data import atomic_numbers,covalent_radii,atomic_masses
import numpy as np
from collections import Counter
from .utils import primitive_atoms2molcryst, primitive_atoms2communities
class Atomset:
    def __init__(self,positions,symbols,tag):
        self.symbols = symbols
        self.position = np.mean(positions,axis=0)
        self.relative_positions = positions - self.position
        self.tag = tag

    def __len__(self):
        return len(self.symbols)

    def to_atoms(self):
        return Atoms(symbols=self.symbols,positions=self.positions)

    def rotate(self,phi,theta,psi):
        rot1 = np.array([[cos(phi),-1*sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]])
        rot2 = np.array([[cos(theta), 0, -1*sin(theta)],[0,1,0],[sin(theta), 0, cos(theta)]])
        rot3 = np.array([[1,0,0],[0,cos(psi),-1*sin(psi)],[0,sin(psi),cos(psi)]])
        rotMat = rot1@rot2@rot3
        self.relative_positions = self.relative_positions@rotMat
        return rotMat
    @property
    def positions(self):
        return self.position + self.relative_positions

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
        self.atoms = atoms
        self.mols = []
        #if len(atoms)>0:
        if detector == 1:
            tags, offsets = primitive_atoms2molcryst(atoms, coef)
        elif detector == 2:
            tags, offsets = primitive_atoms2communities(atoms, coef)
        self.tags = tags
        self.atoms.set_tags(tags)
        self.unique_tags = np.unique(tags)
        # add offsets
        oldPos = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        self.atoms.set_positions(oldPos + np.dot(offsets, cell))

        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        for tag in self.unique_tags:
            indices = np.where(tags == tag)[0]
            pos = [positions[i] for i in indices]
            sym = [symbols[i] for i in indices]
            self.mols.append(Atomset(pos,sym,tag))
        self.n = len(self.mols)

    def get_positions(self):
        cop_pos = np.array([mol.position for mol in self.mols])
        return cop_pos

    def set_positions(self, positions, **kwargs):
        for i,mol in enumerate(self.mols):
            indices = np.where(self.tags == mol.tag)
            self.atoms.positions[indices] = positions[i] + mol.relative_positions
            mol.position = positions[i]

    def get_scaled_positions(self):
        cop_pos = self.get_positions()
        scl_pos = self.atoms.cell.scaled_positions(cop_pos)
        return scl_pos

    def set_scaled_positions(self, scaled_positions):
        positions = np.dot(scaled_positions,self.atoms.cell)
        self.set_positions(positions)

    def set_cell(self, cell):
        cell = np.array(cell)
        scl_pos = self.get_scaled_positions()
        new_pos = np.dot(scl_pos, cell)
        self.atoms.set_cell(cell)
        self.set_positions(new_pos)

    def get_atomic_numbers(self):
        return np.array([mol.number for mol in self.mols])

    def get_forces(self, *args, **kwargs):
        f = self.atoms.get_forces()
        forces = np.zeros((self.n, 3))
        for mol in self.mols:
            indices = np.where(self.tags == mol.tag)
            forces[i] = np.sum(f[indices], axis=0)
        return forces

    def get_masses(self):
        masses = np.array([mol.mass for mol in self.mols])
        return masses

    def __len__(self):
        return len(self.mols)

    def __iter__(self):
        for mol in self.mols:
            yield mol

    def __getitem__(self,i):
        return self.mols[i]

    def append(self,mol):
        mol.tag = len(self.mols)
        self.mols.append(mol)
        self.atoms.extend(mol.to_atoms())
        newtags = np.array([mol.tag]*len(mol))
        self.tags = np.concatenate((self.tags,newtags))

    def to_atoms(self):
        positions = []
        symbols = []
        for mol in self.mols:
            symbols.extend(mol.symbols)
            positions.extend(mol.positions)
        atoms = Atoms(symbols=symbols,positions=positions,pbc=self.atoms.pbc,cell=self.atoms.cell)
        return atoms

# #TODO from ase, need to change
# def isolate_components(atoms, kcutoff=None):
#     if kcutoff is None:
#         intervals = analyze_dimensionality(atoms, method='RDA', merge=False)
#         m = intervals[0]
#         if m.b == float("inf"):
#             kcutoff = m.a + 0.1
#         else:
#             kcutoff = (m.a + m.b) / 2
#     data = {}
#     components, all_visited, ranks = traverse_graph(atoms, kcutoff)

#     for k, v in all_visited.items():
#         if v is None:
#             continue
#         v = sorted(list(v))
#         key = tuple(np.unique([c for c, offset in v]))
#         for c in key:
#             components[np.where(components == c)] = key[0]
#             if c in all_visited.keys():
#                 all_visited[c] = None

#     return components


if __name__ == '__main__':
    from magus.population import Individual
    from magus.offspring_creator import CutAndSplicePairing
    from ase.io import read,write
    from magus.utils import EmptyClass
    from magus.readparm import read_parameters
    parameters = read_parameters('input.yaml')
    p = EmptyClass()
    for key, val in parameters.items():
        setattr(p, key, val)
    cutandsplice = CutAndSplicePairing()
    a = read('H2O.cif')
    ind = Individual(p)
    ind1 = ind(a)
    ind2 = ind(a)
    ind = cutandsplice.cross(ind1,ind2)
    write('new.cif',ind.atoms)
