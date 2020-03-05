from ase.geometry.dimensionality import analyze_dimensionality
from ase.geometry.dimensionality.isolation import traverse_graph
from ase.ga.utilities import gather_atoms_by_tag
from ase.atoms import Atoms
from ase.data import atomic_numbers,covalent_radii
import numpy as np

class Atomset:
    def __init__(self,positions,symbols):
        self.symbols = symbols
        self.position = np.mean(positions,axis=0)
        self.relative_positions = positions - self.position
    
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
    def number(self):
        numbers = [atomic_numbers[symbol] for symbol in self.symbols]
        radius = [covalent_radii[number] for number in numbers]
        return numbers[np.argmax(radius)]

class Molfilter:
    def __init__(self, atoms):
        self.atoms = atoms
        self.mols = []
        if len(atoms)>0:
            tags = isolate_components(atoms)
            self.tags = tags
            self.atoms.set_tags(tags)
            gather_atoms_by_tag(self.atoms)
            self.unique_tags = np.unique(tags)

            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            for tag in self.unique_tags:
                indices = np.where(tags == tag)[0]
                pos = [positions[i] for i in indices]
                sym = [symbols[i] for i in indices]
                self.mols.append(Atomset(pos,sym))
        self.n = len(self.mols)
        
    def get_positions(self):
        cop_pos = np.array([mol.position for mol in self.mols])
        return cop_pos

    def set_positions(self, positions, **kwargs):
        for i,mol in enumerate(self.mols):
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

    def get_atomic_numbers(self)：
        return np.array([mol.number for mol in self.mols])

    def get_forces(self, *args, **kwargs):
        f = self.atoms.get_forces()
        forces = np.zeros((self.n, 3))
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            forces[i] = np.sum(f[indices], axis=0)
        return forces

    def get_masses(self):
        m = self.atoms.get_masses()
        masses = np.zeros(self.n)
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            masses[i] = np.sum(m[indices])
        return masses

    def __len__(self):
        return len(self.mols)

    def __iter__(self):
        for mol in self.mols:
            yield mol

    def __getitem__(self,i):
        return self.mols[i]
    
    def append(self,mol):
        self.mols.append(mol)
    
    def to_atoms(self):
        positions = []
        symbols = []
        for mol in self.mols:
            symbols.extend(mol.symbols)
            positions.extend(mol.positions)
        atoms = Atoms(symbols=symbols,positions=positions,pbc=self.atoms.pbc,cell=self.atoms.cell)
        return atoms
        
#TODO steal from ase, need to change
def isolate_components(atoms, kcutoff=None): 
    if kcutoff is None:
        intervals = analyze_dimensionality(atoms, method='RDA')
        m = intervals[0]
        if m.b == float("inf"):
            kcutoff = m.a + 0.1
        else:
            kcutoff = max((m.a + m.b) / 2, m.b - 0.1)
    data = {}
    components, all_visited, ranks = traverse_graph(atoms, kcutoff)

    for k, v in all_visited.items():
        if v is None:
            continue
        v = sorted(list(v))
        key = tuple(np.unique([c for c, offset in v]))
        for c in key:
            components[np.where(components == c)] = key[0]
            all_visited[c] = None

    return components


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
