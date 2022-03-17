import logging
import spglib
from ase import Atoms, Atom
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
from ase.constraints import FixAtoms
from magus.utils import *
from .molecule import Molfilter
from ..fingerprints import get_fingerprint
from ..comparators import get_comparator


log = logging.getLogger(__name__)
__all__ = ['Bulk']


def get_Ind(p_dict):
    if p_dict['searchType'] == 'bulk':
        Ind = Bulk
    Ind.set_parameters(**p_dict)
    return Ind

def check_new_atom(atoms, np, symbol, distance_dict):
    distances = get_distances(atoms.positions, np, cell=atoms.cell, pbc=atoms.pbc)[1]
    for s, d in zip(atoms.get_chemical_symbols(), distances):
        if d < distance_dict[(s, symbol)]:
            return False
    else:
        return True

# TODO weighten
def to_target_formula(atoms, target_formula, distance_dict, max_n_try=100): 
    symbols = atoms.get_chemical_symbols()
    toadd, toremove = {}, {}
    for s in target_formula:
        if symbols.count(s) < target_formula[s]:
            toadd[s] = target_formula[s] - symbols.count(s)
        elif symbols.count(s) > target_formula[s]:
            toremove[s] = symbols.count(s) - target_formula[s]
    rep_atoms = atoms.copy()
    #remove before add
    while toremove:
        del_symbol = np.random.choice(list(toremove.keys()))
        del_index = np.random.choice([atom.index for atom in rep_atoms if atom.symbol == del_symbol])
        if toadd:
            #if some symbols need to add, change symbol directly
            add_symbol = np.random.choice(list(toadd.keys()))
            remain_index = [i for i in range(len(rep_atoms)) if i != del_index]
            pos = rep_atoms.positions[del_index]
            if check_new_atom(rep_atoms[remain_index], pos, add_symbol, distance_dict):
                rep_atoms[del_index].symbol = add_symbol
                toadd[add_symbol] -= 1
                if toadd[add_symbol] == 0:
                    toadd.pop(add_symbol)
        else:
            del rep_atoms[del_index]
        toremove[del_symbol] -= 1
        if toremove[del_symbol] == 0:
            toremove.pop(del_symbol)

    while toadd:
        add_symbol = np.random.choice(list(toadd.keys()))
        for _ in range(max_n_try):
            # select a center atoms
            center_atom = rep_atoms[np.random.randint(0, len(rep_atoms))]
            basic_r = distance_dict[(center_atom.symbol, add_symbol)]
            radius = basic_r * (1 + np.random.uniform(0, 0.3))
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            new_pos = center_atom.position + radius * np.array([np.sin(theta) * np.cos(phi), 
                                                                np.sin(theta) * np.sin(phi),
                                                                np.cos(theta)])
            if check_new_atom(rep_atoms, new_pos, add_symbol, distance_dict):
                rep_atoms.append(Atom(symbol=add_symbol, position=new_pos))
                toadd[add_symbol] -= 1
                if toadd[add_symbol] == 0:
                    toadd.pop(add_symbol)
                break
        else:
            return Atoms()
    return rep_atoms


class Individual(Atoms):
    @classmethod
    def set_parameters(cls, **parameters):
        # symbols is a property of atoms, will raise Error if set symbols here
        cls.all_parameters = parameters
        Requirement = ['symbol_numlist_pool', 'symprec', 'fp_calc', 'comparator']
        Default={
            'n_repair_try': 3, 
            'max_attempts': 50,
            'add_symmetry': False, 
            'check_seed': True,
            'min_lattice': [0., 0., 0., 45., 45., 45.],
            'max_lattice': [99, 99, 99, 135, 135, 135],
            'd_ratio': 1.,
            'distance_matrix': None,
            'radius': None,
            'max_forces': 50.,
            'max_enthalpy': 100.,
            }
        check_parameters(cls, parameters, Requirement, Default)
        cls.fp_calc = get_fingerprint(parameters)
        cls.comparator = get_comparator(parameters)
        # atoms.symbols has been used by ase
        cls.symbol_list = parameters['symbols']
        cls.distance_dict = get_distance_dict(cls.symbol_list, cls.radius, cls.d_ratio, cls.distance_matrix)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'origin' not in self.info:
            self.info['origin'] = 'Unknown'
        if self.info['origin'] == 'seed' and not self.check_seed:
            self.check_list = []
        else:
            self.check_list = ['check_cell', 'check_distance', 'check_formula', 'check_forces', 'check_enthalpy']
        self.info['fitness'] = {}
        self.info['used'] = 0     # time used in heredity

    def __eq__(self, obj):
        return self.comparator.looks_like(self, obj)

    def to_save(self):
        atoms = self.copy()
        atoms.set_calculator(None)
        atoms.info['type'] = self.__class__.__name__
        if 'trajs' in atoms.info:
            del atoms.info['trajs']
        return atoms

    # TODO avoid repetitive computation 
    @property
    def fingerprint(self):
        if 'fingerprint' not in self.info:
            self.info['fingerprint'] = self.fp_calc.get_all_fingerprints(self)[0]
        return self.info['fingerprint']

    def find_spg(self):
        spg = spglib.get_spacegroup(self, self.symprec)
        pattern = re.compile(r'\(.*\)')
        try:
            spg = pattern.search(spg).group()
            spg = int(spg[1:-1])
        except:
            spg = 1
        self.info['spg'] = spg
        pri_atoms = spglib.find_primitive(self, symprec=self.symprec)
        if pri_atoms:
            cell, positions, numbers = pri_atoms
            self.info['priNum'] = numbers
            self.info['priVol'] = abs(np.linalg.det(cell))
        else:
            self.info['priNum'] = self.get_atomic_numbers()
            self.info['priVol'] = self.get_volume()

    def add_symmetry(self):
        std_atoms = spglib.standardize_cell(self, symprec=self.symprec)
        if std_atoms:
            cell, positions, numbers = std_atoms
            self.set_cell(cell)
            self.set_scaled_positions(positions)
            self.set_atomic_numbers(numbers)
            return True
        return False

    @property
    def numlist(self):
        return [self.get_chemical_symbols().count(s) for s in self.symbol_list] 

    @property
    def ball_volume(self):
        return sum([v * n for v, n in zip(self.volume, self.numlist)])

    @property
    def volume_ratio(self):
        return self.get_volume() / self.ball_volume

    def check(self, atoms=None):
        atoms = atoms or self
        origin = atoms.info['origin'] if 'origin' in atoms.info else 'Unknown'
        for f in self.check_list:
            if not getattr(self, f)(atoms):
                log.debug("Fail in {}, origin = {}".format(f, origin))
                return False
        return True

    def check_forces(self, atoms=None):
        atoms = atoms or self
        if 'forces' in atoms.info:
            return np.max(np.abs(atoms.info['forces'])) < self.max_forces
        return True

    def check_enthalpy(self, atoms=None):
        atoms = atoms or self
        if 'enthalpy' in atoms.info:
            return atoms.info['enthalpy'] < self.max_enthalpy
        return True

    def check_cell(self, atoms=None):
        atoms = atoms or self
        # angle between edges
        edge_angles = atoms.cell.angles()    
        edge_angles_ok = np.all([*(45 <= edge_angles), *(edge_angles <= 135)])
        # angle between edge and surface
        cos_ = np.cos(edge_angles / 180 * np.pi)
        sin_ = np.sin(edge_angles / 180 * np.pi)         
        X = np.sum(cos_ ** 2) - 2 * np.prod(cos_)
        surface_angles = np.arccos(np.sqrt(X - cos_**2) / sin_) / np.pi * 180
        surface_angles_ok = np.all([*(45 <= surface_angles), *(surface_angles <= 135)])
        return edge_angles_ok and surface_angles_ok

    def check_distance(self, atoms=None):
        atoms = atoms or self
        i_indices = neighbor_list('i', atoms, self.distance_dict, max_nbins=100.0)
        return len(i_indices) == 0

    def check_formula(self, atoms=None):
        atoms = atoms or self
        symbols_numlist = np.array(self.numlist)
        for possible_symbols_numlist in self.symbol_numlist_pool:
            if np.all(symbols_numlist == possible_symbols_numlist):
                return True
        else:
            return False

    def sort(self):
        atoms = self[self.numbers.argsort()]
        self.__init__(atoms)

    def merge_atoms(self):
        # exclude atoms in the order of their number of neighbours 
        i = neighbor_list('i', self, self.distance_dict, max_nbins=100.0)
        while len(i) > 0:
            i_ = np.argmax(np.bincount(i))   # remove the atom with the most neighbours 
            del self[i_]
            i = neighbor_list('i', self, self.distance_dict, max_nbins=100.0)

    def get_target_formula(self, n=1):
        symbols = self.get_chemical_symbols()
        distances = self.symbol_numlist_pool - np.array([symbols.count(s) for s in self.symbol_list])
        # pretend to delete atoms instesd of add atoms
        grades = np.sum(np.where(distances > 0, 1, -0.5) * distances, axis=1)
        target_numlists = self.symbol_numlist_pool[np.argsort(grades)[:n]]
        target_formulas = [{s: numlist[i] for i, s in enumerate(self.symbol_list)}
                                          for numlist in target_numlists]
        return target_formulas

    def repair_atoms(self, n=3):
        self.merge_atoms()         # merge atoms too close before repair it
        if len(self) == 0:
            log.debug("Empty crystal after merging!")
            return False
        for target_formula in self.get_target_formula():
            rep_atoms = to_target_formula(self, target_formula, self.distance_dict)
            if len(rep_atoms) > 0:
                self.__init__(rep_atoms)
                self.sort()
                return True
        else:
            return False
