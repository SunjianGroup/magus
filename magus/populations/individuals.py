import logging
import spglib
from ase import Atoms, Atom
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
from magus.utils import *
from .molecule import Molfilter
from ..fingerprints import get_fingerprint
from ..comparators import get_comparator
import ase.build


log = logging.getLogger(__name__)
__all__ = ['Bulk', 'Layer', 'Chain','Confined2D','Confined1D']


def get_Ind(p_dict):
    if p_dict['structureType'] == 'bulk':
        Ind = Bulk
    elif p_dict['structureType'] == 'layer':
        Ind = Layer
    elif p_dict['structureType'] == 'confined_2d':
        Ind = Confined2D
    elif p_dict['structureType'] == 'confined_1d':
        Ind = Confined1D
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
def to_target_formula(atoms, target_formula, distance_dict, max_n_try=10): 
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
            'n_repair_try': 5, 
            'max_attempts': 50,
            'check_seed': False,
            'min_lattice': [0., 0., 0., 45., 45., 45.],
            'max_lattice': [99, 99, 99, 135, 135, 135],
            'd_ratio': 1.,
            'distance_matrix': None,
            'radius': None,
            'max_forces': 50.,
            'max_enthalpy': 100.,
            'full_ele': True,
            'max_length_ratio': 8,
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
#            self.check_list = ['check_distance']
            if self.full_ele:
                self.check_list.append('check_full')
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
        if 'compare_info' in atoms.info:
            del atoms.info['compare_info']
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
        pri_atoms = spglib.standardize_cell(self, symprec=self.symprec, to_primitive=True)
        if pri_atoms:
            cell, positions, numbers = pri_atoms
            self.info['priNum'] = numbers
            self.info['priVol'] = abs(np.linalg.det(cell))
        else:
            self.info['priNum'] = self.get_atomic_numbers()
            self.info['priVol'] = self.get_volume()

    def add_symmetry(self, keep_n_atoms=True, to_primitive=False):
        std_para = spglib.standardize_cell(self, symprec=self.symprec, to_primitive=to_primitive)
        if std_para is None:
            return False
        std_atoms = Atoms(cell=std_para[0], scaled_positions=std_para[1], numbers=std_para[2])
        if keep_n_atoms:
            if len(self) % len(std_atoms) == 0:
                std_atoms = multiply_cell(std_atoms, len(self) // len(std_atoms))
            elif not to_primitive:
                return self.add_symmetry(keep_n_atoms, to_primitive=True)
        self.set_cell(std_atoms.cell)
        self.set_scaled_positions(std_atoms.get_scaled_positions())
        self.set_atomic_numbers(std_atoms.numbers)
        return True

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
        # atoms cell length
        cell_lengths = atoms.cell.lengths()
        cell_lengths_ok = max(cell_lengths) / min(cell_lengths) < self.max_length_ratio
        # angle between edges
        edge_angles = atoms.cell.angles()    
        edge_angles_ok = np.all([*(30 <= edge_angles), *(edge_angles <= 150)])
        # angle between edge and surface
        cos_ = np.cos(edge_angles / 180 * np.pi)
        sin_ = np.sin(edge_angles / 180 * np.pi)         
        X = np.sum(cos_ ** 2) - 2 * np.prod(cos_)
        surface_angles = np.arccos(np.sqrt(X - cos_**2) / sin_) / np.pi * 180
        surface_angles_ok = np.all([*(30 <= surface_angles), *(surface_angles <= 150)])
        return cell_lengths_ok and edge_angles_ok and surface_angles_ok

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

    def check_full(self, atoms=None):
        atoms = atoms or self
        return 0 not in self.numlist

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
            rep_atoms = to_target_formula(self, target_formula, self.distance_dict, self.n_repair_try)
            if len(rep_atoms) > 0:
                self.__init__(rep_atoms)
                self.sort()
                return True
        else:
            return False


class Bulk(Individual):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Default = {
            'mol_detector': 0, 
            'bond_ratio': 1.1,
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list]}
        check_parameters(cls, parameters, [], Default)
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])

    def __init__(self, *args, **kwargs):
        if 'symbols' in kwargs:
            if isinstance(kwargs['symbols'], Molfilter):
                kwargs['symbols'] = kwargs['symbols'].to_atoms()
        if len(args) > 0:
            if isinstance(args[0], Molfilter):
                args = list(args)
                args[0] = args[0].to_atoms()
        super().__init__(*args, **kwargs)

    def for_heredity(self):
        atoms = self.copy()
        if self.mol_detector > 0:
            atoms = Molfilter(atoms, detector=self.mol_detector, coef=self.bond_ratio)
        return atoms



class Layer(Individual):
    @staticmethod
    def translate_to_bottom(atoms):
        new_atoms = atoms.copy()
        p = atoms.get_scaled_positions()
        z = sorted(p[:, 2])
        z.append(z[0] + 1)
        p[:, 2] -= z[np.argmax(np.diff(z))] + np.max(np.diff(z)) - 1
        p[:, 2] -= np.min(p[:, 2]) - 1e-8   # Prevent numerical error
        new_atoms.set_scaled_positions(p)
        return new_atoms

    @staticmethod
    def remove_vacuum(atoms, thickness=1.5):
        new_atoms = Layer.translate_to_bottom(atoms)
        new_cell = new_atoms.get_cell()
        ratio = new_atoms.get_scaled_positions()[:, 2].max() + thickness / new_cell.lengths()[2]
        new_cell[2] *= ratio
        new_atoms.set_cell(new_cell)
        return new_atoms

    @staticmethod
    def add_vacuum(atoms, thickness=10):
        new_atoms = Layer.translate_to_bottom(atoms.copy())
        new_cell = new_atoms.get_cell()
        # some old ase version doesn't have cell.area()
        h = new_atoms.get_volume() / np.linalg.norm(np.cross(new_cell[0], new_cell[1]))
        new_cell[2] *= thickness / h
        new_atoms.set_cell(new_cell)
        p = new_atoms.get_scaled_positions()
        p[:, 2] += 0.5 - (max(p[:, 2]) - min(p[:, 2])) / 2
        new_atoms.set_scaled_positions(p)
        return new_atoms

    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Default = {
            'vacuum_thickness': 10, 
            'bond_ratio': 1.1,
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list]}
        check_parameters(cls, parameters, [], Default)
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])

    def for_heredity(self):
        atoms = Layer.remove_vacuum(self)
        # atoms.set_pbc([True, True, False])
        return atoms

    def for_calculate(self):
        atoms = Layer.add_vacuum(self, self.vacuum_thickness)
        return atoms

    @property
    def volume_ratio(self):
        return self.remove_vacuum(self).get_volume() / self.ball_volume

class Chain(Individual):
    @staticmethod
    def rotate_axis(atoms):
        new_atoms = atoms.copy()
        p = atoms.get_scaled_positions()
        cell = atoms.get_cell()
        new_p = np.roll(p,-1,axis=1)
        new_cell = np.roll(cell,-1,axis=0)
        cell[0] = np.array([20,0.0,0.0])
        cell[1] = np.array([0.0,20,0.0])
        cell[2] = np.array([0.0,0.0,new_cell[2][0]])
        print(cell)
        new_atoms.set_cell(cell)
        new_atoms.set_scaled_positions(new_p)
        return new_atoms

    @staticmethod
    def translate_to_bottom(atoms):
        new_atoms = atoms.copy()
        p = atoms.get_scaled_positions()
        x = sorted(p[:, 0])
        y = sorted(p[:, 1])
        x.append(x[0] + 1)
        y.append(y[0] + 1)
        p[:, 0]=(p[:,0]-x[np.argmax(np.diff(x))] - np.max(np.diff(x)) + 1)%1+1e-8
        p[:, 1]=(p[:,1]-y[np.argmax(np.diff(y))] - np.max(np.diff(y)) + 1)%1+1e-8
        new_atoms.set_scaled_positions(p)
        return new_atoms

    @staticmethod
    def remove_vacuum(atoms, thickness=1.5):
        new_atoms = Chain.translate_to_bottom(atoms)
        new_cell = new_atoms.get_cell()
        ratiox = new_atoms.get_scaled_positions()[:, 0].max() + thickness / new_cell.lengths()[0]
        ratioy = new_atoms.get_scaled_positions()[:, 1].max() + thickness / new_cell.lengths()[1]
        new_cell[0] *= ratiox
        new_cell[1] *= ratioy
        new_atoms.set_cell(new_cell)
        return new_atoms

    @staticmethod
    def add_vacuum(atoms, thickness=10):
#        na = Chain.rotate_axis(atoms.copy())
        new_atoms = Chain.translate_to_bottom(atoms)
        new_cell = new_atoms.get_cell()
        # some old ase version doesn't have cell.area()
        #h = new_atoms.get_volume() / np.linalg.norm(np.cross(new_cell[0], new_cell[1]))
        #new_cell[2] *= thickness / h
        new_cell[0] = np.array([thickness,0.0,0.0])
        new_cell[1] = np.array([0.0,thickness,0.0])
        new_cell[2] = np.array([0.0,0.0,abs(new_cell[2][2])])
        new_atoms.set_cell(new_cell)
        new_atoms2 = Chain.translate_to_bottom(new_atoms.copy())
        p = new_atoms2.get_scaled_positions()
        p[:, 0] += 0.5 - (max(p[:, 0]) - min(p[:, 0])) / 2
        p[:, 1] += 0.5 - (max(p[:, 1]) - min(p[:, 1])) / 2
        new_atoms2.set_scaled_positions(p)
        return new_atoms2

    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Default = {
            'vacuum_thickness': 10, 
            'bond_ratio': 1.1,
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list]}
        check_parameters(cls, parameters, [], Default)
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])

    def for_heredity(self):
        atoms = Chain.remove_vacuum(self)
        # atoms.set_pbc([True, True, False])
        return atoms

    def for_calculate(self):
        atoms = Chain.add_vacuum(self, self.vacuum_thickness)
        return atoms

    @property
    def volume_ratio(self):
        return self.remove_vacuum(self).get_volume() / self.ball_volume


class Confined2D(Individual):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Default = {
            'vacuum_thickness': 10, 
            'bond_ratio': 1.1,
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list]}
        check_parameters(cls, parameters, [], Default)
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])

    def for_heredity(self):
        atoms = Layer.remove_vacuum(self)
        if self.mol_detector > 0:
            atoms = Molfilter(atoms, detector=self.mol_detector, coef=self.bond_ratio)
        return atoms

    def for_calculate(self):
        atoms = Layer.add_vacuum(self, self.vacuum_thickness)
        return atoms

    @property
    def volume_ratio(self):
        return Layer.remove_vacuum(self).get_volume() / self.ball_volume

class Confined1D(Individual):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Default = {
            'vacuum_thickness': 10, 
            'bond_ratio': 1.1,
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list]}
        check_parameters(cls, parameters, [], Default)
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])

    def for_heredity(self):
        atoms = Chain.remove_vacuum(self)
        if self.mol_detector > 0:
            atoms = Molfilter(atoms, detector=self.mol_detector, coef=self.bond_ratio)
        return atoms

    def for_calculate(self):
        atoms = Chain.add_vacuum(self, self.vacuum_thickness)
        return atoms

    @property
    def volume_ratio(self):
        return Chain.remove_vacuum(self).get_volume() / self.ball_volume
