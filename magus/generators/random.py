import itertools, yaml, logging
from secrets import choice
import numpy as np
from sklearn.decomposition import PCA
from ase import Atoms, build
from ase.io import read, write
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import cellpar_to_cell
from magus.utils import *
from . import gensym


log = logging.getLogger(__name__)


def get_swap_matrix(random_swap_axis):
    M = np.array([
        [[1,0,0],[0,1,0],[0,0,1]],
        [[0,1,0],[1,0,0],[0,0,1]],
        [[0,1,0],[0,0,1],[1,0,0]],
        [[1,0,0],[0,0,1],[0,1,0]],
        [[0,0,1],[1,0,0],[0,1,0]],
        [[0,0,1],[0,1,0],[1,0,0]]])
    if random_swap_axis:
        return M[np.random.randint(6)]
    else:
        return M[0]


def add_atoms(generator, numlist, radius, symbols):
    numbers = []
    for i, num in enumerate(numlist):
        if num > 0:
            generator.AppendAtoms(int(numlist[i]), symbols[i], radius[i], False)
            numbers.extend([atomic_numbers[symbols[i]]] * numlist[i])
    return numbers


def add_moles(generator, numlist, radius, symbols, input_mols, symprec):
    numbers = []
    radius_dict = dict(zip(symbols, radius))
    for i, num in enumerate(numlist):
        if num > 0:
            mole = input_mols[i]
            if len(mole) > 1:
                positions = mole.positions.reshape(-1).tolist()
                symbols = mole.get_chemical_symbols()
                uni_symbols = list({}.fromkeys(symbols).keys())
                assert len(uni_symbols) < 5 
                namearray = [str(s) for s in uni_symbols]
                radius = [radius_dict[symbol] for symbol in uni_symbols]
                numinfo = [symbols.count(s) for s in uni_symbols]

                generator.AppendMoles(int(numlist[i]), mole.get_chemical_formula(),
                                      radius, positions, numinfo, namearray, symprec)

                number = sum([num for num in [[atomic_numbers[s]] * int(n) * numlist[i] 
                                  for s,n in zip(uni_symbols,numinfo)]], [])
                numbers.extend(number)
            else:
                symbol = mole.get_chemical_symbols()[0]
                radius = radius_dict[symbol]
                generator.AppendAtoms(int(numlist[i]), symbol, radius, False)
                numbers.extend([atomic_numbers[symbol]] * numlist[i])
    return numbers


def spg_generate(spg, threshold_dict, numlist, radius, symbols, 
                 min_volume, max_volume, min_lattice, max_lattice, random_swap_axis=True, 
                 dimension=3, max_attempts=50, GetConventional=True, method=1,
                 vacuum=None, choice=None, mol_mode=False, input_mols=None, symprec=None,
                 threshold_mol=1.,
                 *args, **kwargs):
    # set generator
    generator = gensym.Info()
    generator.spg = int(spg)
    generator.spgnumber = 1
    generator.maxAttempts = max_attempts
    generator.dimension = dimension
    if vacuum is not None:
        generator.vacuum = vacuum
    if choice is not None:
        generator.choice = choice
    generator.threshold = 100. # now use threshold_dict instead of threshold
    generator.method = method
    generator.forceMostGeneralWyckPos = False
    generator.UselocalCellTrans = 'y'
    generator.GetConventional = GetConventional
    generator.minVolume = min_volume
    generator.maxVolume = max_volume
    # swap axis
    swap_matrix = get_swap_matrix(random_swap_axis)
    min_lattice = np.kron(np.array([[1,0],[0,1]]), swap_matrix) @ min_lattice
    max_lattice = np.kron(np.array([[1,0],[0,1]]), swap_matrix) @ max_lattice

    generator.SetLatticeMins(min_lattice[0], min_lattice[1], min_lattice[2], min_lattice[3], min_lattice[4], min_lattice[5])
    generator.SetLatticeMaxes(max_lattice[0], max_lattice[1], max_lattice[2], max_lattice[3], max_lattice[4], max_lattice[5])
    if mol_mode:
        generator.threshold_mol = threshold_mol
        numbers = add_moles(generator, numlist, radius, symbols, input_mols, symprec)
    else:
        numbers = add_atoms(generator, numlist, radius, symbols)

    for s1, s2 in itertools.combinations_with_replacement(symbols, 2):
        generator.SpThreshold(s1, s2, threshold_dict[(s1, s2)])

    label = generator.Generate(np.random.randint(1000))
    if label:
        cell = generator.GetLattice(0)
        cell = np.reshape(cell, (3,3))
        cell_ = np.linalg.inv(swap_matrix) @ cell
        Q, L = np.linalg.qr(cell_.T)
        scaled_positions = generator.GetPosition(0)
        scaled_positions = np.reshape(scaled_positions, (-1, 3))
        positions = scaled_positions @ cell @ Q
        if np.linalg.det(L) < 0:
            L[2, 2] *= -1
            positions[:, 2] *= -1
        pbc = np.zeros(3)
        pbc[:dimension] = 1
        atoms = Atoms(cell=L.T, positions=positions, numbers=numbers, pbc=pbc)
        atoms.wrap(pbc=pbc)
        atoms = build.sort(atoms)
        return label, atoms
    else:
        return label, None


# 
# units: units of Generator such as:
#  ['Zn', 'OH'] for ['Zn', 'O', 'H'], [[1, 0, 0], [0, 1, 1]]
#  [']
class SPGGenerator:
    def __init__(self, **parameters):
        self.all_parameters = parameters
        Requirement = ['formula_type', 'symbols', 'formula', 'min_n_atoms', 'max_n_atoms']
        Default = {#'threshold': 1.0,
                   'max_attempts': 50,
                   'method': 1, 
                   'p_pri': 0.,           # probability of generate primitive cell
                   'volume_ratio': 1.5,
                   'n_split': [1],
                   'max_n_try': 100, 
                   'dimension': 3,
                   'ele_size': 0,
                   'min_lattice': [-1, -1, -1, -1, -1, -1],
                   'max_lattice': [-1, -1, -1, -1, -1, -1],
                   'min_volume': -1,
                   'max_volume': -1,
                   'min_volume_ratio': 0.5,
                   'max_volume_ratio': 1.5,
                   'min_n_formula': None,
                   'max_n_formula': None,
                   'd_ratio': 1.,
                   'distance_matrix': None,
                   'spacegroup': np.arange(2, 231),
                   'max_ratio': 1000,    # max ratio in var search, for 10, Zn11(OH) is not allowed
                   'full_ele': True,     # only generate structures with full elements
                   }
        check_parameters(self, parameters, Requirement, Default)
        if self.ele_size > 0:
            assert not self.full_ele, 'fullEle setting is conflict with eleSize'
        if 'radius' in parameters:
            self.radius = parameters['radius']
        else:
            self.radius = [covalent_radii[atomic_numbers[atom]] for atom in self.symbols]
        self.volume = np.array([4 * np.pi * r ** 3 / 3 for r in self.radius])
        self.threshold_dict = get_threshold_dict(self.symbols, self.radius, self.d_ratio, self.distance_matrix)
        self.first_pop = True
        assert self.formula_type in ['fix', 'var'], "formulaType must be fix or var"
        if self.formula_type == 'fix':
            self.formula = [self.formula]
        self.main_info = ['formula_type', 'symbols', 'min_n_atoms', 'max_n_atoms']

    def __repr__(self):
        ret = self.__class__.__name__
        ret += "\n-------------------"
        for info in self.main_info:
            if hasattr(self, info):
                value = getattr(self, info)
                if isinstance(value, dict):
                    value = yaml.dump(value).rstrip('\n').replace('\n', '\n'.ljust(18))
                ret += "\n{}: {}".format(info.ljust(15, ' '), value)
        ret += "\n-------------------\n"
        return ret

    @property
    def units(self):
        return [Atoms(symbols=[s for n, s in zip(f, self.symbols) for _ in range(n)]) for f in self.formula]

    def get_default_formula_pool(self):
        formula_pool = []
        n_atoms = np.array([sum(f) for f in self.formula])
        min_n_formula = np.zeros(len(self.formula))
        max_n_formula = np.floor(self.max_n_atoms / n_atoms).astype('int')
        if self.min_n_formula is not None:
            assert len(self.min_n_formula) == len(self.formula)
            min_n_formula = np.maximum(min_n_formula, self.min_n_formula)
        if self.full_ele:
            min_n_formula = np.maximum(min_n_formula, 1)
        if self.max_n_formula is not None:
            assert len(self.max_n_formula) == len(self.formula)
            max_n_formula = np.minimum(max_n_formula, self.max_n_formula)
        formula_range = [np.arange(minf, maxf + 1) for minf, maxf in zip(min_n_formula, max_n_formula)]
        for combine in itertools.product(*formula_range):
            combine = np.array(combine)
            if not self.min_n_atoms <= np.sum(n_atoms * combine) <= self.max_n_atoms:
                continue
            if np.max(combine) / np.min(combine[combine > 0]) > self.max_ratio:
                continue
            formula_pool.append(combine)
        formula_pool = np.array(formula_pool, dtype='int')
        return formula_pool
    
    @property
    def formula_pool(self):
        if not hasattr(self, 'formula_pool_'):
            formula_pool_file = os.path.join(self.all_parameters['workDir'], 'formula_pool')
            if os.path.exists(formula_pool_file) and os.path.getsize(formula_pool_file) > 0:
                self.formula_pool_ = np.loadtxt(formula_pool_file, dtype=int,ndmin=2)
            else:
                self.formula_pool_ = self.get_default_formula_pool()
                np.savetxt(formula_pool_file, self.formula_pool_, fmt='%i')
        return self.formula_pool_

    @property
    def symbol_numlist_pool(self):
        numlist_pool = self.formula_pool @ self.formula
        return numlist_pool

    def get_numlist(self, formula_pool):
        return np.array(self.formula).T @ formula_pool[np.random.randint(len(formula_pool))]

    def get_n_symbols(self, numlist):
        return {s: n for s, n in zip(self.symbols, numlist)}

    def set_volume_ratio(self, volume_ratio):
        log.info("change volRatio from {} to {}".format(self.volume_ratio, volume_ratio))
        self.volume_ratio = volume_ratio

    def get_volume(self, numlist):
        assert len(numlist) == len(self.volume)
        ball_volume = sum([v * n for v, n in zip(self.volume, numlist)])
        mean_volume = ball_volume * self.volume_ratio
        min_volume = self.min_volume_ratio * mean_volume
        max_volume = self.max_volume_ratio * mean_volume
        if self.min_volume > 0:
            min_volume = self.min_volume
        if self.max_volume > 0:
            max_volume = self.max_volume
        assert min_volume <= max_volume
        return min_volume, max_volume

    def get_min_lattice(self, numlist):
        radius = [r for i, r in enumerate(self.radius) if numlist[i] > 0]
        min_lattice = [2 * np.max(radius)] * 3 + [45.] * 3
        min_lattice = [b if b > 0 else a for a, b in zip(min_lattice, self.min_lattice)]
        return min_lattice

    def get_max_lattice(self, numlist):
        max_volume = self.get_volume(numlist)[1]
        max_lattice = [3 * max_volume ** (1/3)] * 3 + [135] * 3
        max_lattice = [b if b > 0 else a for a, b in zip(max_lattice, self.max_lattice)]
        return max_lattice

    def get_generate_parm(self, spg, numlist):
        min_volume, max_volume = self.get_volume(numlist)
        min_lattice = self.get_min_lattice(numlist)
        max_lattice = self.get_max_lattice(numlist)
        d = {
            'spg': spg,
            'threshold': self.d_ratio,
            'numlist': numlist,
            'min_volume': min_volume,
            'max_volume': max_volume,
            'min_lattice': min_lattice,
            'max_lattice': max_lattice,
        }
        d['GetConventional'] = True if np.random.rand() > self.p_pri else False
        for key in ['threshold_dict', 'radius', 'symbols', 'dimension', 'max_attempts', 'method', 'choice']:
            if hasattr(self, key) and key not in d:
                d[key] = getattr(self, key)
        return d

    def generate_ind(self, spg, numlist, n_split):
        numlist_ = np.ceil(numlist / n_split).astype(np.int32)
        n_symbols, n_symbols_ = self.get_n_symbols(numlist), self.get_n_symbols(numlist_ * n_split)
        residual = {s: n_symbols[s] - n_symbols_[s] for s in self.symbols}
        label, atoms = spg_generate(**self.get_generate_parm(spg, numlist_))
        if label:
            atoms = multiply_cell(atoms, n_split)
            for i, symbol in enumerate(residual):
                while residual[symbol] > 0:
                    candidate = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
                    to_del = np.random.choice(candidate)
                    del atoms[to_del]
                    residual[symbol] -= 1
            atoms = atoms[atoms.numbers.argsort()]
            return label, atoms
        else:
            return label, None

    def generate_pop(self, n_pop, format_filter=None, *args, **kwargs):
        if format_filter is not None:
            formula_pool = list(filter(lambda f: np.all(np.clip(f, 0, 1) == format_filter), 
                                    self.formula_pool))
        else:
            formula_pool = self.formula_pool
        if len(formula_pool) == 0:
            log.debug("No formula in the pool with the format_filter: {}".format(format_filter))
            return []
        build_pop = []
        while n_pop > len(build_pop):
            for _ in range(self.max_n_try):
                spg = np.random.choice(self.spacegroup)
                n_split = np.random.choice(self.n_split)
                numlist = self.get_numlist(formula_pool)
                label, atoms = self.generate_ind(spg, numlist, n_split)
                if label:
                    self.afterprocessing(atoms)
                    build_pop.append(atoms)
                    break
            else:
                n_split = np.random.choice(self.n_split)
                numlist = self.get_numlist(formula_pool)
                label, atoms = self.generate_ind(1, numlist, n_split)
                if label:
                    self.afterprocessing(atoms, *args, **kwargs)
                    build_pop.append(atoms)
        return build_pop

    def afterprocessing(self, atoms, *args, **kwargs):
        atoms.info['symbols'] = self.symbols
        atoms.info['parentE'] = 0.
        atoms.info['origin'] = 'random'
        atoms.info['units'] = self.units
        atoms.info['units_formula'] = get_units_formula(atoms, self.units)
        return atoms


class MoleculeSPGGenerator(SPGGenerator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['input_mols']
        Default = {'symprec':0.1, 'threshold_mol': 1.0}
        check_parameters(self, parameters, Requirement, Default)
        radius_dict = dict(zip(self.symbols, self.radius))
        self.mol_n_atoms, self.mol_radius, self.thickness = [], [], []
        for i, mol in enumerate(self.input_mols):
            if isinstance(mol, str):
                mol = build.sort(read(mol))
            assert isinstance(mol, Atoms), "input molucules must be Atoms or a file path can be read by ASE"
            for s in mol.get_chemical_symbols():
                assert s in self.symbols, "{} of {} not in given symbols".format(s, mol.get_chemical_formula())
            assert not mol.pbc.any(), "Please provide a molecule ranther than a periodic system!"
            self.mol_n_atoms.append(len(mol))
            # get molecule radius
            center = np.mean(mol.positions, 0)
            radius = np.array([radius_dict[s] for s in mol.get_chemical_symbols()])
            distance = np.linalg.norm(mol.positions - center, axis=1)
            self.mol_radius.append(np.max(distance + radius))
            # use thickness?
            if len(mol) > 2:
                pca = PCA(n_components=3).fit(mol.positions)
                new = mol.positions @ pca.components_
                self.thickness.append(np.max(new[:, -1] + radius) - np.min(new[:, -1] - radius))
            else:
                self.thickness.append(2 * np.max(radius))

            self.input_mols[i] = mol

        self.volume = np.array([sum([4 * np.pi * (radius_dict[s]) ** 3 / 3
                                for s in mol.get_chemical_symbols()])
                                for mol in self.input_mols])

    def get_default_formula_pool(self):
        formula_pool = []
        n_atoms = np.array([sum([m * n for m, n in zip(f, self.mol_n_atoms)]) for f in self.formula])
        min_n_formula = np.zeros(len(self.formula))
        max_n_formula = np.floor(self.max_n_atoms / n_atoms).astype('int')
        if self.min_n_formula is not None:
            assert len(self.min_n_formula) == len(self.formula)
            min_n_formula = np.maximum(min_n_formula, self.min_n_formula)
        if self.full_ele:
            min_n_formula = np.maximum(min_n_formula, 1) 
        if self.max_n_formula is not None:
            assert len(self.max_n_formula) == len(self.formula)
            max_n_formula = np.minimum(max_n_formula, self.max_n_formula)
        formula_range = [np.arange(minf, maxf + 1) for minf, maxf in zip(min_n_formula, max_n_formula)]
        for combine in itertools.product(*formula_range):
            n = sum([na * nf for na, nf in zip(n_atoms, combine)])
            if self.min_n_atoms <= n <= self.max_n_atoms:
                formula_pool.append(combine)
        formula_pool = np.array(formula_pool, dtype='int')
        return formula_pool

    @property
    def symbol_numlist_pool(self):
        mol_num_matrix = np.array([[mol.get_chemical_symbols().count(s) for s in self.symbols]
                                                                        for mol in self.input_mols])
        numlist_pool = self.formula_pool @ self.formula @ mol_num_matrix
        return numlist_pool

    def get_min_lattice(self, numlist):
        thickness = [r for i, r in enumerate(self.thickness) if numlist[i] > 0]
        min_lattice = [np.max(thickness)] * 3 + [45.] * 3
        min_lattice = [b if b > 0 else a for a, b in zip(min_lattice, self.min_lattice)]
        return min_lattice

    def get_generate_parm(self, spg, numlist):
        d = super().get_generate_parm(spg, numlist)
        d.update({
            'mol_mode': True,
            'input_mols': self.input_mols,
            'symprec': self.symprec,
            'threshold_mol': self.threshold_mol,
            })
        return d

    def get_n_symbols(self, numlist):
        return {s: sum([n * m.get_chemical_symbols().count(s) for n, m in zip(numlist, self.input_mols)])  
                                                              for s in self.symbols}
    @property
    def units(self):
        units = []
        for f in self.formula:
            u = Atoms()
            for i, n in enumerate(f):
                for _ in range(n):
                    u.extend(self.input_mols[i])
            u = Atoms(u.get_chemical_formula())
            units.append(u) 
        return units


class LayerSPGGenerator(SPGGenerator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['min_thickness', 'max_thickness']
        Default = {
            'symprec':0.1, 
            'threshold_mol': 1.0, 
            'spg_type': 'layer', 
            'vacuum_thickness': 10,
            }
        check_parameters(self, parameters, Requirement, Default)
        if self.spg_type == 'plane':
            self.choice = 0
            self.spacegroup = [spg for spg in self.spacegroup if spg <= 17]
        elif self.spg_type == 'layer':
            self.choice = 1
            self.spacegroup = [spg for spg in self.spacegroup if spg <= 80]
        else:
            raise Exception("Unexcepted spg type '{}', should be 'plane' or 'layer'".format(self.spg_type))

    def get_volume(self, numlist):
        assert len(numlist) == len(self.volume)
        ball_volume = sum([v * n for v, n in zip(self.volume, numlist)])
        mean_volume = ball_volume * self.volume_ratio
        min_volume = self.min_volume_ratio * mean_volume
        max_volume = self.max_volume_ratio * mean_volume
        if self.min_volume > 0:
            min_volume = self.min_volume
        if self.max_volume > 0:
            max_volume = self.max_volume
        assert min_volume <= max_volume
        return min_volume, max_volume

    def get_min_lattice(self, numlist):
        min_lattice = super().get_min_lattice(numlist)
        min_lattice[2] = self.min_thickness 
        return min_lattice

    def get_max_lattice(self, numlist):
        max_lattice = super().get_max_lattice(numlist)
        max_lattice[2] = self.max_thickness 
        return max_lattice

    def get_generate_parm(self, spg, numlist):
        d = super().get_generate_parm(spg, numlist)
        d.update({
            'choice': self.choice,
            'dimension': 2,
            'random_swap_axis': False,
            'vacuum': self.vacuum_thickness,
            })
        return d


#test
if __name__ == '__main__':
    import ase.io
    p=EmptyClass()
    Requirement=['symbols','formula','numFrml']
    p.symbols=['C','H','O','N']
    p.formula=np.array([1,4,1,2])
    p.numFrml=[1]
    p.volRatio=2

    d = {
        'symbols': ['Ti', 'O'], 
        'formula': [1, 2], 
        'min_n_atoms': 12, 
        'max_n_atoms': 24, 
        'spacegroup': np.arange(230), 
        'd_ratio': 0.8, 
        'bond_ratio': 0.8,
        'threshold': 1.0,
        'max_attempts': 50,
        'method': 1, 
        'p_pri': 0.,           # probability of generate primitive cell
        'volume_ratio': 1.5,
        'max_n_try': 100, 
        'dimension': 3,
        'mol_mode': False,
        }
    g = Generator(**d)
