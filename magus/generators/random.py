import itertools, yaml, logging
import numpy as np
from ase import Atoms, build
from ase.io import read, write
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import cellpar_to_cell
from magus.utils import *
# from .reconstruct import reconstruct, cutcell, match_symmetry, resetLattice
from magus.generators import GenerateNew


log = logging.getLogger(__name__)


def find_factor(num):
    i = 2
    while i < np.sqrt(num):
        if num % i == 0:
            break
        i += 1
    if num % i > 0:
        i = num
    return i


def get_swap_matrix():
    M = np.array([
        [[1,0,0],[0,1,0],[0,0,1]],
        [[0,1,0],[1,0,0],[0,0,1]],
        [[0,1,0],[0,0,1],[1,0,0]],
        [[1,0,0],[0,0,1],[0,1,0]],
        [[0,0,1],[1,0,0],[0,1,0]],
        [[0,0,1],[0,1,0],[1,0,0]]])
    return M[np.random.randint(6)]


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
                positions = mole.positions.reshape(-1)
                symbols = mole.get_chemical_symbols()
                uni_symbols = list({}.fromkeys(symbols).keys())
                assert len(uni_symbols) < 5 
                namearray = [str(s) for s in uni_symbols]
                radius = np.array([radius_dict[symbol] for symbol in uni_symbols])
                numinfo = np.array([symbols.count(s) for s in uni_symbols], dtype=float)

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
                 min_volume, max_volume, min_lattice, max_lattice, 
                 dimension=3, max_attempts=50, GetConventional=True, method=1,
                 vacuum=None, choice=None, mol_mode=False, input_mols=None, symprec=None,
                 threshold_mol=1.,
                 *args, **kwargs):
    # set generator
    generator = GenerateNew.Info()
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
    swap_matrix = get_swap_matrix()
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
        atoms = Atoms(cell=L.T, positions=positions, numbers=numbers, pbc=1)
        atoms.wrap(pbc=[1, 1, 1])
        atoms = build.sort(atoms)
        return label, atoms
    else:
        return label, None


# 
# units: units of Generator such as:
#  ['Zn', 'OH'] for ['Zn', 'O', 'H'], [[1, 0, 0], [0, 1, 1]]
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
                   'full_eles': True, 
                   'ele_size': 1,
                   'min_lattice': None,
                   'max_lattice': None,
                   'min_n_formula': None,
                   'max_n_formula': None,
                   'd_ratio': 1.,
                   'distance_matrix': None,
                   'spacegroup': np.arange(2, 231),
                   'max_ratio': 1000,    # max ratio in var search, for 10, Zn11(OH) is not allowed
                   }
        check_parameters(self, parameters, Requirement, Default)
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
                self.formula_pool_ = np.loadtxt(formula_pool_file, dtype=int)
                while len(self.formula_pool_.shape) < 2:
                    self.formula_pool_ = self.formula_pool_[None]
            else:
                self.formula_pool_ = self.get_default_formula_pool()
            np.savetxt(formula_pool_file, self.formula_pool_, fmt='%i')
        return self.formula_pool_

    @property
    def symbol_numlist_pool(self):
        numlist_pool = self.formula_pool @ self.formula
        return numlist_pool

    def get_numlist(self, format_filter=None):
        if format_filter is not None:
            formula_pool = list(filter(lambda f: np.all(np.clip(f, 0, 1) == format_filter), 
                                    self.formula_pool))
        else:
            formula_pool = self.formula_pool
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
        min_volume = 0.5 * mean_volume
        max_volume = 1.5 * mean_volume
        if self.min_lattice is not None:
            min_volume = np.linalg.det(cellpar_to_cell(self.min_lattice))
        if self.max_lattice is not None:
            max_volume = np.linalg.det(cellpar_to_cell(self.max_lattice))
        assert min_volume <= max_volume
        return min_volume, max_volume

    def get_lattice(self, numlist):
        _, max_volume = self.get_volume(numlist)
        min_lattice = [2 * np.max(self.radius)] * 3 + [45.] * 3
        max_lattice = [3 * max_volume ** (1/3)] * 3 + [135] * 3
        if self.min_lattice is not None:
            min_lattice = self.min_lattice
        if self.max_lattice is not None:
            max_lattice = self.max_lattice
        return min_lattice, max_lattice

    def get_generate_parm(self, spg, numlist):
        min_volume, max_volume = self.get_volume(numlist)
        min_lattice, max_lattice = self.get_lattice(numlist)
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
        for key in ['threshold_dict', 'radius', 'symbols', 'dimension', 'max_attempts', 'method', 'vacuum', 'choice']:
            if hasattr(self, key):
                d[key] = getattr(self, key)
        return d

    def generate_ind(self, spg, numlist, n_split):
        numlist_ = np.ceil(numlist / n_split).astype(np.int)
        n_symbols, n_symbols_ = self.get_n_symbols(numlist), self.get_n_symbols(numlist_ * n_split)
        residual = {s: n_symbols[s] - n_symbols_[s] for s in self.symbols}
        label, atoms = spg_generate(**self.get_generate_parm(spg, numlist_))
        if label:
            while n_split > 1:
                i = find_factor(n_split)
                to_expand = np.argmin(atoms.cell.cellpar()[:3])
                expand_matrix = [1, 1, 1]
                expand_matrix[to_expand] = i
                atoms = atoms * expand_matrix
                n_split /= i
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

    def generate_pop(self, n_pop, *args, **kwargs):
        build_pop = []
        while n_pop > len(build_pop):
            for _ in range(self.max_n_try):
                spg = np.random.choice(self.spacegroup)
                n_split = np.random.choice(self.n_split)
                numlist = self.get_numlist(*args, **kwargs)
                label, atoms = self.generate_ind(spg, numlist, n_split)
                if label:
                    self.afterprocessing(atoms)
                    build_pop.append(atoms)
                    break
            else:
                n_split = np.random.choice(self.n_split)
                numlist = self.get_numlist(*args, **kwargs)
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
        self.mol_n_atoms, self.mol_radius = [], []
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

    def get_lattice(self, numlist):
        min_lattice, max_lattice = super().get_lattice(numlist)
        if self.min_lattice is None:
            min_lattice = [2 * np.max(self.mol_radius)] * 3 + [60.] * 3
        return min_lattice, max_lattice

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


class ReconstructGenerator(SPGGenerator):
    def __init__(self, **parameters):
        Requirement = ['symbols', 'input_layers']
        Default = {
            'cutslices': None, 
            'bulk_layernum': 3, 
            'range': 0.5, 
            'relaxable_layernum': 3, 
            'rcs_layernum': 2, 
            'randratio': 0.5,
            'rcs_x': 1, 
            'rcs_y': 1, 
            'direction': None, 
            'rotate': 0, 
            'matrix': None, 
            'extra_c': 1.0,
            'dimension': 2, 
            'choice': 0,}
        check_parameters(self, parameters, Requirement, Default)

        for i, layer in enumerate(self.input_layers):
            if isinstance(layer, str):
                layer = read(layer)
            assert isinstance(layer, Atoms), "input layers must be Atoms or a file path can be read by ASE"
            for s in layer.get_chemical_symbols():
                assert s in self.symbols, "{} of {} not in given symbols".format(s, layer.get_chemical_formula())
            self.input_layers[i] = layer
        
        assert len(self.input_layers) in [2, 3]

        if len(self.input_layers)==3:
            # mode = 'reconstruct'
            self.ref_layer = self.input_layers[2]
        else:
            # mode = 'add atoms'
            self.ref_layer = self.input_layers[1].copy()
        if 'formula' not in parameters:
            assert 'addFormula' in parameters, "must have 'formula' or 'addFormula'"
            ref_symbols = self.ref_layer.get_chemical_symbols()
            parameters['formula'] = [ref_symbols.count(s) + n_add 
                                     for s, n_add in self.symbols]

        ## ?????????????
        setlattice = np.round(setlattice, 3)
        setlattice[3:] = [i if np.round(i) != 60 else 120.0 for i in setlattice[3:]]
        self.symtype = 'hex' if 120 in np.round(setlattice[3:]) else 'orth'

        super().__init__(**parameters)

    def get_default_formula_pool(self):
        if self.p.AtomsToAdd:
            assert len(self.p.AtomsToAdd)== len(self.p.symbols), 'Please check the length of AddAtoms'
            try:
                self.p.AtomsToAdd = split_modifier(self.p.AtomsToAdd)
            except:
                raise RuntimeError("wrong format of atomstoadd")
        if self.p.DefectToAdd:
            try:
                self.p.DefectToAdd =  split_modifier(self.p.DefectToAdd)
            except:
                raise RuntimeError("wrong format of defectstoadd")

    def __init__(self,parameters):
        para_t = EmptyClass()
        Requirement=['layerfile']
        Default={'cutslices': None, 'bulk_layernum':3, 'range':0.5, 'relaxable_layernum':3, 'rcs_layernum':2, 'randratio':0.5,
        'rcs_x':[1], 'rcs_y':[1], 'direction': None, 'rotate': 0, 'matrix': None, 'extra_c':1.0, 
        'dimension':2, 'choice':0 }

        checkParameters(para_t, parameters, Requirement,Default)
        
        if os.path.exists("Ref") and os.path.exists("Ref/refslab.traj") and os.path.exists("Ref/layerslices.traj"):
            log.info("Used layerslices in Ref.")
        else:
            if not os.path.exists("Ref"):
                os.mkdir('Ref')
            #here starts to get Ref/refslab to calculate refE            
            ase.io.write("Ref/refslab.traj", ase.io.read(para_t.layerfile), format = 'traj')
            #here starts to split layers into [bulk, relaxable, rcs]
            originatoms = ase.io.read(para_t.layerfile)
            layernums = [para_t.bulk_layernum, para_t.relaxable_layernum, para_t.rcs_layernum]
            cutcell(originatoms, layernums, totslices = para_t.cutslices, direction= para_t.direction,rotate = para_t.rotate, vacuum = para_t.extra_c, matrix = para_t.matrix)
            #layer split ends here    

        self.range=para_t.range
        
        self.ind=RcsInd(parameters)

        #here get new parameters for self.generator 
        _parameters = copy.deepcopy(parameters)
        _parameters.attach(para_t)
        self.layerslices = ase.io.read("Ref/layerslices.traj", index=':', format='traj')
        
        setlattice = []
        if len(self.layerslices)==3:
            #mode = 'reconstruct'
            self.ref = self.layerslices[2]
            vertical_dis = self.ref.get_scaled_positions()[:,2].copy()
            mincell = self.ref.get_cell().copy()
            mincell[2] *= (np.max(vertical_dis) - np.min(vertical_dis))*1.2
            setlattice = list(cell_to_cellpar(mincell))
        else:
            #mode = 'add atoms'
            para_t.randratio = 0
            self.ref = self.layerslices[1].copy()
            lattice = self.ref.get_cell().copy()
            lattice [2]/= para_t.relaxable_layernum
            self.ref.set_cell(lattice)
            setlattice = list(cell_to_cellpar(lattice))

        setlattice = np.round(setlattice, 3)
        setlattice[3:] = [i if np.round(i) != 60 else 120.0 for i in setlattice[3:]]
        self.symtype = 'hex' if 120 in np.round(setlattice[3:]) else 'orth'

        self.reflattice = list(setlattice).copy()
        target = self.ind.get_targetFrml()
        _symbol = [s for s in target]
        requirement = {'minLattice': setlattice, 'maxLattice':setlattice, 'symbols':_symbol}

        for key in requirement:
            if not hasattr(_parameters, key):
                setattr(_parameters,key,requirement[key])
            else:
                if getattr(_parameters,key) == requirement[key]:
                    pass
                else:
                    logging.info("warning: change user defined {} to {} to match rcs layer".format(key, requirement[key]))
                    setattr(_parameters,key,requirement[key])

        self.rcs_generator =Generator(_parameters)
        self.rcs_generator.p.choice =_parameters.choice
        #got a generator! next put all parm together except changed ones

        self.p = EmptyClass()
        self.p.attach(para_t)
        self.p.attach(self.rcs_generator.p)

        origindefault={'symbols':parameters.symbols}
        origindefault['minLattice'] = parameters.minLattice if hasattr(parameters, 'minLattice') else None
        origindefault['maxLattice'] = parameters.maxLattice if hasattr(parameters, 'maxLattice') else None

        for key in origindefault:
            if not hasattr(self.p, key):
                pass
            else:
                setattr(self.p,key,origindefault[key])
        
        #some other settings
        minFrml = int(np.ceil(self.p.minAt/sum(self.p.formula)))
        maxFrml = int(self.p.maxAt/sum(self.p.formula))
        self.p.numFrml = list(range(minFrml, maxFrml + 1))
        self.threshold = self.p.dRatio
        self.maxAttempts = 100

    def afterprocessing(self,ind,nfm, origin, size):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = self.p.formula
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = origin
        ind.info['size'] = size

        return ind
        
    def update_volume_ratio(self, volume_ratio):
        pass
        #return self.rcs_generator.update_volume_ratio(volume_ratio)
    def Generate_ind(self,spg,numlist):
        return self.rcs_generator.Generate_ind(spg,numlist)

    def reconstruct(self, ind):

        c=reconstruct(self.range, ind.copy(), self.threshold, self.maxAttempts)
        label, pos=c.reconstr()
        numbers=[]
        if label:
            for i in range(len(c.atomnum)):
                numbers.extend([atomic_numbers[c.atomname[i]]]*c.atomnum[i])
            cell=c.lattice
            pos=np.dot(pos,cell)
            atoms = ase.Atoms(cell=cell, positions=pos, numbers=numbers, pbc=1)
            
            return label, atoms
        else:
            return label, None

    def rand_displacement(self, extraind, bottomind): 
        rots = []
        trs = []
        for ind in list([bottomind, extraind]):
            sym = spglib.get_symmetry_dataset(ind,symprec=0.2)
            if not sym:
                sym = spglib.get_symmetry_dataset(ind)
            if not sym:
                return False, extraind
            rots.append(sym['rotations'])
            trs.append(sym['translations'])

        m = match_symmetry(*zip(rots, trs), z_axis_only = True)
        if not m.has_shared_sym:
            return False, extraind
        _dis_, rot = m.get()
        #_dis_, rot = match_symmetry(*zip(rots, trs)).get() 
        _dis_[2] = 0
        _dis_ = np.dot(-_dis_, extraind.get_cell())

        extraind.translate([_dis_]*len(extraind))
        return True, extraind

    def get_spg(self, kind, grouptype):
        if grouptype == 'layergroup':
            if kind == 'hex':
                #sym = 'c*', 'p6*', 'p3*', 'p-6*', 'p-3*' 
                return [1, 2, 22, 26, 35, 36, 47, 48] + range(65, 81)  + [10, 13, 18]
            else:
                return list(range(1, 65))
        elif grouptype == 'planegroup':
            if kind == 'hex':
                return [1, 2, 5, 9] + list(range(13, 18))
            else:
                return list(range(1, 13))

    def reset_rind_lattice(self, atoms, _x, _y, botp = 'refbot', type = 'bot'):

        refcell = (self.ref * (_x, _y, 1)).get_cell_lengths_and_angles()
        cell = atoms.get_cell_lengths_and_angles()

        if not np.allclose(cell[:2], refcell[:2], atol=0.1):
            return False, None
        if not np.allclose(cell[3:], refcell[3:], atol=0.5):
            #'hex' lattice
            if np.round(refcell[-1] + cell[-1] )==180:
                atoms = resetLattice(atoms = atoms.copy(), expandsize = (4,1,1)).get(np.dot(np.diag([-1, 1, 1]), atoms.get_cell() ))

            else:
                return False, None
        atoms.set_cell(np.dot(np.diag([1,1, refcell[2]/cell[2]]) ,atoms.get_cell()))
        refcell = (self.ref * (_x, _y, 1)).get_cell()
        atoms.set_cell(refcell, scale_atoms = True)
        pos = atoms.get_scaled_positions(wrap = False)
        refpos = self.ref.get_scaled_positions(wrap = True)
        bot = np.min(pos[:,2]) if type == 'bot' else np.mean(pos[:, 2])
        tobot = np.min(refpos[:,2])*atoms.get_cell()[2] if isinstance(botp, str) else botp
        atoms.translate([ tobot - bot*atoms.get_cell()[2]]* len(atoms))
        return True, atoms
        
        
    def reset_generator_lattice(self, _x, _y, spg):
        symtype = 'default'
        if self.symtype == 'hex':
            if (self.rcs_generator.p.choice == 0 and spg < 13) or (self.rcs_generator.p.choice == 1 and spg < 65):
                #for hex-lattice, 'a' must equal 'b'
                if self.reflattice[0] == self.reflattice[1] and _x == _y:    
                    symtype = 'hex'

        if symtype == 'hex':
            self.rcs_generator.p.GetConventional = False
        elif symtype == 'default': 
            self.rcs_generator.p.GetConventional = True

        self.rcs_generator.p.minLattice = list(self.reflattice *np.array([_x, _y]+[1]*4))
        self.rcs_generator.p.maxLattice = self.rcs_generator.p.minLattice
        return symtype


    def get_lattice(self, numlist):
        _, max_volume = self.get_volume(numlist)
        min_lattice = [2 * np.max(self.radius)] * 3 + [45.] * 3
        max_lattice = [3 * max_volume ** (1/3)] * 3 + [135] * 3
        if self.min_lattice is not None:
            min_lattice = self.min_lattice
        if self.max_lattice is not None:
            max_lattice = self.max_lattice

        self.ref = self.layerslices[1].copy()
        lattice = self.ref.get_cell().copy()
        lattice [2]/= para_t.relaxable_layernum
        self.ref.set_cell(lattice)
        setlattice = list(cell_to_cellpar(lattice))

        setlattice = np.round(setlattice, 3)
        setlattice[3:] = [i if np.round(i) != 60 else 120.0 for i in setlattice[3:]]
        self.symtype = 'hex' if 120 in np.round(setlattice[3:]) else 'orth'

        self.reflattice = list(setlattice).copy()
        target = self.ind.get_targetFrml()
        _symbol = [s for s in target]
        requirement = {'minLattice': setlattice, 'maxLattice':setlattice, 'symbols':_symbol}


    def generate_pop(self, n_pop, *args, **kwargs):
        build_pop = []
        while n_pop > len(build_pop):
            for _ in range(self.max_n_try):
                spg = np.random.choice(self.spacegroup)
                n_split = np.random.choice(self.n_split)
                numlist = self.get_numlist(*args, **kwargs)
                label, atoms = self.generate_ind(spg, numlist, n_split)



                self.rcs_generator.p.choice = self.p.choice
                #logging.debug("formula {} of number {} with chosen spg = {}".format(self.rcs_generator.p.symbols, numlist,spg))
                #logging.debug("with maxlattice = {}".format(self.rcs_generator.p.maxLattice))
                label,ind = self.rcs_generator.Generate_ind(spg,numlist)

                if label:
                    #label, ind = self.reset_rind_lattice(ind, _x, _y, botp = 'refbot', type = 'bot')
                    label, ind = self.reset_rind_lattice(ind, _x, _y, botp = 'refbot')
                if label:
                    _bot_ = (self.layerslices[1] * (_x, _y, 1)).copy()
                    _bot_.info['size'] = [_x, _y]
                    
                    label, ind = self.rand_displacement(ind, self.ind.addvacuum(add = 1, atoms = self.ind.addextralayer('bulk', atoms=_bot_, add = 1)))
                if label:
                    self.afterprocessing(ind,nfm,'rand.symmgen', [_x, _y])
                    ind = self.ind.addbulk_relaxable_vacuum(atoms = ind)
                    buildPop.append(ind)
                if not label:
                    tryNum+=1


                if label:
                    self.afterprocessing(atoms)
                    build_pop.append(atoms)
                    break
            else:
                n_split = np.random.choice(self.n_split)
                numlist = self.get_numlist(*args, **kwargs)
                label, atoms = self.generate_ind(1, numlist, n_split)
                if label:
                    self.afterprocessing(atoms, *args, **kwargs)
                    build_pop.append(atoms)
        return build_pop

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
