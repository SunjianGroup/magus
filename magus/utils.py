from __future__ import print_function, division
from enum import unique
import os, re, logging, itertools, traceback
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.geometry import wrap_positions
from ase.data import atomic_numbers, covalent_radii
from scipy.spatial.distance import cdist
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar,cellpar_to_cell
from functools import reduce
from math import gcd
from importlib import import_module
from pathlib import Path
import logging


log = logging.getLogger(__name__)


class Singleton:
    def __init__(self, cls):
        self._cls = cls

    def __call__(self):
        if not hasattr(self, '_instance'):
            self._instance = self._cls()
        return self._instance


def check_new_atom_dist(atoms, newPosition, newSymbol, threshold):
    newPosition = wrap_positions([newPosition],atoms.cell)[0]
    supAts = atoms * [3 if pbc == True else 1 for pbc in atoms.pbc]
    rs = [covalent_radii[num] for num in supAts.get_atomic_numbers()]
    rnew = covalent_radii[atomic_numbers[newSymbol]]
    # Place the new atoms in the centeral cell
    cell = atoms.get_cell()
    centerPos = newPosition + np.dot(atoms.pbc, cell)
    distArr = cdist([centerPos], supAts.get_positions(wrap=True))[0]
    for i, dist in enumerate(distArr):
        if dist/(rs[i]+rnew) < threshold:
            return False
    return True


def camel2snake(name):
    snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", name)
    return snake_case.lower().strip('_')


def snake2camel(name):
    return re.sub(r"(_[a-z])", lambda x: x.group(1)[1].upper(), name)


def check_parameters(instance, parameters, Requirement=[], Default={}):
    name = instance.__class__.__name__
    for key in Requirement:
        if key in parameters:
            setattr(instance, key, parameters[key])
        elif snake2camel(key) in parameters:
            setattr(instance, key, parameters[snake2camel(key)])
        else:
            raise Exception("'{}' must have {}".format(name, key))

    for key in Default.keys():
        if key in parameters:
            setattr(instance, key, parameters[key])
        elif snake2camel(key) in parameters:
            setattr(instance, key, parameters[snake2camel(key)])  
        else:
            setattr(instance, key, Default[key])


# def match_lattice(atoms1,atoms2):
#     """lattice matching , 10.1016/j.scib.2019.02.009
    
#     Arguments:
#         atoms1 {atoms} -- atoms1
#         atoms2 {atoms} -- atoms2
    
#     Returns:
#         atoms,atoms,float,float -- two best matched atoms in z direction
#     """
#     return atoms1, atoms2, 0.5, 0.5
#     #TODO temporary remove
#     #def match_fitness(a1,b1,a2,b2):
#     #    #za lao shi you shu zhi cuo wu
#     #    a1,b1,a2,b2 = np.round([a1,b1,a2,b2],3)
#     #    a1x = np.linalg.norm(a1)
#     #    a2x = np.linalg.norm(a2)
#     #    if a1x*a2x ==0:
#     #        return 1000
#     #    b1x = a1@b1/a1x
#     #    b2x = a2@b2/a2x
#     #    b1y = np.sqrt(b1@b1 - b1x**2)
#     #    b2y = np.sqrt(b2@b2 - b2x**2)
#     #    if b1y*b2y == 0:
#     #        return 1000
#     #    exx = (a2x-a1x)/a1x
#     #    eyy = (b2y-b1y)/b1y
#     #    exy = b2x/b1y-a2x/a1x*b1x/b1y
#     #    return np.abs(exx)+np.abs(eyy)+np.abs(exy)
#     #
#     #def to_matrix(hkl1,hkl2):
#     #    hklrange = [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
#     #    hklrange = [np.array(_) for _ in hklrange]
#     #    for hkl3 in hklrange:
#     #        M = np.array([hkl1,hkl2,hkl3])
#     #        if np.linalg.det(M)>0:
#     #            break
#     #    return M

#     #def standard_cell(atoms):
#     #    newcell = cellpar_to_cell(cell_to_cellpar(atoms.cell))
#     #    T = np.linalg.inv(atoms.cell)@newcell
#     #    atoms.positions = atoms.positions@T
#     #    atoms.cell = newcell
#     #    return atoms
#     #    
#     #cell1,cell2 = atoms1.cell[:],atoms2.cell[:]
#     #hklrange = [(1,0,0),(0,1,0),(0,0,1),(1,-1,0),(1,1,0),(1,0,-1),(1,0,1),(0,1,-1),(0,1,1),(2,0,0),(0,2,0),(0,0,2)]
#     ##TODO ba cut cell jian qie ti ji bu fen gei gai le 
#     #hklrange = [(1,0,0),(0,1,0),(0,0,1)]
#     #hklrange = [np.array(_) for _ in hklrange]
#     #minfitness = 1000
#     #for hkl1,hkl2 in itertools.permutations(hklrange,2):
#     #    for hkl3,hkl4 in itertools.permutations(hklrange,2):
#     #        a1,b1,a2,b2 = hkl1@cell1,hkl2@cell1,hkl3@cell2,hkl4@cell2
#     #        fitness = match_fitness(a1,b1,a2,b2)
#     #        if fitness<minfitness:
#     #            minfitness = fitness
#     #            bestfit = hkl1,hkl2,hkl3,hkl4
#     #newatoms1 = standard_cell(make_supercell(atoms1,to_matrix(bestfit[0],bestfit[1])))
#     #newatoms2 = standard_cell(make_supercell(atoms2,to_matrix(bestfit[2],bestfit[3])))
#     #ratio1 = newatoms1.get_volume()/atoms1.get_volume()
#     #ratio2 = newatoms2.get_volume()/atoms2.get_volume()
#     #return newatoms1,newatoms2,ratio1,ratio2


def stay_in(func):
    def wrapper(*args, **kwargs):
        currdir = os.getcwd()
        func(*args, **kwargs)
        os.chdir(currdir)
    return wrapper


def get_units_numlist(atoms, units):
    """
    get the number of each unit in the atoms
    For example:
    > atoms = Atoms('CNH7')
    > units = [Atoms('CH4'), Atoms('NH3')]
    > get_units_numlist(atoms, units)
    [1, 1]
    """
    # get all unique symbols
    symbols = set([s for a in [*units, atoms]
                     for s in a.get_chemical_symbols()])
    A = [[unit.get_chemical_symbols().count(s) for unit in units] for s in symbols]
    b = np.array([atoms.get_chemical_symbols().count(s) for s in symbols])
    numlist = np.rint(np.linalg.pinv(A) @ b).astype('int')
    # numlist is all zero or any number of symbol not match means the decompose fail
    if (numlist == 0).all() or (A @ numlist != b).any():
        return None
    else:
        return numlist


def get_units_formula(atoms, units):
    """
    get the formula composed by units
    For example:
    > atoms = Atoms('C2NH11')
    > units = [Atoms('CH4'), Atoms('NH3')]
    > get_units_formula(atoms, units)
    (CH4)2(NH3)
    """
    numlist = get_units_numlist(atoms, units)
    if numlist is None:
        return None
    formula = ''
    for n, unit in zip(numlist, units):
        f = unit.get_chemical_formula()
        if len(unit) > 1:
            f = '(' + f + ')'
        if n == 0:
            f = ''
        elif n > 1:
            f = f + str(n)
        formula += f
    return formula


def get_threshold_dict(symbols, radius=None, d_ratio=None, distance_matrix=None):
    """
    get threshold dictionary such as 
    {('Al', 'Al'): 1., ('Al', 'O'): 0.8, ('O', 'Al'): 0.8, ('O', 'O'): 0.6}
    distance is calculate by threshold_dict[(sj, si)] * (ri + rj)
    """
    threshold_dict = {}
    if radius is None:
        radius = [covalent_radii[atomic_numbers[atom]] for atom in symbols]
    for si, sj in itertools.combinations_with_replacement(symbols, 2):
        i, j = symbols.index(si), symbols.index(sj)
        ri, rj = radius[i], radius[j]
        if distance_matrix is None:
            threshold_dict[(si, sj)] = threshold_dict[(sj, si)] = d_ratio
        else:
            threshold_dict[(si, sj)] = threshold_dict[(sj, si)] = distance_matrix[i][j] / (ri + rj)
    return threshold_dict


def get_distance_dict(symbols, radius=None, d_ratio=None, distance_matrix=None):
    """
    get distance dictionary such as 
    {('Al', 'Al'): 2.42, ('Al', 'O'): 1.5, ('O', 'Al'): 1.5, ('O', 'O'): 0.79}
    """
    distance_dict = {}

    if radius is None:
        radius = [covalent_radii[atomic_numbers[atom]] for atom in symbols]
    for si, sj in itertools.combinations_with_replacement(symbols, 2):
        i, j = symbols.index(si), symbols.index(sj)
        ri, rj = radius[i], radius[j]
        if distance_matrix is None:
            distance_dict[(si, sj)] = distance_dict[(sj, si)] = d_ratio * (ri + rj)
        else:
            distance_dict[(si, sj)] = distance_dict[(sj, si)] = distance_matrix[i][j]
    return distance_dict


def get_unique_symbols(frames):
    """
    get unique symbols of given frames
    """
    if isinstance(frames, Atoms):
        frames = [frames]
    return set([s for atoms in frames for s in atoms.symbols])


def get_symbol_dict(atoms, unique_symbols=None):
    """
    return a dict of number of each symbols of an atom, such as 
    {'H': 2, 'O': 1, 'Zn': 0} for atoms=Atoms('H2O'), unique_symbols=['H', 'O', 'Zn']
    """
    symbols = atoms.get_chemical_symbols()
    unique_symbols = unique_symbols or set(symbols)
    return {s: symbols.count(s) for s in unique_symbols}


def get_gcd_formula(atoms):
    symbol_dict = get_symbol_dict(atoms)
    n_formula = reduce(gcd, symbol_dict.values())
    return Atoms([s for s in symbol_dict for _ in range(symbol_dict[s] // n_formula)]).get_chemical_formula()


def read_seeds(seed_file):
    if not os.path.exists(seed_file):
        return []
    if 'traj' in seed_file:
        seedPop = read(seed_file, index=':', format='traj')
    elif 'POSCARS' in seed_file:
        seedPop = read(seed_file, index=':', format='vasp-xdatcar')
    else:
        try:
            seedPop = read(seed_file, index=':')
        except:
            raise Exception("unknown file format: {}".format(seed_file))
    for ind in seedPop:
        ind.info['origin'] = 'seed'
    return seedPop


def find_factor(num):
    i = 2
    while i < np.sqrt(num):
        if num % i == 0:
            break
        i += 1
    if num % i > 0:
        i = num
    return i


def multiply_cell(atoms, n):
    """
    return a structure with atoms n times of the input
    """
    atoms = atoms.copy()
    assert n >= 1 and n % 1 == 0, "n must be an integer >= 1"
    while n > 1:
        i = find_factor(n)
        to_expand = np.argmin(atoms.cell.cellpar()[:3])
        expand_matrix = [1, 1, 1]
        expand_matrix[to_expand] = i
        atoms = atoms * expand_matrix
        n = n // i
    atoms = atoms[atoms.numbers.argsort()]
    return atoms


class Plugin:
    def __init__(self, name):
        self.name = name
        self.plugins = {}

    def keys(self):
        return self.plugins.keys()

    def register(self, plugin_name):
        def wrapper(plugin):
            self.plugins.update({plugin_name: plugin})
            return plugin
        return wrapper

    def __repr__(self):
        ret = self.name
        ret += "\n-------------------"
        for key, value in self.plugins.items():
            ret += "\n{}: {}".format(key.ljust(15, ' '), value.__name__)
        ret += "\n-------------------\n"
        return ret

    def __contains__(self, key):
        return key in self.plugins

    def __getitem__(self, key):
        if key in self.plugins:
            return self.plugins[key]
        raise NotImplementedError("{} has not been registered in {}".format(key, self.name))


def load_plugins(path, PACKAGE_BASE, PACKAGE_LOAD="all", NOT_LOADABLE=None, verbose=False):
    if NOT_LOADABLE is None:
        NOT_LOADABLE = ("__init__.py",)
    if PACKAGE_LOAD == "all":
        PACKAGE_LOAD = Path(path).parent.glob("*.py")
    for module_file in PACKAGE_LOAD:
        if module_file.name not in NOT_LOADABLE:
            module_name = f".{module_file.stem}"
            try:
                import_module(module_name, PACKAGE_BASE)
            except ImportError:
                if verbose:
                    print("Fail when try to import {}{}, because:\n{}".format(PACKAGE_BASE, module_name, traceback.format_exc(1)))


COMPARATOR_PLUGIN = Plugin('comparator')
COMPARATOR_CONNECT_PLUGIN = Plugin('comparator_connect')
FINGERPRINT_PLUGIN = Plugin('fingerprint')
CALCULATOR_PLUGIN = Plugin('calculator')
CALCULATOR_CONNECT_PLUGIN = Plugin('calculator_connect')
