# TODO
# add phasediagram.py for both this and fitness calculator
# change magusphasediagram so that we don't need to convert for pseudo search
import os, re
from pathlib import Path
from ase.atoms import default
from math import gcd
from functools import reduce
from matplotlib import pyplot as plt
import pandas as pd
from ase.io import iread, write
from ase import Atoms
import numpy as np
import spglib as spg
from magus.utils import MagusPhaseDiagram
try:
    from pymatgen import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
except:
    pass
from magus.utils import get_units_numlist, get_units_formula


pd.options.display.max_rows = 100


def convert_glob(filenames):
    p = Path('.')
    consider_glob = []
    for f in filenames:
        consider_glob.extend(map(str, p.glob(f)))
    return consider_glob


def get_frames(filenames):
    for filename in filenames:
        frames = iread(filename, format='traj', index=':')
        for atoms in frames:
            atoms.info['source'] = filename.split('.')[0]
            yield atoms


# ugly version for temp use
def get_units(frames):
    """
    get units of given frames
    """
    # get all symbols and numlist of the symbols of all the structures
    symbols = set([s for atoms in frames
                   for s in atoms.get_chemical_symbols()])
    formula = np.array([
        [atoms.get_chemical_symbols().count(s) for s in symbols]
        for atoms in frames], dtype=float)
    # reduce the formula matrix to row echelon form
    lead = 0
    n_row, n_column = formula.shape
    for r in range(n_row):
        i = r
        # find the first non-zero column
        while lead < n_column and formula[i][lead] == 0.:
            i += 1
            if i == n_row:
                i = r
                lead += 1
        if lead >= n_column:
            break
        # swap two rows like [0, 0, 1] and [0, 1, 1]
        formula[[i, r], :] = formula[[r, i], :]
        formula[r] = formula[r] / formula[r][lead]
        for i in range(n_row):
            if i != r:
                formula[i] = formula[i] - formula[r] * formula[i][lead]
        # avoid numerical fault
        formula[np.where(np.abs(formula) < 1e-7)] = 0.
        lead += 1
    # avoid negative number
    for r in range(n_row):
        for lead in range(n_column):
            if formula[r][lead] < 0.:
                for i in range(n_row - 1, -1, -1):
                    if formula[i][lead] != 0 and i != r:
                        break
                formula[r] = formula[r] - formula[i] * formula[r][lead] / formula[i][lead]       
    units = []
    for f in formula:
        if np.any(f):
            n = 1 
            while np.any(n * f % 1):
                n += 1
            f = (f * n).astype('int')
            units.append(Atoms(symbols=[s for n, s in zip(f, symbols) for _ in range(n)]))
    return units


def get_decompose_name(atoms, units):
    """
    convert atoms to units format for pseudo binary search
    ZnOH -> {'Zn': 1, '(OH)': 1}
    """
    units_numlist = get_units_numlist(atoms, units)
    # int(n) is necessary, or else will raise ValueError in ase
    f = lambda u: u.get_chemical_formula() if len(u) == 1 else '({})'.format(u.get_chemical_formula())
    return {f(u): int(n) for u, n in zip(units, units_numlist)}


class Summary:
    show_features = ['symmetry', 'enthalpy', 'formula', 'priSym']

    def __init__(self, prec=0.1, remove_features=[], add_features=[], formula_type='fix'):
        self.formula_type = formula_type
        if self.formula_type == 'fix':
            self.default_sort = 'enthalpy'
        elif self.formula_type == 'var':
            self.default_sort = 'ehull'
            self.show_features.append('ehull')

        show_features = [feature for feature in self.show_features if feature not in remove_features]
        show_features.extend(add_features)
        self.show_features = show_features
        self.prec = prec

    def set_features(self, atoms):
        atoms.info['cellpar'] = np.round(atoms.cell.cellpar(), 2).tolist()
        atoms.info['lengths'] = atoms.info['cellpar'][:3]
        atoms.info['angles'] = atoms.info['cellpar'][3:]
        atoms.info['volume'] = round(atoms.get_volume(), 3)
        atoms.info['fullSym'] = atoms.get_chemical_formula()
        if self.formula_type == 'var':
            atoms.info['units'] = self.units
            name = get_decompose_name(atoms, self.units)
            ref_e = self.phase_diagram.decompose(**name)[0]
            # convert enthalpy for pseudo binary search, must be careful!
            ehull = atoms.info['enthalpy'] - ref_e / sum(name.values())
            atoms.info['ehull'] = 0 if ehull < 1e-3 else ehull
        if 'units' not in atoms.info:
            atoms.info['units'] = [Atoms(s) for s in list(set(atoms.get_chemical_symbols()))]
        atoms.info['formula'] = get_units_formula(atoms, atoms.info['units'])
        
    def summary(self, filenames, show_number=20, need_sorted=True, sorted_by='Default', reverse=True, save=False, outdir=None):
        filenames = convert_glob(filenames)
        self.prepare_data(filenames)
        show_number = min(len(self.all_frames), show_number)
        self.show_features_table(show_number, reverse, need_sorted, sorted_by)
        if save:
            self.save_atoms(show_number, outdir)
        if self.formula_type == 'var':
            self.plot_phase_diagram()

    def prepare_data(self, filenames):
        self.rows, self.all_frames = [], []
        if len(filenames) > 1 and 'source' not in self.show_features:
            self.show_features.append('source')
        for atoms in get_frames(filenames):
            self.all_frames.append(atoms)
        if self.formula_type == 'var':
            # for var, we may need recalculate units and ehulls
            self.units = get_units(self.all_frames)
            self.phase_diagram = self.get_phase_diagram()
        for atoms in get_frames(filenames):
            self.set_features(atoms)
            self.rows.append([atoms.info[feature] if feature in atoms.info.keys() else None
                                                  for feature in self.show_features])

    def show_features_table(self, show_number=20, reverse=True, need_sorted=True, sorted_by='Default'):
        df = pd.DataFrame(self.rows, columns=self.show_features)
        if need_sorted:
            if sorted_by == 'Default' or sorted_by not in self.show_features:
                sorted_by = self.default_sort
            df = df.sort_values(by=[sorted_by,])
            self.all_frames = [self.all_frames[i] for i in df.index]
        df.index = range(1, len(df) + 1)
        if reverse:
            print(df[:-show_number:-1])
        else:
            print(df[:show_number])

    def save_atoms(self, show_number, outdir):
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        for i in range(show_number):
            posname = os.path.join(outdir, "POSCAR_{}.vasp".format(i + 1))
            write(posname, self.all_frames[i], direct = True, vasp5 = True)

    def get_phase_diagram(self):
        units = self.units
        # Make sure there are values at the vertices, may raise wrong results
        refs = [(get_decompose_name(atoms, units), 1000 * len(atoms)) for atoms in units]
        for atoms in self.all_frames:
            units_numlist = get_units_numlist(atoms, units)
            name = get_decompose_name(atoms, units)
            # convert enthalpy for pseudo binary search, must be careful!
            enthalpy = atoms.info['enthalpy'] * sum(units_numlist)
            refs.append((name, enthalpy))
        return MagusPhaseDiagram(refs, verbose=False)

    def plot_phase_diagram(self):
        ax = self.phase_diagram.plot()
        plt.savefig('PhaseDiagram.png')


class BulkSummary(Summary):
    def set_features(self, atoms):
        super().set_features(atoms)
        atoms.info['symmetry'] = spg.get_spacegroup(atoms, self.prec)
        # sometimes spglib cannot find primitive cell.
        try:
            lattice, scaled_positions, numbers = spg.find_primitive(atoms, self.prec)
            pri_atoms = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers)
        except:
            # if fail to find prim, set prim to raw
            print("Fail to find primitive for structure")
            pri_atoms = atoms
        finally:
            atoms.info['priSym'] = pri_atoms.get_chemical_formula()


class ClusterSummary(Summary):
    show_features = ['symmetry', 'enthalpy', 'formula', 'Eo', 'energy']
    def set_features(self, atoms):
        super().set_features(atoms)
        molecule = Molecule(atoms.symbols,atoms.get_positions())
        atoms.info['symmetry'] = PointGroupAnalyzer(molecule, self.prec).sch_symbol

def summary(*args, filenames=[], prec=0.1, remove_features=[], add_features=[], 
            need_sorted=True, sorted_by='Defalut', reverse=False,
            show_number=20, save=False, outdir='.', var=False, atoms_type='bulk',
            **kwargs):
    formula_type = 'var' if var else 'fix'
    summary_dict = {
        'bulk': BulkSummary,
        'cluster': ClusterSummary
        }
    s = summary_dict[atoms_type](prec=prec, 
                                 remove_features=remove_features, add_features=add_features, 
                                 formula_type=formula_type)
    s.summary(filenames, show_number, need_sorted, sorted_by, reverse, save, outdir)
