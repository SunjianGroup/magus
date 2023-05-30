import os, re
from pathlib import Path
from ase.atoms import default
from math import gcd
from functools import reduce
from matplotlib import pyplot as plt
import pandas as pd
from ase.io import iread, write, read
from ase import Atoms
import numpy as np
import spglib as spg
from magus.phasediagram import PhaseDiagram, get_units
try:
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
except:
    pass
from magus.utils import get_units_formula


pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
# pd.set_option('max_colwidth', 30)
# pd.set_option('width', 120)
        

def expand_path(path_pattern):
    p = Path(path_pattern).expanduser()
    parts = p.parts[p.is_absolute():]
    return Path(p.root).glob(str(Path(*parts)))


def convert_glob(filenames):
    """
    to support path including asterisk wildcard such as */results/good.traj or **/good.traj
    """
    p = Path('.')
    consider_glob = []
    for f in filenames:
        consider_glob.extend(map(str, expand_path(f)))
    return consider_glob


def get_frames(filenames):
    for filename in filenames:
        try:
            frames = read(filename, index=':')
        except:
            print('Fail to read {}'.format(filename))
            continue
        for atoms in frames:
            atoms.info['source'] = filename.split('.')[0]
            yield atoms


class Summary:
    show_features = ['symmetry', 'enthalpy', 'formula', 'priFormula']

    def __init__(self, prec=0.1, remove_features=[], add_features=[], formula_type='fix', boundary=[]):
        self.formula_type = formula_type
        if self.formula_type == 'fix':
            self.default_sort = ['enthalpy']
        elif self.formula_type == 'var':
            self.default_sort = ['ehull', 'enthalpy']
            self.show_features.append('ehull')
            self.boundary = [Atoms(formula) for formula in boundary]

        show_features = [feature for feature in self.show_features if feature not in remove_features]
        show_features.extend(add_features)
        self.show_features = show_features
        self.prec = prec

    def set_features(self, atoms):
        atoms.info['cellpar'] = np.round(atoms.cell.cellpar(), 2).tolist()
        atoms.info['lengths'] = atoms.info['cellpar'][:3]
        atoms.info['angles'] = atoms.info['cellpar'][3:]
        atoms.info['volume'] = round(atoms.get_volume(), 3)
        atoms.info['fullSym'] = atoms.get_chemical_formula(empirical=True)
        if self.formula_type == 'var':
            ehull = atoms.info['enthalpy'] - self.phase_diagram.decompose(atoms)
            atoms.info['ehull'] = 0 if ehull < 1e-3 else ehull
        if 'units' not in atoms.info:
            atoms.info['units'] = [Atoms(s) for s in list(set(atoms.get_chemical_symbols()))]
        if hasattr(self, 'units'):
            atoms.info['formula'] = get_units_formula(atoms, self.units)
        else:
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
            if len(self.boundary) == 0:
                self.units = get_units(self.all_frames)
                assert self.units is not None, "Fail to find units, please assign units by '-u'"
            else:
                self.units = self.boundary
            self.phase_diagram = self.get_phase_diagram()
        for atoms in get_frames(filenames):
            self.set_features(atoms)
            self.rows.append([atoms.info[feature] if feature in atoms.info.keys() else None
                                                  for feature in self.show_features])

    def show_features_table(self, show_number=20, reverse=True, need_sorted=True, sorted_by='Default'):
        df = pd.DataFrame(self.rows, columns=self.show_features)
        if need_sorted and sorted_by != ['None']:
            if sorted_by == 'Default':
                sorted_by = self.default_sort
            df = df.sort_values(by=sorted_by)
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
        pd = PhaseDiagram(self.all_frames, boundary=self.units)
        for unit in self.units:
            a = unit.copy()
            a.info['enthalpy'] = 1000
            pd.append(a)
        return pd

    def plot_phase_diagram(self):
        ax = self.phase_diagram.plot()
        plt.savefig('PhaseDiagram.png')


class BulkSummary(Summary):
    def set_features(self, atoms):
        super().set_features(atoms)
        atoms.info['symmetry'] = spg.get_spacegroup(atoms, self.prec)
        # sometimes spglib cannot find primitive cell.
        try:
            lattice, scaled_positions, numbers = spg.find_primitive(atoms, symprec=self.prec)
            pri_atoms = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers)
            lattice, scaled_positions, numbers = spg.standardize_cell(atoms, symprec=self.prec)
            std_atoms = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers)
        except:
            # if fail to find prim, set prim to raw
            print("Fail to find primitive for structure")
            pri_atoms = atoms
            std_atoms = atoms
        finally:
            atoms.info['priFormula'] = pri_atoms.get_chemical_formula()
            if hasattr(self, 'units'):
                atoms.info['priFormula'] = get_units_formula(pri_atoms, self.units)
                atoms.info['stdFormula'] = get_units_formula(std_atoms, self.units)
            else:
                atoms.info['priFormula'] = get_units_formula(pri_atoms, atoms.info['units'])
                atoms.info['stdFormula'] = get_units_formula(std_atoms, atoms.info['units'])


class ClusterSummary(Summary):
    show_features = ['symmetry', 'enthalpy', 'formula', 'Eo', 'energy']
    def set_features(self, atoms):
        super().set_features(atoms)
        molecule = Molecule(atoms.symbols,atoms.get_positions())
        atoms.info['symmetry'] = PointGroupAnalyzer(molecule, self.prec).sch_symbol

def summary(*args, filenames=[], prec=0.1, remove_features=[], add_features=[], 
            need_sorted=True, sorted_by='Defalut', reverse=False, boundary=[],
            show_number=20, save=False, outdir='.', var=False, atoms_type='bulk',
            **kwargs):
    formula_type = 'var' if var else 'fix'
    summary_dict = {
        'bulk': BulkSummary,
        'cluster': ClusterSummary
        }
    s = summary_dict[atoms_type](prec=prec, 
                                 remove_features=remove_features, add_features=add_features, 
                                 formula_type=formula_type, boundary=boundary)
    s.summary(filenames, show_number, need_sorted, sorted_by, reverse, save, outdir)
