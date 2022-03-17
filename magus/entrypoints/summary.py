import os, re
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


def get_frames(filenames):
    for filename in filenames:
        frames = iread(filename, format='traj', index=':')
        for atoms in frames:
            atoms.info['source'] = filename.split('.')[0]
            yield atoms


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
        atoms.info['formula'] = get_units_formula(atoms, atoms.info['units'])

    def summary(self, filenames, show_number=20, need_sorted=True, sorted_by='Default', reverse=True, save=False, outdir=None):
        self.prepare_data(filenames)
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
            self.set_features(atoms)
            self.all_frames.append(atoms)
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

    def plot_phase_diagram(self):
        def get_reduce_formula(atoms):
            numlist = np.array(get_units_numlist(atoms, units))
            numlist = numlist / reduce(gcd, numlist)
            return tuple(numlist.astype(int))
        # Make sure there are values at the vertices, may raise wrong results
        units = self.all_frames[0].info['units']
        for u in units:
            u.info['enthalpy'] = 100 / len(u)
        refs_dict = {get_reduce_formula(u): u for u in units}
        # remove the same formula, only remain the lower enthalpy ones for 2d3
        for atoms in self.all_frames:
            name = get_reduce_formula(atoms)
            if name not in refs_dict:
                refs_dict[name] = atoms
            if atoms.info['enthalpy'] < refs_dict[name].info['enthalpy']:
                refs_dict[name] = atoms
        refs = []
        # plot all for 2 units
        to_plot = refs_dict.values() if len(units) > 2 else self.all_frames
        for atoms in to_plot:
            units_numlist = get_units_numlist(atoms, units)
            # int(n) is necessary, or else will raise ValueError in ase
            f = lambda u: u.get_chemical_formula() if len(u) == 1 else '({})'.format(u.get_chemical_formula())
            name = {f(u): int(n) for u, n in zip(units, units_numlist)}
            base_enthalpy = sum([refs_dict[get_reduce_formula(u)].info['enthalpy'] * len(u) * n 
                                 for u, n in zip(units, units_numlist)])
            enthalpy = atoms.info['enthalpy'] * len(atoms) - base_enthalpy
            enthalpy = enthalpy / len(atoms) * sum(units_numlist)
            refs.append((name, enthalpy))
        ax = MagusPhaseDiagram(refs, verbose=False).plot()
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
