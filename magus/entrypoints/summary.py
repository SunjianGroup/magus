import os, sys
import pandas as pd
from ase.io import read, write
from ase import Atoms
import numpy as np
import spglib as spg
import itertools as it
try:
    from pymatgen import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
except:
    pass

pd.options.display.max_rows = 100

def summary(*args, filenames=[], prec=0.1, remove_features=[], add_features=[], sorted_by='enthalpy',
            save=False, outdir='.', reverse=False, show_number=20, cluster = False,
            **kwargs):

    show_features = [feature for feature in ['symmetry', 'enthalpy', 'parentE', 'origin', 'fullSym', 'priSym', 'Eo', 'energy']\
            if feature not in remove_features]
    show_features.extend(add_features)
    if len(filenames) > 1 and 'source' not in show_features:
        show_features.append('source')
    if sorted_by.lower() in ['', 'none']:
        sorted_by = None
    if sorted_by not in show_features:
        print('{} not in show features, auto choose enthalpy as sort feature'.format(sorted_by))
        sorted_by = "enthalpy"

    def get_frames(filenames):
        for filename in filenames:
            frames = read(filename, format='traj', index=':')
            for atoms in frames:
                atoms.info['source'] = filename.split('.')[0]
        yield frames

    def set_features(all_frames):
        for i, atoms in enumerate(all_frames):  
            if cluster:
                molecule = Molecule(atoms.symbols,atoms.get_positions())
                atoms.info['symmetry'] = PointGroupAnalyzer(molecule, prec).sch_symbol
            else:
                atoms.info['symmetry'] = spg.get_spacegroup(atoms, prec)
            atoms.info['cellpar'] = np.round(atoms.cell.cellpar(), 2).tolist()
            atoms.info['lengths'] = atoms.info['cellpar'][:3]
            atoms.info['angles'] = atoms.info['cellpar'][3:]
            atoms.info['volume'] = round(atoms.get_volume(), 3)
            atoms.info['fullSym'] = atoms.get_chemical_formula()

            # sometimes spglib cannot find primitive cell.
            try:
                lattice, scaled_positions, numbers = spg.find_primitive(atoms, prec)
                pri_atoms = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers)
            except:
                # if fail to find prim, set prim to raw
                print("Fail to find primitive for structure {}".format(i))
                pri_atoms = atoms
            finally:
                atoms.info['priSym'] = pri_atoms.get_chemical_formula()

            atoms.info['row'] = [atoms.info[feature] if feature in atoms.info.keys() else None \
                                for feature in show_features]
        yield atoms.info['row']

    all_frames = it.chain(get_frames(filenames))    
    alldata = set_features(all_frames)
    df = pd.DataFrame(alldata, columns=show_features)
    
    if sorted_by is not None:
        df = df.sort_values(by=[sorted_by,])

    if save:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for i, index in enumerate(df.index):
            posname = os.path.join(outdir, "POSCAR_{}.vasp".format(i+1))
            write(posname, all_frames[index], direct = True, vasp5 = True)

    df.index = range(1, len(df)+1)
    if reverse:
        print(df[:-show_number:-1])
    else:
        print(df[:show_number])
