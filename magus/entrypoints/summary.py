import os, sys
import pandas as pd
from ase.io import read, write
from ase import Atoms
import numpy as np
import spglib as spg


pd.options.display.max_rows = 50

def summary(*args, filenames=[], prec=0.1, add_features=[], sorted_by='enthalpy',
            save=False, outdir='.', reverse=False, show_number=20,
            **kwargs):
    all_frames = []
    for filename in filenames:
        frames = read(filename, format='traj', index=':')
        for atoms in frames:
            atoms.info['source'] = filename.split('.')[0]
        all_frames.extend(frames)
    show_features = ['symmetry', 'enthalpy', 'parentE', 'origin', 'fullSym', 'priSym']
    show_features.extend(add_features)
    if len(filenames) > 1 and 'source' not in show_features:
        show_features.append('source')
    try:
        key_index = show_features.index(sorted_by)
    except:
        print('{} not in show features, auto choose enthalpy as sort feature'.format(sorted_by))
        key_index = show_features.index('enthalpy')
    for i, atoms in enumerate(all_frames):  
        atoms.info['symmetry'] = spg.get_spacegroup(atoms, prec)
        atoms.info['cellpar'] = np.round(atoms.cell.cellpar(), 2).tolist()
        atoms.info['lengths'] = atoms.info['cellpar'][:3]
        atoms.info['angles'] = atoms.info['cellpar'][3:]
        lattice, scaled_positions, numbers = spg.find_primitive(atoms, prec)
        pri_atoms = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers)
        atoms.info['priSym'] = pri_atoms.get_chemical_formula()
        volume = atoms.get_volume()
        atoms.info['volume'] = round(volume, 3)
        atoms.info['fullSym'] = atoms.get_chemical_formula()
        atoms.info['row'] = [atoms.info[feature] if feature in atoms.info.keys() else None \
                             for feature in show_features]
    all_frames.sort(key=lambda atoms:atoms.info['row'][key_index] or 100)
    if save:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for i, atoms in eunmerate(all_frames):
            posname = os.path.join(outdir, "POSCAR_{}.vasp".format(i+1))
            write(posname, atoms, direct = True, vasp5 = True)
    alldata = [atoms.info['row'] for atoms in all_frames[:show_number]]
    df = pd.DataFrame(alldata, columns=show_features)
    df.index += 1
    if reverse:
        print(df[::-1])
    else:
        print(df)
