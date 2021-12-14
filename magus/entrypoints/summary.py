import os, sys
import pandas as pd
from ase.io import read, write
from ase import Atoms
import numpy as np
import spglib as spg
try:
    from pymatgen import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
except:
    pass

pd.options.display.max_rows = 100

def summary(*args, filenames=[], prec=0.1, remove_features=[], add_features=[], sorted_by='enthalpy',
            save=False, outdir='.', reverse=False, show_number=20, cluster = False,
            **kwargs):
    all_frames = []
    for filename in filenames:
        frames = read(filename, format='traj', index=':')
        for atoms in frames:
            atoms.info['source'] = filename.split('.')[0]
        all_frames.extend(frames)
    show_features = [feature for feature in ['symmetry', 'enthalpy', 'parentE', 'origin', 'fullSym', 'priSym', 'Eo', 'energy']\
            if feature not in remove_features]
    show_features.extend(add_features)
    if len(filenames) > 1 and 'source' not in show_features:
        show_features.append('source')
    if sorted_by.lower() in ['', 'none']:
        sorted_by = None
    else:
        try:
            key_index = show_features.index(sorted_by)
        except:
            print('{} not in show features, auto choose enthalpy as sort feature'.format(sorted_by))
            key_index = show_features.index('enthalpy')
    for i, atoms in enumerate(all_frames):  
        if cluster:
            molecule = Molecule(atoms.symbols,atoms.get_positions())
            atoms.info['symmetry'] = PointGroupAnalyzer(molecule, prec).sch_symbol
        else:
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
        #Si, C = float(atoms.info['fullSym'][-2:]) - 64, float(atoms.info['fullSym'][1:3]) - 64
        #si= -43.301929/8 
        #c = -60.267208/4 - si
        c = -72.773010 / 8
        si = -60.267208/4 - c
        #atoms.info['Eo'] = atoms.info['energy'] + 969.675849 - Si * si - C * c
        #print("{}\t{}\t{}".format(atoms.info['energy'], Si, C))
        atoms.info['row'] = [atoms.info[feature] if feature in atoms.info.keys() else None \
                             for feature in show_features]
    
    alldata = [atoms.info['row'] for atoms in all_frames]
    df = pd.DataFrame(alldata, columns=show_features)
    if sorted_by is not None:
        df = df.sort_values(by=[show_features[key_index]])

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
        
