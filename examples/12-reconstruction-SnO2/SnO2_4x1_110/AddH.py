import ase.io
import numpy as np

atoms = ase.io.read('Ref/layerslices.traj', index = ':')
posrange = [[0.5, 0.6], [0.75,0.85]]
bondlength = [2.586,4.113]
for i, r in enumerate(posrange): 
    newatoms = atoms[0].copy()
    pos = newatoms.get_scaled_positions()
    index = [i for i, p in enumerate(pos) if (p[2]< r[0] or p[2]> r[1]) or newatoms[i].symbol == 'H']
    del newatoms[index]
    c = newatoms.get_cell_lengths_and_angles()[2]
    bl = bondlength[i]
    newatoms.translate([-bl/c * newatoms.get_cell()[2]]*len(newatoms))
    for at in newatoms:
        at.symbol = 'H'
    atoms[0] += newatoms

ase.io.write('layerslices.traj', atoms, format = 'traj')


