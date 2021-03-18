import ase.io
import numpy as np

atoms = ase.io.read('Ref/layerslices.traj', index = ':')
bondlength = 1.54746
newatoms = atoms[0].copy()
pos = newatoms.get_scaled_positions()
minpos = np.min(pos[:,2])
index = [i for i, p in enumerate(pos) if (p[2] - minpos)>0.05]
del newatoms[index]
c = newatoms.get_cell_lengths_and_angles()[2]
newatoms.translate([-bondlength/c * newatoms.get_cell()[2]]*len(newatoms))
for at in newatoms:
    at.symbol = 'H'
atoms[0] += newatoms

ase.io.write('Ref/layerslices.traj', atoms, format = 'traj')


