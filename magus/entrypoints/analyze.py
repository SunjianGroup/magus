import sys
import ase.io
import numpy as np
import os
import spglib

def analyze(filename, *args, **kwargs):
    for i in range(1,100):
        if os.path.exists('{}/raw{}.traj'.format(filename,i)):
            pop = ase.io.read('{}/raw{}.traj'.format(filename,i), index = ':')
            energy = np.array([ind.info['energy'] for ind in pop])
            print('Gen {}, meanE = {}, minE = {}, maxE = {}'.format(i, np.mean(energy), np.min(energy), np.max(energy)))
        else:
            break
    best = ase.io.read('{}/good.traj'.format(filename),index = ':')
    energy = np.array([ind.info['energy'] for ind in best])
    minE = np.min(energy)
    index = np.where(minE ==energy)[0][0]
    print('\nAllBest E = {}, origin = {}, symmetry = {}'.format(minE, best[index].info['origin'], spglib.get_spacegroup(best[index], 0.2)))

    best = ase.io.read('{}/best.traj'.format(filename),index = ':')
    energy = np.array([ind.info['energy'] for ind in best])
    for i, e in enumerate(energy):
        if i == 0 :
            print('Best ind: \ngen 1, E = {}, origin = {}, symmetry = {}, fullsym = {}'.format(e, best[i].info['origin'], spglib.get_spacegroup(best[i], 0.2), best[i].get_chemical_formula()))
        else:
            if e < energy[i-1]:
                print('gen {}, E = {}, origin = {}, symmetry = {}, fullsym = {}'.format(i+1, e, best[i].info['origin'], spglib.get_spacegroup(best[i], 0.2), best[i].get_chemical_formula()))
