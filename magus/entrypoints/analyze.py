import sys
import ase.io
import numpy as np
import os
import spglib
import pandas as pd

def analyze(filename, to_excel = None, *args, **kwargs):
    print('Energy by generation (raw):\n')
    df_gen = {'mean':[], 'min':[], 'max':[]}
    i = 0
    for i in range(1,1000):
        if os.path.exists('{}/raw{}.traj'.format(filename,i)):
            pop = ase.io.read('{}/raw{}.traj'.format(filename,i), index = ':')
            energy = np.array([ind.info['energy'] for ind in pop])
            creater = [ind.info['origin'] for ind in pop]
            deltae = [(ind.info['enthalpy'] - ind.info['parentE']) if not 'rand' in ind.info['origin'] else np.nan for ind in pop]
            subdf = pd.DataFrame({'energy': energy, 'creater': creater, 'deltae':deltae})
            e = subdf.groupby('creater').mean()

            for origin in set(creater):
                shownum = e['energy'][origin] if 'rand' in origin else e['deltae'][origin]
                origin = origin[:6].lower() if not 'rand' in origin else origin[5:]
                if origin in df_gen:
                    df_gen[origin].append(round(shownum, 3))
                else:
                    df_gen[origin] = [np.nan for _ in range(i)]
                    df_gen[origin][-1] = round(shownum, 3)

            e = {'mean':np.mean(energy), 'min':np.min(energy), 'max':np.max(energy)}
            for key in e:
                df_gen[key].append(round(e[key], 3))
            for key in df_gen:
                #print("{}, {}, {}".format(i, key, len(df_gen[key])))
                if not len(df_gen[key]) == i:
                    df_gen[key].append(np.nan)
        else:
            break
    df = pd.DataFrame(df_gen, index = list(range(1, i)))
    print(df)

    if not to_excel is None:
        if '.xlsx' not in to_excel:
            to_excel += '.xlsx'
        df.to_excel(to_excel, sheet_name="Sheet1")

    best = ase.io.read('{}/good.traj'.format(filename),index = ':')
    energy = np.array([ind.info['energy'] for ind in best])
    minE = np.min(energy)
    index = np.where(minE ==energy)[0][0]
    print('\nAllBest E = {}, origin = {}, symmetry = {}'.format(minE, best[index].info['origin'], spglib.get_spacegroup(best[index], 0.2)))
    print('\n============================================================\n')
    best = ase.io.read('{}/best.traj'.format(filename),index = ':')
    energy = np.array([round(ind.info['energy'], 3) for ind in best])
    print('Best ind: \n')
    df_best = pd.DataFrame({'generation':[], 'energy':[], 'origin':[], 'symmetry':[], 'fullsym':[]})
    for i, e in enumerate(energy):
        if i == 0 or e < energy[i-1]:
            subdf = pd.DataFrame({'generation': [str(i+1)], 'energy': [e], 'origin': [best[i].info['origin']], 'symmetry':[spglib.get_spacegroup(best[i], 0.2)], 'fullsym':[best[i].get_chemical_formula()]})
            df_best = df_best.append(subdf)
    print(df_best)
        
            

