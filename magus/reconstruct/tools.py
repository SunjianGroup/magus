import ase.io
import logging
from ..parameters import magusParameters


def getslab(filename = 'Ref/layerslices.traj', slabfile = 'slab.vasp', *args, **kwargs):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')

    m = magusParameters('input.yaml')
    
    pop = ase.io.read(filename, index = ':', format = 'traj')
    ind, rcs = (m.Population).Ind, None
    #rcs-magus
    if len(pop) == 3:
        rcs = ind(pop[2])
        rcs.buffer = True
        rcs.bulk_layer, rcs.buffer_layer = pop[0], pop[1]
        
    #ads-magus
    elif len(pop) == 2:
        rcs = ind(pop[1])
        rcs.buffer = False
        rcs.bulk_layer = pop[0]

    if not slabfile is None:
        ase.io.write(slabfile, rcs.for_calculate(), format = 'vasp',vasp5=True,direct = True)
    else:
        return rcs.for_calculate()

import sys
import ase.io
import numpy as np
import os
import spglib
import pandas as pd
import matplotlib.pyplot as plt


def analyze(filename, to_excel = None, to_plt = None, fit_key = 'enthalpy', *args, **kwargs):
    print('Energy statistics by generation (raw):\n')
    df_gen = {'Mean':[], 'Min':[], 'Max':[]}
    i = 0
    for i in range(1,1000):
        if os.path.exists('{}/raw{}.traj'.format(filename,i)):
            pop = ase.io.read('{}/raw{}.traj'.format(filename,i), index = ':')
            energy = np.array([ind.info['energy'] for ind in pop])
            fullsym =  [ind.get_chemical_formula() for ind in pop] 
            fitness = [ind.info[fit_key] for ind in pop]
            creater = [ind.info['origin'] for ind in pop]
            deltaE = [(ind.info['enthalpy'] - ind.info['parentE']) if not ('rand' in ind.info['origin'] or 'seed' in ind.info['origin']) else np.nan for ind in pop]
            kwargs = {'energy': energy, 'creater': creater,'deltaE':deltaE, fit_key:fitness, 'fullsym': fullsym}
            
            subdf = pd.DataFrame(kwargs)

            e = subdf.groupby('creater').mean(numeric_only=True)

            for origin in set(creater):
                shownum = e[fit_key][origin] if 'rand' in origin else e['deltaE'][origin]
                if origin == 'random':
                    pass
                elif 'rand' in origin and '.' in origin:
                    origin = origin[5:]
                else:
                    origin = origin.replace('Mutation', '').replace('Pairing', '').lower() 
                if origin in df_gen:
                    df_gen[origin].append(round(shownum, 3))
                else:
                    df_gen[origin] = [np.nan for _ in range(i)]
                    df_gen[origin][-1] = round(shownum, 3)
            
            e = {'Mean':np.mean(subdf[fit_key]), 'Min':np.min(subdf[fit_key]), 'Max':np.max(subdf[fit_key])}
        
            elekeys = list(set(fullsym)) 
            for ele in elekeys:
                e[ele] = subdf.groupby('fullsym').get_group(ele).min()[fit_key]
                if ele not in df_gen.keys():
                    df_gen[ele] = []
            
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

    if not to_plt is None:
        if '.svg' not in to_plt:
            to_plt += '.svg'
        from matplotlib.font_manager import FontProperties

        font = FontProperties()
        font.set_name('Times New Roman')
        
        #figure 2
        
        labels = [str(s) for s in df.columns if s not in ['Mean', 'Min', 'Max', 'random'] + elekeys]
        x = np.linspace(1, np.max(df.index), np.max(df.index))

        for label in labels:
            plt.plot(x, df[label].values,label=label)  # Plot some data on the (implicit) axes.
        

        plt.xlabel('generation',fontsize = 14)
        plt.ylabel('delta E',fontsize = 14)
        plt.title("delta E of heredity operators", fontsize = 14)
        plt.legend(fontsize = 14)
        plt.savefig(to_plt)


    best = ase.io.read('{}/good.traj'.format(filename),index = ':')
    energy = np.array([ind.info['energy'] for ind in best])
    minE = np.min(energy)
    index = np.where(minE ==energy)[0][0]
    print('============================================================')
    print('AllBest E = {}, origin = {}, symmetry = {}'.format(np.round(minE,6), best[index].info['origin'], spglib.get_spacegroup(best[index], 0.2)))
    print('============================================================')

from magus.reconstruct.generator import SurfaceGenerator
def mine_substrate(filename = 'Ref/layerslices.traj', *args, **kwargs):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')
    substrate = ase.io.read(filename, index = -1, format = 'traj')
    SurfaceGenerator.mine_substrate_spg(substrate)
