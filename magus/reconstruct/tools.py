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

import ase.io
import numpy as np
import os
import spglib
import pandas as pd
import matplotlib.pyplot as plt

def shown_name(opertion_name):
    if 'rand' in opertion_name:
        return opertion_name
    #elif 'rand' in opertion_name and '.' in opertion_name:
    #    return opertion_name[5:]
    else:
        return opertion_name.replace('Mutation', '').replace('Pairing', '').lower() 

def analyze(filename, to_excel = None, to_plt = None, fit_key = 'enthalpy', add_label = [], *args, **kwargs):
    print('Energy statistics by generation (raw):\n')
    df_gen = {'Mean':[], 'Min':[], 'Max':[]}
    i = 0
    all_data = {fit_key: []}
    for i in range(1,1000):
        if os.path.exists('{}/raw{}.traj'.format(filename,i)):
            pop = ase.io.read('{}/raw{}.traj'.format(filename,i), index = ':')
            energy = np.array([ind.info['energy'] for ind in pop])
            fullsym =  [ind.get_chemical_formula() for ind in pop] 
            fitness = [ind.info[fit_key] for ind in pop]
            creater = [shown_name(ind.info['origin']) for ind in pop]
            deltaE = [(ind.info['enthalpy'] - ind.info['parentE']) if not ('rand' in ind.info['origin'] or 'seed' in ind.info['origin']) else np.nan for ind in pop]
            kwargs = {'energy': energy, 'creater': creater,'deltaE':deltaE, fit_key:fitness, 'fullsym': fullsym}
            
            subdf = pd.DataFrame(kwargs)

            # add Mean, Min, Max, and Min for each formula
            e = {'Mean':np.mean(subdf[fit_key]), 'Min':np.min(subdf[fit_key]), 'Max':np.max(subdf[fit_key])}
            all_data[fit_key].append (subdf[fit_key].values)
            elekeys = list(set(fullsym)) 
            for ele in elekeys:
                e[ele] = subdf.groupby('fullsym').get_group(ele).min()[fit_key]
                if ele not in df_gen.keys():
                    df_gen[ele] = []
            for key in e:
                df_gen[key].append(round(e[key], 3))


            # add by each creator 
            e = subdf.groupby('creater')

            for origin, data in e:
                select_column = data[fit_key] if 'rand' in origin else data['deltaE']
                shownum = select_column.mean()

                if origin in df_gen:
                    df_gen[origin].append(round(shownum, 3))
                    all_data[origin].append(select_column.values)
                else:
                    df_gen[origin] = [np.nan for _ in range(i)]
                    df_gen[origin][-1] = round(shownum, 3)
                    all_data[origin] = [[] for _ in range(i-1)]
                    all_data[origin].append(select_column.values)

            
            for key in df_gen:
                #print("{}, {}, {}".format(i, key, len(df_gen[key])))
                if not len(df_gen[key]) == i:
                    df_gen[key].append(np.nan)
                    all_data[key].append([])
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

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.figure(figsize=(6,6))

        #figure 2
        plt.subplot(2, 1, 2)
        if add_label:
            labels = add_label
        else:
            labels = [str(s) for s in df.columns if s not in ['Mean', 'Min', 'Max', 'random'] + elekeys]
        x = np.linspace(1, np.max(df.index), np.max(df.index))

        colors = ['red', 'blue', 'green', 'orange', 'black', 'purple']

        for i, label in enumerate(labels):
            data = all_data[label]
            delta_positions= (i - len(labels) /2 + 0.5 )/6
            plt.boxplot(data, patch_artist = False, widths=0.2,
                        medianprops={'color': colors[i]},
                        boxprops= {'color': colors[i]}, 
                        whiskerprops= {'color': colors[i]},
                        capprops = {'color': colors[i]},
                        showfliers=False,
                        positions=delta_positions +x) 
            plt.plot(x +delta_positions, df[label].values, label = label, color = colors[i], marker = 'v')

        plt.xlabel('generation',fontsize = 14)
        plt.ylabel(r"$\Delta$E",fontsize = 14)
        plt.xticks(np.arange(0, len(x)+1), list (range(0,len(x)+1)))
        plt.xlim(0.5, len(x)+0.5)
        plt.title(r"$\Delta$E of generation operators", fontsize = 14)
        plt.legend(loc = (1.01,0.2),prop = {'size':10}, fontsize = 14)

        #figure 1
        plt.subplot(2, 1, 1)
        data = all_data[fit_key]
        plt.boxplot(data, patch_artist = False, medianprops={'color': 'black'}, positions = x)
        plt.plot(x, df['Mean'].values, label = "Mean", color = 'red', marker = 'v')
        plt.xlabel('generation',fontsize = 14)
        plt.ylabel('raw fitness',fontsize = 14)
        plt.xticks(np.arange(0, len(x)+1), list (range(0,len(x)+1)))
        plt.xlim(0.5, len(x)+0.5)
        plt.title("fitness evolution", fontsize = 14)
        plt.legend(loc = (1.01,0.5),prop = {'size':10}, fontsize = 14)
    
        plt.tight_layout()
        #plt.show()
        plt.savefig(to_plt)
    best = ase.io.read('{}/good.traj'.format(filename),index = ':')
    energy = np.array([ind.info['energy'] for ind in best])
    minE = np.min(energy)
    index = np.where(minE ==energy)[0][0]
    print('============================================================')
    print('AllBest E = {}, origin = {}, symmetry = {}'.format(np.round(minE,6), best[index].info['origin'], spglib.get_spacegroup((best[index].cell, best[index].get_scaled_positions(), best[index].numbers), 0.2)))
    print('============================================================')

from magus.reconstruct.generator import SurfaceGenerator
def mine_substrate(filename = 'Ref/layerslices.traj', *args, **kwargs):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')
    substrate = ase.io.read(filename, index = -1, format = 'traj')
    SurfaceGenerator.mine_substrate_spg(substrate)

from magus.reconstruct.utils import cutcell
def inputslab(filename = 'inputslab.vasp', sliceslab = [], **kwargs):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')
    logging.info(f"Reading input slab file: {filename}")
    logging.info(f"Slice positions at {sliceslab}")
    slab = ase.io.read(filename)
    slab.wrap()
    assert len(sliceslab) ==4, "len(sliceslab) must be 4"
    
    sp = slab.get_scaled_positions()[:,2]

    Ref = []

    struct = ['bulk', 'buffer', 'top']
    for i in range(3): 
        _s = slab[[a for a, p in enumerate(sp) if sliceslab [i]< p <= sliceslab[i+1]]].copy()
        _s.positions -= sliceslab [i] * _s.cell[2]
        _s.cell[2] *= sliceslab [i+1] - sliceslab[i]
        #_s.wrap()
        Ref.append(_s)
        logging.info(f"Input {struct[i]} with {len(_s)} atoms")

    ase.io.write('Ref/layerslices.traj', Ref)
    logging.info("Done!")
    