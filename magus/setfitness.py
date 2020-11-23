from __future__ import print_function, division
import os
import logging
import numpy as np
from .utils import find_spg, symbols_and_formula
from .xrd import compare_xrd
import copy
from ase.phasediagram import PhaseDiagram
from .reconstruct import RCSPhaseDiagram

def fix_fitness(Pop):
    for ind in Pop.pop:
        ind.info['enthalpy'] = ind.atoms.info['enthalpy']
        ind.info['fitness']['enthalpy'] = -ind.atoms.info['enthalpy']


def var_fitness(Pop):
    # pop = Pop.allPopallPop.pop
    pop = Pop.pop
    name = [ind.atoms.get_chemical_formula() for ind in pop]
    enth = [ind.atoms.info['enthalpy']*len(ind.atoms) for ind in pop]
    refs = list(zip(name, enth))
    symbols = Pop.p.symbols
    # To make sure that the phase diagram can be constructed, we add elements with high energies.
    for sym in symbols:
        refs.append((sym, 100))
    pd = PhaseDiagram(refs, verbose=False)
    for ind in Pop.pop:
        refE = pd.decompose(ind.atoms.get_chemical_formula())[0]
        ehull = ind.atoms.info['enthalpy'] - refE/len(ind.atoms)
        if ehull < 1e-4:
            ehull = 0
        ind.atoms.info['ehull'] = ehull
        ind.info['ehull'] = ehull
        ind.info['enthalpy'] = ind.atoms.info['enthalpy']
        ind.info['fitness']['ehull'] = -ehull


def rcs_fitness(Pop):
# modified from var_fitness

    symbols = Pop.Individual.p.symbols

    #to remove H atom in bulk layer
    if symbols[0] =='H': 
        symbols = symbols[1:]

    pop = Pop.pop
    mark = 'fix'

    if len(symbols) >1 and Pop.Individual.p.AtomsToAdd:
        for atomnum in Pop.Individual.p.AtomsToAdd:
            if len(atomnum)>1:
                mark = 'var'
                break
        
    if mark == 'fix':
        refE_perAtom  = Pop.Individual.p.refE/np.sum([Pop.Individual.p.refFrml[s] for s in Pop.Individual.p.refFrml])

        for ind in pop:
            scale = 1.0 / ind.info['size'][0] / ind.info['size'][1]
            surfaceE = (ind.atoms.info['enthalpy']-refE_perAtom)*len(ind.atoms)*scale
            ind.info['Eo'] = surfaceE
            ind.info['enthalpy'] = ind.atoms.info['enthalpy']
            ind.info['fitness']['enthalpy'] = -surfaceE

    else:

        refE_perUnit = Pop.Individual.p.refE / Pop.Individual.p.refFrml[symbols[1]]
        ref_num0 =  1.0*Pop.Individual.p.refFrml[symbols[0]] / Pop.Individual.p.refFrml[symbols[1]]
        '''
        define Eo = E_slab - numB*E_ref, [E_ref = energy of unit A(a/b)B]
        define delta_n = numA - numB *(a/b)
        '''
        delta_n = []
        Eo = []
        for ind in pop:
            scale = 1.0 / ind.info['size'][0] / ind.info['size'][1]
            symbol, formula = symbols_and_formula(ind.atoms)
            frml = {s:i for s,i in zip(symbol, formula)}
            delta_n.append( (frml [symbols[0]] - frml[symbols[1]]*ref_num0) *scale)
            Eo.append((ind.atoms.info['enthalpy']*len(ind.atoms) -frml[symbols[1]]*refE_perUnit)*scale)

        refs = list(zip(delta_n, Eo))
        # To make sure that the phase diagram can be constructed, we add elements with high energies.
        refs.append((-ref_num0, 100))
        refs.append((1, 100))
        pd = RCSPhaseDiagram(refs)
        for i in range(len(pop)):
            refEo = pd.decompose(delta_n[i])[0]
            ehull =  Eo[i] - refEo
            if ehull < 1e-4:
                ehull = 0
            pop[i].atoms.info['ehull'] = ehull
            pop[i].info['ehull'] = ehull
            pop[i].info['enthalpy'] = pop[i].atoms.info['enthalpy']
            pop[i].info['fitness']['ehull'] = -ehull
            pop[i].info['Eo'] = Eo[i]


def set_fit_calcs(parameters):
    calcs = []
    if parameters.calcType == 'fix':
        calcs.append(fix_fitness)
    if parameters.calcType == 'var':
        calcs.append(var_fitness)
    elif parameters.calcType=='rcs':
        calcs.append(rcs_fitness)
    if parameters.calcType=='clus':
        calcs = [fix_fitness]
    return calcs
