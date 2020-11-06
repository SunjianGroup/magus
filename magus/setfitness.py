from __future__ import print_function, division
import os
import logging
import numpy as np
from .utils import find_spg
from .xrd import compare_xrd
import copy
from ase.phasediagram import PhaseDiagram

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
    mark = 'fix'

    if len(symbols) >1 and Pop.Individual.p.AtomsToAdd:
        for atomnum in Pop.Individual.p.AtomsToAdd:
            if len(atomnum)>1:
                mark = 'var'
                break
        
    if mark == 'fix':
        fix_fitness(Pop)
    else:
        var_fitness(Pop)

def set_fit_calcs(parameters):
    calcs = []
    if parameters.calcType == 'fix':
        calcs.append(fix_fitness)
    if parameters.calcType == 'var':
        calcs.append(var_fitness)
    if parameters.calcType=='rcs':
        calcs.append(var_fitness)
    if parameters.calcType=='clus':
        calcs.append(fix_fitness)
    return calcs
