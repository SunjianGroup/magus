from __future__ import print_function, division
import os
import logging
from .renewstruct import find_spg

def calc_fitness(Pop, parameters):
    fitPop = Pop[:]
    fitPop = find_spg(fitPop, 1)
    calcType = parameters['calcType']

    for ind in fitPop:
        # gap = ind.info['gap']

        if calcType == 'fix':
            enthalpy = ind.info['enthalpy']
            ftn1 = enthalpy   # define fitness1
        elif calcType == 'var':
            ehull = ind.info['ehull']
            ftn1 = ehull

#        ftn2 = gap if gap > 1.0 else 1.0 # define fitness2
#        ftn2 = abs(gap -1.)
#        ftn2 = enthalpy
        ftn2 = ftn1

        ind.info['fitness1'] = ftn1
        ind.info['fitness2'] = ftn2

    return fitPop
