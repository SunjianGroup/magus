from __future__ import print_function, division
import os
import logging
import numpy as np
# from .renewstruct import find_spg
from .utils import find_spg
from .xrd import compare_xrd

def calc_fitness(Pop, parameters):
    fitPop = Pop[:]
    # fitPop = find_spg(fitPop, 1)
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

def calc_fitness_xrd(Pop, parameters):
    fitPop = Pop[:]
    calcType = parameters['calcType']
    xrdFile = parameters['xrdFile']
    workDir = parameters['workDir']
    lamb = parameters['xrdLamb']
    expData = np.loadtxt("{}/{}".format(workDir, xrdFile))

    for ind in fitPop:

        if calcType == 'fix':
            enthalpy = ind.info['enthalpy']
            ftn1 = enthalpy   # define fitness1
        elif calcType == 'var':
            ehull = ind.info['ehull']
            ftn1 = ehull

        xrdScore = compare_xrd(ind, expData, lamb)
        ind.info['xrdScore'] = xrdScore
        ftn2 = xrdScore


        ind.info['fitness1'] = ftn1
        ind.info['fitness2'] = ftn2

    return fitPop
