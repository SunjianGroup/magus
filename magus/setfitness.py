from __future__ import print_function, division
import os
import logging
import numpy as np
# from .renewstruct import find_spg
from .utils import find_spg
from .xrd import compare_xrd
import copy

def calc_fitness(pop, mode='fix'):
    for ind in pop:
        enthalpy = ind.info['enthalpy']
        if mode == 'fix':
            ftn1 = enthalpy
            ind.info['fitness1'] = ftn1
            ind.info['fitness2'] = ftn1
        elif mode == 'var':
            ftn1 = ind.info['ehull']
            ind.info['fitness1'] = ftn1
            ind.info['fitness2'] = ftn1
