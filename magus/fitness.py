from __future__ import print_function, division
import os
import logging
import numpy as np
from .utils import find_spg
from .xrd import compare_xrd
import copy
from ase.phasediagram import PhaseDiagram
import abc


class FitnessCalculator(abc.ABC):
    @abc.abstractmethod
    def calc(self, Pop):
        pass


class EnthalpyFitness(FitnessCalculator):
    def calc(self, Pop):
        for ind in Pop:
            ind.info['enthalpy'] = ind.atoms.info['enthalpy']
            ind.info['fitness']['enthalpy'] = -ind.atoms.info['enthalpy']


class EhullFitness(FitnessCalculator):
    def calc(self, Pop):
        name = [ind.atoms.get_chemical_formula() for ind in Pop]
        enth = [ind.atoms.info['enthalpy']*len(ind.atoms) for ind in Pop]
        refs = list(zip(name, enth))
        symbols = Pop.p.symbols
        # To make sure that the phase diagram can be constructed, we add elements with high energies.
        for sym in symbols:
            refs.append((sym, 100))
        pd = PhaseDiagram(refs, verbose=False)
        for ind in Pop:
            refE = pd.decompose(ind.atoms.get_chemical_formula())[0]
            ehull = ind.atoms.info['enthalpy'] - refE/len(ind.atoms)
            if ehull < 1e-4:
                ehull = 0
            ind.atoms.info['ehull'] = ehull
            ind.info['ehull'] = ehull
            ind.info['enthalpy'] = ind.atoms.info['enthalpy']
            ind.info['fitness']['ehull'] = -ehull

fit_dict = {
    'Enthalpy': EnthalpyFitness(),
    'Ehull': EhullFitness(),
}
