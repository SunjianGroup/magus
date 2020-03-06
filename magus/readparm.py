from __future__ import print_function, division
from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell
import ase.io
import math
import os
import yaml
import logging
from functools import reduce
import numpy as np

from .localopt import *
from .initstruct import BaseGenerator,read_seeds,VarGenerator,build_mol_struct
from .writeresults import write_dataset, write_results, write_traj
from .utils import *

from .queue import JobManager
from .renew import BaseEA, BOEA
from .population import Population
#ML module
#from .machinelearning import LRmodel
from .offspring_creator import *
###############################


def read_all_parameters(inputFile):
    with open(inputFile) as f:
        p_dict = yaml.load(f)
    p = EmptyClass()
    p.workDir = os.getcwd()
    p.resultsDir = os.path.join(p.workDir,'results')
    p.calcDir = os.path.join(p.workDir,'calcFold')
    for key, val in p_dict.items():
        setattr(p, key, val)

    Requirement = ['calcType','calculator','popSize','numGen','saveGood']
    Default = {'spacegroup':np.arange(1,231),'initSize':p.popSize,'molMode':False}
    checkParameters(p,p,Requirement,{})

    p.initSize = p.popSize
    expandSpg = []
    for item in p.spacegroup:
        if isinstance(item, int):
            if 1 <= item <= 230:
                expandSpg.append(item)
        if isinstance(item, str):
            assert '-' in item, 'Please check the format of spacegroup'
            s1, s2 = item.split('-')
            s1, s2 = int(s1), int(s2)
            assert 1 <= s1 < s2 <= 230, 'Please check the format of spacegroup'
            expandSpg.extend(list(range(s1, s2+1)))
    p.spgs = expandSpg

    if p.molMode:
        assert hasattr(p,'molFile'), 'Please define molFile'
        assert hasattr(p,'molFormula'), 'Please define molFormula'
        mols = [ase.io.read("{}/{}".format(p.workDir, f), format='xyz') for f in p.molFile]
        molSymbols = set(reduce(lambda x,y: x+y, [ats.get_chemical_symbols() for ats in mols]))
        assert molSymbols == set(p.symbols), 'Please check the compositions of molecules'
        if p.molType == 'fix':
            molFrmls = np.array([get_formula(mol, p.symbols) for mol in mols])
            p.formula = np.dot(p.molFormula, molFrmls).tolist()
        p.molList = [{'numbers': ats.get_atomic_numbers(),
                    'positions': ats.get_positions()}
                    for ats in mols]
        p.molNum = len(p.molFile)
    return p

def get_atoms_generator(parameters):
    if parameters.calcType == 'fix':
        return BaseGenerator(parameters)
    elif parameters.calcType == 'var':
        return VarGenerator(parameters)
    else:
        raise Exception("Undefined calcType '{}'".format(parameters.calcType))

def get_pop_generator(parameters):
    cutandsplice = CutAndSplicePairing()
    perm = PermMutation()
    lattice = LatticeMutation()
    ripple = RippleMutation()
    slip = SlipMutation()

    num = 2*int(parameters['popSize']/3)+1
    Requirement = []
    Default = {'cutNum':num,'permNum': num, 'rotNum': num,
        'slipNum': num,'latNum': num, 'ripNum': num}
    checkParameters(parameters,parameters,Requirement,Default)
    numlist = [parameters.cutNum,parameters.permNum,parameters.latNum,parameters.ripNum,parameters.slipNum]
    oplist = [cutandsplice,perm,lattice,ripple,slip]
    
    popgen = PopGenerator(numlist,oplist,parameters)
    return popgen

def get_calculator(parameters):
    p = EmptyClass()
    for key, val in parameters.maincalculator.items():
        setattr(p, key, val)
    if p.calculator == 'vasp':
        MainCalculator = VaspCalculator(p)
    elif p.calculator == 'lj':
        MainCalculator = LJCalculator(p)
    elif p.calculator == 'emt':
        MainCalculator = EMTCalculator(p)
    elif p.calculator == 'gulp':
        MainCalculator = GULPCalculator(p)
    elif p.calculator == 'xtb':
        MainCalculator = XTBCalculator(p)
    elif p.calculator == 'quip':
        MainCalculator = QUIPCalculator(p)
    elif p.calculator == 'lammps':
        MainCalculator = LAMMPSCalculator(p)
    else:
        raise Exception("Undefined calculator '{}'".format(p.calculator))
    return MainCalculator

def get_population(parameters):
    return Population(parameters)

if __name__ == '__main__':
    parm = read_parameters('input.yaml')
    # print(parm['numFrml'])

