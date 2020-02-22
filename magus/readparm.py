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
#from .offspring_creator import *
###############################


def read_parameters(inputFile):
    #Initialize

    parameters = yaml.load(open(inputFile))
    # parameters = yaml.load(open(inputFile), Loader=yaml.FullLoader)
    p = EmptyClass()
    p.workDir = os.getcwd()
    p.resultsDir = os.path.join(p.workDir,'results')
    
    for key, val in parameters.items():
        setattr(p, key, val)

    # Default parameters
    dParms = {
        'mode': 'serial',
        'spacegroup': list(range(1, 231)),
        'repairtryNum': 10,
        'eleSize': 1,
        'fullEles': False,
        'setAlgo': 'ea',
        'volRatio': 2,
        'dRatio': 0.7,
        'exeCmd': "",
        'initSize': parameters['popSize'],
        'jobPrefix':"",
        'latDisps': list(range(1,5)),
        'ripRho': [0.5, 1, 1.5, 2],
        'molDetector': 0,
        'cutNum': 2*int(parameters['popSize']/3)+1,
        'permNum': int(parameters['popSize']/3)+1,
        'rotNum': int(parameters['popSize']/3)+1,
        'slipNum': int(parameters['popSize']/3)+1,
        'latNum': int(parameters['popSize']/3)+1,
        'ripNum': int(parameters['popSize']/3)+1,
        'grids': [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
        'bondRatio': 1.15,
        'bondRange': [1., 1.1, 1.2],
        'waitTime': 60,
        'maxRelaxTime': 1200,
        'xrdFile': None,
        'xrdLamb': 0.6,
        'fixCell': False,
        'setCellPar': [1,1,1,90,90,90],
        'molMode': False,
        'molType': 'fix',
        'chkMol': False,
        'molScaleCell': False,
        'fastcp2k': False,
        'maxRelaxStep': 0.1,
        'optimizer': 'bfgs',
        'goodehull': 0.1,
        'gp_factor': 1,
        'updateVol': True,
        'addSym': False,
        'symprec': 0.2,
        'compress': False,
        'cRatio': 0.8,
        'cutoff': 4.0,
        'mlRelax': False,
        'ZernikeNmax': 4,
        'ZernikeLmax': None,
        'ZernikeNcut': 4,
        'ZernikeDiag': True,
        'kernelType': 'dot',
        'savetmp': False,
        'tourRatio': 0.1,
    }

    for key, val in dParms.items():
        if key not in parameters.keys():
            setattr(p, key, val)
            parameters[key] = val

    # Check parameters
    assert p.mode in ['serial', 'parallel'], "Undefined mode"
    assert p.calcType in ['fix', 'var'], "Undefined calcType"
    assert p.calculator in ['lj', 'emt', 'vasp', 'gulp', 'xtb', 'cp2k', 'lammps', 'quip'], "Undefined calculator"
    if p.calculator in ['lj', 'emt', 'xtb']:
        assert p.mode == 'serial', "The calculator only support serial mode"
    assert p.randFrac <= 1, 'randFrac should be lower than 1'
    if p.ZernikeLmax:
        assert p.ZernikeLmax <= p.ZernikeNmax

    p.setAlgo = p.setAlgo.lower()

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
    p.spacegroup = expandSpg

    if p.calculator == 'vasp':
        if not hasattr(p, 'ppLabel'):
            p.ppLabel = ['' for _ in p.symbols]

    if p.molMode:
        assert 'molFile' in parameters.keys(), 'Please define molFile'
        assert 'molFormula' in parameters.keys(), 'Please define molFormula'
        if p.chkMol:
            assert p.molDetector == 1, 'molDetecotr should be 1 if chkMol is True'
        assert p.calcType == 'fix', 'Variable composition molecule crystal search in not available'
        assert p.molType == 'fix', 'Variable composition molecule crystal search in not available'
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

    if p.calcType == 'fix':
        p.minFrml = int(math.ceil(p.minAt/sum(p.formula)))
        p.maxFrml = int(p.maxAt/sum(p.formula))
        p.numFrml = list(range(p.minFrml, p.maxFrml + 1))
    else:
        p.numFrml = [1]

    # inverse matrix of dot(formula, formula.T)
    # for var mode
    if p.calcType == 'fix':
        tmpFrml = np.array([p.formula])
    else:
        tmpFrml = np.array(p.formula)

    assert np.linalg.matrix_rank(tmpFrml) <= tmpFrml.shape[1], "Please check input formula"

    # check epsArr, stepArr for ASE optimization, e.g. CP2K
    if 'epsArr' in parameters.keys():
        assert len(p.epsArr) == p.calcNum
    if 'stepArr' in parameters.keys():
        assert len(p.stepArr) == p.calcNum

    parmList = [
                'calcType',
                'calculator',
                'setAlgo',
                'spacegroup',
                'popSize',
                'numGen',
                'symbols',
                'formula',
                #'ppLabel',
                'numFrml',
                'minAt',
                'maxAt',
                'workDir',
                'resultsDir',
                'pressure',
                'calcNum',
                'numParallel',
                # 'pickPareto',
                'saveGood',
                'queueName',
                'numCore',
                'randFrac',
                ]
    if p.molMode:
        parmList.extend(['molList', 'molNum'])

    if p.calculator == 'vasp':
        parmList.extend(['ppLabel',])

    for parm in parmList:
        parameters[parm] = getattr(p, parm)

    with open('{}/allParameters.yaml'.format(p.workDir), 'w') as f:
        f.write(yaml.dump(parameters, default_flow_style=False))

    return parameters

def get_atoms_generator(parameters):
    if parameters.calcType == 'fix':
        return BaseGenerator(parameters)
    elif parameters.calcType == 'var':
        return VarGenerator(parameters)

def get_pop_generator(parameters):
    cutandsplice = CutAndSplicePairing()
    perm = PermMutation()
    lattice = LatticeMutation()
    ripple = RippleMutation()
    slip = SlipMutation()

    numlist = [parameters.cutNum,parameters.permNum,parameters.latNum,parameters.ripNum,parameters.slipNum]
    oplist = [cutandsplice,perm,lattice,ripple,slip]
    
    popgen = PopGenerator(numlist,oplist,parameters)
    return popgen

def get_calculator(parameters):
    if parameters.calculator == 'vasp':
        MainCalculator = VaspCalculator(parameters)
    elif parameters.calculator == 'lj':
        MainCalculator = LJCalculator(parameters)
    elif parameters.calculator == 'emt':
        MainCalculator = EMTCalculator(parameters)
    elif parameters.calculator == 'gulp':
        MainCalculator = GULPCalculator(parameters)
    elif parameters.calculator == 'xtb':
        MainCalculator = XTBCalculator(parameters)
    elif parameters.calculator == 'quip':
        MainCalculator = QUIPCalculator(parameters)
    elif parameters.calculator == 'lammps':
        MainCalculator = LAMMPSCalculator(parameters)

    return MainCalculator

def get_population(parameters):
    return Population(parameters)

if __name__ == '__main__':
    parm = read_parameters('input.yaml')
    # print(parm['numFrml'])

