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
from .utils import EmptyClass, get_formula

###############################


def read_parameters(inputFile):
    #Initialize

    parameters = yaml.load(open(inputFile))
    p = EmptyClass()
    p.workDir = os.getcwd()

    for key, val in parameters.items():
        setattr(p, key, val)

    # for parm in parameters.keys():
    #     exec "{0} = parameters['{0}']".format(parm)
    # if 'spacegroup' not in parameters.keys():
    #     p.spacegroup = range(1,231)

    # Default parameters
    dParms = {
        'spacegroup': list(range(1, 231)),
        'eleSize': 1,
        'fullEles': False,
        'volRatio': 2,
        'dRatio': 0.7,
        'exeCmd': "",
        'initSize': parameters['popSize'],
        'jobPrefix':"",
        'permNum': 4,
        'latDisps': list(range(1,5)),
        'ripRho': [0.5, 1, 1.5, 2],
        'molDetector': 0,
        'rotNum': 5,
        'cutNum': 5,
        'slipNum': 5,
        'latNum': 5,
        'grids': [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
        'bondRatio': 1.1,
        'bondRange': [0.9, 0.95, 1., 1.05, 1.1, 1.15],
        'maxRelaxTime': 20,
        'xrdFile': None,
        'xrdLamb': 0.6,
        'fixCell': False,
        'setCellPar': [1,1,1,90,90,90],
        'molMode': False,
        'molType': 'fix',
        'chkMol': False,
        'molScaleCell': False,
    }

    for key, val in dParms.items():
        if key not in parameters.keys():
            setattr(p, key, val)
            parameters[key] = val

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

    if p.setAlgo == 'bbo':
        sumFrac = sum([p.migrateFrac, p.mutateFrac, p.randFrac])
        p.migrateFrac = p.migrateFrac/sumFrac
        p.mutateFrac = p.mutateFrac/sumFrac
        p.randFrac = p.randFrac/sumFrac

    assert p.randFrac <= 1, 'randFrac should be lower than 1'
    if p.molMode:
        assert 'molFile' in parameters.keys(), 'Please define molFile'
        assert 'molFormula' in parameters.keys(), 'Please define molFormula'
        assert p.molDetector == 1, 'molDetecotr should be 1'
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
    p.invFrml = np.linalg.inv(np.dot(tmpFrml, tmpFrml.T))
    p.invFrml = p.invFrml.tolist()



    ############ BBO Parameters#############
    p.bboParm = dict()
    p.bboParm['migrateFrac'] = p.migrateFrac
    p.bboParm['mutateFrac'] = p.mutateFrac
    p.bboParm['randFrac'] = p.randFrac
    p.bboParm['grids'] = p.grids

    ############ Krig Parameters #######
    p.krigParm = dict()
    p.krigParm['randFrac'] = p.randFrac
    p.krigParm['permNum'] = p.permNum
    p.krigParm['rotNum'] = p.rotNum
    p.krigParm['cutNum'] = p.cutNum
    p.krigParm['slipNum'] = p.slipNum
    p.krigParm['latDisps'] = p.latDisps
    p.krigParm['latNum'] = p.latNum
    p.krigParm['ripRho'] = p.ripRho
    p.krigParm['grids'] = p.grids
    p.krigParm['kind'] = 'lcb'
    p.krigParm['kappa'] = p.kappa
    p.krigParm['kappaLoop'] = p.kappaLoop
    p.krigParm['xi'] = 0
    p.krigParm['scale'] = p.scale
    p.krigParm['parent_factor'] = p.parent_factor

    ##############################
    parmList = [
                'calcType',
                'calculator',
                'spacegroup',
                'popSize',
                'numGen',
                'symbols',
                'formula',
                'numFrml',
                'invFrml',
                'minAt',
                'maxAt',
                'workDir',
                'pressure',
                'calcNum',
                'numParallel',
                # 'pickPareto',
                'bboParm',
                'krigParm',
                'saveGood',
                'queueName',
                'numCore',
                'randFrac',
                'mutateFrac',
                'migrateFrac',
                ]
    if p.molMode:
        parmList.extend(['molList', 'molNum'])

    for parm in parmList:
        parameters[parm] = getattr(p, parm)

    with open('{}/allParameters.yaml'.format(p.workDir), 'w') as f:
        f.write(yaml.dump(parameters, default_flow_style=False))

    return parameters


if __name__ == '__main__':
    parm = read_parameters('input.yaml')
    # print(parm['numFrml'])

