from __future__ import print_function, division
from ase.data import atomic_numbers
import math
import os
import yaml
import logging

###############################


def read_parameters(inputFile):
    #Initialize

    workDir = os.getcwd()
    parameters = yaml.load(open(inputFile))
    for parm in parameters.keys():
        exec "{0} = parameters['{0}']".format(parm)

    if 'spacegroup' not in parameters.keys():
        spacegroup = range(1,231)

    sumFrac = sum([migrateFrac, mutateFrac, randFrac])
    migrateFrac = migrateFrac/sumFrac
    mutateFrac = mutateFrac/sumFrac
    randFrac = randFrac/sumFrac
    if calcType == 'fix':
        minFrml = int(math.ceil(minAt/sum(formula)))
        maxFrml = int(maxAt/sum(formula))
        numFrml = range(minFrml, maxFrml + 1)
    else:
        numFrml = [1]
    ############ BBO Parameters#############
    bboParm = dict()
    bboParm['migrateFrac'] = migrateFrac
    bboParm['mutateFrac'] = mutateFrac
    bboParm['randFrac'] = randFrac
    bboParm['grids'] = grids

    ############ Krig Parameters #######
    krigParm = dict()
    krigParm['randFrac'] = randFrac
    krigParm['permRates'] = [0.2, 0.4, 0.6, 0.8]
    krigParm['grids'] = ([1, 1, 2],[1, 2, 1],[2, 1, 1],)
    krigParm['kind'] = 'lcb'
    krigParm['kappa'] = kappa
    krigParm['kappaLoop'] = kappaLoop
    krigParm['xi'] = 0
    krigParm['scale'] = scale
    krigParm['parent_factor'] = parent_factor

    ##############################
    parmList = [
                'calcType',
                'calculator',
                'popSize',
                'numGen',
                'symbols',
                'formula',
                'numFrml',
                'grids',
                'minAt',
                'maxAt',
                'spacegroup',
                'workDir',
                'pressure',
                'exeCmd',
                'calcNum',
                'numParallel',
                # 'pickPareto',
                'bboParm',
                'krigParm',
                'saveGood',
                'queueName',
                'numCore',
                ]
    for parm in parmList:
        parameters[parm] = locals()[parm]

    with open('{}/allParameters.yaml'.format(workDir), 'w') as f:
        f.write(yaml.dump(parameters, default_flow_style=False))

    return parameters


if __name__ == '__main__':
    parm = read_parameters('input.yaml')
    print(parm['numFrml'])

