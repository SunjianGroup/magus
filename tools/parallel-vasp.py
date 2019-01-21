#!/usr/bin/env python

from __future__ import print_function, division
import os, shutil, subprocess, sys
import ase.io
import yaml

def generate_input():
    parameters = {
        'inputDir': os.getcwd(),
        'structFile': '',
        'inFormat': '',
        'calcNum': 1,
        'queueName': '',
        'numParallel': 1,
        'numCore': 1,
        'xc': '',
        'pressure': 0,
        'symbols': [],
        'ppLabel': [],
        'prefix': 'calcVasp',
    }

    return parameters



def simple_calc_vasp_parallel(parameters):

    inputDir = parameters['inputDir']
    structFile = parameters['structFile']
    inFormat = parameters['inFormat']
    calcNum = parameters['calcNum']
    queueName = parameters['queueName']
    numParallel = parameters['numParallel']
    numCore = parameters['numCore']
    xc = parameters['xc']
    pressure = parameters['pressure']
    symbols = parameters['symbols']
    ppLabel = parameters['ppLabel']
    prefix = parameters['prefix']

    vaspSetup = dict(zip(symbols, ppLabel))

    calcPop = ase.io.read(structFile, index=':', format=inFormat)



    popLen = len(calcPop)
    eachLen = popLen//numParallel
    remainder = popLen%numParallel

    runArray = []
    for i in range(numParallel):
        tmpList = [ i + numParallel*j for j in range(eachLen)]
        if i < remainder:
            tmpList.append(numParallel*eachLen + i)
        runArray.append(tmpList)



    runJobs = []
    for i in range(numParallel):
        if not os.path.exists("{}{}".format(prefix, i)):
            os.mkdir("{}{}".format(prefix, i))
        os.chdir("{}{}".format(prefix, i))

        tmpPop = [calcPop[j] for j in runArray[i]]

        ase.io.write('initPop.traj', tmpPop)
        for j in range(1, calcNum + 1):
            shutil.copy("{}/INCAR_{}".format(inputDir, j), 'INCAR_{}'.format(j))
        # shutil.copy("../run_vasp.py", "run_vasp.py")
        with open('vaspSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(vaspSetup))

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J Vasp_%s\n"% (queueName, numCore, i))
        f.write("python -m csp.runvasp {} {} vaspSetup.yaml {} initPop.traj optPop.traj > vasplog".format(calcNum, xc, pressure))
        f.close()

        jobID = subprocess.check_output("bsub < parallel.sh", shell=True).split()[1]
        jobID = jobID[1: -1]
        runJobs.append(jobID)

        os.chdir('..')

    return runJobs


if __name__ == '__main__':
    # print(sys.argv)
    assert len(sys.argv) == 3
    mode = sys.argv[1]
    assert mode in ('sub', 'read', 'gen')
    paramFile = sys.argv[2]

    # generate a blank parameter file
    if mode == 'gen':
        parameters = generate_input()
        with open(paramFile, 'w') as f:
            f.write(yaml.dump(parameters))
        sys.exit()


    parameters = yaml.load(open(paramFile))

    if 'prefix' not in parameters.keys():
        parameters['prefix'] = 'calcVasp'

    if mode == 'sub':
        runJobs = simple_calc_vasp_parallel(parameters)
        print(runJobs)
    elif mode == 'read':
        optTraj = []
        prefix = parameters['prefix']
        for i in range(parameters['numParallel']):
            optTraj.extend(ase.io.read('{}{}/optPop.traj'.format(prefix, i), index=':'))
        ase.io.write('optTraj.traj', optTraj)
