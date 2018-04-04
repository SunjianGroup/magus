# Parallel MultiObject Optimize
from __future__ import print_function, division
import random
import logging
import os
import sys
import shutil
import time
import json
from scipy.spatial.distance import cdist
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from .localopt import *
from .renewstruct import *
from .initstruct import build_struct, read_seeds, varcomp_2elements, varcomp_build
from .readvasp import *
from .setfitness import *
from .writeresults import *
from .fingerprint import calc_all_fingerprints, calc_one_fingerprint
from .bayes import atoms_util
from .readparm import read_parameters


def check_jobs(statFile='currentStat.json'):
    if not os.path.exists(statFile):
        curStat = dict()
        curStat['curGen'] = 1
        return True, curStat
    else:
        with open(statFile, 'r') as f:
            curStat = json.load(f)
        runJobs = curStat['runJobs']
        jobStat = jobs_stat(runJobs)
        curStat['jobStat'] = jobStat
        logging.info("jobStat: %s"% (jobStat))
        if 'RUN' in jobStat:
            return False, curStat
        else:
            logging.info("Generation End")
            return True, curStat

def csp_loop(curStat):
    """
    crystal structure prediction loop
    INPUT: parameters and current
    """
    # Initialize
    # logging.info("Reading parameters")
    # for key, val in parameters.items():
    #     exec "%s = parameters['%s']" % (key, key)

    if 'runJobs' not in curStat.keys():
        # curStat = dict()
        # curStat['curGen'] = 0
        initial = True
    else:
        initial = False
    curGen = curStat['curGen']

    if not initial:
        jobStat = curStat['jobStat']
        if calculator == 'vasp':
            optPop = read_parallel_results(jobStat, parameters, prefix='calcVasp')
        elif calculator == 'gulp':
            optPop = read_parallel_results(jobStat, parameters, prefix='calcGulp')
        logging.debug("optPop length: {}".format(len(optPop)))
        logging.info('calc_structs finish')

        logging.info("check distance")
        optPop = check_dist(optPop, 0.5)
        logging.info("check survival: {}".format(len(optPop)))

        # Initialize paretoPop, goodPop
        if curGen > 1:
            paretoPop = read_yaml("{}/results/pareto{}.yaml".format(workDir, curGen-1))
            goodPop = read_yaml("{}/results/good.yaml".format(workDir))
        else:
            paretoPop = list()
            goodPop = list()

        #Convex Hull
        if calcType == 'var':
            allPop = optPop + paretoPop
            allPop = convex_hull(allPop)
            optLen = len(optPop)
            optPop = allPop[:optLen]
            paretoPop = allPop[optLen:]

        optPop = calc_fitness(optPop, parameters)
        logging.info('calc_fitness finish')
        for ind in optPop:
            # logging.debug("formula: {}".format(ind.get_chemical_formula()))
            logging.debug("optPop {strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format( strFrml=ind.get_chemical_formula(), **ind.info))

        # Calculate fingerprints
        optPop = calc_all_fingerprints(optPop, parameters)
        for ind in optPop:
            initFp = np.atleast_2d(ind.info['initFp'])
            curFp = np.atleast_2d(ind.info['fingerprint'])
            relaxD = cdist(initFp, curFp)[0, 0]
            ind.info['relaxD'] = relaxD

        # os.chdir('Compare')
        # logging.info('optPop_duplicate start')
        optPop = del_duplicate(optPop)
        # logging.info('optPop_duplicate finish')
        # os.chdir('..')



        logging.info('pareto_front')
        paretoPop = pareto_front(optPop + paretoPop, e=0)

        # logging.info('pareto_duplicate start')
        paretoPop = del_duplicate(paretoPop)
        # logging.info('pareto_duplicate finish')

        for ind in paretoPop:
            logging.debug("paretoPop {strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format( strFrml=ind.get_chemical_formula(), **ind.info))
            # logging.info('paretoPop enthalpy: %s, fit1: %s, fit2: %s' %tuple([ind.info[attr] for attr in ('enthalpy', 'fitness1', 'fitness2')]))

        ### save good individuals
        logging.info('goodPop')
        goodPop = calc_dominators(optPop + goodPop)
        goodPop = del_duplicate(goodPop)
        if len(goodPop) > popSize:
            goodPop = sorted(goodPop, key=lambda x:x.info['dominators'])[:popSize]

        ### keep best
        logging.info('keepPop')
        labels, keepPop = clustering(goodPop, saveGood)

        ### write results
        write_results(optPop, curGen, 'gen')
        write_results(paretoPop, curGen, 'pareto')
        write_results(goodPop,'','good')
        write_results(keepPop, curGen, 'keep')
        shutil.copy('log.txt', 'results/log.txt')

        ### write dataset
        write_dataset(optPop)


        # logging.info("curGen: %s"% (curGen))
        curGen += 1
        curStat['curGen'] = curGen
        logging.info("===== Generation {} =====".format(curGen))


    if initial:

        if os.path.exists("results"):
            for i in range(1, 100):
                if not os.path.exists("results{}".format(i)):
                    shutil.move("results", "results{}".format(i))
                    break
        
        os.mkdir("results")
        shutil.copy("allParameters.yaml", "results/allParameters.yaml")

        logging.info("===== Generation 1 =====")
        if calcType == 'fix':
            initPop = build_struct(popSize, symbols, formula, numFrml)
        elif calcType == 'var':
            logging.debug('calc var')
            initPop = varcomp_build(popSize - len(symbols), symbols, minAt, maxAt)
            # logging.debug("initPop length: {}".format(len(initPop)))
            # initPop = varcomp_2elements(popSize - len(symbols), symbols, minAt, maxAt)
            for sybl in symbols:
                initPop.extend(build_struct(1, [sybl], [1]))

        logging.debug("initPop length: {}".format(len(initPop)))
        initPop.extend(read_seeds(parameters))

    else:
        bboPop = del_duplicate(optPop + goodPop)

        if setAlgo == 'bayes':
            mainAlgo = Kriging(bboPop, curGen, parameters)
            #subAlgo = BBO(bboPop, parameters)
            ## for grid in random.sample(grids, 2):
            #for grid in grids:
            #    subAlgo.set_grid(grid)
            #    subAlgo.bbo_cutcell()
            #    tmpPop = subAlgo.get_bboPop()
            #    mainAlgo.add(tmpPop)

            # mainAlgo.add(build_struct(popSize, symbols, formula, numFrml))

            mainAlgo.generate()
            mainAlgo.fit_gp()
            mainAlgo.select()
            initPop = mainAlgo.get_nextPop()

        elif setAlgo == 'bbo':

        ### BBO
            mainAlgo = BBO(bboPop, parameters)
            mainAlgo.bbo_cutcell()
            initPop = mainAlgo.get_bboPop()

        # ### Kriging
        # algo = Kriging(bboPop)
        # algo.generate()
        # algo.add(tmpPop)
        # algo.fit_gp()
        # algo.select()
        # initPop = algo.get_nextPop()



        if len(initPop) < popSize:
            logging.debug("random structures out of Kriging")
            if calcType == 'fix':
                initPop.extend(build_struct(popSize - len(initPop), symbols, formula, numFrml))
            if calcType == 'var':
                initPop.extend(varcomp_build(popSize - len(initPop), symbols, minAt, maxAt))
                # initPop.extend(varcomp_2elements(popSize - len(initPop), symbols, minAt, maxAt))

    ### Initail check
    # initPop = check_dist(initPop, 0.7)

    ### Initail fingerprint
    for ind in initPop:
        if 'fingerprint' in ind.info.keys():
            initFp = ind.info['fingerprint']
        else:
            initFp = calc_one_fingerprint(ind, parameters)
        ind.info['initFp'] = initFp

    ### Save Initail
    write_results(initPop, curGen, 'init')
    if not os.path.exists('calcFold'):
        os.mkdir('calcFold')
    os.chdir('calcFold')

    if calculator == 'gulp':
        runJobs = calc_gulp_parallel(calcNum, initPop, parameters)
    elif calculator == 'vasp':
        # runJobs = calc_vasp_parallel(setincar.calcs, initPop, parameters)
        runJobs = calc_vasp_parallel(calcNum, initPop, parameters)
    os.chdir(workDir)

    curStat['runJobs'] = runJobs
    return curStat


### Main ###

logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(message)s")
parameters = read_parameters('input.yaml')
for key, val in parameters.items():
    vars()[key] = val
# numGen = parameters['numGen']
while True:
    doneOr, curStat = check_jobs()
    logging.info(time.ctime())
    logging.debug("doneOr:%s"%(doneOr))

    if doneOr:
        # All Done
        if curStat['curGen'] > numGen:
            logging.info("All Done")
            break
        
        curStat = csp_loop(curStat)
        with open('currentStat.json', 'w') as f:
            json.dump(curStat, f)

    time.sleep(waitTime)
