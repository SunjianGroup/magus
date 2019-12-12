# Parallel MultiObject Optimize
from __future__ import print_function, division
import random, logging, os, sys, shutil, time, json
import argparse
import numpy as np
from scipy.spatial.distance import cdist
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from csp.localopt import generate_calcs, calc_gulp_parallel, calc_vasp_parallel, jobs_stat, read_parallel_results
from csp.renewstruct import del_duplicate, Kriging, BBO, pareto_front, convex_hull, check_dist, calc_dominators
from csp.initstruct import build_struct, read_seeds, varcomp_2elements, varcomp_build, build_mol_struct
# from .readvasp import *
from csp.setfitness import calc_fitness
from csp.writeresults import write_dataset, write_results, write_traj
from csp.fingerprint import calc_all_fingerprints, calc_one_fingerprint, clustering
from csp.bayes import atoms_util
from csp.readparm import read_parameters
from csp.utils import EmptyClass, calc_volRatio, check_mol_pop


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

def csp_loop(curStat, parameters):
    """
    crystal structure prediction loop
    INPUT: parameters and current
    """

    p = EmptyClass
    for key, val in parameters.items():
        setattr(p, key, val)

    if p.molMode:
        p.inputMols = [Atoms(**molInfo) for molInfo in p.molList]

    if 'runJobs' not in curStat.keys():
        initial = True
    else:
        initial = False
    curGen = curStat['curGen']

    if not initial:
        jobStat = curStat['jobStat']
        if p.calculator == 'vasp':
            optPop = read_parallel_results(jobStat, parameters, prefix='calcVasp')
        elif p.calculator == 'gulp':
            optPop = read_parallel_results(jobStat, parameters, prefix='calcGulp')
        logging.info("optPop length: {}".format(len(optPop)))
        logging.info('calc_structs finish')

        # Save raw data
        write_results(optPop, curGen, 'raw')

        logging.info("check distance")
        optPop = check_dist(optPop, p.dRatio)
        logging.info("check survival: {}".format(len(optPop)))

        if p.chkMol:
            logging.info("check mols")
            optPop = check_mol_pop(optPop, p.inputMols, p.bondRatio)
            logging.info("check survival: {}".format(len(optPop)))

        # Initialize paretoPop, goodPop
        if curGen > 1:
            # paretoPop = ase.io.read("{}/results/pareto{}.traj".format(p.workDir, curGen-1), format='traj', index=':')
            goodPop = ase.io.read("{}/results/good.traj".format(p.workDir), format='traj', index=':')
            keepPop = ase.io.read("{}/results/keep{}.traj".format(p.workDir, curGen-1), format='traj', index=':')
        else:
            # paretoPop = list()
            goodPop = list()
            keepPop = list()

        #Convex Hull
        allPop = optPop + goodPop + keepPop
        if p.calcType == 'var':
            allPop = convex_hull(allPop)

        allPop = calc_fitness(allPop, parameters)
        logging.info('calc_fitness finish')

        optLen = len(optPop)
        optPop = allPop[:optLen]

        for ind in optPop:
            # logging.info("formula: {}".format(ind.get_chemical_formula()))
            logging.info("optPop {strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format( strFrml=ind.get_chemical_formula(), **ind.info))

        # Calculate fingerprints
        logging.debug('calc_all_fingerprints begin')
        optPop = calc_all_fingerprints(optPop, parameters)
        logging.info('calc_all_fingerprints finish')
        for ind in optPop:
            initFp = np.atleast_2d(ind.info['initFp'])
            curFp = np.atleast_2d(ind.info['fingerprint'])
            relaxD = cdist(initFp, curFp)[0, 0]
            ind.info['relaxD'] = relaxD

        # os.chdir('Compare')
        # logging.info('optPop_duplicate start')
        logging.info('del_duplicate optPop begin')
        optPop = del_duplicate(optPop)
        logging.info('del_duplicate optPop finish')
        # logging.info('optPop_duplicate finish')
        # os.chdir('..')



        logging.info('pareto_front')
        # paretoPop = pareto_front(optPop + paretoPop, e=0)
        paretoPop = pareto_front(allPop, e=0)

        # logging.info('pareto_duplicate start')
        paretoPop = del_duplicate(paretoPop)
        # logging.info('pareto_duplicate finish')

        for ind in paretoPop:
            logging.info("paretoPop {strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format( strFrml=ind.get_chemical_formula(), **ind.info))
            # logging.info('paretoPop enthalpy: %s, fit1: %s, fit2: %s' %tuple([ind.info[attr] for attr in ('enthalpy', 'fitness1', 'fitness2')]))

        ### save good individuals
        logging.info('goodPop')
        goodPop = calc_dominators(allPop)
        goodPop = del_duplicate(goodPop)
        goodPop = sorted(goodPop, key=lambda x:x.info['dominators'])
        if p.calcType == 'fix':
            if len(goodPop) > p.popSize:
                goodPop = sorted(goodPop, key=lambda x:x.info['dominators'])[:p.popSize]
        elif p.calcType == 'var':
            goodPop = [ind for ind in goodPop if ind.info['ehull']<=0.1]

        # if len(goodPop) > p.popSize:
        #     goodPop = sorted(goodPop, key=lambda x:x.info['dominators'])[:p.popSize]

        ### keep best
        logging.info('keepPop')
        labels, keepPop = clustering(goodPop, p.saveGood)

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
        assert p.molMode, 'molMode should be True'

        if p.molType == 'fix':
            # inputMols = [Atoms(**molInfo) for molInfo in p.molList]
            initPop = build_mol_struct(p.initSize, p.symbols, p.formula, p.inputMols, p.molFormula, p.numFrml, p.spacegroup, fixCell=p.fixCell, setCellPar=p.setCellPar)


        logging.info("initPop length: {}".format(len(initPop)))

    else:
        bboPop = del_duplicate(optPop + keepPop)


        # renew volRatio
        volRatio = sum([calc_volRatio(ats) for ats in optPop])/len(optPop)
        p.volRatio = 0.5*(volRatio + p.volRatio)
        logging.debug("p.volRatio: {}".format(p.volRatio))

        if p.setAlgo == 'bayes':
            mainAlgo = Kriging(bboPop, curGen, parameters)
            mainAlgo.generate()
            mainAlgo.fit_gp()
            mainAlgo.select(enFilter=True)
            initPop = mainAlgo.get_nextPop()

        # elif p.setAlgo == 'mlpot':
        #     mainAlgo = PotKriging(bboPop, curGen, parameters)

        #     mainAlgo.generate()
        #     mainAlgo.fit_gp()
        #     mainAlgo.select()
        #     initPop = mainAlgo.get_nextPop()


        elif p.setAlgo == 'bbo':

        ### BBO
            mainAlgo = BBO(bboPop, parameters)
            mainAlgo.bbo_cutcell()
            initPop = mainAlgo.get_bboPop()

        logging.debug("initLen: {}".format(len(initPop)))
        write_results(initPop, curGen, 'test_init')

        # check mol crystal
        if p.chkMol:
            logging.info("check mols")
            initPop = check_mol_pop(initPop, p.inputMols, p.bondRatio)
            logging.info("check survival: {}".format(len(initPop)))

        if len(initPop) < p.popSize:
            logging.info("random structures out of Kriging")
            if p.molType == 'fix':
                # inputMols = [Atoms(**molInfo) for molInfo in p.molList]
                initPop.extend(build_mol_struct(p.popSize - len(initPop), p.symbols, p.formula, p.inputMols, p.molFormula, p.numFrml, p.spacegroup, fixCell=p.fixCell, setCellPar=p.setCellPar))



    # # fix cell
    if p.fixCell:
        for ind in initPop:
            ind.set_cell(p.setCellPar, scale_atoms=True)

    # read seeds
    if initial:
        initPop.extend(read_seeds(parameters))
    else:
        initPop.extend(read_seeds(parameters, 'Seeds/POSCARS_{}'.format(curGen)))

    ### Initail check
    initPop = check_dist(initPop, p.dRatio)

    ### Initial fingerprint
    for ind in initPop:
        if 'fingerprint' in ind.info.keys():
            initFp = ind.info['fingerprint']
        else:
            initFp = calc_one_fingerprint(ind, parameters)
        ind.info['initFp'] = initFp

    ### Save Initial
    write_results(initPop, curGen, 'init')
    if not os.path.exists('calcFold'):
        os.mkdir('calcFold')
    os.chdir('calcFold')

    if curStat['curGen'] > p.numGen:
        pass
    else:
        if p.calculator == 'gulp':
            runJobs = calc_gulp_parallel(p.calcNum, initPop, parameters)
        elif p.calculator == 'vasp':
            # runJobs = calc_vasp_parallel(setincar.calcs, initPop, parameters)
            runJobs = calc_vasp_parallel(p.calcNum, initPop, parameters)
        os.chdir(p.workDir)

        # logging.debug('runJobs: {}'.format(runJobs))
        for n, jobID in enumerate(runJobs):
            if type(jobID) is bytes:
                runJobs[n] = jobID.decode()

        curStat['runJobs'] = runJobs

    return curStat


### Main ###
parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="print debug information", action='store_true', default=False)
args = parser.parse_args()
if args.debug:
    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(message)s")
    logging.info('Debug mode')
else:
    logging.basicConfig(filename='log.txt', level=logging.INFO, format="%(message)s")

parameters = read_parameters('input.yaml')
numGen = parameters['numGen']
waitTime = parameters['waitTime']
# for key, val in parameters.items():
#     # setattr(p, key, val)
#     vars()[key] = val

while True:
    doneOr, curStat = check_jobs()
    logging.info(time.ctime())
    logging.info("doneOr:%s"%(doneOr))

    if doneOr:
        # All Done

        curStat = csp_loop(curStat, parameters)
        with open('currentStat.json', 'w') as f:
            json.dump(curStat, f)

        if curStat['curGen'] > numGen:
            logging.info("All Done")
            break

    time.sleep(waitTime)
