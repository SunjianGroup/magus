# Serial MultiObject Optimize
from __future__ import print_function, division
import random, logging, os, sys, shutil, time, json
import numpy as np
import argparse
from scipy.spatial.distance import cdist
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from .localopt import generate_calcs, calc_gulp, calc_vasp, generate_mopac_calcs, calc_mopac, generate_cp2k_calcs, calc_cp2k, generate_cp2k_params, calc_cp2k_params, generate_xtb_calcs, calc_xtb
from .renewstruct import Kriging, BBO, pareto_front, convex_hull, calc_dominators
from .initstruct import read_seeds, build_mol_struct
# from .readvasp import *
from .setfitness import calc_fitness
from .writeresults import write_dataset, write_results
from .fingerprint import calc_all_fingerprints, calc_one_fingerprint, clustering
from .bayes import atoms_util
from .readparm import read_parameters
from .utils import *
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="print debug information", action='store_true', default=False)
args = parser.parse_args()
if args.debug:
    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(message)s")
    logging.info('Debug mode')
else:
    logging.basicConfig(filename='log.txt', level=logging.INFO, format="%(message)s")
parameters = read_parameters('input.yaml')
p = EmptyClass()
for key, val in parameters.items():
    setattr(p, key, val)
    # vars()[key] = val
if p.molMode:
    p.inputMols = [Atoms(**molInfo) for molInfo in p.molList]


for curGen in range(1, p.numGen+1):
    logging.info("===== Generation {} =====".format(curGen))

    # 1st generation
    if curGen == 1:
        if os.path.exists("results"):
            for i in range(1, 100):
                if not os.path.exists("results{}".format(i)):
                    shutil.move("results", "results{}".format(i))
                    break

        os.mkdir("results")
        shutil.copy("allParameters.yaml", "results/allParameters.yaml")

        assert p.molMode, 'molMode should be True'

        if p.molType == 'fix':
            # inputMols = [Atoms(**molInfo) for molInfo in p.molList]
            initPop = build_mol_struct(p.initSize, p.symbols, p.formula, p.inputMols, p.molFormula, p.numFrml, p.spacegroup, fixCell=p.fixCell, setCellPar=p.setCellPar)



        logging.info("initPop length: {}".format(len(initPop)))
        initPop.extend(read_seeds(parameters))

    else:
        bboPop = del_duplicate(optPop + keepPop)
        logging.debug('bboPop: {}'.format(len(bboPop)))

        # renew volRatio
        logging.debug("Renew volRatio")
        volRatio = sum([calc_volRatio(ats) for ats in optPop])/len(optPop)
        p.volRatio = 0.5*(volRatio + p.volRatio)
        logging.debug("p.volRatio: {}".format(p.volRatio))

        if p.setAlgo == 'bayes':
            mainAlgo = Kriging(bboPop, curGen, parameters)

            mainAlgo.generate()
            mainAlgo.fit_gp()
            if p.calculator in ['vasp', 'cp2k', 'xtb']:
                mainAlgo.select()
            elif p.calculator in ['gulp', 'mopac']:
                mainAlgo.select(enFilter=False)
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
        # if p.chkMol:
        #     logging.info("check mols")
        #     initPop = check_mol_pop(initPop, p.inputMols, p.bondRatio)
        #     logging.info("check survival: {}".format(len(initPop)))


        if len(initPop) < p.popSize:
            logging.info("random structures out of Kriging")
            if p.molType == 'fix':
                # inputMols = [Atoms(**molInfo) for molInfo in p.molList]
                initPop.extend(build_mol_struct(p.popSize - len(initPop), p.symbols, p.formula, p.inputMols, p.molFormula, p.numFrml, p.spacegroup, fixCell=p.fixCell, setCellPar=p.setCellPar))

        initPop.extend(read_seeds(parameters, 'Seeds/POSCARS_{}'.format(curGen)))

    # fix cell
    # if p.fixCell:
    #     for ind in initPop:
    #         ind.set_cell(p.setCellPar, scale_atoms=True)

    ### Initail check
    initPop = check_dist(initPop, p.dRatio)

    ### Save Initial
    write_results(initPop, curGen, 'init')

    ### Initial fingerprint
    for ind in initPop:
        if 'fingerprint' in ind.info.keys():
            initFp = ind.info['fingerprint']
        else:
            initFp = calc_one_fingerprint(ind, parameters)
        ind.info['initFp'] = initFp


    ### Calculation
    if not os.path.exists('calcFold'):
        # os.mkdir('calcFold')
        shutil.copytree('inputFold', 'calcFold')
    os.chdir('calcFold')
    if p.calculator == 'gulp':
        optPop = calc_gulp(p.calcNum, initPop, p.pressure, p.exeCmd, "{}/inputFold".format(p.workDir))
    elif p.calculator == 'vasp':
        calcs = generate_calcs(p.calcNum, parameters)
        optPop = calc_vasp(calcs, initPop)
    elif p.calculator == 'mopac':
        calcs = generate_mopac_calcs(p.calcNum, parameters)
        optPop = calc_mopac(calcs, initPop, p.pressure)
    elif p.calculator == 'xtb':
        calcs = generate_xtb_calcs(p.calcNum, parameters)
        optPop = calc_xtb(calcs, initPop, p.pressure, p.epsArr, p.stepArr, p.maxRelaxTime, p.maxRelaxStep, p.optimizer)
    elif p.calculator == 'cp2k':
        if p.fastcp2k:
            calcs = generate_cp2k_calcs(p.calcNum, parameters)
            optPop = calc_cp2k(calcs, initPop, p.pressure, p.epsArr, p.stepArr, p.maxRelaxTime, p.maxRelaxStep, p.optimizer)
        else:
            calcParam = generate_cp2k_params(p.calcNum, parameters)
            optPop = calc_cp2k_params(calcParam, initPop, p.pressure, p.epsArr, p.stepArr, p.maxRelaxTime, p.maxRelaxStep, p.optimizer)

    os.chdir(p.workDir)

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
    # allPop = optPop + paretoPop + goodPop
    if p.calcType == 'var':
        allPop = convex_hull(allPop)

    allPop = calc_fitness(allPop, parameters)
    logging.info('calc_fitness finish')

    optLen = len(optPop)
    # paretoLen = len(paretoPop)
    optPop = allPop[:optLen]
    # paretoPop = allPop[optLen:optLen+paretoLen]
    # goodPop = allPop[optLen:]

    for ind in optPop:
        # logging.debug("formula: {}".format(ind.get_chemical_formula()))
        logging.info("optPop {strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format( strFrml=ind.get_chemical_formula(), **ind.info))

    # Calculate fingerprints
    optPop = calc_all_fingerprints(optPop, parameters)
    for ind in optPop:
        initFp = np.atleast_2d(ind.info['initFp'])
        curFp = np.atleast_2d(ind.info['fingerprint'])
        relaxD = cdist(initFp, curFp)[0, 0]
        ind.info['relaxD'] = relaxD

    optPop = del_duplicate(optPop)

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
