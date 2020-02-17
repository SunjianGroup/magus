import random, logging, os, sys, shutil, time, json
import argparse
import numpy as np
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from .localopt import VaspCalculator,XTBCalculator,LJCalculator,EMTCalculator,GULPCalculator
from .initstruct import BaseGenerator,read_seeds,VarGenerator,build_mol_struct
from .writeresults import write_dataset, write_results, write_traj
from .readparm import *
from .utils import *
import copy
from .queue import JobManager
from .setfitness import calc_fitness
from .renew import BaseEA, BOEA
#ML module
from .machinelearning import LRmodel
from .offspring_creator import PopGenerator
"""
Pop:class,poplulation
pop:list,a list of atoms
Population:Population(pop) --> Pop
"""
class Magus:
    def __init__(self,parameters):
        self.parameters = parameters
        self.Generator = get_atoms_generator(parameters)
        self.Algo = get_pop_generator(parameters)
        self.MainCalculator = get_calculator(parameters)
        self.Population = get_population(parameters)
        self.ML=LRmodel(parameters)
        self.get_fitness = calc_fitness
        self.initPops = []
        self.relaxPops = []
        self.goodPop=[]
        self.keepPop=[]
        self.curgen=0

    def run(self):
        self.Initialize()
        for _ in range(self.parameters.numGen):
            self.Onestep()

    def Initialize(self):
        if os.path.exists("results"):
            i=1
            while os.path.exists("results{}".format(i)):
                i+=1
            shutil.move("results", "results{}".format(i))
        os.mkdir("results")

        shutil.copy("allParameters.yaml", "results/allParameters.yaml")
        logging.info("===== Generation {} =====".format(self.curgen))
        if not self.parameters.molMode:
            initpop = self.Generator.Generate_pop(self.parameters.initSize)
            initPop = self.Population(initpop,'initpop',self.curgen)
            if self.parameters.calcType == 'var':
                p_=copy.deepcopy(self.parameters)
                # Generate simple substance in variable mode
                for n, sybl in enumerate(self.parameters.symbols):
                    p_.symbols = [sybl]
                    p_.formula = [1]
                    p_.numFrml = list(range(1,p_.maxAt+1))
                    g_=BaseGenerator(p_)
                    elePop = g_.Generate_pop(p_.eleSize)
                    eleFrml = np.zeros(len(self.parameters.symbols))
                    eleFrml[n] = 1
                    eleFrml = eleFrml.astype(np.int)
                    for eleInd in elePop:
                        eleInd.info['symbols'] = self.parameters.symbols
                        tmpFrml = eleFrml*len(eleInd)
                        eleInd.info['formula'] = tmpFrml.tolist()
                        eleInd.info['numOfFormula'] = 1
                    initPop.extend(elePop)
        else:
            initPop = build_mol_struct(self.parameters.initSize, self.parameters.symbols, self.parameters.formula, self.inputMols, self.parameters.molFormula, self.parameters.numFrml, self.parameters.spacegroup, fixCell=self.parameters.fixCell, setCellPar=self.parameters.setCellPar)
        logging.info("initPop length: {}".format(len(initPop)))
        initPop.save('init')
        self.initPops.append(initPop)
        relaxpop = self.MainCalculator.relax(initPop.frames)
        relaxPop = self.Population(relaxpop,'relaxpop',self.curgen)
        relaxPop.save('raw')
        self.relaxPops.append(relaxPop)

    def Onestep(self):
        relaxPop = self.relaxPops[-1]
        goodPop = self.goodPops
        keepPop = self.keepPops

        relaxPop.check()
        
        if self.parameters.chkMol:
            logging.info("check mols")
            relaxPop = check_mol_pop(relaxPop, self.inputMols, self.parameters.bondRatio)
            logging.info("check survival: {}".format(len(relaxPop)))

        relaxPop.del_duplicate()

        if self.parameters.calcType == 'var':
            relaxPop, goodPop, keepPop = convex_hull_pops(relaxPop, goodPop, keepPop)

        for Pop in [relaxPop,goodPop,keepPop]:
            self.get_fitness(Pop, mode=self.parameters.calcType)
        logging.info('calc_fitness finish')

        for ind in relaxPop:
            logging.info("{strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format(strFrml=ind.get_chemical_formula(), **ind.info))

        # Calculate fingerprints
        logging.debug('calc_all_fingerprints begin')
        self.ML.get_fp(relaxPop)
        if self.parameters.mlRelax:
            self.ML.updatedataset(relaxPop)
            self.ML.train()
            scfPop = self.MainCalculator.scf(relaxPop)
            logging.info("loss:\nenergy_mse:{}\tenergy_r2:{}\tforce_mse:{}\tforce_r2:{}".format(*self.ML.get_loss(scfPop)[:4]))
        logging.info('calc_all_fingerprints finish')
        # Write relaxPop
        relaxPop.save('gen')

        ### save good individuals
        logging.info('goodPop')
        goodPop = relaxPop + goodPop + keepPop
        goodPop.del_duplicate()
        goodPop.select(self.parameters.popSize)

        ### keep best
        logging.info('keepPop')
        _, keeppop = goodPop.clustering(self.parameters.saveGood)
        keepPop = self.Population(keeppop,'keeppop',self.curgen)
        ### write good and keep pop
        goodPop.save('good','')
        goodPop.save('savegood')
        keepPop.save('keep')
        shutil.copy('{}/log.txt'.format(self.parameters.workDir), '{}/results/log.txt'.format(self.parameters.workDir))

        curPop = relaxPop + keepPop
        curPop.del_duplicate()

        # renew volRatio
        volRatio = relaxPop.get_volRatio()
        self.Generator.updatevolRatio(0.5*(volRatio + self.Generator.volRatio))

        initpop = self.Algo.next_pop(curPop)
        ### Initail check
        # initPop = check_dist(initPop, self.parameters.dRatio)
        logging.debug("Generated by Algo: {}".format(len(initpop)))

        if len(initpop) < self.parameters.popSize:
            logging.info("random structures")
            initpop.extend(self.Generator.Generate_pop(self.parameters.popSize-len(initpop)))


        self.curgen+=1
        logging.info("===== Generation {} =====".format(self.curgen))


        ### Initial fingerprint
        # self.ML.get_fp(initPop)

        ### Save Initial
        initPop = self.Population(initpop,'initpop',self.curgen)
        initPop.save()

        if self.parameters.mlRelax:
            ### mlrelax
            for _ in range(10):
                relaxPop = self.ML.relax(initPop)
                relaxPop = check_dist(relaxPop, self.parameters.dRatio)
                scfPop = self.MainCalculator.scf(relaxPop)
                loss = self.ML.get_loss(scfPop)
                if loss[1]>0.8:
                    logging.info('ML Gen{}\tEnergy Error:{}'.format(_,loss[1]))
                    break
                logging.info('QAQ ML Gen{}\tEnergy Error:{}'.format(_,loss[1]))
                self.ML.updatedataset(scfPop)
                write_results(self.ML.dataset,'','dataset',self.parameters.resultsDir)
                self.ML.train()

            else:
                relaxPop = self.MainCalculator.relax(initPop)
                logging.info('Turn to main calculator')
        else:
            relaxpop = self.MainCalculator.relax(initPop.frames)
            relaxPop = self.Population(relaxpop,'relaxpop',self.curgen)

        # save raw date before checking
        relaxPop.save('raw')

        self.relaxPops.append(relaxPop)
        self.goodPop=copy.deepcopy(goodPop)
        self.keepPop=copy.deepcopy(keepPop)


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

m=Magus(p)
m.run()
