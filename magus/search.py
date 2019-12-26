import random, logging, os, sys, shutil, time, json
import argparse
import numpy as np
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from .localopt import VaspCalculator,XTBCalculator,LJCalculator,EMTCalculator,GULPCalculator
from .initstruct import BaseGenerator,read_seeds,VarGenerator,build_mol_struct
from .writeresults import write_dataset, write_results, write_traj
from .readparm import read_parameters
from .utils import *
import copy
from .queue import JobManager
from .setfitness import calc_fitness
from .renew import BaseEA
#ML module
from .machinelearning import LRmodel

class Magus:
    def __init__(self,parameters):
        self.parameters=parameters
        if self.parameters.calcType == 'fix':
            self.Generator=BaseGenerator(parameters)
        elif self.parameters.calcType == 'var':
            self.Generator=VarGenerator(parameters)
        # molecule mode
        if self.parameters.molMode:
            self.inputMols = [Atoms(**molInfo) for molInfo in self.parameters.molList]
        self.Algo=BaseEA(parameters)
        if self.parameters.calculator == 'vasp':
            self.MainCalculator = VaspCalculator(parameters)
        elif self.parameters.calculator == 'lj':
            self.MainCalculator = LJCalculator(parameters)
        elif self.parameters.calculator == 'emt':
            self.MainCalculator = EMTCalculator(parameters)
        elif self.parameters.calculator == 'gulp':
            self.MainCalculator = GULPCalculator(parameters)
        elif self.parameters.calculator == 'xtb':
            self.MainCalculator = XTBCalculator(parameters)
        self.ML=LRmodel(parameters)
        self.get_fitness=calc_fitness
        self.pop=[]
        self.goodPop=[]
        self.keepPop=[]
        self.curgen=1

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
        # if not os.path.exists("calcFold"):
        #     os.mkdir("calcFold")
        self.parameters.resultsDir=os.path.join(self.parameters.workDir,'results')
        shutil.copy("allParameters.yaml", "results/allParameters.yaml")
        logging.info("===== Generation {} =====".format(self.curgen))
        if not self.parameters.molMode:
            initPop = self.Generator.Generate_pop(self.parameters.initSize)
            if self.parameters.calcType == 'var':
                p_=copy.deepcopy(self.parameters)
                # Generate simple substance in variable mode
                for sybl in self.parameters.symbols:
                    p_.symbols=[sybl]
                    p_.formula=np.array([[1]])
                    p_.numFrml = list(range(1,p_.maxAt+1))
                    g_=BaseGenerator(p_)
                    initPop.extend(g_.Generate_pop(p_.eleSize))
        else:
            initPop = build_mol_struct(self.parameters.initSize, self.parameters.symbols, self.parameters.formula, self.inputMols, self.parameters.molFormula, self.parameters.numFrml, self.parameters.spacegroup, fixCell=self.parameters.fixCell, setCellPar=self.parameters.setCellPar)
        logging.info("initPop length: {}".format(len(initPop)))

        write_results(initPop, self.curgen, 'init',self.parameters.resultsDir)

        relaxPop=self.MainCalculator.relax(initPop)
        write_results(relaxPop, self.curgen, 'debug', self.parameters.resultsDir)

        logging.info("check distance")
        relaxPop = check_dist(relaxPop, self.parameters.dRatio)
        logging.info("check survival: {}".format(len(relaxPop)))


        if self.parameters.chkMol:
            logging.info("check mols")
            relaxPop = check_mol_pop(relaxPop, self.inputMols, self.parameters.bondRatio)
            logging.info("check survival: {}".format(len(relaxPop)))

        if self.parameters.calcType == 'var':
            relaxPop = convex_hull(relaxPop)

        self.get_fitness(relaxPop, mode=self.parameters.calcType)
        for ind in relaxPop:
            logging.info("{strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format(strFrml=ind.get_chemical_formula(), **ind.info))
        write_results(relaxPop, self.curgen, 'gen', self.parameters.resultsDir)

        logging.debug("Calculating fingerprint")
        self.ML.get_fp(relaxPop)
        if self.parameters.mlRelax:
            self.ML.updatedataset(relaxPop)
            self.ML.train()
            logging.info("loss:\nenergy_mse:{}\tenergy_r2:{}\tforce_mse:{}\tforce_r2:{}".format(*self.ML.get_loss(relaxPop)[:4]))
        self.pop=copy.deepcopy(relaxPop)


    def Onestep(self):
        self.curgen+=1
        logging.info("===== Generation {} =====".format(self.curgen))
        relaxPop = self.pop
        goodPop = self.goodPop
        keepPop = self.keepPop
        # try:
        #     goodPop = ase.io.read("{}/good.traj".format(self.parameters.resultsDir), format='traj', index=':')
        #     keepPop = ase.io.read("{}/keep{}.traj".format(self.parameters.resultsDir, self.curgen-1), format='traj', index=':')
        # except:
        #     goodPop = list()
        #     keepPop = list()

        for Pop in [relaxPop,goodPop,keepPop]:
            self.get_fitness(Pop)
        logging.info('calc_fitness finish')


        # Calculate fingerprints
        logging.debug('calc_all_fingerprints begin')
        self.ML.get_fp(relaxPop)
        logging.info('calc_all_fingerprints finish')

        logging.info('del_duplicate relaxPop begin')
        relaxPop = del_duplicate(relaxPop, symprec=self.parameters.symprec)
        logging.info('del_duplicate relaxPop finish')


        ### save good individuals
        logging.info('goodPop')
        goodPop = calc_dominators(relaxPop+goodPop)
        goodPop = del_duplicate(goodPop, symprec=self.parameters.symprec)
        goodPop = sorted(goodPop, key=lambda x:x.info['dominators'])

        if len(goodPop) > self.parameters.popSize:
            goodPop = goodPop[:self.parameters.popSize]

        ### keep best
        logging.info('keepPop')
        labels, keepPop = clustering(goodPop, self.parameters.saveGood)

        ### write results
        # write_results(relaxPop, self.curgen, 'gen', self.parameters.resultsDir)
        write_results(goodPop, '', 'good',self.parameters.resultsDir)
        write_results(keepPop, self.curgen, 'keep',self.parameters.resultsDir)
        shutil.copy('{}/log.txt'.format(self.parameters.workDir), '{}/results/log.txt'.format(self.parameters.workDir))

        ### write dataset
        write_dataset(relaxPop)

        curPop = del_duplicate(relaxPop + keepPop, symprec=self.parameters.symprec)

        # renew volRatio
        volRatio = sum([calc_volRatio(ats) for ats in relaxPop])/len(relaxPop)
        self.Generator.updatevolRatio(0.5*(volRatio + self.Generator.volRatio))
        logging.debug("volRatio: {}".format(self.Generator.volRatio))

        self.Algo.generate(curPop)
        initPop=self.Algo.select()
        ### Initail check
        # initPop = check_dist(initPop, self.parameters.dRatio)
        logging.debug("Generated by Algo: {}".format(len(initPop)))

        if len(initPop) < self.parameters.popSize:
            logging.info("random structures")
            initPop.extend(self.Generator.Generate_pop(self.parameters.initSize-len(initPop)))



        ### Initial fingerprint
        # self.ML.get_fp(initPop)

        ### Save Initial
        write_results(initPop,self.curgen, 'init',self.parameters.resultsDir)


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
            relaxPop = self.MainCalculator.relax(initPop)

        # save raw date before checking
        write_results(relaxPop, self.curgen, 'raw',self.parameters.resultsDir)

        logging.info("check distance")
        relaxPop = check_dist(relaxPop, self.parameters.dRatio)
        logging.info("check survival: {}".format(len(relaxPop)))

        if self.parameters.chkMol:
            logging.info("check mols")
            relaxPop = check_mol_pop(relaxPop, self.inputMols, self.parameters.bondRatio)
            logging.info("check survival: {}".format(len(relaxPop)))

        if self.parameters.calcType == 'var':
            relaxPop, goodPop, keepPop = convex_hull_pops(relaxPop, goodPop, keepPop)

        self.get_fitness(relaxPop)
        for ind in relaxPop:
            logging.info("{strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format(strFrml=ind.get_chemical_formula(), **ind.info))
        write_results(relaxPop, self.curgen, 'gen',self.parameters.resultsDir)

        self.ML.get_fp(relaxPop)
        if self.parameters.mlRelax:
            self.ML.updatedataset(relaxPop)
            self.ML.train()
            scfPop = self.MainCalculator.scf(relaxPop)
            logging.info("loss:\nenergy_mse:{}\tenergy_r2:{}\tforce_mse:{}\tforce_r2:{}".format(*self.ML.get_loss(scfPop)[:4]))

        self.pop=copy.deepcopy(relaxPop)
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
