import random, logging, os, sys, shutil, time, json
import argparse
import numpy as np
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from .localopt import VaspCalculator,xtbCalculator,LJCalculator
from .initstruct import BaseGenerator,read_seeds,VarGenerator
from .writeresults import write_dataset, write_results, write_traj
from .readparm import read_parameters
from .utils import EmptyClass, calc_volRatio
import copy
from .queue import JobManager
from .setfitness import calc_fitness
from .renew import Kriging, calc_dominators, del_duplicate, clustering, check_dist
#ML module
from .machinelearning import LRmodel
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction

class Magus:
    def __init__(self,parameters):
        self.parameters=parameters
        self.Generator=BaseGenerator(parameters)
        self.Algo=Kriging(parameters)
        #self.MainCalculator=xtbCalculator(parameters)
        #self.MainCalculator=VaspCalculator(parameters)
        self.MainCalculator = LJCalculator(parameters)
        self.ML=LRmodel(parameters)
        self.get_fitness=calc_fitness
        self.pop=[]
        self.curgen=0

    def run(self):
        self.Initialize()
        for _ in range(self.parameters.numGen):
            self.Onestep()

    def Initialize(self):
        if os.path.exists("results"):
            i=0
            while os.path.exists("results{}".format(i)):
                i+=1
            shutil.move("results", "results{}".format(i))
        os.mkdir("results")
        self.parameters.resultsDir=os.path.join(self.parameters.workDir,'results')
        shutil.copy("allParameters.yaml", "results/allParameters.yaml")

        logging.info("===== Initializition =====")
        initPop = self.Generator.Generate_pop(self.parameters.initSize)
        logging.info("initPop length: {}".format(len(initPop)))

        write_results(initPop, 'init0',self.parameters.resultsDir)

        relaxPop=self.MainCalculator.relax(initPop)
        write_results(relaxPop, 'debug0',self.parameters.resultsDir)
        
        logging.info("check distance")
        relaxPop = check_dist(relaxPop, self.parameters.dRatio)
        logging.info("check survival: {}".format(len(relaxPop)))

        self.get_fitness(relaxPop)
        for ind in relaxPop:
            logging.info("optPop {strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format(strFrml=ind.get_chemical_formula(), **ind.info))
        write_results(relaxPop, 'relax0',self.parameters.resultsDir)

        self.ML.get_fp(relaxPop)
        self.ML.updatedataset(relaxPop)
        self.ML.train()
        logging.info("loss:{}".format(self.ML.get_loss(relaxPop)))
        self.pop=copy.deepcopy(relaxPop)
        

    def Onestep(self):
        self.curgen+=1
        logging.info("===== Generation {} =====".format(self.curgen))
        optPop=self.pop
        try:
            goodPop = ase.io.read("{}/good.traj".format(self.parameters.resultsDir), format='traj', index=':')
            keepPop = ase.io.read("{}/keep{}.traj".format(self.parameters.resultsDir, self.curgen-1), format='traj', index=':')
        except:
            goodPop = list()
            keepPop = list()

        for Pop in [optPop,goodPop,keepPop]:
            self.get_fitness(Pop)
        logging.info('calc_fitness finish')


        # Calculate fingerprints
        logging.debug('calc_all_fingerprints begin')
        self.ML.get_fp(optPop)
        logging.info('calc_all_fingerprints finish')

        logging.info('del_duplicate optPop begin')
        optPop = del_duplicate(optPop)
        logging.info('del_duplicate optPop finish')


        ### save good individuals
        logging.info('goodPop')
        goodPop = calc_dominators(optPop+goodPop)
        goodPop = del_duplicate(goodPop)
        goodPop = sorted(goodPop, key=lambda x:x.info['dominators'])

        if len(goodPop) > self.parameters.popSize:
            goodPop = goodPop[:self.parameters.popSize]

        ### keep best
        logging.info('keepPop')
        labels, keepPop = clustering(goodPop, self.parameters.saveGood)

        ### write results
        write_results(optPop,  'gen{}'.format(self.curgen),self.parameters.resultsDir)
        write_results(goodPop, 'good',self.parameters.resultsDir)
        write_results(keepPop, 'keep{}'.format(self.curgen),self.parameters.resultsDir)
        shutil.copy('log.txt', 'results/log.txt')

        ### write dataset
        write_dataset(optPop)

        bboPop = del_duplicate(optPop + keepPop)

        # renew volRatio
        volRatio = sum([calc_volRatio(ats) for ats in optPop])/len(optPop)
        self.Generator.updatevolRatio(0.5*(volRatio + self.Generator.volRatio))
        logging.debug("volRatio: {}".format(self.Generator.volRatio))

        initPop=self.Algo.generate(bboPop)

        if len(initPop) < self.parameters.popSize:
            logging.info("random structures out of Kriging")
            initPop.extend(self.Generator.Generate_pop(self.parameters.initSize-len(initPop)))

        ### Initail check
        initPop = check_dist(initPop, self.parameters.dRatio)

        ### Initial fingerprint
        self.ML.get_fp(initPop)

        ### Save Initial
        write_results(initPop,'init{}'.format(self.curgen),self.parameters.resultsDir)


        ### mlrelax
        for _ in range(10):
            relaxPop = self.ML.relax(initPop)
            scfPop = self.MainCalculator.scf(relaxPop)
            loss = self.ML.get_loss(scfPop)
            if loss[1]>0.8:
                logging.info('di {} dai , neng liang wu cha {} , ke yi de'.format(_,loss[1]))
                break
            logging.info('QAQ di {} dai le , neng liang wu cha {}'.format(_,loss[1]))
            self.ML.updatedataset(scfPop)
            self.ML.train()
            
        else:
            relaxPop = self.MainCalculator.relax(initPop)
            logging.info('hai shi yong di yi xing le')


        logging.info("check distance")
        relaxPop = check_dist(relaxPop, self.parameters.dRatio)
        logging.info("check survival: {}".format(len(relaxPop)))

        self.get_fitness(relaxPop)
        for ind in relaxPop:
            logging.info("optPop {strFrml} enthalpy: {enthalpy}, fit1: {fitness1}, fit2: {fitness2}".format(strFrml=ind.get_chemical_formula(), **ind.info))
        write_results(relaxPop, 'relax0',self.parameters.resultsDir)

        self.ML.get_fp(relaxPop)
        self.ML.updatedataset(relaxPop)
        self.ML.train()
        scfPop = self.MainCalculator.scf(relaxPop)
        logging.info("loss:{}".format(self.ML.get_loss(scfPop)))
        self.pop=copy.deepcopy(relaxPop)



        
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