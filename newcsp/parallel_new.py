import random, logging, os, sys, shutil, time, json
import argparse
import numpy as np
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from .localopt import VaspCalculator,xtbCalculator
from .initstruct import BaseGenerator,read_seeds,VarGenerator
from .writeresults import write_dataset, write_results, write_traj
from .readparm import read_parameters
from .utils import EmptyClass, calc_volRatio
import copy
from .queue import JobManager
from .setfitness import calc_fitness
from .renew import Kriging
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
        self.MainCalculator=VaspCalculator(parameters)
        self.ML=LRmodel(parameters)
        self.get_fitness=calc_fitness
        self.pop=[]

    def Initialize(self):
        # if os.path.exists("results"):
        #     i=0
        #     while os.path.exists("results{}".format(i)):
        #         i+=1
        #     shutil.move("results", "results{}".format(i))
        # os.mkdir("results")
        # self.parameters.resultsDir=os.path.join(self.parameters.workDir,'results')
        # shutil.copy("allParameters.yaml", "results/allParameters.yaml")

        # logging.info("===== Initializition =====")
        # initPop = self.Generator.Generate_pop(self.parameters.initSize)
        # logging.info("initPop length: {}".format(len(initPop)))

        # #initPop.extend(read_seeds(parameters))
        # write_results(initPop, 'init')

        # relaxPop=self.MainCalculator.relax(initPop)

        # """
        # logging.info("check distance")
        # relaxPop = check_dist(relaxPop, self.parameters.dRatio)
        # logging.info("check survival: {}".format(len(relaxPop)))
        # """
        
        # self.ML.get_fp(relaxPop)
        # self.get_fitness(relaxPop)

        # write_results(relaxPop, 'relax',self.parameters.resultsDir)
        self.parameters.resultsDir=os.path.join(self.parameters.workDir,'results')
        relaxPop=ase.io.read('results/relax.traj',':')
        self.ML.get_fp(relaxPop)
        self.get_fitness(relaxPop)
        self.ML.updatedataset(relaxPop)
        self.ML.train()
        logging.info("loss:{}".format(self.ML.get_loss(relaxPop)))
        self.pop=copy.deepcopy(relaxPop)
        

    def Onestep(self):
        curPop=self.pop
        initPop=self.Algo.generate(curPop)
        logging.info("generate:{}".format(len(initPop)))
        for _ in range(10):
            write_results(initPop,'gen-init',self.parameters.resultsDir)
            logging.info("ML relax {}".format(_))
            relaxPop=self.ML.relax(initPop)
            write_results(relaxPop,'gen-relax',self.parameters.resultsDir)
            self.get_fitness(relaxPop)
            relaxPop=sorted(relaxPop,key=lambda x:x.info['fitness1'])
            selectPop=relaxPop[:min(len(relaxPop),self.parameters.popSize)]
            logging.info('shabi{}'.format(len(selectPop)))
            selectPop=self.MainCalculator.scf(selectPop)
            write_results(selectPop,'select',self.parameters.resultsDir)
            logging.info("MSE:{}".format(self.ML.get_loss(selectPop)))
            if self.ML.get_loss(selectPop)[0]>0.1:
                self.ML.updatedataset(selectPop)
                self.ML.train()
            else:
                break
    

        
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
m.Initialize()
m.Onestep()