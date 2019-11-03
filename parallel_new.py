from initstruct import BaseGenerator
from .localopt import generate_calcs, calc_gulp_parallel, calc_vasp_parallel, jobs_stat, read_parallel_results

import random, logging, os, sys, shutil, time, json
import argparse
import numpy as np
from scipy.spatial.distance import cdist
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from .localopt import generate_calcs, calc_gulp_parallel, calc_vasp_parallel, jobs_stat, read_parallel_results
from .renewstruct import del_duplicate, Kriging, PotKriging, BBO, pareto_front, convex_hull, check_dist, calc_dominators
from .initstruct import BaseGenerator,read_seeds,VarGenerator
from .setfitness import calc_fitness
from .writeresults import write_dataset, write_results, write_traj
from .fingerprint import calc_all_fingerprints, calc_one_fingerprint, clustering
from .bayes import atoms_util
from .readparm import read_parameters
from .utils import EmptyClass, calc_volRatio
import copy
from queue import JobManager



class Magus:
    def __init__(self,parameters):
        self.parameters=parameters
        self.bjobs=JobManager()
        self.Generator=BaseGenerator(parameters)
        self.MainCalculator=calc_vasp_parallel

    def Initialize(self):
        if os.path.exists("results"):
            i=0
            while os.path.exists("results{}".format(i)):
                i+=1
            shutil.move("results", "results{}".format(i))
        os.mkdir("results")
        shutil.copy("allParameters.yaml", "results/allParameters.yaml")

        logging.info("===== Initializition =====")
        initPop = self.Generator.Generate_pop(self.parameters.initSize)
        logging.info("initPop length: {}".format(len(initPop)))

        #initPop.extend(read_seeds(parameters))
        write_results(initPop, 'init')

        if not os.path.exists('calcFold'):
            os.mkdir('calcFold')
        os.chdir('calcFold')

        self.MainCalculator(self.parameters.calcNum, initPop, parameters, self.bjobs)
        self.bjobs.WaitJobsDone(self.parameters.waitTime)
        os.chdir(p.workDir)
        self.ML.updatedatabase(initPop)
        self.ML.train()

    def Onestep(self):
        self.GetNewPopulation()
        for _ in range(10):
            self.ML.Relax()
            self.MainCalculator(p.calcNum, initPop, parameters, self.bjobs)
            if self.ML.getloss(initPop)>0.2:
                self.ML.updatedatabase(initPop)
                self.ML.retrain()
            else:
                break
        

        

parameters=getparameters
m=Magus(parameters)
m.initialize()
