import random, logging, os, sys, shutil, time, json
import argparse
import copy
import numpy as np
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from .initstruct import read_seeds
from .utils import *
from .machinelearning import LRmodel
from .parameters import magusParameters
from .writeresults import write_results
from .formatting.mtp import dump_cfg, load_cfg
from .calculator.mtp import MTPCalculator, TwostageMTPCalculator
from .machinelearning import MTPmodel
"""
Pop:class,poplulation
pop:list,a list of atoms
Population:Population(pop) --> Pop
"""

#TODO change read parameters
class Magus:
    def __init__(self,parameters):
        self.parameters = parameters.parameters
        self.Generator = parameters.get_AtomsGenerator()
        self.Algo = parameters.get_PopGenerator()
        self.MainCalculator = parameters.get_MainCalculator()
        self.Population = parameters.get_Population()
        self.MTPCalculator = TwostageMTPCalculator(self.MainCalculator, parameters.parameters)
        self.MLmodel = MTPmodel(parameters.parameters)
        self.curgen = 1
        self.bestlen = []
        self.kappa = 3.0

    def run(self):
        self.Initialize()
        for gen in range(self.parameters.numGen-1):
            self.Onestep()

    def mkdir_results(self):
        if os.path.exists("results"):
            i=1
            while os.path.exists("results{}".format(i)):
                i+=1
            shutil.move("results", "results{}".format(i))
        os.mkdir("results")

    def save_parameters(self):
        self.parameters.save('allparameters.yaml')
        self.parameters.save('results/allparameters.yaml')

    def read_seeds(self, curgen):
        seedpop = read_seeds(self.parameters, '{}/Seeds/POSCARS_{}'.format(self.parameters.workDir, curgen))
        seedPop = self.Population(seedpop,'seedpop',self.curgen)
        if self.parameters.chkSeed:
            seedPop.check()
        return seedPop

    def Initialize(self):
        self.mkdir_results()
        self.save_parameters()

        logging.info("===== Generation {} =====".format(self.curgen))
        # get initial pop
        initpop = self.Generator.Generate_pop(self.parameters.initSize,initpop=True)
        initPop = self.Population(initpop,'initpop',self.curgen)
        logging.info("initPop length: {}".format(len(initPop)))
        seedPop = self.read_seeds(self.curgen)
        initPop.extend(seedPop)
        initPop.save('initPop', self.curgen)

        scfpop = self.MainCalculator.scf(initPop.frames)
        scfPop = self.Population(scfpop,'scfPop',self.curgen)
        scfPop.check()
        scfPop.del_duplicate()
        scfPop.save('scfPop', self.curgen)

        self.MLmodel.updatedataset(scfPop.frames)
        self.MLmodel.train()
        logging.info("loss:\nenergy_rmse:{}\tenergy_r2:{}\n"
            "force_rmse:{}\tforce_r2:{}".format(*self.MLmodel.get_loss(scfPop.frames)[:4]))

        self.curPop = scfPop

        if self.parameters.goodSeed:
            logging.info("Please be careful when you set goodSeed=True. \nThe structures in {} will be add to relaxPop without relaxation.".format(self.parameters.goodSeedFile))
            goodseedpop = read_seeds(self.parameters,'{}/Seeds/{}'.format(self.parameters.workDir, self.parameters.goodSeedFile), goodSeed=self.parameters.goodSeed)
            scfPop.extend(goodseedpop)
            scfPop.del_duplicate()
            # relaxPop.check()

        self.bestPop = self.Population([],'bestPop')

        scfPop.calc_dominators()
        scfPop.save('gen', self.curgen)

        logging.info('construct goodPop')
        goodPop = scfPop
        goodPop.del_duplicate()
        goodPop.calc_dominators()
        goodPop.select(self.parameters.popSize)
        goodPop.save('good','')
        goodPop.save('savegood')
        self.goodPop = goodPop

        logging.info("best ind:")
        bestind = goodPop.bestind()
        self.bestPop.extend(bestind)
        self.bestlen.append(len(self.bestPop))
        for ind in bestind:
            logging.info("{strFrml} enthalpy: {enthalpy}, fit: {fitness}, dominators: {dominators}"\
                .format(strFrml=ind.atoms.get_chemical_formula(), **ind.info))
        self.bestPop.save('best','')

        logging.info('construct keepPop')
        _, keeppop = goodPop.clustering(self.parameters.saveGood)
        # initialize keepPop here, so that the inds have identity
        self.keepPop = self.Population(keeppop,'keepPop', self.curgen)
        self.keepPop.save('keep', self.curgen)

    def Onestep(self):
        #TODO make update parameters more reasonable
        self.update_parameters()
        kappa = self.kappa
        curPop = self.curPop
        goodPop = self.goodPop
        keepPop = self.keepPop
        self.curgen+=1
        logging.info("===== Generation {} =====".format(self.curgen))
        #######  get next Pop  #######
        # renew volRatio
        volRatio = curPop.get_volRatio()
        self.Generator.updatevolRatio(0.5*(volRatio + self.Generator.p.volRatio))

        initPop = self.Algo.next_Pop(curPop + keepPop)
        logging.info("Generated by Algo: {}".format(len(initPop)))
        if len(initPop) < self.parameters.popSize:
            logging.info("random structures:{}".format(self.parameters.popSize-len(initPop)))
            addpop = self.Generator.Generate_pop(self.parameters.popSize-len(initPop))
            logging.debug("addpop: {}".format(len(addpop)))
            initPop.extend(addpop)

        #read seeds
        seedpop = read_seeds(self.parameters, '{}/Seeds/POSCARS_{}'.format(self.parameters.workDir, self.curgen))
        seedPop = self.Population(seedpop,'seedpop',self.curgen)
        if self.parameters.chkSeed:
            seedPop.check()
        initPop.extend(seedPop)
        initPop.del_duplicate()
        # Save Initial
        initPop.save()

        #######  local relax by ML  #######
        relaxpop = self.MTPCalculator.relax(initPop.frames)
        relaxPop = self.Population(relaxpop,'relaxpop',self.curgen)
        relaxPop.save("mlraw", self.curgen)
        relaxPop.check()
        # find spg before delete duplicate
        relaxPop.find_spg()
        relaxPop.del_duplicate()
        relaxPop.save("mlgen", self.curgen)

        #######  compare target and predict energy  #######
        scfpop = self.MainCalculator.scf(relaxPop.frames)
        scfPop = self.Population(scfpop)
        logging.info("loss:\nenergy_mse:{:.5f}\tenergy_r2:{:.5f}\n"
            "force_mse:{:.5f}\tforce_r2:{:.5f}".format(*self.MLmodel.get_loss(scfPop.frames)[:4]))
        scfPop.save("scfmlgen", self.curgen)
    
        #######  test ML   #######
        testpop = ase.io.read('{}/test.traj'.format(self.parameters.workDir), ':')
        logging.info("loss:\n"
                "energy_mse:{:.5f}\tenergy_r2:{:.5f}\n"
                "force_mse:{:.5f}\tforce_r2:{:.5f}\n"
                "stress_mse:{:.5f}\tstress_r2:{:.5f}\n".format(*self.MLmodel.get_loss(testpop)))
        
        #######  goodPop and keepPop  #######
        logging.info('construct goodPop')
        # goodPop = relaxPop + goodPop + keepPop
        goodPop = scfPop + goodPop + keepPop # use true energy
        goodPop.del_duplicate()
        goodPop.calc_dominators()
        goodPop.select(self.parameters.popSize)
        goodPop.save('good','')

        logging.info("good ind:")
        for ind in goodPop.pop:
            logging.debug("{strFrml} enthalpy: {enthalpy}, fit: {fitness}, dominators: {dominators}"\
                .format(strFrml=ind.atoms.get_chemical_formula(), **ind.info))
        logging.info("best ind:")
        bestind = goodPop.bestind()
        self.bestPop.extend(bestind)
        self.bestlen.append(len(self.bestPop))
        for ind in bestind:
            logging.info("{strFrml} enthalpy: {enthalpy}, fit: {fitness}, dominators: {dominators}"\
                .format(strFrml=ind.atoms.get_chemical_formula(), **ind.info))
        self.bestPop.save('best','')
        #self.bestPop.save('best',self.curgen)
        # keep best
        logging.info('construct keepPop')
        _, keepPop = goodPop.clustering(self.parameters.saveGood)
        keepPop = self.Population(keepPop,'keeppop',self.curgen)
        keepPop.save('keep')

        curPop = relaxPop
        curPop.del_duplicate()
        curPop.save('gen')
        self.curPop = curPop
        self.goodPop = goodPop
        self.keepPop = keepPop

    def update_parameters(self):
        if self.MainCalculator.p.mode == 'parallel':
            with open('results/allparameters.yaml') as f:
                d = yaml.load(f)
            if d['MainCalculator']['queueName'] != self.MainCalculator.p.queueName:
                logging.warning('*****************\nBe careful, {} is replaced by {}\n*****************'.format(self.MainCalculator.p.queueName, d['MainCalculator']['queueName']))
                self.MainCalculator.p.queueName = d['MainCalculator']['queueName']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="print debug information", action='store_true', default=False)
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')
        logging.info('Debug mode')
    else:
        logging.basicConfig(filename='log.txt', level=logging.INFO, format="%(message)s")

    parameters = magusParameters('input.yaml')
    m=Magus(parameters)
    m.run()
