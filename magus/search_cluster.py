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
from .localopt import OrcaCalculator
"""
Pop:class,poplulation
pop:list,a list of atoms
Population:Population(pop) --> Pop
"""
#TODO build mol struct
#TODO change read parameters
class Magus:
    def __init__(self,parameters):
        self.parameters = parameters.parameters
        self.Generator = parameters.get_AtomsGenerator()
        self.Algo = parameters.get_PopGenerator()
        self.MainCalculator = parameters.get_MainCalculator()
        self.Population = parameters.get_Population()
        self.CheckCal = OrcaCalculator(self.parameters)
        if self.parameters.useml:
            self.ML = parameters.get_MLCalculator()

        self.curgen = 1
        self.bestlen = []
        self.allPop = self.Population([],'allPop')
        self.Population.allPop = self.allPop

    def run(self):
        self.Initialize()
        for gen in range(self.parameters.numGen-1):
            self.Onestep()
            #if gen > 5 and self.bestlen[gen] == self.bestlen[gen-5]:
            #    logging.info('converged')
            #    break

    def Initialize(self):
        if os.path.exists("results"):
            i=1
            while os.path.exists("results{}".format(i)):
                i+=1
            shutil.move("results", "results{}".format(i))
        os.mkdir("results")
        self.parameters.save('allparameters.yaml')
        self.parameters.save('results/allparameters.yaml')

        #shutil.copy("allParameters.yaml", "results/allParameters.yaml")
        logging.info("===== Generation {} =====".format(self.curgen))
        initpop = self.Generator.Generate_pop(self.parameters.initSize,initpop=True)

        initPop = self.Population(initpop,'initpop',self.curgen)
        logging.info("initPop length: {}".format(len(initPop)))

        #read seeds
        seedpop = read_seeds(self.parameters, '{}/Seeds/POSCARS_{}'.format(self.parameters.workDir, self.curgen))
        seedPop = self.Population(seedpop,'seedpop',self.curgen)
        if self.parameters.chkSeed:
            seedPop.check()
        initPop.extend(seedPop)

        initPop.save()

        prelaxpop = self.MainCalculator.relax(initPop.frames)
        prelaxPop = self.Population(prelaxpop,'prelaxpop',self.curgen)
        # save raw pop before check and del_duplicates
        prelaxPop.save('pre_raw')
        relaxPop = self.CheckCal.relax(prelaxPop.frames)
        relaxPop = self.Population(relaxPop,'relaxpop',self.curgen)
        relaxPop.save('raw')
        relaxPop.check()
        relaxPop.del_duplicate()
        relaxPop.calc_dominators()
        if self.parameters.useml:
            self.ML.updatedataset(relaxPop.all_frames)
            self.ML.train()
            logging.info("loss:\nenergy_mse:{}\tenergy_r2:{}\nforce_mse:{}\tforce_r2:{}".format(*self.ML.get_loss(relaxPop.all_frames)[:4]))
            #scfpop = self.MainCalculator.scf(relaxPop.frames)
            #scfPop = self.Population(scfpop,'scfpop',self.curgen)
            #logging.info("loss:\nenergy_mse:{}\tenergy_r2:{}\nforce_mse:{}\tforce_r2:{}".format(*self.ML.get_loss(scfPop.frames)[:4]))

        if self.parameters.goodSeed:
            logging.info("Please be careful when you set goodSeed=True. \nThe structures in {} will be add to relaxPop without relaxation.".format(self.parameters.goodSeedFile))
            goodseedpop = read_seeds(self.parameters,'{}/Seeds/{}'.format(self.parameters.workDir, self.parameters.goodSeedFile), goodSeed=self.parameters.goodSeed)
            relaxPop.extend(goodseedpop)
            relaxPop.del_duplicate()
            # relaxPop.check()

        self.curPop = relaxPop
        self.allPop.extend(self.curPop)
        # self.goodPop = self.Population([],'goodPop',self.curgen)
        # self.keepPop = self.Population([],'keepPop',self.curgen)
        self.bestPop = self.Population([],'bestPop')

        
        relaxPop.save('gen')


        logging.info('construct goodPop')
        # goodpop = []
        # goodpop.extend(relaxpop)
        # goodPop = self.Population(goodpop,'goodPop',self.curgen)
        goodPop = relaxPop
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
        #self.bestPop.save('best',self.curgen)

        logging.info('construct keepPop')
        _, keeppop = goodPop.clustering(self.parameters.saveGood)
        # initialize keepPop here, so that the inds have identity
        self.keepPop = self.Population(keeppop,'keepPop',self.curgen)
        self.keepPop.save('keep')

    def Onestep(self):
        #TODO make update parameters more reasonable
        self.update_parameters()
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

        # Save Initial
        initPop.save()

        #######  relax  #######
        if self.parameters.mlRelax:
            relaxpop = self.ML.relax(initPop.frames)
            relaxPop = self.Population(relaxpop,'relaxpop',self.curgen)
            relaxPop.save("mlraw")
            relaxPop.check()
            relaxPop.del_duplicate()
            relaxPop.save("mlgen")
            logging.info("Using MainCalculator to relax survivals after ML relaxation")
            relaxpop = self.MainCalculator.relax(relaxPop.frames)
            relaxPop = self.Population(relaxpop,'relaxpop',self.curgen)
            self.ML.updatedataset(relaxPop.all_frames)
            write_results(self.ML.dataset,'','dataset',self.parameters.mlDir)
            self.ML.train()
            logging.info("loss:\nenergy_mse:{}\tenergy_r2:{}\nforce_mse:{}\tforce_r2:{}".format(*self.ML.get_loss(relaxPop.frames)[:4]))
            #for _ in range(10):
                #relaxpop = self.ML.relax(initPop.frames)
                #relaxPop = self.Population(relaxpop,'relaxpop',self.curgen)
                #relaxPop.check()
                #scfpop = self.MainCalculator.scf(relaxPop.frames)
                #loss = self.ML.get_loss(scfpop)
                #logging.info('ML Gen{}\tEnergy Error:{}'.format(_,loss[1]))
                #break
                #if loss[1]>0.8:
                #    logging.info('Good fit, ml relax adapt')
                #    break
                #logging.info('Bad fit, retraining...')
                #self.ML.updatedataset(scfpop)
                #write_results(self.ML.dataset,'','dataset',self.parameters.mlDir)
                #self.ML.train()
            #else:
            #    logging.info('Cannot fit, turn to main calculator')
            #    relaxpop = self.MainCalculator.relax(initPop.frames)
            #    relaxPop = self.Population(relaxpop,'relaxpop',self.curgen)
        else:
            prelaxpop = self.MainCalculator.relax(initPop.frames)
            if self.parameters.useml:
                loss = self.ML.get_loss(relaxpop)
                logging.info('ML Energy Error:{}'.format(loss[1]))
                #if loss[1]<0.8:
                #    logging.info('Bad fit, retraining...')
                #    self.ML.updatedataset(relaxpop)
                #    write_results(self.ML.dataset,'','dataset',self.parameters.mlDir)
                #    self.ML.train()
            prelaxPop = self.Population(relaxpop,'relaxpop',self.curgen)

        # save raw date before checking
        prelaxPop.save('pre_raw')
        relaxPop = self.CheckCal.relax(prelaxPop.frames)
        relaxPop = self.Population(relaxPop,'relaxpop',self.curgen)
        relaxPop.save('raw')
        relaxPop.check()
        # find spg before delete duplicate
        relaxPop.find_spg()
        relaxPop.del_duplicate()
        self.allPop.extend(relaxPop)

        #######  goodPop and keepPop  #######
        logging.info('construct goodPop')
        goodPop = relaxPop + goodPop + keepPop
        goodPop.del_duplicate()
        goodPop.calc_dominators()
        goodPop.select(self.parameters.popSize)
        goodPop.save('good','')
        goodPop.save('savegood')
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
            if 'queueName' in d['MainCalculator']:
                if d['MainCalculator']['queueName'] != self.MainCalculator.p.queueName:
                    logging.warning('*****************\nBe careful, {} is replaced by {}\n*****************'.format(self.MainCalculator.p.queueName, d['MainCalculator']['queueName']))
                    self.MainCalculator.p.queueName = d['MainCalculator']['queueName']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="print debug information", action='store_true', default=False)
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(message)s")
        logging.info('Debug mode')
    else:
        logging.basicConfig(filename='log.txt', level=logging.INFO, format="%(message)s")

    parameters = magusParameters('input.yaml')
    m=Magus(parameters)
    m.run()
