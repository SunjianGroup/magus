import random, logging, os, sys, shutil, time, json
import argparse
import copy
import numpy as np
from ase.data import atomic_numbers
from ase import Atoms, Atom
import ase.io
from magus.initstruct import read_seeds
from .search import Magus
"""
Pop:class, poplulation
pop:list, a list of atoms
Population: Population(pop) --> Pop
"""
#TODO 
# use same log among different functions
# change read parameters
# early converged


log = logging.getLogger(__name__)


class MLMagus(Magus):
    def __init__(self, parameters, atoms_generator, pop_generator,
                 main_calculator, Population, ml_calculator, restart=False):
        super().__init__(parameters, atoms_generator, pop_generator,
                         main_calculator, Population, restart)
        self.ml_calculator = ml_calculator
        log.debug('ML Calculator information:')
        log.debug(ml_calculator.__str__())
        self.get_initial_pot(epoch=ml_calculator.init_times)

    def get_initial_pot(self, epoch=1):
        if epoch == 0:
            log.warning("skip initial train, please make sure you have an trained "
                        "potential, otherwise you should make 'init_times' > 0")
            return
        log.info('try to get the initial potential, will repeat {} times'.format(epoch))
        for i in range(epoch):
            log.info('\tepoch {}'.format(i + 1))
            # get random populations
            random_pop = self.atoms_generator.Generate_pop(self.parameters.poolSize, initpop=True)
            log.info("\tRandom generate population with {} strutures\n"
                     "\tSelecting...".format(len(random_pop)))
            # select to add
            select_pop = self.ml_calculator.select(random_pop)
            log.info("\tDone! {} are selected\n\tscf...".format(len(select_pop)))
            scf_pop = self.main_calculator.scf(select_pop)
            self.ml_calculator.updatedataset(scf_pop)
            log.info("\tDone! {} structures in the dataset\n\ttraining...".format(len(self.ml_calculator.trainset)))
            self.ml_calculator.train()
        log.info('Done!')

    def select_to_relax(self, frames, init_num=5, min_num=20):
        try:
            ground_enthalpy = self.goodPop.bestind()[0].atoms.info['enthalpy']
        except:
            ground_enthalpy = min([atoms.info['enthalpy'] for atoms in frames])
        min_num = min(len(frames), min_num)
        trainset = self.ml_calculator.trainset
        energy_mse = self.ml_calculator.get_loss(trainset)[0]
        select_enthalpy = max(ground_enthalpy + init_num * energy_mse, 
                              sorted([atoms.info['enthalpy'] for atoms in frames])[min_num - 1])
        log.info('select good structures to relax\n'
                 '\tground enthalpy: {}\tenergy mse: {}\tselect enthaly: {}'
                 ''.format(ground_enthalpy, energy_mse, select_enthalpy))
        to_relax = [atoms for atoms in frames if atoms.info['enthalpy'] <= select_enthalpy]
        return to_relax            
    
    def select_to_add(self, frames):
        trainset = self.ml_calculator.trainset
        energy_mae = self.ml_calculator.get_loss(trainset)[0]
        frames_ = self.ml_calculator.calc_efs(frames)
        to_add = []
        log.debug('compare begin...\ntarget\tpredict\n')
        for i, (atoms, atoms_) in enumerate(zip(frames, frames_)):
            target_energy = atoms.info['energy'] / len(atoms)
            predict_energy = atoms_.info['energy'] / len(atoms_)
            log.debug("{:.5f}\t{:.5f}\n".format(target_energy, predict_energy))
            error_per_atom = target_energy - predict_energy
            if abs(error_per_atom) > energy_mae:
                to_add.append(atoms)
        return to_add

    def one_step(self):
        self.set_volume_ratio()
        initPop = self.get_initPop()
        initPop.save()
        #######  local relax by ML  #######
        relaxpop = self.ml_calculator.relax(initPop.frames)
        relaxPop = self.Population(relaxpop,'relaxpop',self.curgen)
        relaxPop.save("mlraw", self.curgen)
        relaxPop.check()
        # find spg before delete duplicate
        relaxPop.find_spg()
        relaxPop.del_duplicate()
        relaxPop.calc_dominators()
        relaxPop.save("mlgen", self.curgen)
        if self.parameters.DFTRelax:
            #######  select cfgs to do dft relax  #######
            to_relax = self.select_to_relax(relaxPop.frames)
            #######  compare target and predict energy  #######   
            dft_relaxed_pop = self.main_calculator.relax(to_relax)
            relax_step = sum([atoms.info['relax_step'][-1] for atoms in dft_relaxed_pop])
            log.info('DFT relax {} structures with {} scf'.format(len(dft_relaxed_pop), relax_step))
            DFTRelaxedPop = self.Population(dft_relaxed_pop, 'dft_relaxed_pop', self.curgen)
            DFTRelaxedPop.find_spg()
            DFTRelaxedPop.del_duplicate()
            self.curPop = DFTRelaxedPop
            to_add = self.select_to_add(dft_relaxed_pop)
            self.ml_calculator.updatedataset(to_add)
            self.ml_calculator.train()
        else:
            self.curPop = relaxPop
        self.curPop.save('gen', self.curgen)
        self.set_goodPop()
        self.goodPop.save('good', '')
        self.goodPop.save('good', self.curgen)
        self.set_keepPop()
        self.keepPop.save('keep', self.curgen)
        self.update_bestPop()
        self.bestPop.save('best', '')

