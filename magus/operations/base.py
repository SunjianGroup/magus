#TODO how to select, za lun pan du a ?
import numpy as np
import logging, copy
from ase import Atoms, Atom 
from ase.geometry import cell_to_cellpar,cellpar_to_cell,get_duplicate_atoms
from ase.neighborlist import NeighborList
from ase.data import covalent_radii,chemical_symbols
from .population import Population
from .molecule import Molfilter
import ase.io
from .utils import *
from spglib import get_symmetry_dataset
from collections import Counter


log = logging.getLogger(__name__)


class OffspringCreator:
    Default = {'tryNum': 50}
    def __init__(self, **parameters):
        self.all_parameters = parameters
        check_parameters(self, parameters, self.Requirement, self.Default)
        self.descriptor = self.__class__.__name__

    def get_new_individual(self):
        pass


class Mutation(OffspringCreator):
    def mutate(self, ind):
        if isinstance(ind, Molecule):
            self.mutate_mol(ind)
        else:
            self.mutate_bulk(ind)

    def mutate_mol(self, ind):
        raise NotImplementedError("{} cannot apply in molmode".format(self.descriptor))

    def mutate_bulk(self, ind):
        raise NotImplementedError("{} cannot apply in bulk".format(self.descriptor))

    def get_new_individual(self, ind):
        for _ in range(self.tryNum):
            newind = self.mutate(ind)
            if newind is None:
                continue
            newind.parents = [ind]
            newind.merge_atoms()
            if newind.repair_atoms():
                break
        else:
            log.debug('fail {} in {}'.format(self.descriptor, ind.info['identity']))
            return None
        log.debug('success {} in {}'.format(self.descriptor, ind.info['identity']))
        # remove some parent infomation
        rmkeys = ['enthalpy', 'spg', 'priVol', 'priNum', 'ehull', 'energy','forces']
        for k in rmkeys:
            if k in newind.info.keys():
                del newind.info[k]

        newind.info['parents'] = [ind.info['identity']]
        newind.info['parentE'] = ind.info['enthalpy']
        newind.info['pardom'] = ind.info['dominators']
        newind.info['origin'] = self.descriptor
        return newind


class Crossover(OffspringCreator):
    def cross(self, ind1, ind2):
        if isinstance(ind1, Molecule):
            self.cross_mol(ind)
        else:
            self.cross_bulk(ind)

    def cross_mol(self, ind):
        raise NotImplementedError("{} cannot apply in molmode".format(self.descriptor))

    def cross_bulk(self, ind):
        raise NotImplementedError("{} cannot apply in bulk".format(self.descriptor))

    def get_new_individual(self, ind1, ind2):
        for _ in range(self.tryNum):
            newind = self.cross(ind1, ind2)
            if newind is None:
                continue
            newind.parents = [ind1, ind2]
            newind.merge_atoms()
            if newind.repair_atoms():
                break
        else:
            log.debug('fail {} between {} and {}'.format(self.descriptor, ind1.info['identity'], ind2.info['identity']))
            return None
        log.debug('success {} between {} and {}'.format(self.descriptor, ind1.info['identity'], ind2.info['identity']))
        # remove some parent infomation
        rmkeys = ['enthalpy', 'spg', 'priVol', 'priNum', 'ehull','energy','forces']
        for k in rmkeys:
            if k in newind.atoms.info.keys():
                del newind.atoms.info[k]

        newind.info['parents'] = [ind1.info['identity'], ind2.info['identity']]
        newind.info['parentE'] = 0.5 * (ind1.info['enthalpy'] + ind2.info['enthalpy'])
        newind.info['pardom'] = 0.5 * (ind1.info['dominators'] + ind2.info['dominators'])
        newind.info['origin'] = self.descriptor
        return newind
