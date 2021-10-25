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


class PopGenerator:
    def __init__(self,numlist,oplist,parameters):
        self.oplist = oplist
        self.numlist = numlist
        self.p = EmptyClass()
        Requirement = ['popSize','saveGood','molDetector', 'calcType']
        Default = {'chkMol': False,'addSym': False,'randFrac': 0.0}
        checkParameters(self.p,parameters,Requirement,Default)

    def clustering(self, clusterNum):
        Pop = self.Pop
        labels,_ = Pop.clustering(clusterNum)
        uqLabels = list(sorted(np.unique(labels)))
        subpops = []
        for label in uqLabels:
            subpop = [ind for j,ind in enumerate(Pop.pop) if labels[j] == label]
            subpops.append(subpop)

        self.uqLabels = uqLabels
        self.subpops = subpops

    def get_pairs(self, Pop, crossNum ,clusterNum, tryNum=50,k=0.3):
        ##################################
        #temp
        #si ma dang huo ma yi
        k = 2 / len(Pop)
        ##################################
        pairs = []
        labels,_ = Pop.clustering(clusterNum)
        fail = 0
        while len(pairs) < crossNum and fail < tryNum:
            #label = np.random.choice(self.uqLabels)
            #subpop = self.subpops[label]
            label = np.random.choice(np.unique(labels))
            subpop = [ind for j,ind in enumerate(Pop.pop) if labels[j] == label]

            if len(subpop) < 2:
                fail+=1
                continue

            dom = np.array([ind.info['dominators'] for ind in subpop])
            edom = np.exp(-k*dom)
            p = edom/np.sum(edom)
            pair = tuple(np.random.choice(subpop,2,False,p=p))
            if pair in pairs:
                fail+=1
                continue
            pairs.append(pair)
        return pairs

    def get_inds(self,Pop,mutateNum,k=0.3):
        #Pop = self.Pop
        ##################################
        #temp
        #si ma dang huo ma yi
        k = 2 / len(Pop)
        ##################################
        dom = np.array([ind.info['dominators'] for ind in Pop.pop])
        edom = np.exp(-k*dom)
        p = edom/np.sum(edom)
        # mutateNum = min(mutateNum,len(Pop))
        if mutateNum > len(Pop):
            return np.random.choice(Pop.pop,mutateNum,True,p=p)
        else:
            return np.random.choice(Pop.pop,mutateNum,False,p=p)

    def generate(self,Pop,saveGood):
        # calculate dominators before checking formula
        Pop.calc_dominators()

        #remove bulk_layer and relaxable_layer before crossover and mutation
        if self.p.calcType=='rcs':
            Pop = Pop.copy()
            Pop.removebulk_relaxable_vacuum()
        if self.p.calcType=='clus':
            Pop.randrotate()
        if self.p.calcType == 'var':
            Pop.check_full()
        #TODO move addsym to ind
        if self.p.addSym:
            Pop.add_symmetry()
        newPop = Pop([],'initpop',Pop.gen+1)

        operation_keys = list(self.oplist.keys())
        for key in operation_keys:
            op = self.oplist[key]
            num = self.numlist[key]
            if num == 0:
                continue
            log.debug('name:{} num:{}'.format(op.descriptor,num))
            if op.optype == 'Mutation':
                mutate_inds = self.get_inds(Pop,num)
                for i,ind in enumerate(mutate_inds):
                    #if self.p.molDetector != 0 and not hasattr(atoms, 'molCryst'):
                    if self.p.molDetector != 0:
                        if not hasattr(ind, 'molCryst'):
                            ind.to_mol()
                    atoms = op.get_new_individual(ind, chkMol=self.p.chkMol)
                    if atoms:
                        newPop.append(atoms)
            elif op.optype == 'Crossover':
                cross_pairs = self.get_pairs(Pop,num,saveGood)
                #cross_pairs = self.get_pairs(Pop,num)
                for i,parents in enumerate(cross_pairs):
                    if self.p.molDetector != 0:
                        for ind in parents:
                            if not hasattr(ind, 'molCryst'):
                                ind.to_mol()
                    atoms = op.get_new_individual(parents,chkMol=self.p.chkMol)
                    if atoms:
                        newPop.append(atoms)
            log.debug("popsize after {}: {}".format(op.descriptor, len(newPop)))

        if self.p.calcType == 'var':
            newPop.check_full()
        if self.p.calcType=='rcs':
            newPop.addbulk_relaxable_vacuum()
        #newPop.save('testnew')
        newPop.check()
        return newPop

    def select(self,Pop,num,k=0.3):
        ##################################
        #temp
        #si ma dang huo ma yi
        #k = 2 / len(Pop)
        ##################################
        if num < len(Pop):
            # pardom = np.array([ind.info['pardom'] for ind in Pop.pop])
            # edom = np.e**(-k*pardom)
            # p = edom/np.sum(edom)
            # Pop.pop = list(np.random.choice(Pop.pop,num,False,p=p))
            Pop.pop = list(np.random.choice(Pop.pop, num, False))
            return Pop
        else:
            return Pop
        

    def next_Pop(self,Pop):
        saveGood = self.p.saveGood
        popSize = int(self.p.popSize*(1-self.p.randFrac))
        newPop = self.generate(Pop,saveGood)
        return self.select(newPop,popSize)

class MLselect(PopGenerator):
    def __init__(self, numlist, oplist, calc,parameters):
        super().__init__(numlist, oplist, parameters)
        self.calc = calc

    def select(self,Pop,num,k=0.3):
        predictE = []
        if num < len(Pop):
            for ind in Pop:
                ind.atoms.set_calculator(self.calc)
                ind.info['predictE'] = ind.atoms.get_potential_energy()
                predictE.append(ind.info['predictE'])
                ind.atoms.set_calculator(None)

            dom = np.argsort(predictE)
            edom = np.exp(-k*dom)
            p = edom/np.sum(edom)
            Pop.pop = np.random.choice(Pop.pop,num,False,p=p)
            return Pop
        else:
            return Pop
