import math
import numpy as np
import logging, copy
from ase import Atoms, Atom 
from ase.constraints import voigt_6_to_full_3x3_strain as v2f
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


class CutAndSplicePairing(Crossover):
    """ 
    A cut and splice operator for bulk structures.

    For more information, see also:

    * `Glass, Oganov, Hansen, Comp. Phys. Comm. 175 (2006) 713-720`__

      __ https://doi.org/10.1016/j.cpc.2006.07.020

    * `Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387`__

      __ https://doi.org/10.1016/j.cpc.2010.07.048
    """
    Default = {'tryNum': 50, 'cut_disp': 0, 'best_match': False}

    def cross(self, ind1, ind2):
        if self.best_match:
            M1, M2 = match_lattice(ind1, ind2)
            axis = 2
        else:
            axis = np.random.choice([0, 1, 2])
            atoms1 = ind1.for_mutate
            atoms2 = ind2.for_mutate

        atoms1.set_scaled_positions(atoms1.get_scaled_positions() + np.random.rand(3))
        atoms2.set_scaled_positions(atoms2.get_scaled_positions() + np.random.rand(3))
 
        cut_cell   = 0.5 * (atoms1.get_cell()   + atoms2.get_cell())
        cut_volume = 0.5 * (atoms1.get_volume() + atoms2.get_volume())
        cut_cellpar = cell_to_cellpar(cut_cell)
        ratio = cut_volume / abs(np.linalg.det(cut_cell))
        cut_cellpar[:3] = [length * ratio ** (1/3) for length in cut_cellpar[:3]]

        cut_atoms = atoms1.__class__(cell=cut_cellpar, pbc=True,)

        scaled_positions = []
        cut_position = [0, 0.5 + self.cut_disp * np.random.uniform(-0.5, 0.5), 1]

        for n, atoms in enumerate([atoms1, atoms2]):
            spositions = atoms.get_scaled_positions()
            for i, atom in enumerate(atoms):
                if cut_position[n] <= spositions[i, axis] < cut_position[n+1]:
                    cut_atoms.append(atom)
                    scaled_positions.append(spositions[i])
        if len(scaled_positions) == 0:
            return None

        cut_atoms.set_scaled_positions(scaled_positions)

        return ind1.__class__(cut_atoms)


class ReplaceBallPairing(Crossover):
    """
    replace some atoms in a ball
    """
    parm = {'tryNum': 50, 'cut_range': [1, 2]}

    def cross(self, ind1, ind2):
        """
        replace some atoms in a ball
        """
        cut_radius = np.random.uniform(*self.cutrange)
        atoms1, atoms2 = ind1.for_mutate, ind2.for_mutate
        i, j = np.random.randint(len(atoms1)), np.random.randint(len(atoms2))
        newatoms = atoms1.__class__(pbc=atoms1.pbc, cell=atoms1.cell)

        nl = NeighborList(cutoffs=[cutR/2]*len(atoms1), skin=0, self_interaction=True, bothways=True)
        nl.update(atoms1)
        indices, _ = nl.get_neighbors(i)
        for index,atom in enumerate(atoms1):
            if index not in indices:
                newatoms.append(atom)

        nl = NeighborList(cutoffs=[cutR/2]*len(atoms2), skin=0, self_interaction=True, bothways=True)
        nl.update(atoms2)
        indices, _ = nl.get_neighbors(j)
        atoms2.positions += atoms1.positions[i]-atoms2.positions[j]
        newatoms.extend(atoms2[indices])
        cutInd = ind1(newatoms)
        cutInd.parents = [ind1 ,ind2]
        return newatoms

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
