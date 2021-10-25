import numpy as np
import os,re,itertools,random
from collections import Counter
from scipy.spatial.distance import cdist, pdist
import ase.io
from ase import Atoms
from ase.data import covalent_radii,atomic_numbers
from ase.neighborlist import neighbor_list
from ase.atom import Atom
from ase.atoms import Atoms
import spglib
from .utils import *
import logging
from sklearn import cluster
from .descriptor import ZernikeFp
import copy
from .molecule import Molfilter
from ase.constraints import FixAtoms
# from .reconstruct import fixatoms, weightenCluster
from ase import neighborlist
from scipy import sparse

#clusters
try:
    from pymatgen import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
except:
    pass


log = logging.getLogger(__name__)


class Population:
    """
    a class of atoms population
    """
    batch_operation = [
        'find_spg', 'add_symmetry', 'removebulk_relaxable_vacuum', 'addbulk_relaxable_vacuum', 'randrotate'
        ]

    @classmethod
    def set_parameters(cls, **parameters):
        cls.all_parameters = parameters
        Requirement = ['results_dir', 'pop_size']
        Default = {'check_seed': False}
        check_parameters(cls, parameters, Requirement, Default)

    def __init__(self, pop, name='temp', gen=None):
        self.pop = [ind if isinstance(ind, Individual) else self.Individual(ind) for ind in pop]
        self.name = name
        self.gen = gen
        log.debug('construct Population {} with {} individual'.format(name, len(pop)))
        for i, ind in enumerate(pop):
            ind.info['identity'] = (name, i)
        return newPop

    def __iter__(self):
        for i in self.pop:
            yield i

    def __getitem__(self, i):
        return self.pop[i]

    def __len__(self):
        return len(self.pop)

    def __add__(self, other):
        newPop = self.copy()
        newPop.extend(other)
        return newPop

    def __iadd__(self, other):
        self.extend(other)

    def __contains__(self, ind):
        ind = ind if isinstance(ind, Individual) else self.Individual(ind)
        for ind_ in self.pop:
            if ind == ind_:
                return True
        else:
            return False

    def __getattr__(self, name):
        # batch operations
        if name in self.batch_operation:
            def f(*arg, **kwargs):
                for ind in self.pop:
                    getattr(ind, name)(*arg, **kwargs)
            return f
        else:
            raise AttributeError("{} is not defined in 'Population'".format(name))

    def append(self, ind):
        ind = ind if isinstance(ind, Individual) else self.Individual(ind)
        ind.info['identity'] = [self.name, len(self.pop)]
        self.pop.append(ind)
        return True
        #谁删的啊，为啥来着？
        #for ind_ in self.pop:
        #    if ind == ind_:
        #        return False
        #else:
        #    self.pop.append(ind)
        #    return True

    def extend(self, pop):
        for ind in pop:
            self.append(ind)

    def copy(self):
        newpop = [ind.copy() for ind in self.pop]
        return self.__class__(newpop, name=self.name, gen=self.gen)

    def save(self, filename=None, gen=None, savedir=None):
        filename = self.name if filename is None else filename
        gen = self.gen if gen is None else gen
        savedir = self.results_dir if savedir is None else savedir
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        pop = []
        for ind in self.pop:
            atoms = ind.to_save()
            pop.append(atoms)
        ase.io.write("{}/{}{}.traj".format(savedir, filename, gen), pop, format='traj')
        log.debug("save {}{}.traj".format(filename,gen))

    @property
    def volume_ratio(self):
        return np.mean([ind.volume_ratio for ind in self.pop])

    @property
    def frames(self):
        return [ind.atoms for ind in self.pop]

    @property
    def all_frames(self):
        pop = []
        for ind in self.pop:
            atoms = ind.atoms
            if 'trajs' in atoms.info:
                traj = atoms.info['trajs'][-1]
                for frame in traj:
                    pop.append(read_atDict(frame))
            else:
                pop.append(atoms)
        return pop

    def calc_dominators(self):
        self.calc_fitness()
        domLen = len(self.pop)
        for ind1 in self.pop:
            dominators = -1 #number of individuals that dominate the current ind
            for ind2 in self.pop:
                for key in ind1.info['fitness']:
                    if ind1.info['fitness'][key] > ind2.info['fitness'][key]:
                        break
                else:
                    dominators += 1

            ind1.info['dominators'] = dominators
            ind1.info['MOGArank'] = dominators + 1
            ind1.info['sclDom'] = (dominators) / domLen

    def calc_fitness(self):
        for fit_calc in self.fit_calcs:
            fit_calc.calc(self)

    def del_duplicate(self):
        self.calc_dominators()
        log.debug('del_duplicate {} begin, popsize:{}'.format(self.name, len(self.pop)))
        newpop = []
        # sort the pop so the better individual will be remained
        self.pop = sorted(self.pop, key=lambda x: x.info['dominators'])
        for ind1 in self.pop:
            for ind2 in newpop:
                if ind1 == ind2:
                    break
            else:
                newpop.append(ind1)
        log.debug('del_duplicate survival: {}'.format(len(newpop)))
        self.pop = newpop

    def check(self):
        log.debug("check population {}, popsize:{}".format(self.name, len(self.pop)))
        checkpop = []
        for ind in self.pop:
            if not ind.need_check() or ind.check():
                checkpop.append(ind)
        log.debug("check survival: {}".format(len(checkpop)))
        self.pop = checkpop

    def clustering(self, n_clusters):
        """
        clustering by fingerprints
        TODO may not be a class method
        """
        pop = [ind.copy() for ind in self.pop]
        if n_clusters >= len(pop):
            return np.arange(len(pop)), pop

        fp = np.array([ind.fingerprint for ind in pop])
        labels = cluster.KMeans(n_clusters=n_clusters).fit_predict(fp)
        goodpop = [None] * n_clusters
        for label, ind in zip(labels, pop):
            if goodpop[label] is None:
                goodpop[label] = ind
            else:
                if ind.info['dominators'] < goodpop[label].info['dominators']:
                    goodpop[label] = ind
        return labels, goodpop

    def select(self, n, delete_highE=False, high=0.6):
        self.calc_dominators()
        self.pop = sorted(self.pop, key=lambda x: x.info['dominators'])
        if len(self) > n:
            self.pop = self.pop[:n]
        if delete_highE:
            enthalpys = [ind.atoms.info['enthalpy'] for ind in self.pop]
            high *= np.min(enthalpys)
            logging.debug("select without enthalpy higher than {} eV/atom, pop length before selecting: {}".format(high, len(self.pop)))
            self.pop = [ind for ind in self.pop if ind.atoms.info['enthalpy'] <= high]
            logging.debug("select end with pop length: {}".format(len(self.pop)))

    def bestind(self):
        self.calc_dominators()
        dominators = np.array([ind.info['dominators'] for ind in self.pop])
        best_i = np.where(dominators == np.min(dominators))[0]
        bestInds = [self.pop[i] for i in best_i]
        # Write generation of bestind
        for ind in bestInds:
            ind.info['gen'] = self.gen
        return  bestInds
        #return [self.pop[i] for i in best_i]

    def fill_up_with_random(self):
        raise NotImplementedError


class FixPopulation(Population):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        cls.Individual = set_ind(parameters)
        cls.fit_calcs = fit_calcs

    def fill_up_with_random(self):
        n_random = self.pop_size - len(self)
        addpop = self.atoms_generator.generate_pop(n_random)
        self.pop.extend(addpop)


class VarPopulation(Population):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        check_parameters(cls, parameters, [], {'ele_size': 0})
        cls.Individual = set_ind(parameters)
        cls.fit_calcs = fit_calcs

    def fill_up_with_random(self):
        n_units = len(self.atoms_generator.formula[0])
        d_n_random = {format_filter: self.ele_size for format_filter in itertools.product([0, 1], repeat=n_units)}
        d_n_random[tuple([0] * n_units)] = 0
        d_n_random[tuple([1] * n_units)] = self.pop_size
        for ind in self.pop:
            d_n_random[tuple(np.clip(ind.numlist, 0, 1))] -= 1
        for format_filter, n_random in d.items():
            if n_random > 0:
                addpop = self.atoms_generator.generate_pop(n_random, format_filter=format_filter)
                self.pop.extend(addpop)


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
