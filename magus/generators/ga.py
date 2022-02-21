# TODO
# how to set k in edom
import itertools, copy, logging
import numpy as np
from magus.utils import *
# from .reconstruct import reconstruct, cutcell, match_symmetry, resetLattice


log = logging.getLogger(__name__)


##################################
# How to select parents?
#
# How Evolutionary Crystal Structure Prediction Works—and Why. 
#   Acc. Chem. Res. 44, 227–237 (2011).
# XtalOpt: An open-source evolutionary algorithm for crystal structure prediction. 
#   Computer Physics Communications 182, 372–387 (2011).
# A genetic algorithm for first principles global structure optimization of supported nano structures.
#   The Journal of Chemical Physics 141, 044711 (2014).
#
# For now, we use a scheme similar to oganov's, because it just use rank information and can be easily extend to multi-target search.
##################################

class GAGenerator:
    def __init__(self, numlist, oplist, **parameters):
        self.oplist = oplist
        self.numlist = numlist
        self.n_next = int(parameters['popSize'] * (1 - parameters['randFrac']))
        self.n_cluster = parameters['n_cluster']
        self.add_sym = parameters['addSym']

    def __repr__(self):
        ret = self.__class__.__name__
        ret += "\n-------------------"
        c, m = "\nCrossovers:", "\nMutations:"
        for op, num in zip(self.oplist, self.numlist):
            if op.n_input == 1:
                m += "\n {}: {}".format(op.__class__.__name__.ljust(20, ' '), num)
            elif op.n_input == 2:
                c += "\n {}: {}".format(op.__class__.__name__.ljust(20, ' '), num)
        ret += m + c
        ret += "\nNumber of cluster      : {}".format(self.add_sym) 
        ret += "\nAdd symmertry before GA: {}".format(self.add_sym) 
        ret += "\n-------------------\n"
        return ret

    def get_pairs(self, pop, n, n_try=50, k=2):
        # k = k / len(pop)
        k = 0.3
        dom = np.array([ind.info['dominators'] for ind in pop])
        edom = np.exp(-k * dom)
        labels, _ = pop.clustering(self.n_cluster)
        fail = 0; choosed = []
        # first try to only choose pairs in a cluster, if fail, random choose 
        while len(choosed) < n and fail < n_try:
            label = np.random.choice(np.unique(labels))
            indices = np.where(labels == label)[0]
            if len(indices) < 2:
                fail += 1
                continue
            p = edom[indices] / sum(edom[indices])
            i, j = np.random.choice(indices, 2 , False, p=p)
            pop[i].info['used'] += 1; pop[j].info['used'] += 1
            edom[i] *= 0.9; edom[j] *= 0.9
            choosed.append((pop[i].copy(), pop[j].copy()))
        indices = np.arange(len(pop))
        while len(choosed) < n:
            p = edom / np.sum(edom)
            i, j = np.random.choice(indices, 2 , False, p=p)
            pop[i].info['used'] += 1; pop[j].info['used'] += 1
            edom[i] *= 0.9; edom[j] *= 0.9
            choosed.append((pop[i].copy(), pop[j].copy()))
        return choosed

    def get_inds(self, pop, n, k=2):
        # k = k / len(pop)
        k = 0.3
        dom = np.array([ind.info['dominators'] for ind in pop])
        edom = np.exp(-k * dom)
        p = edom / np.sum(edom)
        choosed = []
        while len(choosed) < n:
            i = np.random.choice(len(pop), p=p)
            pop[i].info['used'] += 1
            p[i] *= 0.8
            p /= sum(p)
            choosed.append(pop[i].copy())
        return choosed

    def generate(self, pop):
        log.debug(self)
        # calculate dominators before checking formula
        pop.calc_dominators()
        if self.add_sym:
            pop.add_symmetry()
        newpop = pop.__class__([], name='init')
        for op, num in zip(self.oplist, self.numlist):
            if num == 0:
                continue
            log.debug('name:{} num:{}'.format(op.descriptor, num))
            if op.n_input == 1:
                cands = self.get_inds(pop, num)
            elif op.n_input == 2:
                cands = self.get_pairs(pop, num)
            for cand in cands:
                newind = op.get_new_individual(cand)
                if newind is not None:
                    newpop.append(newind)
            log.debug("popsize after {}: {}".format(op.descriptor, len(newpop)))
        newpop.check()
        return newpop

    def select(self, pop, num):
        if num < len(pop):
            pop = pop[np.random.choice(len(pop), num, False)]
        return pop

    def get_next_pop(self, pop, n_next=None):
        n_next = n_next or self.n_next
        newpop = self.generate(pop)
        return self.select(newpop, n_next)
