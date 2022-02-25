# TODO
# how to set k in edom
import itertools, copy, logging
import numpy as np
from magus.utils import *
import prettytable as pt
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
    def __init__(self, op_list, op_prob, **parameters):
        assert len(op_list) == len(op_prob), "number of operations and probabilities not match"
        assert np.sum(op_prob) > 0 and np.all(op_prob >= 0), "unreasonable probability are given"
        self.op_list = op_list
        self.op_prob = op_prob / np.sum(op_prob)
        self.n_next = int(parameters['popSize'] * (1 - parameters['randFrac']))
        self.n_cluster = parameters['n_cluster']
        self.add_sym = parameters['addSym']

    def __repr__(self):
        ret = self.__class__.__name__
        ret += "\n-------------------"
        c, m = "\nCrossovers:", "\nMutations:"
        for op, prob in zip(self.op_list, self.op_prob):
            if op.n_input == 1:
                m += "\n {}: {}".format(op.__class__.__name__.ljust(20, ' '), prob)
            elif op.n_input == 2:
                c += "\n {}: {}".format(op.__class__.__name__.ljust(20, ' '), prob)
        ret += m + c
        ret += "\nNumber of cluster      : {}".format(self.add_sym) 
        ret += "\nAdd symmertry before GA: {}".format(self.add_sym) 
        ret += "\n-------------------\n"
        return ret

    def get_pair(self, pop, k=2, n_try=50, history_punish=1.):
        assert 0 < history_punish <= 1, "history_punish should between 0 and 1"
        # k = k / len(pop)
        k = 0.3
        dom = np.array([ind.info['dominators'] for ind in pop])
        edom = np.exp(-k * dom)
        used = np.array([ind.info['used'] for ind in pop])
        labels, _ = pop.clustering(self.n_cluster)
        fail = 0

        while fail < n_try:
            label = np.random.choice(np.unique(labels))
            indices = np.where(labels == label)[0]
            if len(indices) < 2:
                fail += 1
                continue
            prob = edom[indices] * history_punish ** used[indices]
            prob = prob / sum(prob)
            i, j = np.random.choice(indices, 2 , False, p=prob)
            pop[i].info['used'] += 1
            pop[j].info['used'] += 1
            return pop[i].copy(), pop[j].copy()

        indices = np.arange(len(pop))
        prob = edom[indices] * history_punish ** used[indices]
        prob = prob / sum(prob)
        i, j = np.random.choice(indices, 2 , False, p=prob)
        pop[i].info['used'] += 1
        pop[j].info['used'] += 1
        return pop[i].copy(), pop[j].copy()

    def get_ind(self, pop, k=2, history_punish=1.):
        # k = k / len(pop)
        k = 0.3
        dom = np.array([ind.info['dominators'] for ind in pop])
        edom = np.exp(-k * dom)
        used = np.array([ind.info['used'] for ind in pop])
        prob = edom * history_punish ** used
        prob = prob / sum(prob)
        choosed = []
        i = np.random.choice(len(pop), p=prob)
        pop[i].info['used'] += 1
        return pop[i].copy()

    def generate(self, pop, n=None):
        n = n or self.n_next
        log.debug(self)
        # calculate dominators before checking formula
        pop.calc_dominators()
        # add symmetry before crossover and mutation
        if self.add_sym:
            pop.add_symmetry()
        newpop = pop.__class__([], name='init')
        op_choosed_num = [0] * len(self.op_list)
        op_success_num = [0] * len(self.op_list)
        while len(newpop) < n:
            i = np.random.choice(len(self.op_list), p=self.op_prob)
            op_choosed_num[i] += 1
            op = self.op_list[i]
            if op.n_input == 1:
                cand = self.get_ind(pop)
            elif op.n_input == 2:
                cand = self.get_pair(pop)
            newind = op.get_new_individual(cand)
            if newind is not None:
                op_success_num[i] += 1
                newpop.append(newind)
        table = pt.PrettyTable()
        table.field_names = ['Operator', 'Probability ', 'SelectedTimes', 'SuccessNum']
        for i in range(len(self.op_list)):
            table.add_row([self.op_list[i].descriptor, 
                           self.op_prob[i],
                           op_choosed_num[i],
                           op_success_num[i]])
        log.info(table)
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
