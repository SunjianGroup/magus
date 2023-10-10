# TODO
# how to set k in edom
import logging
import numpy as np
from magus.utils import *
import prettytable as pt
from collections import defaultdict
import yaml
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

def f_prob(func_name = 'exp', k = 0.3):

    def exp(dom):
        return np.exp(-k * dom)
    def liner(dom):
        """
        [https://doi.org/10.1063/1.3097197]
        p[i] = p1 - (i - 1) p1 / c ; recommand value c: 2/3 population size
        """
        return 1 - (dom - 1) / (k * len(dom))
        
    if func_name == 'exp':
        return exp
    elif func_name == 'liner':
        return liner
    else:
        raise Exception("Unknown function name {}".format(func_name))


class GAGenerator:
    def __init__(self, op_list, op_prob, **parameters):
        Requirement = ['pop_size', 'n_cluster']
        Default={'rand_ratio': 0.3, 'add_sym': True, 'history_punish':1.0, 'k': 0.3, 'choice_func': 'exp'}
        check_parameters(self, parameters, Requirement, Default)

        assert len(op_list) == len(op_prob), "number of operations and probabilities not match"
        assert np.sum(op_prob) > 0 and np.all(op_prob >= 0), "unreasonable probability are given"
        self.op_list = op_list
        self.op_prob = op_prob / np.sum(op_prob)

        self.gen = 1

    def __repr__(self):
        ret = self.__class__.__name__
        ret += "\n-------------------"
        c, m = "\nCrossovers:", "\nMutations:"
        for op, prob in zip(self.op_list, self.op_prob):
            if op.n_input == 1:
                m += "\n {}: {:>5.2f}%".format(op.__class__.__name__.ljust(20, ' '), prob * 100)
            elif op.n_input == 2:
                c += "\n {}: {:>5.2f}%".format(op.__class__.__name__.ljust(20, ' '), prob * 100)
        ret += m + c
        ret += "\nRandom Ratio         : {:.2%}".format(self.rand_ratio)
        ret += "\nNumber of cluster    : {}".format(self.n_cluster)
        ret += "\nAdd symmertry        : {}".format(self.add_sym)
        if self.history_punish != 1.0:
            ret += "\nHistory punishment   : {}".format(self.history_punish)
        ret += "\nSelection function   {}; k = {}".format(self.choice_func, self.k)
        ret += "\n-------------------\n"
        return ret

    @property
    def n_next(self):
        return int(self.pop_size * (1 - self.rand_ratio))

    def get_parents(self, pop, n_input):
        if n_input == 1:
            return self.get_ind(pop)
        elif n_input == 2:
            return self.get_pair(pop)

    def get_pair(self, pop, n_try=50):
        history_punish = self.history_punish
        assert 0 < history_punish <= 1, "history_punish should between 0 and 1"

        dom = np.array([ind.info['dominators'] for ind in pop])
        edom = (f_prob(k = self.k))(dom)
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
            i, j = np.random.choice(indices, 2 , p=prob)
            pop[i].info['used'] += 1
            pop[j].info['used'] += 1
            return pop[i].copy(), pop[j].copy()

        indices = np.arange(len(pop))
        prob = edom[indices] * history_punish ** used[indices]
        prob = prob / sum(prob)
        i, j = np.random.choice(indices, 2 , p=prob)
        pop[i].info['used'] += 1
        pop[j].info['used'] += 1
        return pop[i].copy(), pop[j].copy()

    def get_ind(self, pop):
        history_punish = self.history_punish
        
        dom = np.array([ind.info['dominators'] for ind in pop])
        edom = (f_prob(k = self.k))(dom)
        used = np.array([ind.info['used'] for ind in pop])
        prob = edom * history_punish ** used
        prob = prob / sum(prob)
        choosed = []
        i = np.random.choice(len(pop), p=prob)
        pop[i].info['used'] += 1
        return pop[i].copy()

    def generate(self, pop, n):
        log.debug(self)
        # Add symmetry before crossover and mutation
        if self.add_sym:
            pop.add_symmetry()
        newpop = pop.__class__([], name='init', gen=self.gen)
        op_choosed_num = [0] * len(self.op_list)
        op_success_num = [0] * len(self.op_list)
        # Ensure that the operator is selected at least once
        # for i, op in enumerate(self.op_list):
        #     op_choosed_num[i] += 1
        #     cand = self.get_parents(pop, op.n_input)
        #     newind = op.get_new_individual(cand)
        #     if newind is not None:
        #         op_success_num[i] += 1
        #         newpop.append(newind)
        while len(newpop) < n:
            i = np.random.choice(len(self.op_list), p=self.op_prob)
            op_choosed_num[i] += 1
            op = self.op_list[i]
            cand = self.get_parents(pop, op.n_input)
            newind = op.get_new_individual(cand)
            if newind is not None:
                op_success_num[i] += 1
                newpop.append(newind)
        table = pt.PrettyTable()
        table.field_names = ['Operator', 'Probability ', 'SelectedTimes', 'SuccessNum']
        for i in range(len(self.op_list)):
            table.add_row([self.op_list[i].descriptor,
                           '{:.2%}'.format(self.op_prob[i]),
                           op_choosed_num[i],
                           op_success_num[i]])
        log.info("OP infomation: \n" + table.__str__())
        newpop.check()
        return newpop

    def select(self, pop, num):
        if num < len(pop):
            pop = pop[np.random.choice(len(pop), num, False)]
        return pop

    def get_next_pop(self, pop, n_next=None):
        # calculate dominators before choose structures
        pop.del_duplicate()
        pop.calc_dominators()
        n_next = n_next or self.n_next
        self.gen += 1
        newpop = self.generate(pop, n_next)
        return self.select(newpop, n_next)
    
    def save_all_parm_to_yaml(self):
        d = {}
        for op, prob in zip(self.op_list, self.op_prob):
            d[op.__class__.__name__] = {}
            d[op.__class__.__name__]['prob'] = float(prob)
            for k in op.Default.keys():
                d[op.__class__.__name__][k] = getattr(op, k)
        
        d['rand_ratio'] = self.rand_ratio
        d['n_cluster'] = self.n_cluster
        d['add_sym'] = self.add_sym
        d['history_punish'] = self.history_punish
        d['choice_func'] = self.choice_func
        d['k'] = self.k
        
        with open('gaparm.yaml', 'w') as f:
            f.write(yaml.dump(d))
        return 


class AutoOPRatio(GAGenerator):
    def __init__(self, op_list, op_prob, **parameters):
        Default = {'good_ratio': 0.6, 'auto_random_ratio': True}
        check_parameters(self, parameters, [], Default)
        super().__init__(op_list, op_prob, **parameters)

    def change_op_ratio(self, pop):
        total_nums = defaultdict(int)
        good_nums = defaultdict(int)
        for ind in pop:
            origin = ind.info['origin']
            if origin == 'seed':
                continue
            if not self.auto_random_ratio and origin == 'random':
                continue
            total_nums[origin] += 1
            if ind.info['dominators'] < len(pop) * self.good_ratio:
                good_nums[origin] += 1
        op_grade = {op: good_nums[op] ** 2 / total_nums[op] for op in total_nums if total_nums[op] > 0}
        table = pt.PrettyTable()
        table.field_names = ['Operator', 'Total ', 'Good', 'Grade']
        for op in self.op_list:
            grade = op_grade[op.descriptor] if op.descriptor in op_grade else 0
            table.add_row([op.descriptor,
                           total_nums[op.descriptor],
                           good_nums[op.descriptor],
                           np.round(grade, 3)])
        if self.auto_random_ratio and self.gen > 2:
            grade = op_grade['random'] if 'random' in op_grade else 0
            table.add_row(['random', total_nums['random'], good_nums['random'], np.round(grade, 3)])
        log.debug("OP grade: \n" + table.__str__())
        if self.auto_random_ratio and self.gen > 2:
            if 'random' not in op_grade:
                op_grade['random'] = 0
            self.rand_ratio = 0.5 * (op_grade['random'] / sum(op_grade.values()) + self.rand_ratio)
            del op_grade['random']
        for i, op in enumerate(self.op_list):
            if op.descriptor in op_grade:
                self.op_prob[i] = 0.5 * (op_grade[op.descriptor] / sum(op_grade.values()) + self.op_prob[i])
            else:
                self.op_prob[i] = 0.5 * self.op_prob[i]
        self.op_prob /= np.sum(self.op_prob)


    def get_next_pop(self, pop, n_next=None):
        pop.calc_dominators()
        if self.gen > 1:
            self.change_op_ratio(pop)
            #self.save_all_parm_to_yaml()
        n_next = n_next or self.n_next
        newpop = self.generate(pop, n_next)
        self.gen += 1
        return self.select(newpop, n_next)
