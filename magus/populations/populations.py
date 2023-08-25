import copy
import os, itertools, logging, numbers
from numpy.core.numeric import indices
import numpy as np
from sklearn import cluster
import ase.io
from magus.utils import check_parameters, get_units_numlist
from .individuals import Individual, get_Ind
from ..fitness import get_fitness_calculator
from ..generators import get_random_generator


log = logging.getLogger(__name__)
__all__ = ['FixPopulation', 'VarPopulation']


class Population:
    """
    a class of atoms population
    """
    batch_operation = [
        'find_spg', 'add_symmetry', 
        ]
    
    @classmethod
    def set_parameters(cls, **parameters):
        cls.all_parameters = parameters
        Requirement = ['results_dir', 'pop_size', 'symbols', 'formula', 'units']
        Default = {'check_seed': False}
        check_parameters(cls, parameters, Requirement, Default)
        if 'atoms_generator' not in parameters:
            cls.atoms_generator = get_random_generator(parameters)
        else:
            cls.atoms_generator = parameters['atoms_generator']
        parameters['symbol_numlist_pool'] = cls.atoms_generator.symbol_numlist_pool
        cls.Ind = get_Ind(parameters)
        cls.fit_calcs = get_fitness_calculator(parameters)

    def __init__(self, pop, name='temp', gen=''):
        self.pop = [ind if isinstance(ind, Individual) else self.Ind(ind) for ind in pop]
        self.name = name
        self.gen = gen
        log.debug('construct Population {} with {} individual'.format(name, len(pop)))
        for i, ind in enumerate(self.pop):
            if 'identity' not in ind.info:
                ind.info['identity'] = "{}{}-{}".format(name, gen, i)

    def __repr__(self):
        ret = self.__class__.__name__
        ret += "\n-------------------"
        ret += "\nInd Type           : {}".format(self.Ind.__name__)
        ret += "\nInd Numbers        : {}".format(len(self))
        ret += "\nPopulation Size    : {}".format(self.pop_size)
        ret += "\nDistance Dict      : {}".format(self.Ind.distance_dict)
        ret += "\n-------------------"
        return ret

    def __iter__(self):
        for i in self.pop:
            yield i

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            return self.pop[i]
        else:
            newpop = self.copy()
            if isinstance(i, slice):
                newpop.pop = newpop.pop[i]
            else:
                indices = np.array(i)
                if indices.dtype == bool:
                    try:
                        indices = np.arange(len(self))[indices]
                    except IndexError:
                        raise IndexError('length of item mask '
                                        'mismatches that of {0} '
                                        'object'.format(self.__class__.__name__))
                newpop.pop = [newpop.pop[i] for i in indices]
            return newpop
        
    def __setitem__(self, index, ind):
        self.pop[index] = ind.copy()

    def __len__(self):
        return len(self.pop)

    def __add__(self, other):
        newPop = self.copy()
        newPop.extend(other)
        return newPop

    def __iadd__(self, other):
        self.extend(other)

    def __contains__(self, ind):
        ind = ind if isinstance(ind, Individual) else self.Ind(ind)
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
        ind = ind if isinstance(ind, Individual) else self.Ind(ind)
        if 'identity' not in ind.info:
            ind.info['identity'] = "{}{}-{}".format(self.name, self.gen, len(self.pop))
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
            atoms.info['units'] = self.units
            pop.append(atoms)
        ase.io.write("{}/{}{}.traj".format(savedir, filename, gen), pop, format='traj')
        log.debug("save {}{}.traj".format(filename,gen))

    @property
    def volume_ratio(self):
        return np.mean([ind.volume_ratio for ind in self.pop])

    def calc_dominators(self):
        log.debug("calculating dominators...")
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
        log.debug("calculating fitness...")
        for fit_calc in self.fit_calcs:
            fit_calc.calc(self)

    def del_duplicate(self):
        self.calc_dominators()
        log.debug('del_duplicate {} begin, popsize:{}'.format(self.name, len(self.pop)))
        newpop = []
        # sort the pop so the better individual will be remained
        self.pop = sorted(self.pop, key=lambda x: (x.info['dominators'], x.info['gen']))
        for ind in self.pop:
            if not ind == newpop:
                newpop.append(ind)
        log.debug('del_duplicate survival: {}'.format(len(newpop)))
        self.pop = newpop

    def check(self):
        log.debug("check population {}, popsize:{}".format(self.name, len(self.pop)))
        checkpop = []
        for ind in self.pop:
            if ind.check():
                checkpop.append(ind)
        log.debug("check survival: {}".format(len(checkpop)))
        self.pop = checkpop

    def clustering(self, n_clusters):
        """
        clustering by fingerprints
        TODO may not be a class method
        """
        pop = [copy.copy(ind) for ind in self.pop]
        if n_clusters >= len(pop):
            return np.arange(len(pop)), pop

        fp = np.array([ind.fingerprint for ind in pop])
        labels = cluster.KMeans(n_clusters=n_clusters).fit_predict(fp)
        # TODO fix bug: clustering may fail if there are dulplicate structures
        # goodpop = [None] * n_clusters
        goodpop = [None] * len(set(labels))
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
        return  bestInds
        #return [self.pop[i] for i in best_i]

    def fill_up_with_random(self):
        raise NotImplementedError


class FixPopulation(Population):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)

    def fill_up_with_random(self):
        n_random = self.pop_size - len(self)
        add_frames = self.atoms_generator.generate_pop(n_random)
        self.extend(add_frames)


class VarPopulation(Population):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        check_parameters(cls, parameters, [], {'ele_size': 0})

    def fill_up_with_random(self):
        units = self.atoms_generator.units
        n_units = len(units)
        d_n_random = {format_filter: self.ele_size for format_filter in itertools.product([0, 1], repeat=n_units)}
        d_n_random[tuple([0] * n_units)] = 0
        d_n_random[tuple([1] * n_units)] = self.pop_size
        for ind in self.pop:
            d_n_random[tuple(np.clip(get_units_numlist(ind, units), 0, 1))] -= 1
        for format_filter, n_random in d_n_random.items():
            if n_random > 0:
                add_frames = self.atoms_generator.generate_pop(n_random, format_filter=format_filter)
                self.extend(add_frames)
