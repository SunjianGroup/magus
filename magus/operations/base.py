import logging, yaml
from ase import Atoms 
from magus.utils import *
from magus.populations.individuals import *


log = logging.getLogger(__name__)


class OffspringCreator:
    main_info = [] # info need to be print
    Requirement = []
    Default = {'tryNum': 50}
    def __init__(self, **parameters):
        self.all_parameters = parameters
        check_parameters(self, parameters, self.Requirement, self.Default)
        self.descriptor = self.__class__.__name__

    def __repr__(self):
        d = {info: getattr(self, info) if hasattr(self, info) else None for info in self.main_info}
        out  = self.__class__.__name__ + ':\n'
        out += yaml.dump(d)
        return out

    def get_new_individual(self):
        raise NotImplementedError


class AdjointOP(OffspringCreator):
    def __init__(self, oplist):
        self.n_input = max([op.n_input for op in oplist])
        self.oplist = oplist

    def get_new_individual(self, inds):
        for op in self.oplist:
            if isinstance(inds, Atoms):
                assert op.n_input == 1
                inds = op.get_new_individual(inds)
            else:
                assert len(inds) == 2
                if op.n_input == 1:
                    inds = [op.get_new_individual(inds[0]), op.get_new_individual(inds[1])]
                elif op.n_input == 2:
                    inds = op.get_new_individual(inds)
        return inds


class Mutation(OffspringCreator):
    n_input = 1
    def mutate(self, ind):
        if isinstance(ind, Bulk):
            return self.mutate_bulk(ind)
        elif isinstance(ind, Layer):
            return self.mutate_layer(ind)
        else:
            pass

    def mutate_bulk(self, ind):
        raise NotImplementedError("{} cannot apply in bulk".format(self.descriptor))

    def mutate_layer(self, ind):
        raise NotImplementedError("{} cannot apply in layer".format(self.descriptor))

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
        newind.info = {}
        newind.info['parents'] = [ind.info['identity']]
        newind.info['parentE'] = ind.info['enthalpy']
        newind.info['pardom'] = ind.info['dominators']
        newind.info['origin'] = self.descriptor
        newind.info['fitness'] = {}
        newind.info['used'] = 0
        return newind


class Crossover(OffspringCreator):
    n_input = 2
    def cross(self, ind1, ind2):
        if isinstance(ind1, Bulk):
            return self.cross_bulk(ind1, ind2)
        elif isinstance(ind1, Layer):
            return self.cross_layer(ind1, ind2)
        else:
            pass

    def cross_bulk(self, ind1, ind2):
        raise NotImplementedError("{} cannot apply in bulk".format(self.descriptor))

    def cross_layer(self, ind1, ind2):
        raise NotImplementedError("{} cannot apply in layer".format(self.descriptor))

    def get_new_individual(self, parents):
        ind1, ind2 = parents
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
        newind.info = {}
        newind.info['parents'] = [ind1.info['identity'], ind2.info['identity']]
        newind.info['parentE'] = 0.5 * (ind1.info['enthalpy'] + ind2.info['enthalpy'])
        newind.info['pardom'] = 0.5 * (ind1.info['dominators'] + ind2.info['dominators'])
        newind.info['origin'] = self.descriptor
        newind.info['fitness'] = {}
        newind.info['used'] = 0
        return newind
