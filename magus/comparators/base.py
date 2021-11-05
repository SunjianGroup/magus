import abc
from magus.utils import COMPARATOR_CONNECT_PLUGIN


class Comparator(abc.ABC):
    @abc.abstractmethod
    def looks_like(self, a1, a2):
        pass


@COMPARATOR_CONNECT_PLUGIN.register('and')
class AndGate(Comparator):
    def __init__(self, comparator_list):
        self.comparator_list = comparator_list

    def looks_like(self, a1, a2):
        for comparator in self.comparator_list:
            if not comparator.looks_like(a1, a2):
                return False
        return True


@COMPARATOR_CONNECT_PLUGIN.register('or')
class OrGate(Comparator):
    def __init__(self, comparator_list):
        self.comparator_list = comparator_list

    def looks_like(self, a1, a2):
        for comparator in self.comparator_list:
            if comparator.looks_like(a1, a2):
                return True
        return False
