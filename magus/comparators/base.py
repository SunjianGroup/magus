import abc


class Comparator(abc.ABC):
    @abc.abstractmethod
    def looks_like(self, a1, a2):
        pass


class AndGate(Comparator):
    def __init__(self, comparator_list):
        self.comparator_list = comparator_list

    def looks_like(self, a1, a2):
        for comparator in self.comparator_list:
            if not comparator.looks_like(a1, a2):
                return False
        return True


class OrGate(Comparator):
    def __init__(self, comparator_list):
        self.comparator_list = comparator_list

    def looks_like(self, a1, a2):
        for comparator in self.comparator_list:
            if comparator.looks_like(a1, a2):
                return True
        return False
