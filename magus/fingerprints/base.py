import abc


class FingerprintCalculator(abc.ABC):
    @abc.abstractmethod
    def get_all_fingerprints(self, atoms):
        pass
