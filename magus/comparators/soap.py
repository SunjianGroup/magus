from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize
from magus.utils import COMPARATOR_PLUGIN


@COMPARATOR_PLUGIN.register('soap')
class SoapComparator:
    def __init__(self, symbols, threshold=0.95, rcut=6.0, nmax=8, lmax=6, periodic=True, **kwargs):
        self.soap = SOAP(species=symbols, periodic=periodic, rcut=rcut, nmax=nmax, lmax=lmax)
        self.kernel = REMatchKernel(metric="linear", alpha=1, threshold=1e-6)
        self.threshold = threshold

    def looks_like(self, ind1, ind2):
        features_1 = normalize(self.soap.create(ind1))
        features_2 = normalize(self.soap.create(ind2))
        similarity = self.kernel.create([features_1, features_2])[0][1]
        return similarity > self.threshold
