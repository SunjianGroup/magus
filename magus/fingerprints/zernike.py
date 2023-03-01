import numpy as np
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list
from magus.utils import FINGERPRINT_PLUGIN
from .base import FingerprintCalculator

#@FINGERPRINT_PLUGIN.register('zernike')
class ZernikeFp(FingerprintCalculator):
    def __init__(self, symbols, cutoff=4.0, nmax=8, lmax=None, ncut=4, diag=True, eleParm=None, **kwargs):
        self.cutoff = cutoff
        lmax = lmax or nmax
        assert lmax <= nmax
        elems = [atomic_numbers[element] for element in symbols]
        eleParm = eleParm or list(range(100))
        self.eleDic = {}
        for i, ele in enumerate(elems):
            self.eleDic[ele] = i
        self.numEles = len(elems)
        self.part = lrpot.CalculateFingerprints_part(cutoff, nmax, lmax, ncut, diag)
        self.Nd=self.part.Nd
        self.totNd = self.Nd * self.numEles
        self.part.eleParm = eleParm

    def get_all_fingerprints(self, atoms):
        Nat = len(atoms)
        totNd = self.Nd * self.numEles
        self.part.totNd=totNd
        self.part.SetNat(Nat)

        nl = neighbor_list('ijdD', atoms, self.cutoff, max_nbins=10)
        sortNl = [[] for _ in range(Nat)]
        for i,j,d,D in zip(*nl):                                 #All numbers must be double here
            sortNl[i].extend([i, j, d, D[0], D[1], D[2], atoms.numbers[j]])

        for ith in range(Nat):
            self.part.SetNeighbors(ith, sortNl[ith]) 
        #Finish the loop above before starting the loop below

        eFps = np.zeros((Nat, totNd))
        fFps = np.zeros((Nat, Nat, 3 ,totNd))
        sFps = np.zeros((Nat, 3, 3 ,totNd))

        for i in range(Nat):
            cenEleInd = self.eleDic[atoms.numbers[i]]
            self.part.get_fingerprints(i, cenEleInd)
            eFps[i] = self.part.GeteFp()                           #returns list of length totNd                         #returns array of Nat*3*totNd
            fFps[i] = np.array(self.part.GetfFps()).reshape(Nat,3,totNd)    #returns list of length Nat*3*totNd
            sFps[i] = np.array(self.part.GetsFps()).reshape(3,3,totNd) #returns list of length (3,3,totNd)
        sFps = sFps[:,[0,1,2,1,0,0],[0,1,2,2,2,1],:]
        sFps = np.zeros_like(sFps)
        eFps = np.sum(eFps,axis=0)
        fFps = -np.sum(fFps,axis=0).reshape(Nat*3,totNd)
        return eFps, fFps , sFps
