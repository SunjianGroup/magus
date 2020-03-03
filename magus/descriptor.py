######################################################################
import numpy as np
from ase.neighborlist import NeighborList, neighbor_list, NewPrimitiveNeighborList
from ase.data import atomic_numbers
from ase import io
from . import lrpot
##############################################################################

class CalculateFingerprints:
    def __init__(self):
        pass

    def get_all_fingerprints(self,atoms):
        pass

class ZernikeFp(CalculateFingerprints):
    def __init__(self, cutoff, nmax, lmax, ncut, elems, diag=False, norm=False ,eleParm=None):
        if not lmax:
            lmax = nmax
        assert lmax <= nmax
        self.cutoff=cutoff
        self.nmax = nmax
        self.lmax = lmax
        self.ncut = ncut
        self.elems = elems
        self.diag = diag
        # parameters of elements
        self.norm = norm
        if not eleParm:
            eleParm = list(range(100))

        self.elems = elems
        eleDic = {}
        for i, ele in enumerate(elems):
            eleDic[ele] = i
        self.eleDic = eleDic

        self.numEles = len(elems)

        self.part=lrpot.CalculateFingerprints_part(cutoff, nmax, lmax, ncut, diag)

        self.Nd=self.part.Nd
        self.totNd = self.Nd * self.numEles

        self.part.SeteleParm(1.0*np.array(eleParm)) #All numbers must be double here


    def get_all_fingerprints(self,atoms):

        Nat = len(atoms)
        totNd = self.Nd * self.numEles
        self.part.totNd=totNd
        self.part.SetNat(Nat)

        nl = neighbor_list('ijdD', atoms, self.cutoff)
        sortNl = [[] for _ in range(Nat)]
        for i,j,d,D in zip(*nl):                                 #All numbers must be double here
            sortNl[i].extend([i*1.0, j*1.0, d, D[0],D[1],D[2],atoms.numbers[j]])

        for ith in range(Nat):
            self.part.SetNeighbors(ith, np.array(sortNl[ith])) 
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
        return eFps, fFps , sFps