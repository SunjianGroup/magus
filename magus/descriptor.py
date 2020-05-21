######################################################################
import numpy as np
from ase.neighborlist import NeighborList, neighbor_list, NewPrimitiveNeighborList
from ase.data import atomic_numbers
from ase import io
from . import lrpot
from .utils import checkParameters
from .fingerprint import Fingerprint
##############################################################################
"""
TODO Efps Ffps should be calculate separately
"""
class CalculateFingerprints:
    def __init__(self):
        pass

    def get_all_fingerprints(self,atoms):
        pass

class ZernikeFp(CalculateFingerprints):
    def __init__(self, parameters):
        
        Requirement = ['symbols']
        Default = {'cutoff': 4.0,'nmax': 4,'lmax':None,'ncut':4,'diag':True,'eleParm':None}
        checkParameters(self,parameters,Requirement,Default)
        self.elems = [atomic_numbers[element] for element in self.symbols]
        if not self.lmax:
            self.lmax = self.nmax
        assert self.lmax <= self.nmax
       
        # parameters of elements
        if not self.eleParm:
            self.eleParm = list(range(100))

        eleDic = {}
        for i, ele in enumerate(self.elems):
            eleDic[ele] = i
        self.eleDic = eleDic

        self.numEles = len(self.elems)

        self.part=lrpot.CalculateFingerprints_part(\
            self.cutoff, self.nmax, self.lmax, self.ncut, self.diag)

        self.Nd=self.part.Nd
        self.totNd = self.Nd * self.numEles

        self.part.SeteleParm(1.0*np.array(self.eleParm)) #All numbers must be double here


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

        eFps = np.sum(eFps,axis=0)
        fFps = -np.sum(fFps,axis=0).reshape(Nat*3,totNd)
        return eFps, fFps , sFps

class GofeeFp(CalculateFingerprints):
    def __init__(self, parameters):
        #self.fingerprint = Fingerprint()
        if hasattr(parameters,'descriptor_parm'):
            self.fingerprint = Fingerprint(**parameters.descriptor_parm)
        else:
            self.fingerprint = Fingerprint()

    def get_all_fingerprints(self,atoms):
        eFps = self.fingerprint.get_feature(atoms)
        fFps = self.fingerprint.get_featureGradient(atoms)
        sFps = 0
        return eFps, fFps , sFps