import numpy as np
from . import GenerateNew
from ase.data import atomic_numbers, covalent_radii
import ase,ase.io
import copy
from .utils import *
import os

class BaseGenerator:
    def __init__(self,p):
        self.p=p
        Requirement=['symbols','formula','numFrml']
        Default={'threshold':1.0,'maxAttempts':50,'method':2,'volRatio':1.5,'spgs':np.arange(1,231),'maxtryNum':100}
        self.checkParameters(Requirement,Default)
        self.radius = p.raidus if hasattr(p,'radius') else [covalent_radii[atomic_numbers[atom]] for atom in self.symbols]
        
        self.meanVolume = p.meanVolume if hasattr(p,'meanVolume') else 4*np.pi/3*np.sum(np.array(self.radius)**3*np.array(self.formula))*self.volRatio/sum(self.formula)
        self.minVolume = p.minVolume if hasattr(p,'minVolume') else self.meanVolume*0.5
        self.maxVolume = p.maxVolume if hasattr(p,'maxVolume') else self.meanVolume*1.5
        """
        self.minLen=p.minLen if hasattr(p,'minLen') else [2*np.max(self.radius)]*3
        self.maxLen=p.maxLen if hasattr(p,'maxLen') else [(self.maxVolume*np.max(self.numFrml))**(1./3)]*3
        """

    def updatevolRatio(self,volRatio):
        self.meanVolume *= volRatio/self.volRatio
        self.minVolume *= volRatio/self.volRatio
        self.maxVolume *= volRatio/self.volRatio
        self.volRatio=volRatio

    def checkParameters(self,Requirement=[],Default={}):
        for key in Requirement:
            if not hasattr(self.p, key):
                raise Exception("Mei you '{}' wo suan ni ma?".format(key))
            setattr(self,key,getattr(self.p,key))

        for key in Default.keys():
            if not hasattr(self.p,key):
                setattr(self,key,Default[key])
            else:
                setattr(self,key,getattr(self.p,key))

    def getVolumeandLattice(self,numlist):
        minVolume = self.minVolume*np.sum(numlist)
        maxVolume = self.maxVolume*np.sum(numlist)
        minLattice= [2*np.max(self.radius)]*3+[60]*3
        maxLattice= [maxVolume/2/np.max(self.radius)]*3+[120]*3
        return minVolume,maxVolume,minLattice,maxLattice

    def Generate_ind(self,spg,numlist):
        spg=int(spg)
        numType = len(self.formula)
        generator = GenerateNew.Info()
        generator.spg = spg
        generator.spgnumber = 1
        generator.maxAttempts = self.maxAttempts
        generator.threshold=self.threshold
        generator.method=self.method
        # generator.forceMostGeneralWyckPos=True
        
        minVolume,maxVolume,minLattice,maxLattice=self.getVolumeandLattice(numlist)
        generator.minVolume = minVolume
        generator.maxVolume = maxVolume
        generator.SetLatticeMins(minLattice[0], minLattice[1], minLattice[2], minLattice[3], minLattice[4], minLattice[5])
        generator.SetLatticeMaxes(maxLattice[0], maxLattice[1], maxLattice[2], maxLattice[3], maxLattice[4], maxLattice[5])

        numbers=[]
        for i in range(numType):
            generator.AppendAtoms(int(numlist[i]), str(i), self.radius[i], False)
            numbers.extend([atomic_numbers[self.symbols[i]]]*numlist[i])

        label = generator.PreGenerate()
        if label:
            cell = generator.GetLattice(0)
            cell = np.reshape(cell, (3,3))
            positions = generator.GetPosition(0)
            positions = np.reshape(positions, (-1, 3))
            positions = np.dot(positions,cell)
            atoms = ase.Atoms(cell=cell, positions=positions, numbers=numbers, pbc=1)

            return label, atoms
        else:
            return label, None

    def afterprocessing(self,ind,nfm):
        ind.info['symbols'] = self.symbols
        ind.info['formula'] = self.formula
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['Origin'] = 'random'
        return ind

    def Generate_pop(self,popSize):
        buildPop = []
        tryNum=0
        while tryNum<self.maxtryNum and popSize > len(buildPop):
            nfm = np.random.choice(self.numFrml)
            spg = np.random.choice(self.spgs)
            numlist=np.array(self.formula)*nfm
            label,ind = self.Generate_ind(spg,numlist)
            if label:
                self.afterprocessing(ind,nfm)
                buildPop.append(ind)
            else:
                tryNum+=1

        # Allow P1 structure
        if popSize > len(buildPop):
            for i in range(popSize - len(buildPop)):
                nfm = np.random.choice(self.numFrml)
                spg = np.random.choice(self.spgs)
                numlist=np.array(self.formula)*nfm
                label,ind = self.Generate_ind(spg,numlist)
                if label:
                    self.afterprocessing(ind,nfm)
                    buildPop.append(ind)
                else:
                    label,ind = self.Generate_ind(1,numlist)
                    if label:
                        self.afterprocessing(ind,nfm)
                        buildPop.append(ind)
        return buildPop

class LayerGenerator(BaseGenerator):
    def __init__(self, p):
        super().__init__(p)
        super().checkParameters(Requirement=['cmax'])

        self.cmax=p.cmax
        self.d=p.d if hasattr(p,'d') else 15
        self.minLen=p.minLen if hasattr(p,'minLen') else [2*np.max(self.radius)]*3
        amax=(self.maxVolume*np.max(self.numFrml)/self.cmax)**(1./2)
        self.maxLen=p.maxLen if hasattr(p,'maxLen') else [amax,amax,self.cmax]

    def addVacuumlayer(self,ind):
        c=ind.get_cell()[2]
        c_=ind.get_reciprocal_cell()[2]
        k=self.d*np.linalg.norm(c_)/np.dot(c,c_)+1
        ind.cell[2]*=k

    def afterprocessing(self,ind,nfm): 
        super().afterprocessing(ind,nfm)
        #self.addVacuumlayer(ind)

class MoleculeGenerator(BaseGenerator):
    pass


class VarGenerator(BaseGenerator):
    def __init__(self,p):
        super().__init__(p)
        super().checkParameters(Requirement=['minAt','maxAt'],Default={'fullEles':True})
        self.projection_matrix=np.dot(self.formula.T,np.linalg.pinv(self.formula.T))
     

    def afterprocessing(self,ind):
        ind.info['symbols'] = self.symbols
        ind.info['formula'] = self.formula
        ind.info['numOfFormula'] = 1
        ind.info['parentE'] = 0
        ind.info['Origin'] = 'random'
        return ind

    def Generate_pop(self,popSize):
        buildPop = []
        for i in range(popSize):
            for j in range(self.maxtryNum):
                numAt = np.random.randint(self.minAt, self.maxAt+1)
                numlist = np.random.rand(len(self.symbols))
                numlist *= numAt/np.sum(numlist)
                numlist = np.rint(np.dot(self.projection_matrix,numlist)).astype(np.int)
                if np.sum(numlist) < self.minAt or np.sum(numlist) > self.maxAt or (self.fullEles and 0 in numlist) or np.sum(numlist<0)>0:
                    continue

                spg = np.random.choice(self.spgs)

                label,ind = self.Generate_ind(spg,numlist)
                if label:
                    self.afterprocessing(ind)
                    buildPop.append(ind)
                    break
                else:
                    continue
        return buildPop


def read_seeds(parameters, seedFile='Seeds/POSCARS'):
    seedPop = []
    setSym = parameters['symbols']
    setFrml = parameters['formula']
    minAt = parameters['minAt']
    maxAt = parameters['maxAt']
    calcType = parameters['calcType']

    if os.path.exists(seedFile):
        readPop = ase.io.read(seedFile, index=':', format='vasp-xdatcar')
        if len(readPop) > 0:
            logging.info("Reading Seeds ...")

        seedPop = read_bare_atoms(readPop, setSym, setFrml, minAt, maxAt, calcType)

    logging.info("Read Seeds: %s"%(len(seedPop)))
    return seedPop


#test
if __name__ == '__main__':
    class EmptyClass:
        def __init__(self):
            pass
    import ase.io
    p=EmptyClass()
    Requirement=['symbols','formula','numFrml']
    p.symbols=['C','H','O','N']
    p.formula=np.array([1,4,1,2])
    p.numFrml=[1]
    p.volRatio=2

    g=BaseGenerator(p)
    buildind=g.Generate_pop(10)
    ase.io.write('a.traj',buildind)
