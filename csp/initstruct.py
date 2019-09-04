import numpy as np
from . import GenerateNew
from ase.data import atomic_numbers, covalent_radii
import ase,ase.io
import copy
from .utils import *

class BaseGenerator:
    def __init__(self,p):
        self.p=p
        Requirement=['symbols','formula','numFrml']
        Default={'threshold':1.0,'maxAttempts':50,'method':2,'volRatio':1.5,'spgs':np.arange(1,231),'maxtryNum':100}
        self.checkParameters(Requirement,Default)
        self.radius = p.raidus if hasattr(p,'radius') else [covalent_radii[atomic_numbers[atom]] for atom in self.symbols]
        self.formula=np.array(self.formula)
        """
        self.meanVolume = p.meanVolume if hasattr(p,'meanVolume') else 4*np.pi/3*np.sum(np.array(self.radius)**3*np.array(self.formula))*self.volRatio
        self.minVolume = p.minVolume if hasattr(p,'minVolume') else self.meanVolume*0.5
        self.maxVolume = p.maxVolume if hasattr(p,'maxVolume') else self.meanVolume*1.5
        self.minLen=p.minLen if hasattr(p,'minLen') else [2*np.max(self.radius)]*3
        self.maxLen=p.maxLen if hasattr(p,'maxLen') else [(self.maxVolume*np.max(self.numFrml))**(1./3)]*3
        """

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

    def Generate_ind(self,spg,numlist):
        spg=int(spg)
        numType = len(self.formula)
        generator = GenerateNew.Info()
        generator.spg = spg
        generator.spgnumber = 1
        generator.maxAttempts = self.maxAttempts
        generator.threshold=self.threshold
        generator.method=self.method
        generator.UselocalCellTrans = 'y'
        # generator.forceMostGeneralWyckPos=True

        """
        generator.minVolume = self.minVolume*nfm
        generator.maxVolume = self.maxVolume*nfm
        generator.SetLatticeMins(self.minLen[0], self.minLen[1], self.minLen[2], 60, 60, 60)
        generator.SetLatticeMaxes(self.maxLen[0], self.maxLen[1], self.maxLen[2], 120, 120 ,120)
        """
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
        logging.debug(self.numFrml)
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
                numlist=self.formula[i]*nfm
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
     

    def afterprocessing(self,ind,numlist):
        ind.info['symbols'] = self.symbols
        ind.info['formula'] = numlist
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
                    self.afterprocessing(ind,numlist)
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
    p.symbols=['Na','B','F']
    p.formula=np.array([[1,0,0],[0,1,0],[0,0,1]])
    p.numFrml=[2,4]
    p.minAt=45
    p.maxAt=50
    g=VarGenerator(p)
    buildind=g.Generate_pop(10)
    for i,a in enumerate(buildind):
        ase.io.write('{}.cif'.format(i),a)
