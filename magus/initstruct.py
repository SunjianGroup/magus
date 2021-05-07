import numpy as np
try:
    from . import GenerateNew
except:
    import GenerateNew
from ase.data import atomic_numbers, covalent_radii
from ase import Atoms,build
from ase.spacegroup import Spacegroup
from ase.geometry import cellpar_to_cell,cell_to_cellpar
from scipy.spatial.distance import cdist, pdist
import ase,ase.io
import copy
import logging
from .utils import *


log = logging.getLogger(__name__)


class Generator:
    def __init__(self,parameters):
        self.p = EmptyClass()
        Requirement=['symbols','formula','minAt','maxAt','spgs','dRatio','fixCell','setCellPar', 'bondRatio', 'molMode']
        Default={'threshold':1.0,'maxAttempts':50,'method':1,
        'volRatio':1.5,'maxtryNum':100,'minLattice':None,'maxLattice':None, 'dimension':3, 'choice':0}
        checkParameters(self.p,parameters,Requirement,Default)
        radius = [float(covalent_radii[atomic_numbers[atom]]) for atom in self.p.symbols]
        checkParameters(self.p,parameters,[],{'radius':radius})

    def update_volume_ratio(self, volume_ratio):
        log.debug("change volRatio from {} to {}".format(self.p.volRatio, volume_ratio))
        self.p.volRatio = volume_ratio

    def get_swap(self):
        M = np.array([
            [[1,0,0],[0,1,0],[0,0,1]],
            [[0,1,0],[1,0,0],[0,0,1]],
            [[0,1,0],[0,0,1],[1,0,0]],
            [[1,0,0],[0,0,1],[0,1,0]],
            [[0,0,1],[1,0,0],[0,1,0]],
            [[0,0,1],[0,1,0],[1,0,0]]])
        return M[np.random.randint(6)]

    def getVolumeandLattice(self,numlist):
        # Recalculate atomic radius, considering the change of radius in molecular crystal mode
        atomicR = [float(covalent_radii[atomic_numbers[atom]]) for atom in self.p.symbols]
        Volume = np.sum(4*np.pi/3*np.array(atomicR)**3*np.array(numlist))*self.p.volRatio
        minVolume = Volume*0.5
        maxVolume = Volume*1.5
        minLattice= [2*np.max(self.p.radius)]*3+[60]*3
        # maxLattice= [maxVolume/2/np.max(self.p.radius)]*3+[120]*3
        maxLattice= [maxVolume**(1./3)]*3+[120]*3
        if self.p.minLattice:
            minLattice = self.p.minLattice
            minVolume = np.linalg.det(cellpar_to_cell(minLattice)) 
        if self.p.maxLattice:
            maxLattice = self.p.maxLattice
            maxVolume = np.linalg.det(cellpar_to_cell(maxLattice)) 
        if self.p.fixCell:
            minLattice = self.p.setCellPar
            minVolume = np.linalg.det(cellpar_to_cell(minLattice))
            maxLattice = [l+0.01 for l in minLattice]
            maxVolume = np.linalg.det(cellpar_to_cell(maxLattice))
        return minVolume,maxVolume,minLattice,maxLattice

    def Generate_ind(self,spg,numlist):
        spg=int(spg)
        numType = len(numlist)
        generator = GenerateNew.Info()
        generator.spg = spg
        generator.spgnumber = 1
        generator.maxAttempts = self.p.maxAttempts
        generator.dimension = self.p.dimension
        try:
            if self.p.vacuum:
                generator.vacuum = self.p.vacuum
        except:
            pass
        generator.choice = self.p.choice
        if self.p.molMode:
            generator.threshold=self.p.bondRatio
        else:
            generator.threshold=self.p.dRatio
        generator.method=self.p.method
        generator.forceMostGeneralWyckPos=False
        generator.UselocalCellTrans = 'y'
        generator.GetConventional = True

        minVolume,maxVolume,minLattice,maxLattice=self.getVolumeandLattice(numlist)
        # TODO should be encapsulated into HanYu code
        swap_matrix = self.get_swap() 
        minLattice = np.kron(np.array([[1,0],[0,1]]), swap_matrix) @ minLattice
        maxLattice = np.kron(np.array([[1,0],[0,1]]), swap_matrix) @ maxLattice

        generator.minVolume = minVolume
        generator.maxVolume = maxVolume
        generator.SetLatticeMins(minLattice[0], minLattice[1], minLattice[2], minLattice[3], minLattice[4], minLattice[5])
        generator.SetLatticeMaxes(maxLattice[0], maxLattice[1], maxLattice[2], maxLattice[3], maxLattice[4], maxLattice[5])
        numbers=[]
        for i in range(numType):
            if numlist[i] > 0:
                if self.p.molMode:
                    mole = self.p.inputMols[i]
                    if len(mole) > 1:
                        radius = np.array([covalent_radii[atomic_numbers[atom.symbol]] for atom in mole])
                        radius = np.array(list({}.fromkeys(radius).keys()))
                        positions = mole.positions.reshape(-1)
                        symbols = mole.get_chemical_symbols()
                        uni_symbols = list({}.fromkeys(symbols).keys())
                        assert len(uni_symbols)<5 
                        #TODO char array
                        namearray = [str(_s) for _s in uni_symbols]
                        numinfo = np.array([symbols.count(s) for s in uni_symbols],dtype=float)

                        symprec = self.p.symprec
                        generator.threshold_mol = self.p.threshold_mol
                        
                        generator.AppendMoles(int(numlist[i]),mole.get_chemical_formula()\
                            ,radius, positions, numinfo, namearray, symprec)

                        number = sum([num for num in [[atomic_numbers[s]]*int(n)*numlist[i] \
                            for s,n in zip(uni_symbols,numinfo)]],[])
                        numbers.extend(number)
                    else:
                        symbol = mole.get_chemical_symbols()[0]
                        radius = covalent_radii[atomic_numbers[symbol]]
                        generator.AppendAtoms(int(numlist[i]), symbol, radius, False)
                        numbers.extend([atomic_numbers[symbol]]*numlist[i])
                else:
                    generator.AppendAtoms(int(numlist[i]), str(i), self.p.radius[i], False)
                    numbers.extend([atomic_numbers[self.p.symbols[i]]]*numlist[i])

        label = generator.PreGenerate(np.random.randint(1000))

        if label:
            cell = generator.GetLattice(0)
            cell = np.reshape(cell, (3,3))
            cell_ = np.linalg.inv(swap_matrix) @ cell
            Q, L = np.linalg.qr(cell_.T)
            scaled_positions = generator.GetPosition(0)
            scaled_positions = np.reshape(scaled_positions, (-1, 3))
            positions = scaled_positions @ cell @ Q
            if np.linalg.det(L) < 0:
                L[2, 2] *= -1
                positions[:, 2] *= -1
            atoms = ase.Atoms(cell=L.T, positions=positions, numbers=numbers, pbc=1)
            atoms.wrap(pbc=[1, 1, 1])
            atoms = build.sort(atoms)
            return label, atoms
        else:
            return label, None

class BaseGenerator(Generator):
    def __init__(self,parameters):
        super().__init__(parameters)
        minFrml = int(np.ceil(self.p.minAt/sum(self.p.formula)))
        maxFrml = int(self.p.maxAt/sum(self.p.formula))
        self.p.numFrml = list(range(minFrml, maxFrml + 1))

    def afterprocessing(self,ind,nfm):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = self.p.formula
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = 'random'
        return ind

    def Generate_pop(self,popSize,initpop=False):
        buildPop = []
        tryNum=0
        while tryNum<self.p.maxtryNum*popSize and popSize > len(buildPop):
            nfm = np.random.choice(self.p.numFrml)
            spg = np.random.choice(self.p.spgs)
            numlist=np.array(self.p.formula)*nfm
            label,ind = self.Generate_ind(spg,numlist)
            if label:
                self.afterprocessing(ind,nfm)
                buildPop.append(ind)
            else:
                tryNum+=1

        # Allow P1 structure
        if popSize > len(buildPop):
            for _ in range(popSize - len(buildPop)):
                nfm = np.random.choice(self.p.numFrml)
                spg = np.random.choice(self.p.spgs)
                numlist=np.array(self.p.formula)*nfm
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
        self.minLen=p.minLen if hasattr(p,'minLen') else [2*np.max(self.p.radius)]*3
        amax=(self.maxVolume*np.max(self.p.numFrml)/self.cmax)**(1./2)
        self.maxLen=p.maxLen if hasattr(p,'maxLen') else [amax,amax,self.cmax]

    def addVacuumlayer(self,ind):
        c=ind.get_cell()[2]
        c_=ind.get_reciprocal_cell()[2]
        k=self.d*np.linalg.norm(c_)/np.dot(c,c_)+1
        ind.cell[2]*=k

    def afterprocessing(self,ind,nfm):
        super().afterprocessing(ind,nfm)
        #self.addVacuumlayer(ind)

class MoleculeGenerator(Generator):
    def __init__(self,parameters):
        super().__init__(parameters)
        Requirement=['inputMols','molFormula','numFrml']
        Default = {'molMode':True, 'symprec':0.1, 'threshold_mol': 1.0}
        checkParameters(self.p,parameters,Requirement,Default)
        radius = [get_radius(mol) for mol in self.p.inputMols]
        checkParameters(self.p,parameters,[],{'radius':radius})

    def afterprocessing(self,ind,nfm):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = self.p.formula
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = 'random'
        return ind

    def Generate_pop(self,popSize,initpop=False):
        buildPop = []
        tryNum=0
        while tryNum<self.p.maxtryNum*popSize and popSize > len(buildPop):
            nfm = np.random.choice(self.p.numFrml)
            spg = np.random.choice(self.p.spgs)
            numlist=np.array(self.p.molFormula)*nfm
            label,ind = self.Generate_ind(spg,numlist)
            if label:
                self.afterprocessing(ind,nfm)
                buildPop.append(ind)
            else:
                tryNum+=1

        # Allow P1 structure
        if popSize > len(buildPop):
            for _ in range(popSize - len(buildPop)):
                nfm = np.random.choice(self.p.numFrml)
                spg = np.random.choice(self.p.spgs)
                numlist=np.array(self.p.molFormula)*nfm
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


class VarGenerator(Generator):
    def __init__(self,parameters):
        super().__init__(parameters)
        Requirement=['minAt','maxAt']
        Default={'fullEles':True,'eleSize':1}
        checkParameters(self.p,parameters,Requirement,Default)
        # self.projection_matrix=np.dot(self.p.formula.T,np.linalg.pinv(self.p.formula.T))
        self.p.invFrml = np.linalg.pinv(self.p.formula).tolist()

    def afterprocessing(self,ind,numlist,nfm):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = numlist
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = 'random'
        return ind

    def Generate_pop(self,popSize,initpop=False):
        buildPop = []
        for i in range(popSize):
            for j in range(self.p.maxtryNum):
                numAt = np.random.randint(self.p.minAt, self.p.maxAt+1)
                numlist = np.random.rand(len(self.p.symbols))
                numlist *= numAt/np.sum(numlist)
                nfm = np.rint(np.dot(numlist,self.p.invFrml)).astype(np.int)
                #if (nfm<0).any():
                #    continue
                nfm[np.where(nfm<0)] = 0
                numlist = np.dot(nfm,self.p.formula)
                # numlist = np.rint(np.dot(self.projection_matrix,numlist)).astype(np.int)
                if np.sum(numlist) < self.p.minAt or np.sum(numlist) > self.p.maxAt or (self.p.fullEles and 0 in numlist) or np.sum(numlist<0)>0:
                    continue

                spg = np.random.choice(self.p.spgs)

                label,ind = self.Generate_ind(spg,numlist)
                if label:
                    self.afterprocessing(ind,numlist,nfm)
                    buildPop.append(ind)
                    break
                else:
                    continue

        # Generate simple substance in variable mode
        if initpop:
            for n,symbol in enumerate(self.p.symbols):
                for i in range(self.p.eleSize):
                    for j in range(self.p.maxtryNum):
                        numAt = np.random.randint(self.p.minAt, self.p.maxAt+1)
                        numlist = [0]*len(self.p.symbols)
                        numlist[n] = numAt
                        spg = np.random.choice(self.p.spgs)

                        label,ind = self.Generate_ind(spg,numlist)
                        if label:
                            self.afterprocessing(ind,numlist,nfm)
                            buildPop.append(ind)
                            break
                        else:
                            continue
        return buildPop



def read_seeds(parameters, seedFile, goodSeed=False):
    seedPop = []
    setSym = parameters.symbols
    setFrml = parameters.formula
    minAt = parameters.minAt
    maxAt = parameters.maxAt
    calcType = parameters.calcType

    if os.path.exists(seedFile):
        if goodSeed:
            readPop = ase.io.read(seedFile, index=':', format='traj')
        else:
            readPop = ase.io.read(seedFile, index=':', format='vasp-xdatcar')
        if len(readPop) > 0:
            log.info("Reading Seeds ...")

        seedPop = read_bare_atoms(readPop, setSym, setFrml, minAt, maxAt, calcType)
        for ind in seedPop:
            if goodSeed:
                ind.info['origin'] = 'goodseed'
            else:
                ind.info['origin'] = 'seed'
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
