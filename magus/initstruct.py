import numpy as np
from . import GenerateNew
from ase.data import atomic_numbers, covalent_radii
from ase import Atoms
from ase.spacegroup import Spacegroup
from ase.geometry import cellpar_to_cell,cell_to_cellpar
from scipy.spatial.distance import cdist, pdist
import ase,ase.io
import copy
from .utils import *

class Generator:
    def __init__(self,parameters):
        self.p = EmptyClass()
        Requirement=['symbols','formula','minAt','maxAt','spgs']
        Default={'threshold':1.0,'maxAttempts':50,'method':2,
        'volRatio':1.5,'maxtryNum':100,'minLattice':None,'maxLattice':None}
        checkParameters(self.p,parameters,Requirement,Default)
        radius = [covalent_radii[atomic_numbers[atom]] for atom in self.p.symbols]
        checkParameters(self.p,parameters,[],{'radius':radius})

    def updatevolRatio(self,volRatio):
        self.p.volRatio=volRatio
        logging.debug("new volRatio: {}".format(self.p.volRatio))

    def getVolumeandLattice(self,numlist):
        Volume = np.sum(4*np.pi/3*np.array(self.p.radius)**3*np.array(numlist))*self.p.volRatio
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
        return minVolume,maxVolume,minLattice,maxLattice

    def Generate_ind(self,spg,numlist):
        spg=int(spg)
        numType = len(numlist)
        generator = GenerateNew.Info()
        generator.spg = spg
        generator.spgnumber = 1
        generator.maxAttempts = self.p.maxAttempts
        generator.threshold=self.p.threshold
        generator.method=self.p.method
        generator.forceMostGeneralWyckPos=False
        generator.UselocalCellTrans = 'y'

        minVolume,maxVolume,minLattice,maxLattice=self.getVolumeandLattice(numlist)
        generator.minVolume = minVolume
        generator.maxVolume = maxVolume
        generator.SetLatticeMins(minLattice[0], minLattice[1], minLattice[2], minLattice[3], minLattice[4], minLattice[5])
        generator.SetLatticeMaxes(maxLattice[0], maxLattice[1], maxLattice[2], maxLattice[3], maxLattice[4], maxLattice[5])

        numbers=[]
        for i in range(numType):
            if numlist[i] > 0:
                generator.AppendAtoms(int(numlist[i]), str(i), self.p.radius[i], False)
                numbers.extend([atomic_numbers[self.p.symbols[i]]]*numlist[i])

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

class MoleculeGenerator(BaseGenerator):
    pass


class VarGenerator(Generator):
    def __init__(self,parameters):
        super().__init__(parameters)
        Requirement=['minAt','maxAt']
        Default={'fullEles':True,'eleSize':1}
        checkParameters(self.p,parameters,Requirement,Default)
        # self.projection_matrix=np.dot(self.p.formula.T,np.linalg.pinv(self.p.formula.T))
        self.p.invFrml = np.linalg.pinv(self.p.formula)

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

def equivalent_sites_rots(spg, scaled_positions, symprec=1e-3):
    """Returns equivalent sites and the relative rotations.

        Parameters:

        spg: init | ASE's Spacegroup
            spacegroup

        scaled_positions: list | array
            One non-equivalent site given in unit cell coordinates.

        symprec: float
            Minimum "distance" betweed two sites in scaled coordinates
            before they are counted as the same site.

        Returns:

        sites: array
            A NumPy array of equivalent sites.
        rots: list
            A list containing possible rotations for all equilvalent sites.
    """

    if isinstance(spg, int):
        spgObj = Spacegroup(spg)
    else:
        spgObj = spg

    assert isinstance(spgObj, Spacegroup)

    pos = np.array(scaled_positions)

    sites = []
    rotsArr = []

    for rot, trans in spgObj.get_symop():
        site = np.mod(np.dot(rot, pos) + trans, 1.)
        if not sites:
            sites.append(site)
            rots = [rot]
            rotsArr.append(rots)
            continue
        t = site - sites
        mask = np.all((abs(t) < symprec) | (abs(abs(t) - 1.0) < symprec), axis=1)
        if np.any(mask):
            inds = np.argwhere(mask).flatten()
            for ind in inds:
                rotsArr[ind].append(rot)
        else:
            sites.append(site)
            rots = [rot]
            rotsArr.append(rots)

    return np.array(sites), rotsArr




def generate_centers_cell(formula, spg, radius, minVol, maxVol):
    assert len(formula) == len(radius)
    print(formula, spg, radius, minVol, maxVol)
    numType = len(formula)
    generator = GenerateNew.Info()
    generator.spg = spg
    generator.spgnumber = 1
    if minVol:
        generator.minVolume = minVol
    if maxVol:
        generator.maxVolume = maxVol
        maxLen = maxVol**(1./3)
        generator.SetLatticeMaxes(maxLen, maxLen, maxLen, 120, 120 ,120)
    generator.maxAttempts = 500
    generator.threshold=1
    generator.method=2
    generator.forceMostGeneralWyckPos=False
    minLen = 2*max(radius)
    generator.SetLatticeMins(minLen, minLen, minLen, 60, 60, 60)
    generator.GetConventional = True
    numbers = []
    for i in range(numType):
        generator.AppendAtoms(formula[i], "{}".format(i), radius[i], False)
        numbers.extend([i]*formula[i])

    label = generator.PreGenerate()
    if label:
        cell = generator.GetLattice(0)
        cell = np.reshape(cell, (3,3))
        positions = generator.GetPosition(0)
        positions = np.reshape(positions, (-1, 3))
        wyckPos = generator.GetWyckPos(0)
        wyckPos = np.reshape(wyckPos, (-1,3))
        wyckName = generator.GetWyckLabel(0)
        wyckNum = [int(n) for n in wyckName]
        return label, cell, numbers, positions, wyckNum, wyckPos
    else:
        return label, None, None, None, None, None

def mol_radius_and_rltPos(atoms):
    pos = atoms.get_positions()
    center = pos.mean(0)
    dists = cdist([center], pos)
    radius = dists.max()
    rltPos = pos - center
    return radius, rltPos

def generate_one_mol_crystal(molFormula, spg, radius, rltPosList, molNumList, minVol=None, maxVol=None, fixCell = False, setCellPar = None,):
    assert len(radius) == len(molFormula) == len(rltPosList) == len(molNumList)
    # numType = len(molFormula)
    label, cell, molIndices, centers, wyckNum, wyckPos = generate_centers_cell(molFormula, spg, radius, minVol, maxVol)
    if fixCell:
        cell = cellpar_to_cell(setCellPar)
    # spgOb = Spacegroup(spg)
    # rotations = spgOb.get_rotations()
    # rotations = [r for r in rotations if (np.dot(r, r.T)==np.eye(3)).all()]
    #logging.debug(radius)
    #logging.debug("{}\t{}\t{}".format(spg, minVol, maxVol))
    # print(pdist(np.dot(centers, cell)))
    if label:
        # tmpAts = Atoms(cell=cell, scaled_positions=centers, numbers=[1]*len(molIndices), pbc=1)
        # print(tmpAts.get_all_distances(mic=True))
        numList = []
        posList = []
        for i, molInd in enumerate(wyckNum):
            randMat = rand_rotMat()
            wyckSite = wyckPos[i]
            sites, rotsArr = equivalent_sites_rots(spg, wyckSite)
            for pos, rots in zip(sites, rotsArr):
                numList.append(molNumList[molInd])
                molPos = np.dot(pos, cell) + np.dot(rltPosList[molInd], np.dot(randMat, random.choice(rots)))
                posList.append(molPos)

        # for i, molInd in enumerate(molIndices):
        #     numList.append(molNumList[molInd])
        #     molPos = np.dot(centers[i], cell) + np.dot(rltPosList[molInd], np.dot(randMat, random.choice(rotations)))
        #     # molPos = np.dot(centers[i], cell) + np.dot(rltPosList[molInd], randMat)
        #     posList.append(molPos)
        pos = np.concatenate(posList, axis=0)
        numbers = np.concatenate(numList, axis=0)
        argsort = np.argsort(numbers)
        numbers = numbers[argsort]
        pos = pos[argsort]
        atoms = Atoms(cell=cell, positions=pos, numbers=numbers, pbc=1)
        return atoms
    else:
        return None

def generate_mol_crystal_list(molList, molFormula, spgList, numStruct, smallRadius=False, fixCell = False, setCellPar = None,):
    assert len(molList) == len(molFormula)
    radius = []
    molNumList = []
    rltPosList = []
    meanVol = 0
    if smallRadius:
        rCoef = random.uniform(0.5, 1)
    else:
        rCoef = 1
    for i, mol in enumerate(molList):
        numbers = mol.get_atomic_numbers()
        rmol, rltPos = mol_radius_and_rltPos(mol)
        rAt = covalent_radii[numbers].max()
        rmol += rAt
        radius.append(rCoef*rmol)
        molNumList.append(numbers)
        rltPosList.append(rltPos)
        vol = 4*np.pi / 3 * rmol**3
        meanVol += vol * molFormula[i]

    minVol = 0.5*meanVol
    maxVol = 3*meanVol

    molPop = []
    for _ in range(numStruct):
        spg = random.choice(spgList)
        # atoms = generate_one_mol_crystal(molFormula, spg, radius, rltPosList, molNumList,)
        atoms = generate_one_mol_crystal(molFormula, spg, radius, rltPosList, molNumList,minVol, maxVol,fixCell=fixCell, setCellPar=setCellPar)
        if atoms:
            molPop.append(atoms)

    return molPop

def build_mol_struct(
    popSize,
    symbols,
    formula,
    inputMols,
    molFormula,
    numFrml = [1],
    spgs = range(1,231),
    tryNum = 10,
    bondRatio = 1.1,
    fixCell = False,
    setCellPar = None,
):
    buildPop = []
    for nfm in numFrml:
        numStruct = popSize//len(numFrml)
        if numStruct == 0:
            break
        inputMolFrml = [nfm*frml for frml in molFormula]
        # randomPop = generate_mol_crystal_list(inputMols, inputMolFrml, spgs, numStruct, smallRadius=False, fixCell=fixCell, setCellPar=setCellPar)
        randomPop = generate_mol_crystal_list(inputMols, inputMolFrml, spgs, numStruct, smallRadius=True, fixCell=fixCell, setCellPar=setCellPar)
        for ind in randomPop:
            ind.info['symbols'] = symbols
            ind.info['formula'] = formula
            ind.info['numOfFormula'] = nfm
            ind.info['molFormula'] = molFormula
            ind.info['parentE'] = 0
            ind.info['origin'] = 'random'
        buildPop.extend(randomPop)

    buildPop = check_mol_pop(buildPop, inputMols, bondRatio)

     # Build structure to fill buildPop
    for _ in range(tryNum):
        if popSize <= len(buildPop):
            break
        randomPop = []
        for _ in range(popSize - len(buildPop)):
            nfm = random.choice(numFrml)
            inputMolFrml = [nfm*frml for frml in molFormula]
            randomPop.extend(generate_mol_crystal_list(inputMols, inputMolFrml, spgs, 1, smallRadius=True,fixCell=fixCell, setCellPar=setCellPar))
            if len(randomPop) > 0:
                randomPop[-1].info['numOfFormula'] = nfm
                randomPop[-1].info['symbols'] = symbols
                randomPop[-1].info['formula'] = formula
                randomPop[-1].info['molFormula'] = molFormula
                randomPop[-1].info['parentE'] = 0
                randomPop[-1].info['origin'] = 'random'
        randomPop = check_mol_pop(randomPop, inputMols, bondRatio)
        buildPop.extend(randomPop)

    logging.debug("Build {} molecular crystals with small radius.".format(len(buildPop)))

    # Build large-radius structure to fill buildPop
    for _ in range(tryNum):
        if popSize <= len(buildPop):
            break
        randomPop = []
        for _ in range(popSize - len(buildPop)):
            nfm = random.choice(numFrml)
            inputMolFrml = [nfm*frml for frml in molFormula]
            randomPop.extend(generate_mol_crystal_list(inputMols, inputMolFrml, spgs, 1,fixCell=fixCell, setCellPar=setCellPar))
            if len(randomPop) > 0:
                randomPop[-1].info['numOfFormula'] = nfm
                randomPop[-1].info['symbols'] = symbols
                randomPop[-1].info['formula'] = formula
                randomPop[-1].info['molFormula'] = molFormula
                randomPop[-1].info['parentE'] = 0
                randomPop[-1].info['origin'] = 'random'
        buildPop.extend(randomPop)

    return buildPop


def read_seeds(parameters, seedFile):
    seedPop = []
    setSym = parameters.symbols
    setFrml = parameters.formula
    minAt = parameters.minAt
    maxAt = parameters.maxAt
    calcType = parameters.calcType

    if os.path.exists(seedFile):
        readPop = ase.io.read(seedFile, index=':', format='vasp-xdatcar')
        if len(readPop) > 0:
            logging.info("Reading Seeds ...")

        seedPop = read_bare_atoms(readPop, setSym, setFrml, minAt, maxAt, calcType)
        for ind in seedPop:
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
