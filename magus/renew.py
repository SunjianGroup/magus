import random, sys, os, re, math, logging, yaml
from collections import Counter
import numpy as np
import spglib
from numpy import pi, sin, cos, tan, sqrt
from numpy import dot
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.gaussian_process import kernels
import ase.io
from ase import Atom, Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.phasediagram import PhaseDiagram
from ase.neighborlist import NeighborList
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.optimize import BFGS, FIRE, BFGSLineSearch, LBFGS, LBFGSLineSearch
from ase.units import GPa
from ase.constraints import UnitCellFilter#, ExpCellFilter
from sklearn import cluster
from .utils import *


class BaseEA:
    def __init__(self, parameters):

        self.parameters = parameters
        self.symbols = parameters.symbols
        self.formula = parameters.formula
        self.saveGood = parameters.saveGood
        self.addSym = parameters.addSym
        self.calcType = parameters.calcType

        # krigParm = parameters.krigParm
        self.randFrac = parameters.randFrac
        self.permNum = parameters.permNum
        self.latDisps = parameters.latDisps
        self.ripRho = parameters.ripRho
        self.rotNum = parameters.rotNum
        self.cutNum = parameters.cutNum
        self.slipNum = parameters.slipNum
        self.latNum = parameters.latNum
        self.grids = parameters.grids

        # self.kind = krigParm['kind']
        # self.xi = krigParm['xi']
        # self.kappaLoop = krigParm['kappaLoop']
        # self.scaled_factor = krigParm['scale']

        self.parent_factor = 0

        self.dRatio = parameters.dRatio
        self.bondRatio = parameters.bondRatio
        self.bondRange = parameters.bondRange
        self.molDetector = parameters.molDetector
        self.molMode = parameters.molMode

        self.newLen = int((self.parameters.popSize*(1-self.parameters.randFrac)))

    def heredity(self, cutNum):
        #curPop = standardize_pop(self.curPop, 1.)
        curPop = self.curPop
        symbols = self.symbols
        grids = self.grids
        labels, goodPop = self.labels, self.goodPop
        hrdPop = list()
        for i in range(self.saveGood):
            goodInd = goodPop[i]
            splitPop = [ind for ind in self.clusters[i] if ind.info['dominators'] > goodInd.info['dominators']]
            splitLen = len(splitPop)
            sampleNum = int(splitLen/2)+1
            logging.debug("splitlen: %s"%(splitLen))
            if splitLen <= 1:
                continue
            for j in range(cutNum):
                grid = random.choice(grids)
                spInd = tournament(splitPop, sampleNum)
                tranPos = spInd.get_scaled_positions() # Displacement
                tranPos += np.array([[random.random(), random.random(), random.random()]]*len(spInd))
                spInd.set_scaled_positions(tranPos)
                spInd.wrap()

                ind1 = cut_cell([spInd, goodInd], grid, symbols, 0.2)
                ind2 = cut_cell([goodInd, spInd], grid, symbols, 0.2)
                ind1 = merge_atoms(ind1, self.dRatio)
                ind2 = merge_atoms(ind2, self.dRatio)


                parentE = 0.5*(sum([ind.info['enthalpy'] for ind in [spInd, goodInd]]))
                parDom = 0.5*(sum([ind.info['sclDom'] for ind in [spInd, goodInd]]))
                ind1.info['parentE'], ind2.info['parentE'] = parentE, parentE
                ind1.info['parDom'], ind2.info['parDom'] = parDom, parDom
                ind1.info['symbols'], ind2.info['symbols'] = symbols, symbols


                if self.calcType == 'fix':
                    nfm = int(round(0.5 * sum([ind.info['numOfFormula'] for ind in [spInd, goodInd]])))
                    ind1.info['formula'], ind2.info['formula'] = self.formula, self.formula
                    ind1.info['numOfFormula'], ind2.info['numOfFormula'] = nfm, nfm
                    ind1 = repair_atoms(ind1, symbols, self.formula, nfm)
                    ind2 = repair_atoms(ind2, symbols, self.formula, nfm)

                pairPop = [ind for ind in [ind1, ind2] if ind]
                hrdPop.extend(del_duplicate(pairPop, compareE=False, report=False))

        return hrdPop

    def permutate(self, permNum):
        permPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(splitLen/2) + 1
            for j in range(permNum):
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                rate = random.uniform(0.4,0.6)
                permInd = exchage_atom(parInd, rate)
                # if mode == 'atom':
                #     permInd = exchage_atom(parInd, rate)
                # elif mode == 'mol':
                #     parMolC = MolCryst(**parInd.info['molDict'])
                #     parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                #     permMolC = mol_exchage(parMolC)
                #     permInd = permMolC.to_atoms()

                # permInd = merge_atoms(permInd, self.dRatio)
                # toFrml = [int(i) for i in parInd.info['formula']]
                # permInd = repair_atoms(permInd, self.symbols, toFrml, parInd.info['numOfFormula'])

                permInd.info['symbols'] = self.symbols
                permInd.info['formula'] = parInd.info['formula']
                permInd.info['parentE'] = parentE
                permInd.info['parDom'] = parDom
                permPop.append(permInd)

        return permPop

    def latmutate(self, latNum, sigma=0.3):
        latPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(0.7*splitLen) + 1
            for j in range(latNum):
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                latInd = gauss_mut(parInd, sigma=sigma, cellCut=0.5)
                # if mode == 'atom':
                #     latInd = gauss_mut(parInd, sigma=sigma, cellCut=0.5)
                # elif mode == 'mol':
                #     parMolC = MolCryst(**parInd.info['molDict'])
                #     parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                #     latMolC = mol_gauss_mut(parMolC, sigma=sigma, cellCut=0, distCut=1.5)
                #     latInd = latMolC.to_atoms()

                latInd.info['symbols'] = self.symbols
                latInd.info['formula'] = parInd.info['formula']
                latInd.info['parentE'] = parentE
                latInd.info['parDom'] = parDom

                latInd = merge_atoms(latInd, self.dRatio)
                toFrml = [int(i) for i in parInd.info['formula']]
                latInd = repair_atoms(latInd, self.symbols, toFrml, parInd.info['numOfFormula'])

                if latInd:
                    latPop.append(latInd)

        return latPop

    def slipmutate(self, slipNum=5):
        slipPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(splitLen/2) + 1
            for j in range(slipNum):
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                slipInd = slip(parInd)
                # if mode == 'atom':
                #     slipInd = slip(parInd)
                # elif mode == 'mol':
                #     parMolC = MolCryst(**parInd.info['molDict'])
                #     parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                #     slipMolC = mol_slip(parMolC)
                #     slipInd = slipMolC.to_atoms()

                slipInd.info['symbols'] = self.symbols
                slipInd.info['formula'] = parInd.info['formula']
                slipInd.info['parentE'] = parentE
                slipInd.info['parDom'] = parDom

                slipInd = merge_atoms(slipInd, self.dRatio)
                slipInd = repair_atoms(slipInd, self.symbols, parInd.info['formula'], parInd.info['numOfFormula'])

                if slipInd:
                    slipPop.append(slipInd)

        return slipPop

    def ripmutate(self, rhos=[0.5, 0.75, 1.]):
        ripPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(splitLen/2) + 1
            for rho in rhos:
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                ripInd = ripple(parInd, rho=rho)

                ripInd.info['symbols'] = self.symbols
                ripInd.info['formula'] = parInd.info['formula']
                ripInd.info['parentE'] = parentE
                ripInd.info['parDom'] = parDom

                ripInd = merge_atoms(ripInd, self.dRatio)
                toFrml = [int(i) for i in parInd.info['formula']]
                ripInd = repair_atoms(ripInd, self.symbols, toFrml, parInd.info['numOfFormula'])

                if ripInd:
                    ripPop.append(ripInd)

        return ripPop

    def generate(self,curPop):
        self.curPop = calc_dominators(curPop)
        self.labels, self.goodPop = clustering(self.curPop, self.saveGood)
        if self.addSym:
            self.goodPop = symmetrize_pop(self.goodPop, 1.)

        self.curLen = len(self.curPop)
        logging.debug("curLen: {}".format(self.curLen))
        assert self.curLen >= self.saveGood, "saveGood should be shorter than length of curPop!"

        self.tmpPop = list()
        self.nextPop = list()

        self.clusters = []
        for i in range(self.saveGood):
            self.clusters.append([ind for n, ind in enumerate(self.curPop) if self.labels[n] == i])

        hrdPop = self.heredity(self.cutNum)
        # hrdPop = []
        permPop = self.permutate(self.permNum)
        latPop = self.latmutate(self.latNum)
        slipPop = self.slipmutate(self.slipNum)
        ripPop = self.ripmutate(self.ripRho)
        logging.debug("hrdPop length: %s"%(len(hrdPop)))
        logging.debug("permPop length: %s"%(len(permPop)))
        logging.debug("latPop length: %s"%(len(latPop)))
        logging.debug("slipPop length: %s"%(len(slipPop)))
        logging.debug("ripPop length: %s"%(len(ripPop)))
        tmpPop = hrdPop + permPop + latPop + slipPop + ripPop
        tmpPop = check_dist(tmpPop, self.dRatio)
        logging.debug("After check distance: {}".format(len(tmpPop)))
        self.tmpPop.extend(tmpPop)
        return self.tmpPop

    def select(self):
        tmpPop = self.tmpPop[:]
        newPop = []
        if self.newLen < len(tmpPop):
            for _ in range(self.newLen):
                # logging.debug("select len tmpPop {}".format(len(tmpPop)))
                newInd = tournament(tmpPop, int(0.5*len(tmpPop))+1, keyword='parDom')
                newPop.append(newInd)
                tmpPop.remove(newInd)

            return newPop
        else:
            return tmpPop

class MLEA(BaseEA):
    def __init__(self, parameters):
        return super().__init__(parameters)

    def select(self):
        return super().select()


def cut_cell(cutPop, grid, symbols, cutDisp=0):
    """
    Cut cells to generate a new structure.
    len(cutPop) = grid[0] * grid[1] * grid[2]
    cutDisp: displacement in cut
    """
    k1, k2, k3 = grid
    numSiv = k1*k2*k3
    siv = [(x, y, z) for x in range(k1) for y in range(k2) for z in range(k3)]
    sivCut = list()
    for i in range(3):
        cutPos = [(x + cutDisp*random.uniform(-0.5, 0.5))/grid[i] for x in range(grid[i])]
        cutPos.append(1)
        cutPos[0] = 0
        sivCut.append(cutPos)

    cutCell = np.zeros((3, 3))
    cutVol = 0
    for otherInd in cutPop:
        cutCell = cutCell + otherInd.get_cell()/numSiv
        cutVol = cutVol + otherInd.get_volume()/numSiv

    cutCellPar = cell_to_cellpar(cutCell)
    ratio = cutVol/abs(np.linalg.det(cutCell))
    if ratio > 1:
        cutCellPar[:3] = [length*ratio**(1/3) for length in cutCellPar[:3]]

    syblDict = dict()
    for sybl in symbols:
        syblDict[sybl] = []
    for k in range(numSiv):
        imAts = [imAtom for imAtom in cutPop[k]
                if sivCut[0][siv[k][0]] <= imAtom.a < sivCut[0][siv[k][0] + 1]
                if sivCut[1][siv[k][1]] <= imAtom.b < sivCut[1][siv[k][1] + 1]
                if sivCut[2][siv[k][2]] <= imAtom.c < sivCut[2][siv[k][2] + 1]
                ]
        for atom in imAts:
            for sybl in symbols:
                if atom.symbol == sybl:
                    syblDict[sybl].append((atom.a, atom.b, atom.c))
    cutPos = []
    strFrml = ''
    for sybl in symbols:
        if len(syblDict[sybl]) > 0:
            cutPos.extend(syblDict[sybl])
            strFrml = strFrml + sybl + str(len(syblDict[sybl]))
    if strFrml == '':
        raise RuntimeError('No atoms in the new cell')


    cutInd = Atoms(strFrml,
        cell=cutCellPar,
        pbc = True,)
    cutInd.set_scaled_positions(cutPos)

    # formula = [len(syblDict[sybl]) for sybl in symbols]
    # cutInd.info['formula'] = formula

    return cutInd




def exchage_atom(parInd, fracSwaps=None):
    if not fracSwaps:
        fracSwaps = 0.5
    maxSwaps = int(fracSwaps*len(parInd))
    if maxSwaps == 0:
        maxSwaps = 1
    numSwaps = random.randint(1, maxSwaps)
    chdInd = parInd.copy()
    chdPos = chdInd.get_scaled_positions().tolist()

    symbols = parInd.get_chemical_symbols()
    symList = list(set(symbols))
    symDict = dict()
    for sym in symList:
        indices = [index for index, atom in enumerate(symbols) if atom is sym]
        random.shuffle(indices)
        symDict[sym] = indices

    exIndices = list()
    for i in range(numSwaps):
        availSym = [sym for sym in symList if len(symDict[sym]) > 0]

        if len(availSym) < 2:
            break

        exSym = random.sample(availSym, 2)
        index0 = symDict[exSym[0]].pop()
        index1 = symDict[exSym[1]].pop()

        exIndices.append((index0, index1))

    for j, k in exIndices:
        chdPos[j], chdPos[k] = chdPos[k], chdPos[j]

    chdInd.set_scaled_positions(np.array(chdPos))
    # chdInd.info['Origin'] = "Exchange"
    return chdInd

def gauss_mut(parInd, sigma=0.5, cellCut=1):
    """
    sigma: Gauss distribution standard deviation
    cellCut: coefficient of gauss distribution in cell mutation
    """
    chdInd = parInd.copy()
    parVol = parInd.get_volume()


    chdCell = chdInd.get_cell()
    latGauss = [random.gauss(0, sigma)*cellCut for i in range(6)]
    strain = np.array([
        [1+latGauss[0], latGauss[1]/2, latGauss[2]/2],
        [latGauss[1]/2, 1+latGauss[3], latGauss[4]/2],
        [latGauss[2]/2, latGauss[4]/2, 1+latGauss[5]]
        ])
    chdCell = chdCell*strain
    cellPar = cell_to_cellpar(chdCell)
    ratio = parVol/abs(np.linalg.det(chdCell))
    cellPar[:3] = [length*ratio**(1/3) for length in cellPar[:3]]
    chdInd.set_cell(cellPar, scale_atoms=True)

    # for at in chdInd:
    #     atGauss = np.array([random.gauss(0, sigma)/sigma for i in range(3)])
    #     # atGauss = np.array([random.gauss(0, sigma)*distCut for i in range(3)])
    #     at.position += atGauss*covalent_radii[atomic_numbers[at.symbol]]

    chdInd.wrap()
    chdInd.info = parInd.info.copy()
    # chdInd.info['Origin'] = 'Mutate'

    return chdInd

def slip(parInd, cut=0.5, randRange=[0.5, 2]):
    '''
    from MUSE
    '''
    chdInd = parInd.copy()
    pos = parInd.get_scaled_positions()
    axis = list(range(3))
    random.shuffle(axis)
    rand1 = random.uniform(*randRange)
    rand2 = random.uniform(*randRange)

    for i in range(len(pos)):
        if pos[i, axis[0]] > cut:
            pos[i, axis[1]] += rand1
            pos[i, axis[2]] += rand2

    chdInd.set_scaled_positions(pos)
    return chdInd

def ripple(parInd, rho=0.3, mu=2, eta=1):
    '''
    from XtalOpt
    '''
    chdInd = parInd.copy()
    pos = parInd.get_scaled_positions()
    axis = list(range(3))
    random.shuffle(axis)

    for i in range(len(pos)):
        pos[i, axis[0]] += rho * cos(2*pi*mu*pos[i, axis[1]] + random.uniform(0, 2*pi)) *\
                            cos(2*pi*eta*pos[i, axis[2]] + random.uniform(0, 2*pi))

    chdInd.set_scaled_positions(pos)
    return chdInd


def tournament(pop, num, keyword='dominators'):
    smpPop = random.sample(pop, num)
    best = smpPop[0]
    for ind in smpPop[1:]:
        if ind.info[keyword] < best.info[keyword]:
            best = ind

    return best



def merge_atoms(atoms, tolerance=0.3,):
    """
    if a pair of atoms are too close, merge them.
    """

    cutoffs = [tolerance * covalent_radii[num] for num in atoms.get_atomic_numbers()]
    nl = neighbor_list("ij", atoms, cutoffs)
    indices = list(range(len(atoms)))
    exclude = []

    # logging.debug("merge_atoms()")
    # logging.debug("number of atoms: {}".format(len(atoms)))
    # logging.debug("{}".format(nl[0]))
    # logging.debug("{}".format(nl[1]))

    # remove self connection
    iArr = []
    jArr = []
    for i, j in zip(*nl):
        if i == j:
            pass
        else:
            iArr.append(i)
            jArr.append(j)


    for i, j in zip(iArr, jArr):
        if i in exclude or j in exclude:
            pass
        else:
            exclude.append(random.choice([i,j]))

    if len(exclude) > 0:
        save = [index for index in indices if index not in exclude]
        # logging.debug("exculde: {}\tsave: {}\n".format(exclude, save))
        mAts = atoms[save]
        mAts.info = atoms.info.copy()
    else:
        mAts = atoms

    return mAts

def repair_atoms(ind, symbols, toFrml, numFrml=1, dRatio=1, tryNum=20):
    """
    sybls: a list of symbols
    toFrml: a list of formula after repair
    """

    # numbers = [atomic_numbers[s] for s in symbols]
    inCt = Counter(ind.get_chemical_symbols())
    toFrml = [numFrml*i for i in toFrml]
    toDict = dict(zip(symbols, toFrml))
    diff = dict()
    for s in symbols:
        diff[s] = toDict[s] - inCt[s]

    sortSym = sorted(symbols, key=lambda x:diff[x])

    repInd = ind.copy()
    posArr = []
    for s in sortSym:
        if diff[s] < 0:
            atList = [atom for atom in repInd if atom.symbol==s]
            delAt = random.sample(atList, inCt[s] - toDict[s])
            delIns = [atom.index for atom in delAt]
            # Save deleted positons
            repPos = repInd.get_positions()
            posArr.extend(repPos[delIns])
            del repInd[delIns]

        elif diff[s] > 0:
            addNum = diff[s]
            if len(posArr) > 0:
                # Try to place the atoms on the previous positions
                rmIns = []
                for i, pos in enumerate(posArr):
                    if addNum == 0:
                        break
                    if check_new_atom_dist(repInd, pos, s, dRatio):
                        addAt = Atom(symbol=s, position=pos)
                        repInd.append(addAt)
                        addNum -= 1
                        rmIns.append(i)
                posArr = [posArr[j] for j in range(len(posArr)) if j not in rmIns]


            for _ in range(addNum):
                for _ in range(tryNum):
                    # select a center atoms
                    centerAt = repInd[random.randint(0,len(repInd)-1)]
                    basicR = covalent_radii[centerAt.number] + covalent_radii[atomic_numbers[s]]
                    # random position in spherical coordination
                    radius = basicR * (dRatio + random.uniform(0,0.3))
                    theta = random.uniform(0,math.pi)
                    phi = random.uniform(0,2*math.pi)
                    pos = centerAt.position + radius*np.array([sin(theta)*cos(phi), sin(theta)*sin(phi),cos(theta)])
                    if check_new_atom_dist(repInd, pos, s, dRatio):
                        addAt = Atom(symbol=s, position=pos)
                        repInd.append(addAt)
                        break
                    else:
                        continue

                else:
                    # logging.debug("Fail in repairing atoms")
                    return None
    repInd = sort_elements(repInd)
    return repInd

    # Still have some bugs, so check the formula before return
    # newFrml = get_formula(repInd, symbols)
    # if (np.array(newFrml) == toFrml).all():
    #     return repInd
    # else:
    #     logging.debug("Wrong formula in repair_atoms")
    #     return None



# if __name__=="__main__":
#     from .readparm import read_parameters
#     from .utils import EmptyClass
#     import ase.io
#     from .setfitness import calc_fitness
#     a=ase.io.read('relax.traj',':')
#     parameters = read_parameters('input.yaml')
#     p = EmptyClass()
#     for key, val in parameters.items():
#         setattr(p, key, val)
#     g=Kriging(p)
#     calc_fitness(a)
#     repop=g.generate(a)
#     from .writeresults import write_dataset, write_results, write_traj
#     write_results(repop, 'result', '.')

