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
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from sklearn import cluster
from .utils import *


class Kriging:
    def __init__(self, parameters):

        self.parameters = parameters
        self.symbols = parameters.symbols
        self.formula = parameters.formula
        self.saveGood = parameters.saveGood
        self.addSym = parameters.addSym

        krigParm = parameters.krigParm
        self.randFrac = krigParm['randFrac']
        self.permNum = krigParm['permNum']
        self.latDisps = krigParm['latDisps']
        self.ripRho = krigParm['ripRho']
        self.rotNum = krigParm['rotNum']
        self.cutNum = krigParm['cutNum']
        self.slipNum = krigParm['slipNum']
        self.latNum = krigParm['latNum']

        self.kind = krigParm['kind']
        self.xi = krigParm['xi']
        self.grids = krigParm['grids']
        self.kappaLoop = krigParm['kappaLoop']
        self.scaled_factor = krigParm['scale']

        self.parent_factor = 0

        self.dRatio = parameters.dRatio
        self.bondRatio = parameters.bondRatio
        self.bondRange = parameters.bondRange

    def heredity(self, cutNum=5, mode='atom'):
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


                nfm = int(round(0.5 * sum([ind.info['numOfFormula'] for ind in [spInd, goodInd]])))
                ind1 = repair_atoms(ind1, symbols, self.formula, nfm)
                ind2 = repair_atoms(ind2, symbols, self.formula, nfm)
                ind1.info['formula'], ind2.info['formula'] = self.formula, self.formula
                ind1.info['numOfFormula'], ind2.info['numOfFormula'] = nfm, nfm

                pairPop = [ind for ind in [ind1, ind2] if ind is not None]
                hrdPop.extend(del_duplicate(pairPop, compareE=False, report=False))

        return hrdPop

    def permutate(self, permNum, mode='atom'):
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
                if mode == 'atom':
                    permInd = exchage_atom(parInd, rate)
                elif mode == 'mol':
                    parMolC = MolCryst(**parInd.info['molDict'])
                    parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                    permMolC = mol_exchage(parMolC)
                    permInd = permMolC.to_atoms()

                # permInd = merge_atoms(permInd, self.dRatio)
                # toFrml = [int(i) for i in parInd.info['formula']]
                # permInd = repair_atoms(permInd, self.symbols, toFrml, parInd.info['numOfFormula'])

                permInd.info['symbols'] = self.symbols
                permInd.info['formula'] = parInd.info['formula']
                permInd.info['parentE'] = parentE
                permInd.info['parDom'] = parDom
                permPop.append(permInd)

        return permPop

    def latmutate(self, latNum, sigma=0.3, mode='atom'):
        latPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(0.7*splitLen) + 1
            for j in range(latNum):
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                if mode == 'atom':
                    latInd = gauss_mut(parInd, sigma=sigma, cellCut=0.5)
                elif mode == 'mol':
                    parMolC = MolCryst(**parInd.info['molDict'])
                    parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                    latMolC = mol_gauss_mut(parMolC, sigma=sigma, cellCut=0, distCut=1.5)
                    latInd = latMolC.to_atoms()

                # latInd = merge_atoms(latInd, self.dRatio)
                # toFrml = [int(i) for i in parInd.info['formula']]
                # latInd = repair_atoms(latInd, self.symbols, toFrml, parInd.info['numOfFormula'])

                latInd.info['symbols'] = self.symbols
                latInd.info['formula'] = parInd.info['formula']
                latInd.info['parentE'] = parentE
                latInd.info['parDom'] = parDom
                latPop.append(latInd)

        return latPop

    def slipmutate(self, slipNum=5, mode='atom'):
        slipPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(splitLen/2) + 1
            for j in range(slipNum):
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                if mode == 'atom':
                    slipInd = slip(parInd)
                elif mode == 'mol':
                    parMolC = MolCryst(**parInd.info['molDict'])
                    parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                    slipMolC = mol_slip(parMolC)
                    slipInd = slipMolC.to_atoms()

                # slipInd = merge_atoms(slipInd, self.dRatio)
                # slipInd = repair_atoms(slipInd, self.symbols, parInd.info['formula'], parInd.info['numOfFormula'])

                slipInd.info['symbols'] = self.symbols
                slipInd.info['formula'] = parInd.info['formula']
                slipInd.info['parentE'] = parentE
                slipInd.info['parDom'] = parDom
                slipPop.append(slipInd)

        return slipPop

    def ripmutate(self, rhos=[0.5, 0.75, 1.], mode='atom'):
        ripPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(splitLen/2) + 1
            for rho in rhos:
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                if mode == 'atom':
                    ripInd = ripple(parInd, rho=rho)

                # ripInd = merge_atoms(ripInd, self.dRatio)
                # toFrml = [int(i) for i in parInd.info['formula']]
                # ripInd = repair_atoms(ripInd, self.symbols, toFrml, parInd.info['numOfFormula'])

                ripInd.info['symbols'] = self.symbols
                ripInd.info['formula'] = parInd.info['formula']
                ripInd.info['parentE'] = parentE
                ripInd.info['parDom'] = parDom
                ripPop.append(ripInd)

        return ripPop

    def generate(self,curPop):
        self.curPop = calc_dominators(curPop)
        self.labels, self.goodPop = clustering(self.curPop, self.saveGood)
        if self.addSym:
            self.goodPop = standardize_pop(self.goodPop, 1.)

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
        self.tmpPop.extend(tmpPop)
        return tmpPop


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

def check_dist(pop, threshold=0.7):
    checkPop = []
    for ind in pop:
    #    ase.io.write('checking.vasp', ind, format='vasp', direct=True, vasp5=True)
        if check_dist_individual(ind, threshold):
            checkPop.append(ind)

    return checkPop

def check_dist_individual(ind, threshold):
    """
    The distance between the atoms should be larger than
    threshold * sumR(the sum of the covalent radii of the two
    corresponding atoms).
    """
    radius = [covalent_radii[number] for number in ind.get_atomic_numbers()]
    cellPar = ind.get_cell_lengths_and_angles()
    vector = cellPar[:3]
    angles = cellPar[-3:]

    minAng = np.array([45]*3)
    maxAng = np.array([135]*3)

    maxBond = 2*max(radius)
    allBonds = 2*sum(radius)
    minVec = np.array([maxBond]*3)
    maxVec = np.array([allBonds]*3)

    checkAng = (minAng < angles).all() and (angles < maxAng).all()
    checkVec = (0.5 * minVec < vector).all()


    if checkAng and checkVec:
        cutoffs = [rad*threshold for rad in radius]
        nl = NeighborList(cutoffs, skin=0, self_interaction=False, bothways=True)
        nl.update(ind)
        nlSum = sum([len(nl.get_neighbors(i)[0]) for i in range(len(ind))])
        return nlSum == 0
    else:
        return False

def repair_atoms(ind, sybls, toFrml, numFrml=1):
    """
    sybls: a list of symbols
    toFrml: a list of formula after repair
    """

    toFrml = [numFrml*i for i in toFrml]
    toDict = dict(zip(sybls, toFrml))
    saveInfo = ind.info

    strFrml = ''
    repPos = []
    for sybl in sybls:
        syblPos = [(at.a, at.b, at.c) for at in ind if at.symbol is sybl]
        curLen = len(syblPos)

        if toDict[sybl] < curLen:
            toDel = random.sample(range(len(syblPos)),
            curLen - toDict[sybl])
            toDel.sort(reverse=True)
            for i in toDel:
                del syblPos[i]

        if toDict[sybl] > curLen:
            for i in range(toDict[sybl] - curLen):
                syblPos.append((random.random(),
                                random.random(),
                                random.random(),))

        repPos.extend(syblPos)
        strFrml = strFrml + sybl + str(toDict[sybl])

    repInd = Atoms(strFrml, cell=ind.get_cell(), pbc=True)
    repInd.set_scaled_positions(repPos)
    repInd.info = saveInfo
    repInd.info['formula'] = toFrml
    #logging.debug(repInd.get_chemical_formula())

    return repInd

def del_duplicate(Pop, compareE=True, tol = 0.2, diffE = 0.005, diffV = 0.05, diffD = 0.01, report=True, mode='naive'):
    dupPop = find_spg(Pop, tol)
    #for ind in Pop:
     #   logging.info("spg: %s" %ind.info['spg'])
    # sort the pop by composion, wait for adding
    # dupPop = compare_fingerprint(Pop, diffD)
    # logging.info("fingerprint survival: %s" %(len(dupPop)))

    dupPop = compare_volume_energy(dupPop, diffE, diffV, compareE=compareE, mode=mode)
    if report:
        logging.info("volume_energy survival: %s" %(len(dupPop)))
    # logging.info("survival: %s Individual" %len(dupPop))
    return dupPop

def compare_volume_energy(Pop, diffE, diffV, ltol=0.1, stol=0.1, angle_tol=5, compareE=True, mode='naive'): #differnce in enthalpy(eV/atom) and volume(%)
    vol_tol = diffV**(1./3)
    priList = []
    for ind in Pop:
        ind.info['vPerAtom'] = ind.get_volume()/len(ind)
        priInfo = ind.info['priInfo']
        if priInfo:
            lattice, scaled_positions, numbers = priInfo
            priAts = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers, pbc=1)
        else:
            priAts = ind.copy()
        priList.append(priAts)


    cmpPop = Pop[:]

    toCompare = [(x,y) for x in range(len(Pop)) for y in range(len(Pop)) if x < y]
    # toCompare = [(x,y) for x in Pop for y in Pop if Pop.index(x) < Pop.index(y)]

    comp = SymmetryEquivalenceCheck(to_primitive=True, angle_tol=angle_tol, ltol=ltol, stol=stol,vol_tol=vol_tol)

    for pair in toCompare:
        s0 = Pop[pair[0]]
        s1 = Pop[pair[1]]
        pri0 = priList[pair[0]]
        pri1 = priList[pair[1]]


        symCt0 = Counter(pri0.numbers)
        symCt1 = Counter(pri1.numbers)
        if symCt0 != symCt1 and mode=='naive':
            continue

        duplicate = True

        # pairV = [Pop[n].info['vPerAtom'] for n in pair]
        pairV = [pri0.get_volume(), pri1.get_volume()]
        deltaV = abs(pairV[0] - pairV[1])/min(pairV)

        pairSpg = [Pop[n].info['spg'] for n in pair]


        if compareE:
            pairE = [Pop[n].info['enthalpy'] for n in pair]
            deltaE = abs(pairE[0] - pairE[1])
            if mode == 'ase':
                try:
                    duplicate = comp.compare(pri0, pri1) and deltaE <= diffE
                except:
                    s = sys.exc_info()
                    logging.info("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
                    logging.info("ASE check fails. Use native check")
                    # ase.io.write('failcomp0.vasp', pri0, vasp5=1, direct=1)
                    # ase.io.write('failcomp1.vasp', pri1, vasp5=1, direct=1)
                    duplicate = duplicate and deltaE <= diffE and pairSpg[0] == pairSpg[1]
            elif mode == 'naive':
                duplicate = duplicate and deltaV <= diffV and deltaE <= diffE and pairSpg[0] == pairSpg[1]
        else:
            if mode == 'ase':
                try:
                    duplicate = comp.compare(pri0, pri1)
                except:
                    s = sys.exc_info()
                    logging.info("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
                    logging.info("ASE check fails. Use native check")
                    # ase.io.write('failcomp0.vasp', pri0, vasp5=1, direct=1)
                    # ase.io.write('failcomp1.vasp', pri1, vasp5=1, direct=1)
                    duplicate = duplicate and deltaV <= diffV and pairSpg[0] == pairSpg[1]
            elif mode == 'naive':
                duplicate = duplicate and deltaV <= diffV and pairSpg[0] == pairSpg[1]
        # logging.debug('pairindex: %s %s, duplicate: %s' % (Pop.index(pair[0]), Pop.index(pair[1]), duplicate))
        # logging.debug('pairindex: %s, duplicate: %s' % (pair, duplicate))

        if duplicate:
            if compareE:
                cmpInd = Pop[pair[0]] if pairE[0] > pairE[1] else Pop[pair[1]]
            else:
                cmpInd = Pop[pair[0]]
            if cmpInd in cmpPop:
                cmpPop.remove(cmpInd)
                # logging.info("remove duplicate")

    return cmpPop

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

    for at in chdInd:
        atGauss = np.array([random.gauss(0, sigma)/sigma for i in range(3)])
        # atGauss = np.array([random.gauss(0, sigma)*distCut for i in range(3)])
        at.position += atGauss*covalent_radii[atomic_numbers[at.symbol]]

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

def standardize_atoms(atoms, symprec=1e-5):
    """
    Use spglib to get standardize cell of atoms
    """

    spgCell = spglib.standardize_cell(atoms, symprec=symprec)
    if spgCell:
        lattice, pos, numbers = spgCell
        stdAts = Atoms(cell=lattice, scaled_positions=pos, numbers=numbers)
        stdAts.info = atoms.info.copy()
    else:
        stdAts = Atoms(atoms)

    return stdAts

def standardize_pop(pop, symprec=1e-5):

    stdPop = list()
    for ind in pop:
        stdInd = standardize_atoms(ind, symprec)
        if len(stdInd) == len(ind):
            stdPop.append(stdInd)
        else:
            stdPop.append(Atoms(ind))

    return stdPop

def tournament(pop, num):
    smpPop = random.sample(pop, num)
    ind = sorted(smpPop, key=lambda x:x.info['dominators'])[0]
    return ind

def calc_dominators(Pop):
    domPop = [ind.copy() for ind in Pop]
    domLen = len(domPop)
    for ind in domPop:
        ftn1 = ind.info['fitness1']
        ftn2 = ind.info['fitness2']
        dominators = 0 #number of individuals that dominate the current ind
        # toDominate = 0 #number of individuals that are dominated by the current ind
        for otherInd in domPop[:]:
            if ((otherInd.info['fitness1'] < ftn1 and otherInd.info['fitness2'] < ftn2)
                or (otherInd.info['fitness1'] <= ftn1 and otherInd.info['fitness2'] < ftn2)
                or (otherInd.info['fitness1'] < ftn1 and otherInd.info['fitness2'] <= ftn2)):

                dominators += 1

        ind.info['dominators'] = dominators
        ind.info['MOGArank'] = dominators + 1
        ind.info['sclDom'] = (dominators)/domLen

    return domPop

def clustering(inPop, numClusters, label=False):
    """
    clustering by fingerprints
    """
    if numClusters >= len(inPop):
        return [i for i in range(len(inPop))], inPop

    fpMat = np.array([ind.info['image_fp'] for ind in inPop])
    km = cluster.KMeans(n_clusters=numClusters,)
    km.fit(fpMat)
    labels = km.labels_

    goodPop = [None]*numClusters
    for label, ind in zip(labels, inPop):
        curBest = goodPop[label]
        if curBest:
            if ind.info['dominators'] < curBest.info['dominators']:
                goodPop[label] = ind
        else:
            goodPop[label] = ind
    return labels, goodPop

if __name__=="__main__":
    from .readparm import read_parameters
    from .utils import EmptyClass
    import ase.io
    from .setfitness import calc_fitness
    a=ase.io.read('relax.traj',':')
    parameters = read_parameters('input.yaml')
    p = EmptyClass()
    for key, val in parameters.items():
        setattr(p, key, val)
    g=Kriging(p)
    calc_fitness(a)
    repop=g.generate(a)
    from .writeresults import write_dataset, write_results, write_traj
    write_results(repop, 'result', '.')

