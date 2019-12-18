from __future__ import print_function, division
import os, subprocess, shutil, math, random, re, logging, fractions, sys
from collections import Counter
import numpy as np
import spglib
from numpy import pi, sin, cos, tan, sqrt
import networkx as nx
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import root
from .crystgraph import quotient_graph, cycle_sums, graphDim, find_communities, find_communities2, remove_selfloops, nodes_and_offsets
try:
    from functools import reduce
    from ase.utils.structure_comparator import SymmetryEquivalenceCheck
except ImportError:
    pass


class EmptyClass:
    pass

class MolCryst:
    def __init__(self, numbers, cell, sclPos, partition, offSets=None, sclCenters=None, rltSclPos=None, info=dict()):
        self.info = info
        if offSets is not None:
            self.numbers, self.cell, self.sclPos, self.offSets = list(map(np.array, [numbers, cell, sclPos, offSets]))
            self.partition = tuple(partition)
            assert len(numbers) == len(sclPos) == len(offSets)
            numAts = len(numbers)
            indSet = set(list(reduce(lambda x, y: x+y, partition)))
            assert indSet == set(range(numAts))
            self.dispPos = self.sclPos + self.offSets

            sclCenters, rltSclPos = zip(*[self.get_center_and_rltSclPos(indices) for indices in partition])
            sclCenters = np.array(sclCenters)
            self.sclCenters = sclCenters - np.floor(sclCenters)
            self.centers = np.dot(self.sclCenters, self.cell)
            self.rltSclPos = rltSclPos
            self.rltPos = [np.dot(pos, self.cell) for pos in rltSclPos]



        elif sclCenters is not None and rltSclPos is not None:
            self.numbers, self.cell, self.sclPos = list(map(np.array, [numbers, cell, sclPos]))
            self.partition = tuple(partition)
            assert len(numbers) == len(sclPos)
            numAts = len(numbers)
            indSet = set(list(reduce(lambda x, y: x+y, partition)))
            assert indSet == set(range(numAts))

            self.sclCenters = sclCenters - np.floor(sclCenters)
            self.centers = np.dot(self.sclCenters, self.cell)
            self.rltSclPos = rltSclPos
            self.rltPos = [np.dot(pos, self.cell) for pos in rltSclPos]


        else:
            raise RuntimeError("Need to set offsets or (sclCenters and rltSclPos)!")

        self.numAts = len(numbers)
        self.numMols = len(partition)
        self.molNums = [self.numbers[p].tolist() for p in self.partition]

    def get_center_and_rltSclPos(self, indices):
        molPos = self.dispPos[indices]
        center = molPos.mean(0)
        rltPos = molPos - center
        return center, rltPos

    def get_sclCenters(self):
        return self.sclCenters[:]

    def get_centers(self):
        return self.centers[:]

    def get_rltPos(self):
        return self.rltPos[:]

    def get_radius(self):
        radius = []
        for pos in self.rltPos:
            rmax = max(np.linalg.norm(pos, axis=1))
            radius.append(rmax)
        return radius


    def copy(self):
        return MolCryst(self.numbers, self.cell, self.sclPos, self.partition, sclCenters=self.sclCenters, rltSclPos=self.rltSclPos, info=self.info.copy())

    def to_dict(self, infoCopy=True):
        molDict = {
            'numbers': self.numbers,
            'cell': self.cell,
            'sclPos': self.sclPos,
            'partition': self.partition,
            'sclCenters': self.sclCenters,
            'rltSclPos': self.rltSclPos,
        }
        if infoCopy:
            molDict['info'] = self.info.copy()
        return molDict

    def get_cell(self):
        return self.cell[:]

    def get_numbers(self):
        return self.numbers[:]

    def get_scaled_positions(self):
        return self.sclPos[:]

    def get_volume(self):
        return np.linalg.det(self.cell)

    def set_cell(self, cell, scale_atoms=False, scale_centers=True):
        self.cell = np.array(cell)
        if scale_atoms:
            self.centers = np.dot(self.sclCenters, self.cell)
            self.rltPos = [np.dot(pos, self.cell) for pos in self.rltSclPos]
        else:
            if scale_centers:
                self.centers = np.dot(self.sclCenters, self.cell)
            self.update_centers_and_rltPos(self.centers, self.rltPos)

    def to_atoms(self):
        return Atoms(numbers=self.numbers, scaled_positions=self.sclPos, cell=self.cell, pbc=1, info=self.info.copy())

    def update_centers_and_rltPos(self, centers=None, rltPos=None):
        invCell = np.linalg.inv(self.cell)
        if centers is not None:
            self.centers = np.array(centers)
            self.sclCenters = np.dot(self.centers, invCell)
        if rltPos is not None:
            self.rltPos = [np.array(pos) for pos in rltPos]
            self.rltSclPos = [np.dot(pos, invCell) for pos in self.rltPos]
        self.update()

    def update_sclCenters_and_rltSclPos(self, sclCenters=None, rltSclPos=None):

        if sclCenters is not None:
            self.sclCenters = np.array(sclCenters)
            self.centers = np.dot(self.sclCenters, self.cell)
        if rltSclPos is not None:
            self.rltSclPos = [np.array(pos) for pos in rltSclPos]
            self.rltPos = [np.dot(pos, self.cell) for pos in self.rltSclPos]
        self.update()

    def update(self):
        molNum = len(self.partition)
        posList = [self.sclCenters[i]+self.rltSclPos[i] for i in range(molNum)]
        tmpSclPos = reduce(lambda x,y: np.concatenate((x,y), axis=0), posList)
        indices = list(reduce(lambda x, y: x+y, self.partition))
        self.sclPos[indices] = tmpSclPos


def atoms2molcryst(atoms, coef=1.1):
    QG = quotient_graph(atoms, coef)
    graphs = nx.connected_component_subgraphs(QG)
    partition = []
    offSets = np.zeros([len(atoms), 3])
    for G in graphs:
        if graphDim(G) == 0 and G.number_of_nodes() > 1:
            nodes, offs = nodes_and_offsets(G)
            partition.append(nodes)
            for i, offSet in zip(nodes, offs):
                offSets[i] = offSet
            # nodes = list(G.nodes())
            # partition.append(nodes)
            # paths = nx.single_source_shortest_path(G, nodes[0])
            # for index, i in enumerate(nodes):
            #     if index is 0:
            #         offSets[i] = [0,0,0]
            #     else:
            #         path = paths[i]
            #         offSet = np.zeros((3,))
            #         for j in range(len(path)-1):
            #             vector = G[path[j]][path[j+1]][0]['vector']
            #             direction = G[path[j]][path[j+1]][0]['direction']
            #             pathDi = (path[j], path[j+1])
            #             if pathDi == direction:
            #                 offSet += vector
            #             elif pathDi[::-1] == direction:
            #                 offSet -= vector
            #             else:
            #                 raise RuntimeError("Error in direction!")
            #         offSets[i] = offSet

        else:
            for i in G.nodes():
                partition.append([i])
                offSets[i] = [0,0,0]

    molC = MolCryst(numbers=atoms.get_atomic_numbers(), cell=atoms.get_cell(),
    sclPos=atoms.get_scaled_positions(), partition=partition, offSets=offSets, info=atoms.info.copy())

    return molC

def atoms2communities(atoms, coef=1.1):
    QG = quotient_graph(atoms, coef)
    graphs = nx.connected_component_subgraphs(QG)
    partition = []
    offSets = np.zeros([len(atoms), 3])
    for SG in graphs:
        G = remove_selfloops(SG)
        if graphDim(G) == 0:
            nodes, offs = nodes_and_offsets(G)
            partition.append(nodes)
            for i, offSet in zip(nodes, offs):
                offSets[i] = offSet
        else:
            # comps = find_communities(G)
            comps = find_communities2(G)
            for indices in comps:
                tmpG = G.subgraph(indices)
                nodes, offs = nodes_and_offsets(tmpG)
                partition.append(nodes)
                for i, offSet in zip(nodes, offs):
                    offSets[i] = offSet

    # logging.debug("atoms2communities partition: {}".format(partition))

    molC = MolCryst(numbers=atoms.get_atomic_numbers(), cell=atoms.get_cell(),
    sclPos=atoms.get_scaled_positions(), partition=partition, offSets=offSets, info=atoms.info.copy())

    return molC


def symbols_and_formula(atoms):

    allSym = atoms.get_chemical_symbols()
    symbols = list(set(allSym))
    numOfSym = lambda sym: len([i for i in allSym if i == sym])
    formula = list(map(numOfSym, symbols))

    return symbols, formula

def get_formula(atoms, symbols):
    allSym = atoms.get_chemical_symbols()
    formula = [allSym.count(s) for s in symbols]
    return formula

def find_spg(Pop, tol): #tol:tolerance

    spgPop = []
    for ind in Pop:
        spg = spglib.get_spacegroup(ind, tol)
        pattern = re.compile(r'\(.*\)')
        try:
            spg = pattern.search(spg).group()
            spg = int(spg[1:-1])
        except:
            spg = 1
        priInfo = spglib.find_primitive(ind, tol)
        ind.info['spg'] = spg
        ind.info['priInfo'] = priInfo
        spgPop.append(ind)

    return spgPop

def extract_atoms(atoms):
    atDict = {
        'numbers': atoms.get_atomic_numbers(),
        'positions': atoms.get_positions(),
        'cell': np.array(atoms.get_cell()),
        'energy': atoms.get_potential_energy(),
        'forces': atoms.get_forces(),
        'stress': atoms.get_stress()
    }

    return atDict

def read_atDict(atDict):
    ats = Atoms(numbers=atDict['numbers'], cell=atDict['cell'], positions=atDict['positions'], pbc=True)
    calc = SinglePointCalculator(ats)
    calc.results['energy'] = atDict['energy']
    calc.results['forces'] = atDict['forces']
    calc.results['stress'] = atDict['stress']
    ats.set_calculator(calc)
    ats.info['energy'] = atDict['energy']
    ats.info['forces'] = atDict['forces']
    ats.info['stress'] = atDict['stress']

    return ats

def calc_volRatio(atoms):
    ballVol = 0
    # for num in atoms.get_atomic_numbers():
    #     ballVol += 4*math.pi/3*(covalent_radii[num])**3

    volRatio = atoms.get_volume()/calc_ball_volume(atoms)
    return volRatio

def calc_ball_volume(atoms):
    ballVol = 0
    for num in atoms.get_atomic_numbers():
        ballVol += 4*math.pi/3*(covalent_radii[num])**3

    return ballVol

def read_bare_atoms(readPop, setSym, setFrml, minAt, maxAt, calcType):
    seedPop = []

    if calcType == 'fix':
        setGcd = reduce(math.gcd, setFrml)
        setRd = [x/setGcd for x in setFrml]


    # if len(readPop) > 0:
    #     logging.debug("Reading Seeds ...")
    for ind in readPop:
        selfSym, selfFrml = symbols_and_formula(ind)
        # logging.debug('selfSym: {!r}'.format(selfSym))
        symDic = dict(zip(selfSym, selfFrml))
        for sym in [s for s in setSym if s not in selfSym]:
            symDic[sym] = 0

        if False not in map(lambda x: x in setSym, selfSym):
            ind.info['symbols'] = setSym
        else:
            logging.debug("ERROR in checking symbols")
            continue

        if calcType == 'var':
            # if minAt <= len(ind) <= maxAt or len(selfSym) < len(setSym):
            formula = [symDic[sym] for sym in setSym]
            ind.info['formula'] = formula
            seedPop.append(ind)
            # else:
            #     logging.debug("ERROR in checking symbols")

        elif calcType == 'fix':
            if len(selfSym) == len(setSym):
                formula = [symDic[sym] for sym in setSym]
                logging.info('formula: {!r}'.format(formula))
                selfGcd = reduce(math.gcd, formula)
                selfRd = [x/selfGcd for x in formula]
                if selfRd == setRd:
                    ind.info['formula'] = setFrml
                    ind.info['numOfFormula'] = int(len(ind)/sum(setFrml))
                    seedPop.append(ind)
                else:
                    logging.debug("ERROR in checking formula")



    # logging.info("Read Seeds: %s"%(len(seedPop)))
    return seedPop

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

    for i, j in zip(*nl):
        if i in exclude or j in exclude:
            continue
        exclude.append(random.choice([i,j]))

    if len(exclude) > 0:
        save = [index for index in indices if index not in exclude]
        # logging.debug("exculde: {}\tsave: {}".format(exclude, save))
        mAts = atoms[save]
        mAts.info = atoms.info.copy()
    else:
        mAts = atoms

    return mAts

def rand_rotMat():
    phi, theta, psi = [2*math.pi*random.uniform(-1,1) for _ in range(3)]
    rot1 = np.array([[cos(phi),-1*sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]])
    rot2 = np.array([[cos(theta), 0, -1*sin(theta)],[0,1,0],[sin(theta), 0, cos(theta)]])
    rot3 = np.array([[1,0,0],[0,cos(psi),-1*sin(psi)],[0,sin(psi),cos(psi)]])
    rotMat = np.dot(np.dot(rot1, rot2), rot3)
    # print("det of rotMat: {}".format(np.linalg.det(rotMat)))
    return rotMat

def check_mol_ind(molInd, inputNums, bondRatio):
    molc = atoms2molcryst(molInd, bondRatio)
    molNums = molInd.get_atomic_numbers()
    counters = [Counter(nums) for nums in inputNums]
    if False in [Counter(molNums[p]) in counters for p in molc.partition]:
        return False
    else:
        return True

def check_mol_pop(molPop, inputMols, bondRatio):
    inputNums = [mol.get_atomic_numbers() for mol in inputMols]
    chkPop = []
    for ind in molPop:
        if check_mol_ind(ind, inputNums, bondRatio):
            chkPop.append(ind)
    return chkPop



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
    if mode == 'ase':
        comp = SymmetryEquivalenceCheck(to_primitive=True, angle_tol=angle_tol, ltol=ltol, stol=stol,vol_tol=vol_tol)

    rmIndices = []
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
                rmInd = pair[0] if pairE[0] > pairE[1] else pair[1]
            else:
                rmInd = pair[0]
            rmIndices.append(rmInd)
            # if cmpInd in cmpPop:
            #     cmpPop.remove(cmpInd)
                # logging.info("remove duplicate")
    cmpPop = [ind for i, ind in enumerate(cmpPop) if i not in rmIndices]

    return cmpPop

def compare_fingerprint(fpPop, diffD):
    """
    Compare indviduals in inPop based on their fingerprints.
    """
    cmpPop = fpPop[:]
    fpList = [ind.info['fingerprint'] for ind in fpPop]
    toCompare = [(x,y) for x in range(len(fpPop)) for y in range(len(fpPop)) if x < y]

    for i, j in toCompare:
        distance = np.linalg.norm(fpList[i] - fpList[j])
        # logging.debug("Index: %s %s, dist: %s" %(i, j, distance))
        if distance < diffD:
            cmpInd = fpPop[i] if fpPop[i].info['enthalpy'] > fpPop[j].info['enthalpy'] else fpPop[j]
            if cmpInd in cmpPop:
                cmpPop.remove(cmpInd)
                logging.debug("remove duplicate")

    return cmpPop

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
    # logging.debug("dupPop: {}".format(len(dupPop)))
    return dupPop


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

    minAng = np.array([30]*3)
    maxAng = np.array([150]*3)

    maxBond = 2*max(radius)
    minVec = np.array([maxBond]*3)

    checkAng = (minAng < angles).all() and (angles < maxAng).all()
    checkVec = (0.5 * minVec < vector).all()

    if checkAng and checkVec:
        cutoffs = [rad*threshold for rad in radius]
        nl = neighbor_list('i', ind, cutoffs)
        return len(nl) == 0
    else:
        return False


def check_dist(pop, threshold=0.7):
    checkPop = []
    for ind in pop:
    #    ase.io.write('checking.vasp', ind, format='vasp', direct=True, vasp5=True)
        if check_dist_individual(ind, threshold):
            checkPop.append(ind)

    return checkPop

def symmetrize_atoms(atoms, symprec=1e-5):
    """
    Use spglib to get standardize cell of atoms
    """

    stdCell = spglib.standardize_cell(atoms, symprec=symprec)
    priCell = spglib.find_primitive(atoms, symprec=symprec)
    if stdCell and len(stdCell[0])==len(atoms):
        lattice, pos, numbers = stdCell
        symAts = Atoms(cell=lattice, scaled_positions=pos, numbers=numbers, pbc=True)
        symAts.info = atoms.info.copy()
    elif priCell and len(priCell[0])==len(atoms):
        lattice, pos, numbers = priCell
        symAts = Atoms(cell=lattice, scaled_positions=pos, numbers=numbers, pbc=True)
        symAts.info = atoms.info.copy()
    else:
        symAts = atoms

    return symAts

def symmetrize_pop(pop, symprec=1e-5):

    # stdPop = list()
    # for ind in pop:
    #     stdInd = symmetrize_atoms(ind, symprec)
    #     if len(stdInd) == len(ind):
    #         stdPop.append(stdInd)
    #     else:
    #         stdPop.append(ind)

    return [symmetrize_atoms(ats, symprec) for ats in pop]


def lower_triangullar_cell(oriInd):
    """
    Convert the cell of origin structure to a triangular matrix.
    """
    cellPar = oriInd.get_cell_lengths_and_angles()
    oriCell = oriInd.get_cell()
    # oriPos =oriInd.get_scaled_positions()
    triInd = oriInd.copy()

    a, b, c, alpha, beta, gamma = cellPar
    alpha *= pi/180.0
    beta *= pi/180.0
    gamma *= pi/180.0
    va = a * np.array([1, 0, 0])
    vb = b * np.array([cos(gamma), sin(gamma), 0])
    cx = cos(beta)
    cy = (cos(alpha) - cos(beta)*cos(gamma))/sin(gamma)
    cz = sqrt(1. - cx*cx - cy*cy)
    vc = c * np.array([cx, cy, cz])
    triCell = np.vstack((va, vb, vc))

#    T = np.linalg.solve(oriCell, triCell)
#    triPos = dot(oriPos, T)

    triInd.set_cell(triCell, scale_atoms=True)
    # triInd.set_scaled_positions(oriPos)
    triInd.info = oriInd.info.copy()

    return triInd



def lda_mol(centers, rltPos, cell, ratio, coefEps=1e-3, ratioEps=1e-3):
    """
    centers: centers of molecules
    rltPos: relative positions with respect to the centers
    cell: crystal cell
    ratio: compress ratio
    coefEps: if a coefficient is less than coefEps, it is set to 0
    ratioEps: check the result of compression
    """
    cartPos = []
    classes = []
    n = 0
    for cen, rlt in zip(centers, rltPos):
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    n += 1
                    offset = np.dot([x,y,z], cell)
                    pos = cen + rlt + offset
                    cartPos.extend(pos.tolist())
                    classes.extend([n]*len(pos))

    clf = LinearDiscriminantAnalysis(n_components=1)
    clf.fit(cartPos, classes)

    # compress vector
    ldaVec = clf.scalings_[:,0]
    ldaVec = ldaVec/np.linalg.norm(ldaVec)
    # print(ldaVec)

    # print(cell)
    # the relative stress tensor
    stress = np.outer(ldaVec, ldaVec)
    f_h = np.dot(np.linalg.inv(cell.T), stress)
    # print(f_h)


    sclF = np.dot(f_h, np.linalg.inv(cell))
    # parameters of the cubic equation
    p3 = -1 * np.linalg.det(sclF)
    p2 = sclF[0,0]*sclF[1,1] + sclF[1,1]*sclF[2,2] + sclF[0,0]*sclF[2,2]
    - sclF[0,1]*sclF[1,0] - sclF[0,2]*sclF[2,0] - sclF[1,2]*sclF[2,1]
    p1 = -1 * np.trace(sclF)
    p0 = 1 - ratio

    coefs = np.array([p3,p2,p1,p0])
    # print(coefs)
    coefs[np.abs(coefs) < coefEps] = 0
    # print(coefs)
    r = np.roots(coefs)
    # print(r)
    c = r[r>0].min()
    initVol = np.linalg.det(cell)
    rdcCell = cell - c*f_h
    rdcVol = np.linalg.det(cell - c*f_h)
    # print("Initial Volume: {}".format(initVol))
    # print("Reduced Volume: {}".format(rdcVol))
    # print("Target ratio: {}, Real ratio: {}".format(ratio, rdcVol/initVol))
    if abs(ratio-rdcVol/initVol) < ratioEps:
        return rdcCell
    else:
        return None

def compress_mol_crystal(molC, minRatio, bondRatio, nsteps=10):
    partition = [set(p) for p in molC.partition]
    ratioArr = np.linspace(1, minRatio, nsteps+1)
    ratioArr = ratioArr[1:]/ratioArr[:-1]
    inMolC = molC

    for ratio in ratioArr:
        centers = inMolC.get_centers()
        rltPos = inMolC.get_rltPos()
        cell = inMolC.get_cell()
        rdcCell = lda_mol(centers, rltPos, cell, ratio)
        outMolC = inMolC.copy()
        if rdcCell is None:
            logging.debug("Too different ratio")
            return inMolC
        else:
            outMolC.set_cell(rdcCell)
            testMolC = atoms2molcryst(outMolC.to_atoms(), bondRatio)
            if False in [set(p) in partition for p in testMolC.partition]:
                # logging.debug('Overlap between molecules')
                return inMolC
            else:
                inMolC = outMolC

    return outMolC

def compress_mol_pop(molPop, volRatio, bondRatio, nsteps=10):
    outPop = []
    for ind in molPop:
        minRatio = volRatio/calc_volRatio(ind)
        logging.debug("minRatio: {}".format(minRatio))
        if minRatio < 1:
            molC = atoms2molcryst(ind, bondRatio)
            outMolC = compress_mol_crystal(molC, minRatio, bondRatio, nsteps)
            outInd = outMolC.to_atoms()
            outInd.info = ind.info.copy()
            outPop.append(outInd)
        else:
            outPop.append(ind)

    return outPop







