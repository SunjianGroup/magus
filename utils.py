from __future__ import print_function, division
import os, subprocess, shutil, math, random, re, logging, fractions
from collections import Counter
import numpy as np
import spglib
from numpy import pi, sin, cos, tan, sqrt
import networkx as nx
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers, covalent_radii
try:
    from functools import reduce
except ImportError:
    pass
from .crystgraph import quotient_graph, cycle_sums, graphDim, find_communities, find_communities2, remove_selfloops, nodes_and_offsets


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
    properties = ['numbers', 'positions', 'cell', 'energy', 'forces', 'stress']
    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    atDict = {}
    for p in properties:
        atDict[p] = locals()[p]

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
    for num in atoms.get_atomic_numbers():
        ballVol += 4*math.pi/3*(covalent_radii[num])**3

    volRatio = atoms.get_volume()/ballVol
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
            if minAt <= len(ind) <= maxAt or len(selfSym) < len(setSym):
                formula = [symDic[sym] for sym in setSym]
                ind.info['formula'] = formula
                seedPop.append(ind)
            else:
                logging.debug("ERROR in checking number of atoms")

        elif calcType == 'fix':
            if len(selfSym) == len(setSym):
                formula = [symDic[sym] for sym in setSym]
                logging.info('formula: {!r}'.format(formula))
                selfGcd = reduce(math.gcd, formula)
                selfRd = [x/selfGcd for x in formula]
                if selfRd == setRd:
                    ind.info['formula'] = setFrml
                    ind.info['numOfFormula'] = len(ind)/sum(setFrml)
                    seedPop.append(ind)
                else:
                    logging.debug("ERROR in checking formula")



    # logging.info("Read Seeds: %s"%(len(seedPop)))
    return seedPop

def merge_atoms(atoms, tolerance=0.3, tryNum=5):
    """
    if a pair of atoms are too close, merge them.
    """

    allDist = atoms.get_all_distances(mic=True)
    numbers = atoms.get_atomic_numbers()
    radius = [covalent_radii[num] for num in numbers]
    indices = list(range(len(atoms)))
    exclude = []

    for _ in range(tryNum):
        save = [index for index in indices if index not in exclude]
        short = 0
        for n, i in enumerate(save):
            for j in save[n+1:]:
                if allDist[i,j] < tolerance*(radius[i] + radius[j]):
                    if i not in exclude and j not in exclude:
                        exclude.append(random.choice([i,j]))
                    short += 1
        if short == 0:
            break

    save = [index for index in indices if index not in exclude]
    mAts = atoms[save]
    mAts.info = atoms.info.copy()

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



