from __future__ import print_function, division
import os, subprocess, shutil, math, random, re, logging, fractions, sys, yaml, itertools
from collections import Counter
import numpy as np
import spglib
from numpy import pi, sin, cos, sqrt
import networkx as nx
from ase import Atoms
from ase.geometry import wrap_positions
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list
from ase.phasediagram import PhaseDiagram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cluster
from scipy.optimize import root
from scipy.spatial.distance import cdist, pdist
from .crystgraph import quotient_graph, cycle_sums, graph_dim, find_communities, find_communities2, remove_selfloops, nodes_and_offsets
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar,cellpar_to_cell
try:
    from functools import reduce
except ImportError:
    pass


def todict(p):
    if isinstance(p,EmptyClass):
        d = {}
        for key,value in p.__dict__.items():
            d[key] = todict(value)
        return d
    else:
        return p
class EmptyClass:
    def attach(self,other):
        if not isinstance(other,EmptyClass):
            raise Exception("zhe mei fa attach a")
        for key,value in other.__dict__.items():
            if not hasattr(self,key):
                setattr(self,key,value)

    def save(self,filename):
        d = todict(self)
        with open(filename,'w') as f:
            yaml.dump(d,f)


class MolCryst:
    def __init__(self, numbers, cell, sclPos, partition, offSets=None, sclCenters=None, rltSclPos=None, info=dict()):
        """
        numbers: atomic numbers for all atoms in the structure, same to ase.Atoms.numbers
        cell: cell of the structure, a 3x3 matrix
        sclPos: scaled positions for all atoms, same to ase.Atoms.scaled_positions
        partition: atom indices for every molecule, like [[0,1], [2,3,4]], which means atoms 0,1 belong to one molecule and atoms 3,4,5 belong to another molecule.
        offSets: cell offsets for every atom, a Nx3 integer matrix
        sclCenters: centers of molecules, in fractional coordination
        rltSclPos: positions of atoms relative to molecule centers, in fractional coodination
        To define a molecule crystal correctly, you must set offSets OR sclCenters and rltSclPos.
        """
        self.info = info
        self.partition = tuple([list(p) for p in partition])
        if offSets is not None:
            self.numbers, self.cell, self.sclPos, self.offSets = list(map(np.array, [numbers, cell, sclPos, offSets]))
            # self.partition = tuple(partition)
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
            # self.numbers, self.cell, self.sclPos = list(map(np.array, [numbers, cell, sclPos]))
            # self.partition = tuple(partition)
            # assert len(numbers) == len(sclPos)
            self.numbers, self.cell = list(map(np.array, [numbers, cell]))
            # sclPos should be calculated from sclCenters and rltSclPos
            self.sclPos = np.zeros((len(numbers), 3))
            numAts = len(numbers)
            indSet = set(list(reduce(lambda x, y: x+y, partition)))
            assert indSet == set(range(numAts))

            self.sclCenters = sclCenters - np.floor(sclCenters)
            self.centers = np.dot(self.sclCenters, self.cell)
            self.rltSclPos = rltSclPos
            self.rltPos = [np.dot(pos, self.cell) for pos in rltSclPos]
            self.update()


        else:
            raise RuntimeError("Need to set offsets or (sclCenters and rltSclPos)!")

        self.numAts = len(numbers)
        self.numMols = len(partition)
        self.molNums = [self.numbers[p].tolist() for p in self.partition]

    def get_center_and_rltSclPos(self, indices):
        """
        Return centers and scaled relative positions for each molecule
        """
        molPos = self.dispPos[indices]
        center = molPos.mean(0)
        rltPos = molPos - center
        return center, rltPos

    def get_sclCenters(self):
        """
        Return scaled relative positions for each molecule
        """
        return self.sclCenters[:]

    def get_centers(self):
        """
        Return centers for each molecule
        """
        return self.centers[:]

    def get_rltPos(self):
        """
        Return relative positions for each molecule
        """
        return self.rltPos[:]

    def get_radius(self):
        """
        Return radius for each molecule
        """
        radius = []
        for pos in self.rltPos:
            rmax = max(np.linalg.norm(pos, axis=1))
            radius.append(rmax)
        return radius

    def copy(self):
        """
        Return a copy of current object
        """
        return MolCryst(self.numbers, self.cell, self.sclPos, self.partition, sclCenters=self.sclCenters, rltSclPos=self.rltSclPos, info=self.info.copy())

    def to_dict(self, infoCopy=True):
        """
        Return a dictionary containing all properties of the current molecular crystal
        """
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
        """
        Return the cell
        """
        return self.cell[:]

    def get_numbers(self):
        """
        Return atomic numbers
        """
        return self.numbers[:]

    def get_scaled_positions(self):
        """
        Return scaled positions of all atoms
        """
        return self.sclPos[:]

    def get_volume(self):
        """
        Return the cell volume
        """
        return np.linalg.det(self.cell)

    def get_mols(self):
        """
        Return all the molecules as a list of ASE's Atoms objects
        """
        mols = []
        for n, indices in enumerate(self.partition):
            mol = Atoms(numbers=self.numbers[indices], positions=self.rltPos[n])
            mols.append(mol)
        return mols

    def set_cell(self, cell, scale_atoms=False, scale_centers=True):
        """
        Set the cell
        scale_atoms: scale all atoms (may change relative positions) or not
        scale_center: scale molecule centers (do not change relative positions) or not
        """
        self.cell = np.array(cell)
        if scale_atoms:
            self.centers = np.dot(self.sclCenters, self.cell)
            self.rltPos = [np.dot(pos, self.cell) for pos in self.rltSclPos]
        else:
            if scale_centers:
                self.centers = np.dot(self.sclCenters, self.cell)
            self.update_centers_and_rltPos(self.centers, self.rltPos)

    def to_atoms(self):
        """
        Return the molecular crystal as an ASE's Atoms object
        """
        return Atoms(numbers=self.numbers, scaled_positions=self.sclPos, cell=self.cell, pbc=1, info=self.info.copy())

    def update_centers_and_rltPos(self, centers=None, rltPos=None):
        """
        Set centers and relative positions in Cartesian coodination
        """
        invCell = np.linalg.inv(self.cell)
        if centers is not None:
            self.centers = np.array(centers)
            self.sclCenters = np.dot(self.centers, invCell)
        if rltPos is not None:
            self.rltPos = [np.array(pos) for pos in rltPos]
            self.rltSclPos = [np.dot(pos, invCell) for pos in self.rltPos]
        self.update()

    def update_sclCenters_and_rltSclPos(self, sclCenters=None, rltSclPos=None):
        """
        Set centers and relative positions in fractional coodination
        """

        if sclCenters is not None:
            self.sclCenters = np.array(sclCenters)
            self.centers = np.dot(self.sclCenters, self.cell)
        if rltSclPos is not None:
            self.rltSclPos = [np.array(pos) for pos in rltSclPos]
            self.rltPos = [np.dot(pos, self.cell) for pos in self.rltSclPos]
        self.update()

    def update(self):
        """
        Update centers and relative positions
        """
        molNum = len(self.partition)
        posList = [self.sclCenters[i]+self.rltSclPos[i] for i in range(molNum)]
        tmpSclPos = reduce(lambda x,y: np.concatenate((x,y), axis=0), posList)
        indices = list(reduce(lambda x, y: x+y, self.partition))
        self.sclPos[indices] = tmpSclPos


def atoms2molcryst(atoms, coef=1.1):
    """
    Convert crystal to molecular crystal
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: MolCryst
    """
    QG = quotient_graph(atoms, coef)
    graphs = nx.connected_component_subgraphs(QG)
    partition = []
    offSets = np.zeros([len(atoms), 3])
    for G in graphs:
        if graph_dim(G) == 0 and G.number_of_nodes() > 1:
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

def primitive_atoms2molcryst(atoms, coef=1.1):
    """
    Convert crystal to molecular crystal
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: tags and offsets
    """
    QG = quotient_graph(atoms, coef)
    graphs = nx.connected_component_subgraphs(QG)
    partition = []
    offSets = np.zeros([len(atoms), 3])
    tags = np.zeros(len(atoms))
    for G in graphs:
        if graph_dim(G) == 0 and G.number_of_nodes() > 1:
            nodes, offs = nodes_and_offsets(G)
            partition.append(nodes)
            for i, offSet in zip(nodes, offs):
                offSets[i] = offSet
        else:
            for i in G.nodes():
                partition.append([i])
                offSets[i] = [0,0,0]

    for tag, p in enumerate(partition):
        for j in p:
            tags[j] = tag

    return tags, offSets


def atoms2communities(atoms, coef=1.1):
    """
    Split crystal to communities
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: MolCryst
    """
    QG = quotient_graph(atoms, coef)
    graphs = nx.connected_component_subgraphs(QG)
    partition = []
    offSets = np.zeros([len(atoms), 3])
    for SG in graphs:
        G = remove_selfloops(SG)
        if graph_dim(G) == 0:
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

def primitive_atoms2communities(atoms, coef=1.1):
    """
    Split crystal to communities
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: tags and offsets
    """
    QG = quotient_graph(atoms, coef)
    graphs = nx.connected_component_subgraphs(QG)
    partition = []
    offSets = np.zeros([len(atoms), 3])
    tags = np.zeros(len(atoms))
    for SG in graphs:
        G = remove_selfloops(SG)
        if graph_dim(G) == 0:
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

    for tag, p in enumerate(partition):
        for j in p:
           tags[j] = tag

    return tags, offSets

def symbols_and_formula(atoms):

    allSym = atoms.get_chemical_symbols()
    symbols = list(set(allSym))
    counter = Counter(allSym)
    formula = [counter[s] for s in symbols]

    return symbols, formula

def get_formula(atoms, symbols):
    allSym = atoms.get_chemical_symbols()
    counter = Counter(allSym)
    formula = [counter[s] for s in symbols]
    return formula

def sort_elements(atoms):
    allSym = atoms.get_chemical_symbols()
    symbols = list(set(allSym))
    info = atoms.info
    atList = []
    for s in symbols:
        atList.extend([atom for atom in atoms if atom.symbol==s])
    sortAts = Atoms(atList)
    sortAts.set_cell(atoms.get_cell())
    sortAts.set_pbc(atoms.get_pbc())
    sortAts.info = info

    return sortAts

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
            ind.info['numOfFormula'] = 1
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

def rand_rotMat():
    """
    Get random rotation matrix
    Return: a 3x3 matrix
    """
    phi, theta, psi = [2*math.pi*random.uniform(-1,1) for _ in range(3)]
    rot1 = np.array([[cos(phi),-1*sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]])
    rot2 = np.array([[cos(theta), 0, -1*sin(theta)],[0,1,0],[sin(theta), 0, cos(theta)]])
    rot3 = np.array([[1,0,0],[0,cos(psi),-1*sin(psi)],[0,sin(psi),cos(psi)]])
    rotMat = np.dot(np.dot(rot1, rot2), rot3)
    # print("det of rotMat: {}".format(np.linalg.det(rotMat)))
    return rotMat

def check_mol_ind(molInd, inputNums, bondRatio):
    """
    Check if the crystal is molecular crystal. Here we only check the atomic numbers of each molecule, do not check the inner connectivity of molecules.
    molInd: (ASE.Atoms) input crystal
    inputNums: atomic numbers for all molecules
    bondRatio: the criterion for connecting atoms
    """
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

def convex_hull(Pop):

    hullPop = Pop[:]
    name = [ind.get_chemical_formula() for ind in Pop]
    enth = [ind.info['enthalpy']*len(ind) for ind in Pop]
    refs = zip(name, enth)

    pd = PhaseDiagram(refs, verbose=False)
    for ind in hullPop:
        refE = pd.decompose(ind.get_chemical_formula())[0]
        ehull = ind.info['enthalpy'] - refE/len(ind)
        ind.info['ehull'] = ehull #if ehull > 1e-6 else 0

    return hullPop

def convex_hull_pops(*popArr):
    """
    popArr: allow for multiple pops
    """
    numPop = len(popArr)
    allPop = []
    popLens = []
    for pop in popArr:
        allPop.extend(pop)
        popLens.append(len(pop))

    hullPop = convex_hull(allPop)
    newArr = []
    start = 0
    for i in range(numPop):
        end = start + popLens[i]
        newArr.append(hullPop[start:end])
        start = end

    return newArr


def compare_structure_energy(Pop, diffE=0.01, diffV=0.05, ltol=0.1, stol=0.1, angle_tol=5, compareE=True, mode='naive'):
    """
    naive mode simply compare the enthalpy, volume and spacegroup
    ase mode only compare the structure, do not compare enthalpy
    ase mode still have bug
    """
    priList = []
    for ind in Pop:
        # ind.info['vPerAtom'] = ind.get_volume()/len(ind)
        priInfo = ind.info['priInfo']
        if priInfo:
            lattice, scaled_positions, numbers = priInfo
            priAts = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers, pbc=1)
        else:
            priAts = ind.copy()
        priList.append(priAts)

    toCompare = [(x,y) for x in range(len(Pop)) for y in range(len(Pop)) if x < y]
    # toCompare = [(x,y) for x in Pop for y in Pop if Pop.index(x) < Pop.index(y)]

    comp = SymmetryEquivalenceCheck(to_primitive=False, angle_tol=angle_tol, ltol=ltol, stol=stol, scale_volume=True)
    # comp = SymmetryEquivalenceCheck(to_primitive=True, angle_tol=angle_tol, ltol=ltol, stol=stol,vol_tol=vol_tol)

    rmIndices = []
    for i,j in toCompare:
        s0 = Pop[i]
        s1 = Pop[j]
        pri0 = priList[i]
        pri1 = priList[j]

        if mode == 'ase':
            if comp.compare(pri0, pri1):
                if i not in rmIndices and j not in rmIndices:
                    rmIndices.append(random.choice([i,j]))

        elif mode == 'naive':
            symCt0 = Counter(pri0.numbers)
            symCt1 = Counter(pri1.numbers)
            # compare formula
            if symCt0 != symCt1:
                continue

            # compare space group
            if s0.info['spg'] != s1.info['spg']:
                continue

            # compare volume
            vol0 = pri0.get_volume()
            vol1 = pri1.get_volume()
            if abs(1-vol0/vol1) > diffV:
                continue

            # compare enthalpy
            if compareE and abs(s0.info['enthalpy'] - s1.info['enthalpy']) > diffE:
                continue

            # remove one of them
            if i not in rmIndices and j not in rmIndices:
                if compareE:
                    rmIndices.append(i if s0.info['enthalpy'] > s1.info['enthalpy'] else j)
                else:
                    rmIndices.append(random.choice([i,j]))

    cmpPop = [ind for n, ind in enumerate(Pop) if n not in rmIndices]

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

def del_duplicate(Pop, compareE=True, symprec = 0.2, diffE = 0.005, diffV = 0.01, diffD = 0.01, report=True, mode='naive'):
    dupPop = find_spg(Pop, symprec)
    #for ind in Pop:
     #   logging.info("spg: %s" %ind.info['spg'])
    # sort the pop by composion, wait for adding
    # dupPop = compare_fingerprint(Pop, diffD)
    # logging.info("fingerprint survival: %s" %(len(dupPop)))

    dupPop = compare_structure_energy(dupPop, diffE, diffV, compareE=compareE, mode=mode)
    if report:
        logging.info("del_duplicate survival: %s" %(len(dupPop)))
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

def check_new_atom_dist(atoms, newPosition, newSymbol, threshold):
    newPosition = wrap_positions([newPosition],atoms.cell)[0]
    supAts = atoms * (3,3,3)
    rs = [covalent_radii[num] for num in supAts.get_atomic_numbers()]
    rnew = covalent_radii[atomic_numbers[newSymbol]]
    # Place the new atoms in the centeral cell
    cell = atoms.get_cell()
    centerPos = newPosition+ np.dot([1,1,1],cell)
    distArr = cdist([centerPos], supAts.get_positions(wrap=True))[0]

    for i, dist in enumerate(distArr):
        if dist/(rs[i]+rnew) < threshold:
            return False

    return True

def check_var_formula(inFrml, setFrmls, minAt, maxAt):
    """
    inFrml: 1-D array (M,)
    setFrmls: 2-D array (N,M)
    M is the number of elements, N is the dimension of formula space.
    """
    if sum(inFrml) < minAt or sum(inFrml) > maxAt:
        return False
    # projection_matrix=np.dot(setFrmls.T,np.linalg.pinv(setFrmls.T))
    rank = np.linalg.matrix_rank(np.concatenate((setFrmls, [inFrml])))
    if rank == len(setFrmls):
        return True
    else:
        return False

def best_formula(inFrml, setFrmls):
    """
    inFrml: 1-D array (M,)
    setFrmls: 2-D array (N,M)
    M is the number of elements, N is the dimension of formula space.
    """
    invF = np.linalg.pinv(setFrmls)
    coef = np.rint(np.dot(inFrml, invF)).astype(np.int)
    newFrml = np.dot(coef, setFrmls).astype(np.int)
    bestFrml = newFrml.tolist()

    return bestFrml


def symmetrize_atoms(atoms, symprec=1e-2):
    """
    Use spglib to get standardize cell of atoms
    """

    stdCell = spglib.standardize_cell(atoms, symprec=symprec)
    priCell = spglib.find_primitive(atoms, symprec=symprec)
    if stdCell and len(stdCell[1])==len(atoms):
        lattice, pos, numbers = stdCell
        symAts = Atoms(cell=lattice, scaled_positions=pos, numbers=numbers, pbc=True)
        symAts.info = atoms.info.copy()
    elif priCell and len(priCell[1])==len(atoms):
        lattice, pos, numbers = priCell
        symAts = Atoms(cell=lattice, scaled_positions=pos, numbers=numbers, pbc=True)
        symAts.info = atoms.info.copy()
    else:
        symAts = atoms

    return symAts

def symmetrize_pop(pop, symprec=1e-2):

    # stdPop = list()
    # for ind in pop:
    #     stdInd = symmetrize_atoms(ind, symprec)
    #     if len(stdInd) == len(ind):
    #         stdPop.append(stdInd)
    #     else:
    #         stdPop.append(ind)

    return [symmetrize_atoms(ats, symprec) for ats in pop]


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

def checkParameters(instance,parameters,Requirement=[],Default={}):
    name = instance.__class__.__name__
    for key in Requirement:
        if not hasattr(parameters, key):
            raise Exception("Mei you '{}' wo suan ni ma {}?".format(key,name))
        setattr(instance,key,getattr(parameters,key))

    for key in Default.keys():
        if not hasattr(parameters,key):
            setattr(instance,key,Default[key])
        else:
            setattr(instance,key,getattr(parameters,key))

def match_lattice(atoms1,atoms2):
    """lattice matching , 10.1016/j.scib.2019.02.009
    
    Arguments:
        atoms1 {atoms} -- atoms1
        atoms2 {atoms} -- atoms2
    
    Returns:
        atoms,atoms,float,float -- two best matched atoms in z direction
    """
    def match_fitness(a1,b1,a2,b2):
        #za lao shi you shu zhi cuo wu
        a1,b1,a2,b2 = np.round([a1,b1,a2,b2],3)
        a1x = np.linalg.norm(a1)
        a2x = np.linalg.norm(a2)
        if a1x*a2x ==0:
            return 1000
        b1x = a1@b1/a1x
        b2x = a2@b2/a2x
        b1y = np.sqrt(b1@b1 - b1x**2)
        b2y = np.sqrt(b2@b2 - b2x**2)
        if b1y*b2y == 0:
            return 1000
        exx = (a2x-a1x)/a1x
        eyy = (b2y-b1y)/b1y
        exy = b2x/b1y-a2x/a1x*b1x/b1y
        return np.abs(exx)+np.abs(eyy)+np.abs(exy)
    
    def to_matrix(hkl1,hkl2):
        hklrange = [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
        hklrange = [np.array(_) for _ in hklrange]
        for hkl3 in hklrange:
            M = np.array([hkl1,hkl2,hkl3])
            if np.linalg.det(M)>0:
                break
        return M

    def standard_cell(atoms):
        newcell = cellpar_to_cell(cell_to_cellpar(atoms.cell))
        T = np.linalg.inv(atoms.cell)@newcell
        atoms.positions = atoms.positions@T
        atoms.cell = newcell
        return atoms
        
    cell1,cell2 = atoms1.cell[:],atoms2.cell[:]
    hklrange = [(1,0,0),(0,1,0),(0,0,1),(1,-1,0),(1,1,0),(1,0,-1),(1,0,1),(0,1,-1),(0,1,1),(2,0,0),(0,2,0),(0,0,2)]
    #TODO ba cut cell jian qie ti ji bu fen gei gai le 
    hklrange = [(1,0,0),(0,1,0),(0,0,1)]
    hklrange = [np.array(_) for _ in hklrange]
    minfitness = 1000
    for hkl1,hkl2 in itertools.permutations(hklrange,2):
        for hkl3,hkl4 in itertools.permutations(hklrange,2):
            a1,b1,a2,b2 = hkl1@cell1,hkl2@cell1,hkl3@cell2,hkl4@cell2
            fitness = match_fitness(a1,b1,a2,b2)
            if fitness<minfitness:
                minfitness = fitness
                bestfit = hkl1,hkl2,hkl3,hkl4
    newatoms1 = standard_cell(make_supercell(atoms1,to_matrix(bestfit[0],bestfit[1])))
    newatoms2 = standard_cell(make_supercell(atoms2,to_matrix(bestfit[2],bestfit[3])))
    ratio1 = newatoms1.get_volume()/atoms1.get_volume()
    ratio2 = newatoms2.get_volume()/atoms2.get_volume()
    return newatoms1,newatoms2,ratio1,ratio2
