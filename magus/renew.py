import random, sys, os, re, math, logging, yaml
from collections import Counter
import numpy as np
import spglib
from numpy import pi, sin, cos, sqrt
from numpy import dot
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.gaussian_process import kernels
import ase.io
from ase import Atom, Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import NeighborList
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.optimize import BFGS, FIRE, BFGSLineSearch, LBFGS, LBFGSLineSearch
from ase.units import GPa
from ase.constraints import UnitCellFilter#, ExpCellFilter
from sklearn import cluster
from .utils import *
from .machinelearning import LRCalculator
from .bayes import UtilityFunction, GP_fit, atoms_util
from .machinelearning import LRmodel, LRCalculator
import itertools
from ase.geometry import cell_to_cellpar,cellpar_to_cell,get_duplicate_atoms
from ase.build import make_supercell

class BaseEA:
    def __init__(self, parameters):

        self.parameters = parameters
        self.symbols = parameters.symbols
        self.formula = parameters.formula
        self.saveGood = parameters.saveGood
        self.addSym = parameters.addSym
        self.symprec = parameters.symprec
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
        self.ripNum = parameters.ripNum
        self.grids = parameters.grids
        if self.parameters.molMode:
            self.inputMols = [Atoms(**molInfo) for molInfo in parameters.molList]

        self.mutDict = {
            'perm': self.permNum,
            'lat': self.latNum,
            'slip': self.slipNum,
            'rip': self.ripNum,
            'rot': self.rotNum,
        }

        # self.kind = krigParm['kind']
        # self.xi = krigParm['xi']
        # self.kappaLoop = krigParm['kappaLoop']
        # self.scaled_factor = krigParm['scale']

        self.parent_factor = 0

        self.dRatio = parameters.dRatio
        self.bondRatio = parameters.bondRatio
        self.bondRange = parameters.bondRange

        self.newLen = int((self.parameters.popSize*(1-self.parameters.randFrac)))

    def heredity(self, cutNum=5):
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
            # sampleNum = int(splitLen/4)+1
            sampleNum = int(self.parameters.tourRatio*splitLen) + 1
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

                if self.parameters.molDetector == 0:
                    hrdInd = cut_cell([spInd, goodInd], grid, symbols, 0.2)
                elif self.parameters.molDetector in [1,2]:
                    spMolC = MolCryst(**spInd.info['molDict'])
                    # spMolC.set_cell(spInd.info['molCell'], scale_atoms=False)
                    goodMolC = MolCryst(**goodInd.info['molDict'])
                    # goodMolC.set_cell(goodInd.info['molCell'], scale_atoms=False)
                    cutAxis = random.randrange(0,3)
                    hrdInd = mol_cut_cell(spMolC, goodMolC, cutAxis)
                hrdInd = merge_atoms(hrdInd, self.dRatio)


                parentE = 0.5*(sum([ind.info['enthalpy'] for ind in [spInd, goodInd]]))
                parDom = 0.5*(sum([ind.info['sclDom'] for ind in [spInd, goodInd]]))
                hrdInd.info['parentE'] = parentE
                hrdInd.info['parDom'] = parDom
                hrdInd.info['symbols'] = symbols
                hrdInd.info['origin'] = 'cut'

                if self.calcType == 'fix':
                    hrdLen = len(hrdInd)
                    if self.parameters.minAt <= hrdLen <= self.parameters.maxAt :
                        nfm = int(round(hrdLen/sum(self.formula)))
                    else:
                        nfm = int(round(0.5 * sum([ind.info['numOfFormula'] for ind in [spInd, goodInd]])))
                    hrdInd.info['formula'] = self.formula
                    hrdInd.info['numOfFormula'] = nfm
                    hrdInd = repair_atoms(hrdInd, symbols, self.formula, nfm)
                elif self.calcType == 'var':
                    curFrml = get_formula(hrdInd, self.symbols)
                    if not check_var_formula(curFrml, self.formula, self.parameters.minAt, self.parameters.maxAt):
                        bestFrml = best_formula(curFrml, self.formula)
                        if self.parameters.minAt <= sum(bestFrml) <= self.parameters.maxAt:
                            hrdInd.info['formula'] = bestFrml
                            hrdInd.info['numOfFormula'] = 1
                            hrdInd = repair_atoms(hrdInd, symbols, bestFrml, 1)
                        else:
                            hrdInd = None
                    else:
                        hrdInd.info['formula'] = get_formula(hrdInd, self.symbols)
                        hrdInd.info['numOfFormula'] = 1

                if hrdInd:
                    hrdPop.append(hrdInd)
                # pairPop = [ind for ind in [ind1, ind2] if ind]
                # hrdPop.extend(del_duplicate(pairPop, compareE=False, report=False, symprec=self.symprec))

        return hrdPop

    def mutation(self, mutDict):
        """
        mutDict: dict, number of structures for different mutation operators
        """
        mutPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(self.parameters.tourRatio*splitLen) + 1
            for mut, mutNum in mutDict.items():
                for _ in range(mutNum):
                    parInd = tournament(splitPop, sampleNum)
                    parentE = parInd.info['enthalpy']
                    parDom = parInd.info['sclDom']
                    if self.parameters.molDetector == 0:
                        if mut == 'perm':
                            mutInd = exchage_atom(parInd)
                        elif mut == 'lat':
                            mutInd = gauss_mut(parInd)
                        elif mut == 'slip':
                            mutInd = slip(parInd, cut=random.random())
                        elif mut == 'rip':
                            mutInd = ripple(parInd, rho=random.uniform(0.5,1.5))
                        else:
                            break
                    elif self.parameters.molDetector in [1,2]:
                        parMolC = MolCryst(**parInd.info['molDict'])
                        # parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                        if mut == 'perm':
                            mutMolC = mol_exchage(parMolC)
                        elif mut == 'lat':
                            mutMolC = mol_gauss_mut(parMolC)
                        elif mut == 'slip':
                            mutMolC = mol_slip(parMolC, cut=random.random())
                        elif mut == 'rip':
                            mutMolC = mol_ripple(parMolC, rho=random.uniform(0.5,1.5))
                        elif mut == 'rot':
                            partLens = [len(p) for p in parMolC.partition]
                            if max(partLens) == 1:
                                continue
                            mutMolC = mol_rotation(parMolC)
                        else:
                            break
                        mutInd = mutMolC.to_atoms()

                    mutInd.info = dict()
                    mutInd.info['symbols'] = parInd.info['symbols']
                    mutInd.info['formula'] = parInd.info['formula']
                    mutInd.info['numOfFormula'] = parInd.info['numOfFormula']
                    mutInd.info['parentE'] = parentE
                    mutInd.info['parDom'] = parDom
                    mutInd.info['origin'] = mut

                    mutInd = merge_atoms(mutInd, self.dRatio)
                    toFrml = [int(i) for i in parInd.info['formula']]
                    mutInd = repair_atoms(mutInd, parInd.info['symbols'], toFrml, parInd.info['numOfFormula'])
                    if mutInd:
                        mutPop.append(mutInd)

        return mutPop

    def generate(self,curPop):
        self.curPop = calc_dominators(curPop)
        if self.parameters.addSym:
            self.curPop = symmetrize_pop(self.curPop, self.symprec)
        # remove ind which do not contain all symbols
        if self.parameters.fullEles and self.parameters.calcType == 'var':
            self.curPop = list(filter(lambda x: 0 not in x.info['formula'], self.curPop))
        # decompose atoms into modulars
        if self.parameters.molDetector in [1,2]:
            if self.parameters.chkMol:
                self.curPop = mol_dict_pop(self.curPop, self.parameters.molDetector, [self.bondRatio])
            else:
                self.curPop = mol_dict_pop(self.curPop, self.parameters.molDetector, self.bondRange)

        self.labels, self.goodPop = clustering(self.curPop, self.parameters.saveGood)

        self.curLen = len(self.curPop)
        logging.debug("curLen: {}".format(self.curLen))
        assert self.curLen >= self.saveGood, "saveGood should be shorter than length of curPop!"

        self.tmpPop = list()
        self.nextPop = list()

        self.clusters = []
        for i in range(self.saveGood):
            self.clusters.append([ind for n, ind in enumerate(self.curPop) if self.labels[n] == i])

        hrdPop = self.heredity(self.cutNum)
        logging.debug("hrdPop length: {}".format(len(hrdPop)))
        mutPop = self.mutation(self.mutDict)
        logging.debug("mutPop length: {}".format(len(mutPop)))
        tmpPop = hrdPop + mutPop
        tmpPop = check_dist(tmpPop, self.dRatio)
        if self.parameters.chkMol:
            tmpPop = check_mol_pop(tmpPop, self.inputMols, self.parameters.bondRatio)
        self.tmpPop.extend(tmpPop)
        return self.tmpPop

    def select(self):
        tmpPop = self.tmpPop[:]
        newPop = []
        if self.newLen < len(tmpPop):
            for _ in range(self.newLen):
                newInd = tournament(tmpPop, int(self.parameters.tourRatio*len(tmpPop)) + 1, keyword='parDom')
                newPop.append(newInd)
                tmpPop.remove(newInd)

            return newPop
        else:
            return tmpPop

class easyMLEA(BaseEA):
    """
    only use ML module to guess energy
    """
    def __init__(self, parameters, ML):
        self.ML = ML
        return super().__init__(parameters)

    def select(self):
        tmpPop = self.tmpPop[:]
        newPop = []
        if self.newLen < len(tmpPop):
            for _ in range(self.newLen):
                newInd = tournament(tmpPop, int(self.parameters.tourRatio*len(tmpPop)) + 1, keyword='enthalpy')
                newPop.append(newInd)
                tmpPop.remove(newInd)
        else:
            newPop = tmpPop
        # remove the enthalpy key
        for ind in newPop:
            ind.info['predictE'] = ind.info['enthalpy']
            del ind.info['enthalpy']

        return newPop

class BOEA(BaseEA):
    """
    Bayesian Optimization plus EA
    """
    def __init__(self, parameters):
        self.ML = LRmodel(parameters)
        return super().__init__(parameters)


    def fit_gp(self):
        # fps, ens = read_dataset()
        fps = [ind.info['image_fp']/np.linalg.norm(ind.info['image_fp']) for ind in self.curPop]
        fps = [fp.flatten().tolist() for fp in fps]
        ens = [ind.info['energy']/len(ind) for ind in self.curPop]

        fps = np.array(fps)
        ens = np.array(ens)
        # logging.debug("ens: {}".format(ens))

        gpParm = {
            'alpha': 1e-5,
            'n_restarts_optimizer': 25,
            'normalize_y': True,
        }

        if self.parameters.kernelType == 'dot':
            kernel = (kernels.DotProduct(sigma_0=0))**2
            gpParm['n_restarts_optimizer'] = 0
        elif self.parameters.kernelType == 'rbf':
            kernel = kernels.RBF()

        gpParm['kernel'] = kernel


        gp = GP_fit(fps, ens, gpParm)
        logging.debug("kernel hyperparameter:\n")
        logging.debug(gp.kernel_.get_params())
        self.gp = gp
        self.fps = fps

    def select(self):
        self.fit_gp()

        tmpPop = self.tmpPop[:]
        self.ML.get_fp(tmpPop)

        gp = self.gp
        for ind in tmpPop:
            normFps = ind.info['image_fp']/np.linalg.norm(ind.info['image_fp'])
            preEn, sigma = gp.predict(normFps.reshape(1,-1), return_std=True)
            preEn = preEn[0]
            sigma = sigma[0]
            ind.info['predictE'] = preEn + self.parameters.pressure * GPa * ind.get_volume()/len(ind)
            ind.info['sigma'] = sigma
            ind.info['utilVal'] = ind.info['predictE'] - self.parameters.kappa * sigma

        # tmpPop = [ind for ind in tmpPop if ind.info['sigma'] > 5e-3]

        newPop = []
        if self.newLen < len(tmpPop):
            for _ in range(self.newLen):
                newInd = tournament(tmpPop, int(self.parameters.tourRatio*len(tmpPop)) + 1, keyword='utilVal')
                newPop.append(newInd)
                tmpPop.remove(newInd)
        else:
            newPop = tmpPop

        return newPop


class MLcutEA(easyMLEA):
    """
    use ML module to help cut cell
    """
    def __init__(self, parameters, ML):
        self.ML = ML
        self.calc = LRCalculator(self.ML.reg,self.ML.cf)
        return super().__init__(parameters)

    def displace(self, cutNum=5):
        #curPop = standardize_pop(self.curPop, 1.)
        curPop = self.curPop
        symbols = self.symbols
        hrdPop = list()
#<<<<<<< magus/renew.py
#        for i in range(self.saveGood):
#            goodInd = goodPop[i]
#            splitPop = [ind for ind in self.clusters[i] if ind.info['dominators'] > goodInd.info['dominators']]
#            splitLen = len(splitPop)
#            sampleNum = int(self.parameters.tourRatio*splitLen) + 1
#            logging.debug("splitlen: %s"%(splitLen))
#            if splitLen <= 1:
#                continue
#            for j in range(cutNum):
#                grid = random.choice(grids)
#                spInd = tournament(splitPop, sampleNum)
#                tranPos = spInd.get_scaled_positions() # Displacement
#                tranPos += np.array([[random.random(), random.random(), random.random()]]*len(spInd))
#                spInd.set_scaled_positions(tranPos)
#                spInd.wrap()
#
#                ind1 = cut_cell_ml([spInd, goodInd], grid, symbols, self.calc, 0.2)
#                ind2 = cut_cell_ml([goodInd, spInd], grid, symbols, self.calc, 0.2)
#                ind1 = merge_atoms(ind1, self.dRatio)
#                ind2 = merge_atoms(ind2, self.dRatio)
#
#
#                parentE = 0.5*(sum([ind.info['enthalpy'] for ind in [spInd, goodInd]]))
#                parDom = 0.5*(sum([ind.info['sclDom'] for ind in [spInd, goodInd]]))
#                ind1.info['parentE'], ind2.info['parentE'] = parentE, parentE
#                ind1.info['parDom'], ind2.info['parDom'] = parDom, parDom
#                ind1.info['symbols'], ind2.info['symbols'] = symbols, symbols
#
#
#                nfm = int(round(0.5 * sum([ind.info['numOfFormula'] for ind in [spInd, goodInd]])))
#                ind1 = repair_atoms(ind1, symbols, self.formula, nfm)
#                ind2 = repair_atoms(ind2, symbols, self.formula, nfm)
#                ind1.info['formula'], ind2.info['formula'] = self.formula, self.formula
#                ind1.info['numOfFormula'], ind2.info['numOfFormula'] = nfm, nfm
#
#                pairPop = [ind for ind in [ind1, ind2] if ind is not None]
#                hrdPop.extend(del_duplicate(pairPop, compareE=False, report=False))
#=======
        r = 3  #radius of the ball to be displaced
        core_enenrgies = {}
        for i,ind in enumerate(curPop):
            energies = np.array(self.calc.get_potential_energies(ind))
            nl = NeighborList(cutoffs=[r*len(ind)], skin=0, self_interaction=True, bothways=True)
            nl.update(ind)
            ind.info['core_enenrgies']=[]
            for j in range(len(ind)):
                core_enenrgies[(i,j)] = np.mean(energies[(nl.get_neighbors(j)[0],)])
                ind.info['core_enenrgies'].append(core_enenrgies[(i,j)])

        for ind in curPop:
            tmp = np.array(ind.info['core_energies'])
            tmp = tmp - np.mean(tmp)
            i = np.random.choice(range(len(ind)),p=np.e**tmp/np.sum(np.e**tmp))
            tmp = np.array(list(core_enenrgies.values()))
            tmp = tmp - np.mean(tmp)
            _, j = np.random.choice(list(core_enenrgies.keys()),p=np.e**tmp/np.sum(np.e**tmp))
            ind2 = curPop[_]
            hrdInd = replaceball(ind,i,ind2,j,r)
            hrdInd = merge_atoms(hrdInd, self.dRatio)
            hrdInd.info['symbols'] = symbols

            if self.calcType == 'fix':
                hrdLen = len(hrdInd)
                if self.parameters.minAt <= hrdLen <= self.parameters.maxAt :
                    nfm = int(round(hrdLen/sum(self.formula)))
                else:
                    nfm = int(round(0.5 * sum([ind.info['numOfFormula'] for ind in [spInd, goodInd]])))
                hrdInd.info['formula'] = self.formula
                hrdInd.info['numOfFormula'] = nfm
                hrdInd = repair_atoms(hrdInd, symbols, self.formula, nfm)
            elif self.calcType == 'var':
                curFrml = get_formula(hrdInd, self.symbols)
                if not check_var_formula(curFrml, self.formula, self.parameters.minAt, self.parameters.maxAt):
                    bestFrml = best_formula(curFrml, self.formula)
                    if self.parameters.minAt <= sum(bestFrml) <= self.parameters.maxAt:
                        hrdInd.info['formula'] = bestFrml
                        hrdInd.info['numOfFormula'] = 1
                        hrdInd = repair_atoms(hrdInd, symbols, bestFrml, 1)
                    else:
                        hrdInd = None
                else:
                    hrdInd.info['formula'] = get_formula(hrdInd, self.symbols)
                    hrdInd.info['numOfFormula'] = 1
#>>>>>>> magus/renew.py

            if hrdInd:
                hrdPop.append(hrdInd)
        return hrdPop

    def select(self):
        tmpLen = len(self.tmpPop)
        tmpPop = self.ML.scf(self.tmpPop[:])
        newPop = []
        for _ in range(self.newLen):
            newInd = tournament(tmpPop, int(self.parameters.tourRatio*len(tmpPop)) + 1, keyword='energy')
            newPop.append(newInd)
            tmpPop.remove(newInd)
        return newPop

"""
TODO rotate the replace ball
how to rotate the ball to make energy lower
"""
def replace_ball(atoms1,i,atoms2,j,cutR):
    """replace some atoms in a ball
    
    Arguments:
        atoms1 {atoms} -- [structure to be replaced]
        i {int} -- [atom index of the centre of ball]
        atoms2 {atoms} -- [structure to replace]
        j {int} -- [atom index of the centre of ball]
        cutR {float} -- [radius of the ball]
    """
    newatoms = Atoms(pbc=atoms1.pbc, cell=atoms1.cell)

    nl = NeighborList(cutoffs=[cutR/2]*len(atoms1), skin=0, self_interaction=True, bothways=True)
    nl.update(atoms1)
    indices, _ = nl.get_neighbors(i)
    for index,atom in enumerate(atoms1):
        if index not in indices:
            newatoms.append(atom)
 
    nl = NeighborList(cutoffs=[cutR/2]*len(atoms2), skin=0, self_interaction=True, bothways=True)
    nl.update(atoms2)
    indices, _ = nl.get_neighbors(j)
    atoms2.positions += atoms1.positions[i]-atoms2.positions[j]
    newatoms.extend(atoms2[indices])
    return newatoms 


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


def cut_cell_new(atoms1, atoms2, cutDisp=0):
    """cut two cells to get a new cell
    
    Arguments:
        atoms1 {atoms} -- atoms1 to be cut
        atoms2 {atoms} -- atoms2 to be cut
    
    Keyword Arguments:
        cutDisp {int} -- dispalacement in cut (default: {0})
    
    Raises:
        RuntimeError: no atoms in new cell
    
    Returns:
        atoms -- generated atoms
    """
    atoms1,atoms2,ratio1,ratio2 = match_lattice(atoms1,atoms2)
    cutCell = (atoms1.get_cell()+atoms2.get_cell())*0.5
    cutCell[2] = (atoms1.get_cell()[2]/ratio1+atoms2.get_cell()[2]/ratio2)*0.5
    cutVol = (atoms1.get_volume()/ratio1+atoms2.get_volume()/ratio2)*0.5
    cutCellPar = cell_to_cellpar(cutCell)
    ratio = cutVol/abs(np.linalg.det(cutCell))
    if ratio > 1:
        cutCellPar[:3] = [length*ratio**(1/3) for length in cutCellPar[:3]]

    cutInd = Atoms(cell=cutCellPar,pbc = True,)
    scaled_positions = []
    cutPos = 0.5+cutDisp*np.random.uniform(-0.5, 0.5)
    for atom in atoms1:
        if 0 <= atom.c < cutPos/ratio1:
            cutInd.append(atom)
            scaled_positions.append([atom.a,atom.b,atom.c])
    for atom in atoms2:
        if 0 <= atom.c < (1-cutPos)/ratio2:
            cutInd.append(atom)
            scaled_positions.append([atom.a,atom.b,atom.c+cutPos/ratio1])

    cutInd.set_scaled_positions(scaled_positions)
    if len(cutInd) == 0:
        raise RuntimeError('No atoms in the new cell')
    return cutInd

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
    for i in range(6):
        if latGauss[i] >= 1 or latGauss[i] <= -1:
            latGauss[i] = sigma
    strain = np.array([
        [1+latGauss[0], latGauss[1]/2, latGauss[2]/2],
        [latGauss[1]/2, 1+latGauss[3], latGauss[4]/2],
        [latGauss[2]/2, latGauss[4]/2, 1+latGauss[5]]
        ])
    chdCell = np.dot(chdCell,strain)
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

def mol_dict_pop(pop, molDetector=1, coefRange=[1.1,], scale_cell=False):
    logging.debug("mol_dict_pop():")
    logging.debug("coef range: {}".format(coefRange))
    molPop = [ind.copy() for ind in pop]
    for ind in molPop:
        maxMolNum = len(ind)
        molC = None
        for coef in coefRange:
            if molDetector == 1:
                tryMolc = atoms2molcryst(ind, coef)
            elif molDetector == 2:
                tryMolc = atoms2communities(ind, coef)
            if tryMolc.numMols <= maxMolNum:
                logging.debug("coef: {}\tnumMols: {}".format(coef, tryMolc.numMols))
                molC = tryMolc
                maxMolNum = tryMolc.numMols
        oriVol = ind.get_volume()
        oriCell = ind.get_cell()
        radius = np.array(molC.get_radius())
        eleRad = covalent_radii[max(ind.get_atomic_numbers())]
        radius += eleRad
        molVol = 4/3 * pi * np.power(radius, 3).sum()
        logging.debug("partition {}".format(molC.partition))
        logging.debug("oriVol: {}\tmolVol: {}".format(oriVol, molVol))
        if scale_cell and molVol > oriVol:
            ratio = float((molVol/oriVol)**(1./3))
            molCell = oriCell*ratio
        else:
            molCell = oriCell
        ind.info['molDict'] = molC.to_dict()
        ind.info['molCell'] = molCell

    return molPop

def mol_gauss_mut(parInd, sigma=0.5, cellCut=1, distCut=0):
    """
    Gaussian mutation for molecule crystal.
    parInd should be a MolCryst object.
    """
    chdInd = parInd.copy()
    parVol = parInd.get_volume()
    rmax = covalent_radii[parInd.get_numbers()].max()

    chdCell = chdInd.get_cell()
    latGauss = [random.gauss(0, sigma)*cellCut for i in range(6)]
    for i in range(6):
        if latGauss[i] >= 1 or latGauss[i] <= -1:
            latGauss[i] = sigma
    strain = np.array([
        [1+latGauss[0], latGauss[1]/2, latGauss[2]/2],
        [latGauss[1]/2, 1+latGauss[3], latGauss[4]/2],
        [latGauss[2]/2, latGauss[4]/2, 1+latGauss[5]]
        ])
    chdCell = np.dot(chdCell,strain)
    cellPar = cell_to_cellpar(chdCell)
    ratio = parVol/abs(np.linalg.det(chdCell))
    cellPar[:3] = [length*ratio**(1/3) for length in cellPar[:3]]
    chdCell = cellpar_to_cell(cellPar)
    chdInd.set_cell(chdCell, scale_atoms=False, scale_centers=True)

    chdCenters = parInd.get_centers()
    atGauss = np.random.normal(0, sigma, chdCenters.shape)*distCut
    # atGauss = np.array([random.uniform(-1*sigma, sigma)*distCut for i in range(3)])
    chdCenters += atGauss*rmax
    chdInd.update_centers_and_rltPos(centers=chdCenters)
    # chdInd.update_sclCenters_and_rltSclPos(sclCenters=chdCenters)
        # at.position += atGauss*covalent_radii[atomic_numbers[at.symbol]]


    return chdInd

def mol_rotation(parInd, sigma=0.1):

    rmax = covalent_radii[parInd.get_numbers()].max()
    chdInd = parInd.copy()

    # rotation
    chdRltPos = []
    for pos in chdInd.rltPos:
        newPos = np.dot(pos, rand_rotMat())
        chdRltPos.append(newPos)

    # mutation
    chdCenters = parInd.get_centers()
    # atGauss = np.random.normal(0, sigma, chdCenters.shape)/sigma
    #chdCenters += atGauss*rmax

    chdInd.update_centers_and_rltPos(rltPos=chdRltPos, centers=chdCenters)
    # chdInd.update_sclCenters_and_rltSclPos(sclCenters=sclCenters)
    return chdInd

def mol_exchage(parInd):


    chdInd = parInd.copy()
    chdCenters = chdInd.get_sclCenters().tolist()
    random.shuffle(chdCenters)
    chdInd.update_sclCenters_and_rltSclPos(sclCenters=chdCenters)

    return chdInd

def mol_slip(parInd, cut=0.5, randRange=[0.2, 0.8]):

    chdInd = parInd.copy()
    sclCenters = parInd.get_sclCenters()
    axis = list(range(3))
    random.shuffle(axis)
    rand1 = random.uniform(*randRange)
    rand2 = random.uniform(*randRange)

    for i in range(len(sclCenters)):
        if sclCenters[i, axis[0]] > cut:
            sclCenters[i, axis[1]] += rand1
            sclCenters[i, axis[2]] += rand2

    chdInd.update_sclCenters_and_rltSclPos(sclCenters=sclCenters)
    return chdInd

def mol_cut_cell(parInd1, parInd2, axis=0):
    """
    Cut two MolCryst to create a new one.
    Return an Atoms object.
    """

    # cell
    cutCell = 0.5*(parInd1.get_cell() + parInd2.get_cell())
    cutVol = 0.5*(parInd1.get_volume() + parInd2.get_volume())
    cutCellPar = cell_to_cellpar(cutCell)
    ratio = cutVol/abs(np.linalg.det(cutCell))
    if ratio > 1:
        cutCellPar[:3] = [length*ratio**(1/3) for length in cutCellPar[:3]]
    cutCell = cellpar_to_cell(cutCellPar)

    # atomic numbers
    numList = []
    # atom's positions
    posList = []

    for n, ind in enumerate([parInd1, parInd2]):
        sclCenters = ind.get_sclCenters()
        centers = ind.get_centers()
        rltPos = ind.get_rltPos()
        numbers = ind.get_numbers()
        for i in range(ind.numMols):
            if 0.5*n <= sclCenters[i, axis] < 0.5*(n+1):
                indices = ind.partition[i]
                molNums = numbers[indices].tolist()
                numList.extend(molNums)
                newCenter = np.dot(sclCenters[i], cutCell)
                molPos =(newCenter + rltPos[i]).tolist()
                posList.extend(molPos)
                # print(rltPos[i])
                # print(molNums)

    cutInd = Atoms(numbers=numList, positions=posList, cell=cutCell, pbc=True)

    return cutInd

def mol_ripple(parInd, rho=0.3, mu=2, eta=1):
    '''
    from XtalOpt
    '''
    chdInd = parInd.copy()
    sclCenters = parInd.get_sclCenters()
    axis = list(range(3))
    random.shuffle(axis)

    for i in range(len(sclCenters)):
        sclCenters[i, axis[0]] += rho * cos(2*pi*mu*sclCenters[i, axis[1]] +
        random.uniform(0, 2*pi))*cos(2*pi*eta*sclCenters[i, axis[2]] + random.uniform(0, 2*pi))

    chdInd.update_sclCenters_and_rltSclPos(sclCenters=sclCenters)
    return chdInd



#=======
#>>>>>>> magus/renew.py
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
    # logging.debug("toDict: {}".format(toDict))
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
                    if len(repInd) == 0:
                        return None
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
"""
TODO add use_tags:   
Whether to use the atomic tags to preserve molecular identity.
"""
from ase.ga.offspring_creator import OffspringCreator, CombinationMutation


def atoms_too_close(atoms, bl, use_tags=False):
    """ Checks if any atoms in a are too close, as defined by
        the distances in the bl dictionary.

        use_tags: whether to use the Atoms tags to disable distance
                  checking within a set of atoms with the same tag.

        Note: if certain atoms are constrained and use_tags is True,
        this method may return unexpected results in case the
        contraints prevent same-tag atoms to be gathered together in
        the minimum-image-convention. In such cases, one should
        (1) release the relevant constraints,
        (2) apply the gather_atoms_by_tag function, and
        (3) re-apply the constraints, before using the
            atoms_too_close function. """
    a = atoms.copy()
    if use_tags:
        gather_atoms_by_tag(a)

    pbc = a.get_pbc()
    cell = a.get_cell()
    num = a.get_atomic_numbers()
    pos = a.get_positions()
    tags = a.get_tags()
    unique_types = sorted(list(set(num)))

    neighbours = []
    for i in range(3):
        if pbc[i]:
            neighbours.append([-1, 0, 1])
        else:
            neighbours.append([0])

    for nx, ny, nz in itertools.product(*neighbours):
        displacement = np.dot(cell.T, np.array([nx, ny, nz]).T)
        pos_new = pos + displacement
        distances = cdist(pos, pos_new)

        if nx == 0 and ny == 0 and nz == 0:
            if use_tags and len(a) > 1:
                x = np.array([tags]).T
                distances += 1e2 * (cdist(x, x) == 0)
            else:
                distances += 1e2 * np.identity(len(a))

        iterator = itertools.combinations_with_replacement(unique_types, 2)
        for type1, type2 in iterator:
            x1 = np.where(num == type1)
            x2 = np.where(num == type2)
            if np.min(distances[x1].T[x2]) < bl[(type1, type2)]:
                return True

    return False


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

