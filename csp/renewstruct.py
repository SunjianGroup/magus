from __future__ import print_function, division
import random, sys, os, re, math, logging, yaml
from collections import Counter
import numpy as np
import spglib
import networkx as nx
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
from .initstruct import build_struct
from .fingerprint import calc_all_fingerprints, clustering
from .bayes import UtilityFunction, GP_fit, atoms_util
from .writeresults import read_dataset
from .dataset import PreProcess
from .atomgp import AtomGP
from .mlpot import MLPot
from .utils import *
from .crystgraph import quotient_graph, cycle_sums, graphDim

class PotKriging:
    def __init__(self, curPop, curGen, parameters):
        krigParm = parameters['krigParm']
        # global
        self.parameters = parameters
        self.calcType = parameters['calcType']
        self.symbols = parameters['symbols']
        self.formula = parameters['formula']
        self.popSize = parameters['popSize']
        self.invFrml = parameters['invFrml']
        self.saveGood = parameters['saveGood']
        self.minAt = parameters['minAt']
        self.maxAt = parameters['maxAt']
        self.randFrac = krigParm['randFrac']
        self.permNum = krigParm['permNum']
        self.latDisps = krigParm['latDisps']
        self.ripRho = krigParm['ripRho']
        self.rotNum = krigParm['rotNum']
        self.addSym = parameters['addSym']
        self.pressure = parameters['pressure']
        self.kind = krigParm['kind']
        self.xi = krigParm['xi']
        self.grids = krigParm['grids']
        self.kappaLoop = krigParm['kappaLoop']
        self.update_kappa(curGen, krigParm['kappa'])
        self.scaled_factor = krigParm['scale']
        self.parent_factor = 0
        self.fullEles = parameters['fullEles']
        self.fpSetup = yaml.load(open("{}/fpFold/fpsetup.yaml".format(parameters['workDir'])))
        self.molDetector = parameters['molDetector']
        # local
        self.curPop = calc_dominators(curPop)
        self.tmpPop = list()
        self.nextPop = list()
        self.gp = None
        self.util = None
        self.y_max = None
        self.labels, self.goodPop = clustering(self.curPop, self.saveGood)
        if self.addSym:
            self.goodPop = standardize_pop(self.goodPop, 1.)


    def get_nextPop(self):
        return self.nextPop

    def update_kappa(self, curGen, kappa):
        kappaLoop = self.kappaLoop
        if kappaLoop == 1:
            self.kappa = kappa
        else:
            remainder = curGen % kappaLoop
            self.kappa = kappa *(1 - remainder/(kappaLoop - 1))

    def heredity(self, saveGood,):
        #curPop = standardize_pop(self.curPop, 1.)
        curPop = self.curPop
        symbols = self.symbols
        grids = self.grids
        labels, goodPop = self.labels, self.goodPop
        hrdPop = list()
        for i in range(saveGood):
            splitPop = [ind for n, ind in enumerate(curPop) if labels[n] == i]
            goodInd = goodPop[i]
            splitLen = len(splitPop)
            logging.debug("splitlen: %s"%(splitLen))
            for spInd in splitPop:

                tranPos = spInd.get_scaled_positions() # Displacement
                tranPos += np.array([[random.random(), random.random(), random.random()]]*len(spInd))
                spInd.set_scaled_positions(tranPos)
                spInd.wrap()

                # grid = random.choice(grids)
                for grid in grids:
                    try:
                        ind1 = cut_cell([spInd, goodInd], grid, symbols, 0.2)
                        ind2 = cut_cell([goodInd, spInd], grid, symbols, 0.2)
                    except:
                        continue

                    parentE = 0.5*(sum([ind.info['enthalpy'] for ind in [spInd, goodInd]]))
                    ind1.info['parentE'] = parentE
                    ind2.info['parentE'] = parentE
                    ind1.info['symbols'], ind2.info['symbols'] = symbols, symbols

                    if self.calcType == 'fix':
                        nfm = int(round(0.5 * sum([ind.info['numOfFormula'] for ind in [spInd, goodInd]])))
                        ind1 = repair_atoms(ind1, symbols, self.formula, nfm)
                        ind2 = repair_atoms(ind2, symbols, self.formula, nfm)
                        ind1.info['formula'], ind2.info['formula'] = self.formula, self.formula
                        ind1.info['numOfFormula'], ind2.info['numOfFormula'] = nfm, nfm

                    elif self.calcType == 'var':
                        npFrml = np.array(self.formula)
                        frml1 = np.array(get_formula(ind1, symbols))
                        coef1 = np.ceil(np.dot(np.dot(frml1, npFrml.T), self.invFrml))
                        tFrml1 = np.dot(coef1, npFrml)
                        tFrml1 = [int(i) for i in tFrml1]
                        logging.debug('tFrml1: {}'.format(tFrml1))
                        if self.minAt <= sum(tFrml1) <= self.maxAt:
                            ind1 = repair_atoms(ind1, symbols, tFrml1)
                            ind1.info['numOfFormula'] = 1
                        else:
                            ind1 = None

                        frml2 = np.array(get_formula(ind2, symbols))
                        coef2 = np.ceil(np.dot(np.dot(frml2, npFrml.T), self.invFrml))
                        tFrml2 = np.dot(coef2, npFrml)
                        tFrml2 = [int(i) for i in tFrml2]
                        logging.debug('tFrml2: {}'.format(tFrml2))
                        if self.minAt <= sum(tFrml2) <= self.maxAt:
                            ind2 = repair_atoms(ind2, symbols, tFrml2)
                            ind2.info['numOfFormula'] = 1
                        else:
                            ind2 = None

                        # make new structures contain all species
                        if self.fullEles:
                            if 0 in tFrml1:
                                ind1 = None
                            if 0 in tFrml2:
                                ind2 = None


                    pairPop = [ind for ind in [ind1, ind2] if ind]
                    hrdPop.extend(del_duplicate(pairPop, compareE=False, report=False))

        return hrdPop


    def permutate(self, permNum):
        goodPop = self.goodPop
        # curPop = self.curPop
        permPop = list()
        for parInd in goodPop:
            parentE = parInd.info['enthalpy']
            for i in range(permNum):
                rate = random.uniform(0.4,0.6)
                permInd = exchage_atom(parInd, rate)
                permInd.info['symbols'] = self.symbols
                permInd.info['formula'] = parInd.info['formula']
                permInd.info['parentE'] = parentE
                permPop.append(permInd)

        return permPop

    def latmutate(self, disps=[1, 2, 3, 4], sigma=0.2):
        goodPop = self.goodPop
        curPop = self.curPop
        latPop = list()
        for parInd in goodPop:
            parentE = parInd.info['enthalpy']
            for disp in disps:
                latInd = gauss_mut(parInd, sigma=sigma, cellCut=disp, distCut=0.5)
                latInd.info['symbols'] = self.symbols
                latInd.info['formula'] = parInd.info['formula']
                latInd.info['parentE'] = parentE
                latPop.append(latInd)

        return latPop

    def slipmutate(self, cuts=[0.3, 0.4, 0.5]):
        goodPop = self.goodPop
        curPop = self.curPop
        slipPop = list()
        for parInd in goodPop:
            parentE = parInd.info['enthalpy']
            for cut in cuts:
                slipInd = slip(parInd, cut=cut)
                slipInd.info['symbols'] = self.symbols
                slipInd.info['formula'] = parInd.info['formula']
                slipInd.info['parentE'] = parentE
                slipPop.append(slipInd)

        return slipPop

    def ripmutate(self, rhos=[0.2, 0.3, 0.4]):
        goodPop = self.goodPop
        curPop = self.curPop
        ripPop = list()
        for parInd in goodPop:
            parentE = parInd.info['enthalpy']
            for rho in rhos:
                ripInd = ripple(parInd, rho=rho)
                ripInd.info['symbols'] = self.symbols
                ripInd.info['formula'] = parInd.info['formula']
                ripInd.info['parentE'] = parentE
                ripPop.append(ripInd)

        return ripPop

    def fit_gp(self):
        images = [read_atDict(ats.info['trajs'][0][0]) for ats in self.goodPop]
        logging.debug("images: {}".format(len(images)))
        pre = PreProcess(images, self.fpSetup, self.symbols)
        trainData = pre.images_data(mod='trainGP', type='list')

        fpDim = len(self.fpSetup['sf2']['eta'])
        gp = AtomGP(input_dim=fpDim*len(self.symbols), theta=0.6, theta_bound=2,)
        gp.read_fpsetup(self.fpSetup)
        gp.train(trainData, noise=[0.003,0.03,0.005], opt=0, fminOptions={'maxiter': 100}, lr=0.05, train_virial=1, noise_bound=[0.05, 0.2, 0.01], sclNoi=0.01, )
        ff = MLPot(self.fpSetup, self.symbols, gp)
        varff = MLPot(self.fpSetup, self.symbols, gp, computeVar=1)


        self.gp = gp
        self.ff = ff
        self.varff = varff
        # self.y_max = max(-ens)
        # self.fps = fps

    def set_factor(self, factor):
        self.scaled_factor = factor

    def generate(self):
        hrdPop = self.heredity(self.saveGood)
        # hrdPop = []
        permPop = self.permutate(self.permNum)
        latPop = self.latmutate(self.latDisps)
        slipPop = self.slipmutate()
        ripPop = self.ripmutate(self.ripRho)
        logging.debug("hrdPop length: %s"%(len(hrdPop)))
        logging.debug("permPop length: %s"%(len(permPop)))
        logging.debug("latPop length: %s"%(len(latPop)))
        logging.debug("slipPop length: %s"%(len(slipPop)))
        logging.debug("ripPop length: %s"%(len(ripPop)))
        tmpPop = hrdPop + permPop + latPop + slipPop + ripPop
        self.tmpPop.extend(tmpPop)

    def add(self, pop):
        self.tmpPop.extend(pop)

    def select(self):
        tmpPop = self.tmpPop
        symbols = self.symbols
        ff = self.ff
        varff = self.varff
        randFrac = self.randFrac
        popSize = self.popSize
        dRatio = self.parameters['dRatio']

        tmpPop = check_dist(tmpPop, dRatio)
        logging.info("tmpPop length: %s after check_dist"%(len(tmpPop)))
        self.util = UtilityFunction(self.kind, self.kappa, self.xi)
        tmpPop = calc_all_fingerprints(tmpPop, self.parameters)
        for ind in tmpPop:
            ind.set_calculator(ff)
            ucf = UnitCellFilter(ind, scalar_pressure=self.pressure*GPa)
            # temp sigma setup, wait for modification
            ind.info['sigma'] = 0
            ###
            dyn = BFGS(ucf, maxstep=0.1)
            dyn.run(steps=100)
            ## compute variance of energy
            ind.set_calculator(varff)
            _ = ind.get_potential_energy()
            ind.info['sigma'] = varff.results['eVar']/len(ind)
            logging.debug("sigma: {}".format(ind.info['sigma']))


            ind.info['predictE'] = (ind.get_potential_energy()+self.pressure*ind.get_volume()/160.2262)/len(ind)
            ind.info['enthalpy'] = ind.info['predictE']
            ind.info['utilVal'] = ind.info['predictE'] - self.kappa * ind.info['sigma']
        tmpPop = list(filter(lambda x: abs(x.info['predictE'] - x.info['parentE']) > 0.001, tmpPop))
        logging.info("tmpPop length: %s after filter"%(len(tmpPop)))

        # nearPop = filter(lambda x: x.info['sigma'] < 1 , tmpPop)
        # logging.debug("nearPop length: {}".format(len(nearPop)))

        if self.calcType is 'fix':
            tmpPop = sorted(tmpPop, reverse=False, key=lambda ind: ind.info['utilVal'] + ind.info['parentE'] * self.parent_factor)
            # tmpPop = sorted(tmpPop, reverse=False, key=lambda ind: ind.info['sigma'] + ind.info['parentE'] * self.parent_factor)
        elif self.calcType is 'var':
            tmpPop = convex_hull(self.goodPop + tmpPop)
            tmpPop = sorted(tmpPop[len(self.goodPop):], reverse=False, key=lambda ind: ind.info['ehull'] - self.kappa * ind.info['sigma'] + ind.info['parentE'] * self.parent_factor)



        krigLen = int((1-randFrac) * popSize)
        if len(tmpPop) > 5 * popSize:
            tmpPop = tmpPop[:krigLen*5]

        # for ind in tmpPop:
        #     ind.info['enthalpy'] = gp.predict(ind.info['fingerprint'])[0]

        # tmpPop = del_duplicate(tmpPop, compareE=True)
        self.tmpPop = tmpPop

        nextPop = tmpPop[:]

        # krigLen = popSize # debug: save all structures

        if len(nextPop) > krigLen:
            self.nextPop = nextPop[:krigLen]
        else:
            self.nextPop = nextPop[:]

        for ind in nextPop:
            ind.info['enthalpy'] = None


class Kriging:
    def __init__(self, curPop, curGen, parameters):
        krigParm = parameters['krigParm']
        # global
        self.parameters = parameters
        self.calcType = parameters['calcType']
        self.symbols = parameters['symbols']
        self.formula = parameters['formula']
        self.invFrml = parameters['invFrml']
        self.popSize = parameters['popSize']
        self.saveGood = parameters['saveGood']
        self.minAt = parameters['minAt']
        self.maxAt = parameters['maxAt']
        self.randFrac = krigParm['randFrac']
        self.permNum = krigParm['permNum']
        self.latDisps = krigParm['latDisps']
        self.ripRho = krigParm['ripRho']
        self.rotNum = krigParm['rotNum']
        self.cutNum = krigParm['cutNum']
        self.slipNum = krigParm['slipNum']
        self.latNum = krigParm['latNum']
        self.addSym = parameters['addSym']
        self.kind = krigParm['kind']
        self.xi = krigParm['xi']
        self.grids = krigParm['grids']
        self.kappaLoop = krigParm['kappaLoop']
        self.update_kappa(curGen, krigParm['kappa'])
        self.scaled_factor = krigParm['scale']
        self.parent_factor = 0
        self.fullEles = parameters['fullEles']
        self.molDetector = parameters['molDetector']
        self.dRatio = parameters['dRatio']
        self.bondRatio = parameters['bondRatio']
        self.bondMin = parameters['bondMin']
        self.bondStep = parameters['bondStep']
        self.bondStepNum = parameters['bondStepNum']
        self.bondRange = parameters['bondRange']
        self.molScaleCell = parameters['molScaleCell']
        self.chkMol = parameters['chkMol']
        self.molMode = parameters['molMode']
        if self.molMode:
            self.inputMols = [Atoms(**molInfo) for molInfo in parameters['molList']]
        # local
        self.curPop = calc_dominators(curPop)
        if self.fullEles:
            self.curPop = list(filter(lambda x: 0 not in x.info['formula'], self.curPop))
        if self.molDetector > 0:
            # bondRange = [self.bondMin + self.bondStep*i for i in range(self.bondStepNum-1)]
            self.curPop = mol_dict_pop(self.curPop, self.molDetector, self.bondRange, self.molScaleCell)

        self.curLen = len(self.curPop)
        logging.debug("curLen: {}".format(self.curLen))
        assert self.curLen >= self.saveGood, "saveGood should be shorter than length of curPop!"
        # if self.curLen < self.saveGood:
        #     raise RuntimeError("saveGood should be shorter than length of curPop!")

        self.tmpPop = list()
        self.nextPop = list()
        self.gp = None
        self.util = None
        self.y_max = None
        self.labels, self.goodPop = clustering(self.curPop, self.saveGood)
        logging.debug("labels: {}".format(self.labels))
        self.clusters = []
        for i in range(self.saveGood):
            self.clusters.append([ind for n, ind in enumerate(self.curPop) if self.labels[n] == i])


        if self.addSym:
            self.goodPop = standardize_pop(self.goodPop, 0.1)



    def get_nextPop(self):
        return self.nextPop

    def update_kappa(self, curGen, kappa):
        kappaLoop = self.kappaLoop
        if kappaLoop == 1:
            self.kappa = kappa
        else:
            remainder = curGen % kappaLoop
            self.kappa = kappa *(1 - remainder/(kappaLoop - 1))

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
                try:
                    if mode == 'atom':
                        ind1 = cut_cell([spInd, goodInd], grid, symbols, 0.2)
                        ind2 = cut_cell([goodInd, spInd], grid, symbols, 0.2)
                        ind1 = merge_atoms(ind1, self.dRatio)
                        ind2 = merge_atoms(ind2, self.dRatio)
                    elif mode == 'mol':
                        spMolC = MolCryst(**spInd.info['molDict'])
                        spMolC.set_cell(spInd.info['molCell'], scale_atoms=False)
                        goodMolC = MolCryst(**goodInd.info['molDict'])
                        goodMolC.set_cell(goodInd.info['molCell'], scale_atoms=False)
                        cutAxis = random.randrange(0,3)
                        ind1 = mol_cut_cell(spMolC, goodMolC, cutAxis)
                        ind2 = mol_cut_cell(goodMolC, spMolC, cutAxis)
                        ind1 = merge_atoms(ind1, self.dRatio)
                        ind2 = merge_atoms(ind2, self.dRatio)
                except:
                    continue

                parentE = 0.5*(sum([ind.info['enthalpy'] for ind in [spInd, goodInd]]))
                parDom = 0.5*(sum([ind.info['sclDom'] for ind in [spInd, goodInd]]))
                ind1.info['parentE'], ind2.info['parentE'] = parentE, parentE
                ind1.info['parDom'], ind2.info['parDom'] = parDom, parDom
                ind1.info['symbols'], ind2.info['symbols'] = symbols, symbols

                if self.calcType == 'fix':
                    nfm = int(round(0.5 * sum([ind.info['numOfFormula'] for ind in [spInd, goodInd]])))
                    ind1 = repair_atoms(ind1, symbols, self.formula, nfm)
                    ind2 = repair_atoms(ind2, symbols, self.formula, nfm)
                    ind1.info['formula'], ind2.info['formula'] = self.formula, self.formula
                    ind1.info['numOfFormula'], ind2.info['numOfFormula'] = nfm, nfm

                elif self.calcType == 'var':
                    npFrml = np.array(self.formula)

                    frml1 = np.array(get_formula(ind1, symbols))
                    coef1 = np.round(np.dot(np.dot(frml1, npFrml.T), self.invFrml))
                    tFrml1 = np.dot(coef1, npFrml)
                    tFrml1 = [int(i) for i in tFrml1]
                    logging.debug('tFrml1: {}'.format(tFrml1))
                    if self.minAt <= sum(tFrml1) <= self.maxAt:
                        ind1 = repair_atoms(ind1, symbols, tFrml1)
                        ind1.info['numOfFormula'] = 1
                    else:
                        ind1 = None

                    frml2 = np.array(get_formula(ind2, symbols))
                    coef2 = np.round(np.dot(np.dot(frml2, npFrml.T), self.invFrml))
                    tFrml2 = np.dot(coef2, npFrml)
                    tFrml2 = [int(i) for i in tFrml2]
                    logging.debug('tFrml2: {}'.format(tFrml2))
                    if self.minAt <= sum(tFrml2) <= self.maxAt:
                        ind2 = repair_atoms(ind2, symbols, tFrml2)
                        ind2.info['numOfFormula'] = 1
                    else:
                        ind2 = None

                    # make new structures contain all species
                    if self.fullEles:
                        if 0 in tFrml1:
                            ind1 = None
                        if 0 in tFrml2:
                            ind2 = None

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

    def latmutate(self, latNum, sigma=0.7, mode='atom'):
        latPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(splitLen/2) + 1
            for j in range(latNum):
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                if mode == 'atom':
                    latInd = gauss_mut(parInd, sigma=sigma, cellCut=1, distCut=0.5)
                elif mode == 'mol':
                    parMolC = MolCryst(**parInd.info['molDict'])
                    parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                    latMolC = mol_gauss_mut(parMolC, sigma=sigma, cellCut=1, distCut=1)
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

    def mol_rotate(self, rotNum):
        rotPop = list()
        for i in range(self.saveGood):
            splitPop = self.clusters[i]
            splitLen = len(splitPop)
            sampleNum = int(splitLen/2) + 1
            for j in range(rotNum):
                parInd = tournament(splitPop, sampleNum)
                parentE = parInd.info['enthalpy']
                parDom = parInd.info['sclDom']
                # parMolC = atoms2molcryst(parInd)
                parMolC = MolCryst(**parInd.info['molDict'])
                parMolC.set_cell(parInd.info['molCell'], scale_atoms=False)
                partition = parMolC.partition
                partLens = [len(p) for p in partition]
                if max(partLens) < 3:
                    break
                rotMolC = mol_rotation(parMolC)
                rotInd = rotMolC.to_atoms()

                # rotInd = merge_atoms(rotInd, self.dRatio)
                # toFrml = [int(i) for i in parInd.info['formula']]
                # rotInd = repair_atoms(rotInd, self.symbols, toFrml, parInd.info['numOfFormula'])

                rotInd.info['symbols'] = self.symbols
                rotInd.info['formula'] = parInd.info['formula']
                rotInd.info['parentE'] = parentE
                rotInd.info['parDom'] = parDom
                rotPop.append(rotInd)

        return rotPop





    def fit_gp(self):
        # fps, ens = read_dataset()
        fps = [ind.info['fingerprint'] for ind in self.curPop]
        fps = [fp.flatten().tolist() for fp in fps]
        ens = [ind.info['enthalpy'] for ind in self.curPop]

        fps = np.array(fps)
        ens = np.array(ens)
        # logging.debug("ens: {}".format(ens))

        maxDist = pdist(fps).max()
        logging.debug("maxDist: {}".format(maxDist))
        # minFps = fps.min(0)
        # maxFps = fps.max(0)

        scaled_factor = self.scaled_factor
        # linear_factor = 0.5
        # logging.debug(scaled_factor * (maxFps - minFps))
        # kernel = kernels.RBF(scaled_factor * (maxFps - minFps))
        kernel = kernels.RBF(length_scale=scaled_factor * maxDist, length_scale_bounds=(scaled_factor*maxDist, 50*maxDist))
        # kernel = kernels.Matern(scaled_factor * maxDist)
        # kernel = kernels.DotProduct()
        # kernel = kernels.RBF(length_scale=scaled_factor * maxDist, length_scale_bounds=(scaled_factor*maxDist, 50*maxDist)) \
        # kernel = kernels.RBF() \
        #        + linear_factor * kernels.DotProduct()

        gpParm = {
            'kernel': kernel,
            'alpha': 1e-5,
            'n_restarts_optimizer': 25,
            'normalize_y': True
        }

        gp = GP_fit(fps, ens, gpParm)
        logging.debug("kernel hyperparameter: {}".format(gp.kernel_.get_params()['length_scale']))
        self.gp = gp
        self.y_max = max(-1*ens)
        self.fps = fps

    def set_factor(self, factor):
        self.scaled_factor = factor

    def generate(self):
        if self.molDetector == 0:
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
        elif self.molDetector in [1,2]:
            hrdPop = self.heredity(self.cutNum, mode='mol')
            rotPop = self.mol_rotate(self.rotNum)
            permPop = self.permutate(self.permNum, mode='mol')
            latPop = self.latmutate(self.latNum, mode='mol')
            slipPop = self.slipmutate(self.slipNum, mode='mol')
            logging.debug("hrdPop length: %s"%(len(hrdPop)))
            logging.debug("rotPop length: %s"%(len(rotPop)))
            logging.debug("permPop length: %s"%(len(permPop)))
            logging.debug("latPop length: %s"%(len(latPop)))
            logging.debug("slipPop length: %s"%(len(slipPop)))
            tmpPop = hrdPop + rotPop + permPop + latPop + slipPop
            self.tmpPop.extend(tmpPop)
        else:
            raise RuntimeError('molDetector should be 0 or 1!')

    def add(self, pop):
        self.tmpPop.extend(pop)

    def select(self, enFilter=True):
        tmpPop = self.tmpPop
        symbols = self.symbols
        gp = self.gp
        y_max = self.y_max
        randFrac = self.randFrac
        popSize = self.popSize
        dRatio = self.dRatio

        if self.fullEles:
            tmpPop = list(filter(lambda x: 0 not in x.info['formula'], tmpPop))

        tmpPop = check_dist(tmpPop, dRatio)
        logging.info("tmpPop length: %s after check_dist"%(len(tmpPop)))
        if self.molDetector == 1 and self.chkMol:
            tmpPop = check_mol_pop(tmpPop, self.inputMols, self.bondRatio)
            logging.info("tmpPop length: %s after check_mol"%(len(tmpPop)))

        self.util = UtilityFunction(self.kind, self.kappa, self.xi)
        tmpPop = calc_all_fingerprints(tmpPop, self.parameters)
        for ind in tmpPop:
            ind.info['predictE'], ind.info['sigma'] = gp.predict(ind.info['fingerprint'].reshape(1, -1), return_std=True)
            ind.info['predictE'], ind.info['sigma'] = ind.info['predictE'][0], ind.info['sigma'][0]
            ind.info['enthalpy'] = ind.info['predictE']
            ind.info['utilVal'] = atoms_util(ind.info['fingerprint'] ,self.util, gp, symbols, y_max)[0]
            ind.info['minDist'] = cdist(ind.info['fingerprint'].reshape(1, -1), self.fps).min()
        if enFilter:
            tmpPop = list(filter(lambda x: abs(x.info['predictE'] - x.info['parentE']) > 0.01, tmpPop))
            logging.info("tmpPop length: %s after filter"%(len(tmpPop)))
        # tmpPop = del_duplicate(tmpPop, compareE=False)
        # logging.info("tmpPop length: %s after del_duplicate"%(len(tmpPop)))

        # nearPop = filter(lambda x: x.info['sigma'] < 1 , tmpPop)
        # logging.debug("nearPop length: {}".format(len(nearPop)))

        if self.calcType is 'fix':
            tmpPop = sorted(tmpPop, reverse=False, key=lambda ind: ind.info['utilVal'] + ind.info['parDom'] * self.parent_factor)
            # tmpPop = sorted(tmpPop, reverse=False, key=lambda ind: ind.info['utilVal'] + ind.info['parentE'] * self.parent_factor)
        elif self.calcType is 'var':
            tmpPop = convex_hull(self.curPop + tmpPop)
            tmpPop = sorted(tmpPop[self.curLen:], reverse=False, key=lambda ind: ind.info['ehull'] - self.kappa * ind.info['sigma'] + ind.info['parDom'] * self.parent_factor)
            # tmpPop = sorted(tmpPop[self.curLen:], reverse=False, key=lambda ind: ind.info['ehull'] - self.kappa * ind.info['sigma'] + ind.info['parentE'] * self.parent_factor)

        krigLen = int((1-randFrac) * popSize)
        if len(tmpPop) > 5 * popSize:
            tmpPop = tmpPop[:krigLen*5]

        # for ind in tmpPop:
        #     ind.info['enthalpy'] = gp.predict(ind.info['fingerprint'])[0]

        # tmpPop = del_duplicate(tmpPop, compareE=True)
        self.tmpPop = tmpPop

        nextPop = tmpPop[:]

        # krigLen = popSize # debug: save all structures

        if len(nextPop) > krigLen:
            self.nextPop = nextPop[:krigLen]
        else:
            self.nextPop = nextPop[:]

        for ind in nextPop:
        #    ind.info['predictE'] = gp.predict(ind.info['fingerprint'])[0]
            ind.info['enthalpy'] = None
            ind.info['ehull'] = None
        #    ind.info['utilVal'] = atoms_util(ind ,self.util, gp, symbols, y_max)[0]

#        ucb = [atoms_util(at ,self.util, gp, symbols, y_max) for at in nextPop]
#        logging.debug("%s"%(ucb))





class BBO:
    def __init__(self, curPop, parameters):
        self.curPop = curPop
        self.parameters = parameters
        for index, ind in enumerate(curPop):
            triInd = lower_triangullar_cell(ind)
            # logging.debug(triInd.get_cell())
            curPop[index] = triInd

        self.bboPop = list()
        bboParm = parameters['bboParm']
        self.migrateFrac = bboParm['migrateFrac']
        self.mutateFrac = bboParm['mutateFrac']
        self.randFrac = bboParm['randFrac']
        self.grids = bboParm['grids']
        self.grid = None

    def get_bboPop(self):
        return self.bboPop

    def set_grid(self, grid):
        self.grid = grid

    def bbo_cutcell(self):

        """
        BBO algorithm;
        cut the cell;
        fixed or variable composition
        """
        # global parameters
        formula = self.parameters['formula']
        symbols = self.parameters['symbols']
        popSize = self.parameters['popSize']
        calcType = self.parameters['calcType']
        saveGood = self.parameters['saveGood']
        dRatio = self.parameters['dRatio']
        migrateFrac = self.migrateFrac
        mutateFrac = self.mutateFrac
        randFrac = self.randFrac
        curPop = self.curPop[:]
        grid = self.grid
        if not grid:
            grid = random.choice(self.grids)

        oriLen = len(curPop) # origin length of curPop
        curPop = calc_dominators(curPop)
        curPop = sorted(curPop, key=lambda x:x.info['dominators'])
        if oriLen > int(round((1 - randFrac)*popSize)):
            curPop = curPop[:int(round((1 - randFrac)*popSize))]
        curLen = len(curPop)

        logging.debug("grid: %s"% grid)
        k1, k2, k3 = grid
        siv = [(x, y, z) for x in range(k1) for y in range(k2) for z in range(k3)]
        sivCut = list()
        for i in range(3):
            cutPos = [(x + 0.3*random.uniform(-1, 1))/grid[i] for x in range(grid[i])]
            cutPos.append(1)
            cutPos[0] = 0
            sivCut.append(cutPos)

        logging.debug('sivCut:')
        logging.debug(sivCut)


        numSiv = k1*k2*k3 # SIV: Suitability index variable
        bboPop = list()
        mutPop = list()

        sumRank = sum([ind.info['MOGArank'] for ind in curPop])
        fitAvg = sumRank/curLen
        fitMin = curPop[0].info['MOGArank']
        fitMax = curPop[-1].info['MOGArank']

        ### save SIV ###
        self.siv = siv
        self.sivCut = sivCut
        self.numSiv = numSiv

        ### goodPop ###
        curPop = standardize_pop(curPop, 1.)
        labels, goodPop = clustering(curPop, saveGood)
        curPop = list(filter(lambda ind: ind not in goodPop, curPop))
        curPop = goodPop + curPop
        for index, ind in enumerate(curPop):
            # logging.debug("index: %s, enthalpy: %s"%(index, ind.info['enthalpy']))
            # if index < saveGood:
            #     logging.debug("%s"%(ind.get_cell()))
            pass

        for index, ind in enumerate(curPop):
            # exAxis = random.sample(['x', 'y', 'z', '-x', '-y', '-z'], 2) # Rotate, exchange two axis
            # ind.rotate(exAxis[0], exAxis[1], rotate_cell=True)
            tranPos = ind.get_scaled_positions() # Displacement
            tranPos += np.array([[random.random(), random.random(), random.random()]]*len(ind))
            ind.set_scaled_positions(tranPos)
            ind.wrap()
            # u = ind.info['MOGArank']/(fitMax + 1)
            # l = 1 - u
            u = index/(curLen - 1)
            l = 1- u
            v = math.factorial(curLen - 1)/(math.factorial(curLen - 1 - index)*math.factorial(index))
            ind.info['u'] = u
            ind.info['l'] = l
            ind.info['v'] = v

        # KNN
        fingerprints = np.array([ind.info['fingerprint'] for ind in curPop])
        # logging.debug("fingerprints shape: %s"%fingerprints.shape)
        tree = BallTree(fingerprints, leaf_size=int(0.6*len(curPop)))

        for index, ind in enumerate(curPop):

            logging.info('\n')

            numIn = int(round(ind.info['u']*numSiv))
            if numIn == 0:
                numIn = 1

            # clustering
            # logging.debug('clustering')
            # iCluster = labels[index]
            # emRates = [tmpInd.info['l'] if labels[i] == iCluster else 0 for i, tmpInd in enumerate(curPop)]
            # logging.debug("emRates:%s"% emRates)

            # KNN
            # if index < saveGood:
            # logging.debug('KNN')
            knn = 1 + int(math.ceil(0.5*len(curPop)))
            neighs = tree.query(np.reshape(fingerprints[index], (1, -1)) , k=knn, return_distance=False)[0]
            # logging.debug("shape: %s" %(repr(neighs.shape)))
            # emRates = [tmpInd.info['l'] if i in neighs else 0 for i, tmpInd in enumerate(curPop)]
            # logging.debug("emRates:%s"% emRates)

            # else:
            #     dist = DistanceMetric.get_metric('euclidean')
            #     r = dist.pairwise([ind.info['fingerprint'], curPop[labels[index]].info['fingerprint']])[0][1]
            #     logging.debug("radius: %s"%(r))
            #     neighs = tree.query_radius(np.reshape(fingerprints[index], (1, -1)), r)[0]
            #     logging.debug("shape: %s" %(repr(neighs.shape)))


            # emRates[index] = 0
            # logging.debug("neighs: %s"%neighs)
            neighs = list(filter(lambda x: x != index, neighs))
            selectPop = [curPop[i] for i in neighs]

            testNum = 50
            for test in range(testNum):
                try:
                    bboInd = self.cutCell(ind, selectPop, numIn)
                    if check_dist_individual(bboInd, dRatio) and bboInd:
                        # logging.debug("parents: %s"%(bboInd.info['parents']))
                        bboPop.append(bboInd)
                        break
                except RuntimeError as e:
                    # logging.debug('no atoms in cell')
                    pass
                if test == testNum - 1:
                    logging.debug("Fail in migration after %s times"%(testNum))

        # bboPop = sorted(bboPop, key=lambda x:sum(x.info['parents']))
        if len(bboPop) > migrateFrac*popSize:
            bboPop = bboPop[:int(round(migrateFrac*popSize))]

        for i in range(int(round(mutateFrac*popSize))):
            mutIndex = random.choice(range(saveGood))
            if random.randint(0, 1):
                mutInd = gauss_mut(curPop[mutIndex], sigma=5, cellCut=10)
                logging.info('index:%s Mutation' % mutIndex)
            else:
                mutInd = exchage_atom(curPop[mutIndex])
                logging.info('index:%s Exchange' % mutIndex)
            mutInd.info['parentE'] = curPop[mutIndex].info['enthalpy']
            bboPop.append(mutInd)

        # mutRates = [ind.info['v'] for ind in curPop]
        # mutIndices= rou_select(mutRates, int(round(mutateFrac*popSize)))
        # for mutIndex in mutIndices:
        #     if random.randint(0, 1):
        #         mutInd = gauss_mut(curPop[mutIndex])
        #         logging.info('index:%s Mutation' % mutIndex)
        #     else:
        #         mutInd = exchage_atom(curPop[mutIndex])
        #         logging.info('index:%s Exchange' % mutIndex)
        #     mutInd.info['parentE'] = curPop[mutIndex].info['enthalpy']
        #     bboPop.append(mutInd)


        self.bboPop = bboPop

    def cutCell(self, curInd, selectPop, numIn):
        siv = self.siv
        sivCut = self.sivCut
        numSiv = self.numSiv
        formula = self.parameters['formula']
        symbols = self.parameters['symbols']
        calcType = self.parameters['calcType']
        invFrml = self.parameters['invFrml']
        minAt = self.parameters['minAt']
        maxAt = self.parameters['maxAt']

        selectRates = [tmpInd.info['l'] for tmpInd in selectPop]
        imIndex = rou_select(selectRates, numIn)
        imPop = [selectPop[i] for i in imIndex]


        for j in range(numSiv - numIn):
            imPop.append(curInd)
        random.shuffle(imPop)

        bboCell = np.zeros((3,3))
        bboVol = 0
        for otherInd in imPop:
            bboCell = bboCell + otherInd.get_cell()/numSiv
            bboVol = bboVol + otherInd.get_volume()/numSiv

        bboCellPar = cell_to_cellpar(bboCell)
        ratio = bboVol/abs(np.linalg.det(bboCell))
        if ratio > 1:
            bboCellPar[:3] = [length*ratio**(1/3) for length in bboCellPar[:3]]
            # logging.debug(bboCellPar)

        syblDict = {}
        for sybl in symbols:
            syblDict[sybl] = []

        for k in range(numSiv):
            imAts = [imAtom for imAtom in imPop[k]
                    if sivCut[0][siv[k][0]] <= imAtom.a < sivCut[0][siv[k][0] + 1]
                    if sivCut[1][siv[k][1]] <= imAtom.b < sivCut[1][siv[k][1] + 1]
                    if sivCut[2][siv[k][2]] <= imAtom.c < sivCut[2][siv[k][2] + 1]
                    ]

            for atom in imAts:
                for sybl in symbols:
                    if atom.symbol == sybl:
                        syblDict[sybl].append((atom.a, atom.b, atom.c))

        bboPos = []
        strFrml = ''
        for sybl in symbols:
            if len(syblDict[sybl]) > 0:
                bboPos.extend(syblDict[sybl])
                strFrml = strFrml + sybl + str(len(syblDict[sybl]))
        if strFrml == '':
            raise RuntimeError('No atoms in the new cell')

        bboInd = Atoms(strFrml,
            cell = bboCellPar,
            pbc = True,)
        bboInd.info['Origin'] = 'Migrate'

        # random.shuffle(bboPos) #exchage atoms in bboPos
        bboInd.set_scaled_positions(bboPos)

            # logging.debug(bboInd.get_chemical_formula())

        # bboInd.info['parents'] = imIndex
        if calcType == 'fix':
            nfm = sum([ats.info['numOfFormula'] for ats in imPop])
            nfm = int(round(nfm/numSiv))

            # logging.debug("Repair bboInd.")
            bboInd = repair_atoms(bboInd, symbols, formula, nfm)
            bboInd.info['symbols'] = symbols
            bboInd.info['formula'] = formula
            bboInd.info['numOfFormula'] = nfm

            parentE = sum([ats.info['enthalpy'] for ats in imPop])/numSiv
            bboInd.info['parentE'] = parentE

        elif calcType == 'var':
            npFrml = np.array(formula)
            frml = np.array(get_formula(bboInd, symbols))
            coef = np.ceil(np.dot(np.dot(frml, npFrml.T), invFrml))
            tFrml = np.dot(coef, npFrml)
            tFrml = [int(i) for i in tFrml]
            logging.debug('tFrml: {}'.format(tFrml))
            if minAt <= sum(tFrml) <= maxAt:
                bboInd = repair_atoms(bboInd, symbols, tFrml)
                bboInd.info['numOfFormula'] = 1
                bboInd.info['symbols'] = symbols
                bboInd.info['formula'] = tFrml
                parentE = sum([ats.info['enthalpy'] for ats in imPop])/numSiv
                bboInd.info['parentE'] = parentE
            else:
                bboInd = None

        return bboInd

    def check_cut(self, atom):
        """
        check whether the atom in the subcell with the indices
        """
        pass

    def bbo_old(self):
        # global parameters
        formula = self.parameters['formula']
        symbols = self.parameters['symbols']
        popSize = self.parameters['popSize']
        calcType = self.parameters['calcType']
        saveGood = self.parameters['saveGood']
        dRatio = self.parameters['dRatio']
        migrateFrac = self.migrateFrac
        mutateFrac = self.mutateFrac
        randFrac = self.randFrac
        curPop = self.curPop[:]
        oriLen = len(curPop) # origin length of curPop
        curPop = calc_dominators(curPop)
        curPop = sorted(curPop, key=lambda x:x.info['dominators'])
        if oriLen > int(round((1 - randFrac)*popSize)):
            curPop = curPop[:int(round((1 - randFrac)*popSize))]
        curLen = len(curPop)

        numSiv = len(curPop[0])
        bboPop = list()
        mutPop = list()

        sumRank = sum([ind.info['MOGArank'] for ind in curPop])
        fitAvg = sumRank/curLen
        fitMin = curPop[0].info['MOGArank']
        fitMax = curPop[-1].info['MOGArank']

        for index, ind in enumerate(curPop):
            tranPos = ind.get_scaled_positions() # Displacement
            tranPos += np.array([[random.random(), random.random(), random.random()]]*len(ind))
            ind.set_scaled_positions(tranPos)
            ind.wrap()
            # u = ind.info['MOGArank']/(fitMax + 1)
            # l = 1 - u
            u = index/(curLen - 1)
            l = 1- u
            v = math.factorial(curLen - 1)/(math.factorial(curLen - 1 - index)*math.factorial(index))
            ind.info['u'] = u
            ind.info['l'] = l
            ind.info['v'] = v

        for index, ind in enumerate(curPop):

            logging.info('\n')

            numIn = int(round(ind.info['u']*numSiv))
            if numIn == 0:
                numIn = 1


            emRates = [tmpInd.info['l'] for tmpInd in curPop]
            emRates[index] = 0
          #  emRates = emRates[:saveGood]
            imIndex = rou_select(emRates, numIn)
            logging.info("imIndex: {}".format(imIndex))
            imPop = [curPop[i] for i in imIndex] # immigration pop
            logging.info('index: %s,numSiv: %s, numIn: %s, u: %s, l: %s'
                %(index, numSiv, numIn, ind.info['u'], ind.info['l']))

            for j in range(numSiv - numIn):
                imPop.append(ind)
            random.shuffle(imPop)
            logging.info([curPop.index(ats) for ats in imPop])

            bboCell = np.zeros((3,3))
            bboVol = 0
            for otherInd in imPop:
                bboCell = bboCell + otherInd.get_cell()/numSiv
                bboVol = bboVol + otherInd.get_volume()/numSiv

            # logging.info("parents cell:")
            # for otherInd in imPop:
            #     logging.info(otherInd.get_cell())
            # logging.info("bbo cell:")
            # logging.info(bboCell)
            # logging.info('\n')

            # logging.info('BBO Cell Parameter:\n')
            bboCellPar = cell_to_cellpar(bboCell)
            # logging.info(bboCellPar)
            ratio = bboVol/abs(np.linalg.det(bboCell))
            if ratio > 1:
                bboCellPar[:3] = [length*ratio**(1/3) for length in bboCellPar[:3]]
            # logging.info(bboCellPar)

            syblDict = {}
            for sybl in symbols:
                syblDict[sybl] = []

            bboPos = []
            for k, imInd in enumerate(imPop):
                imPos = imInd.get_scaled_positions()
                bboPos.append(imPos[k])
            bboPos = np.array(bboPos)

            bboInd = ind.copy()
            bboInd.set_cell(bboCellPar)
            # random.shuffle(bboPos) #exchage atoms in bboPos
            bboInd.set_scaled_positions(bboPos)
            bboInd.info['Origin'] = 'Migrate'

            logging.info(bboInd.get_chemical_formula())

            ase.io.write('bbo%s.vasp'% (index), bboInd, direct=True, vasp5=True)

            if not check_dist_individual(bboInd, dRatio):
                bboInd = gauss_mut(bboInd)
                logging.info("Migrate fail, convert to Mutate")

            bboPop.append(bboInd)

        if len(bboPop) > migrateFrac*popSize:
            bboPop = bboPop[:int(round(migrateFrac*popSize))]

        for i in range(int(round(mutateFrac*popSize))):
            mutIndex = random.choice(range(saveGood))
            if random.randint(0, 1):
                mutInd = gauss_mut(curPop[mutIndex])
                logging.info('index:%s Mutation' % mutIndex)
            else:
                mutInd = exchage_atom(curPop[mutIndex])
                logging.info('index:%s Exchange' % mutIndex)
            mutInd.info['parentE'] = curPop[mutIndex].info['enthalpy']
            bboPop.append(mutInd)

        self.bboPop = bboPop
        # mutRates = [ind.info['v'] for ind in curPop]
        # mutIndices= rou_select(mutRates, int(round(mutateFrac*popSize)))
        # for mutIndex in mutIndices:
        #     if random.randint(0, 1):
        #         mutInd = gauss_mut(curPop[mutIndex])
        #         logging.info('index:%s Mutation' % mutIndex)
        #     else:
        #         mutInd = exchage_atom(curPop[mutIndex])
        #         logging.info('index:%s Exchange' % mutIndex)
        #     mutInd.info['parentE'] = curPop[mutIndex].info['enthalpy']
        #     bboPop.append(mutInd)








def pareto_front(Pop, e=0):    # e: epsilon-dominance
    paretoPop = []

    for ind in Pop[:]:
        ftn1 = ind.info['fitness1']
        ftn2 = ind.info['fitness2']
        addOrNot = True

        for otherInd in Pop[:]:
            # if ind != otherInd:
            #     addOrNot = (ftn1 < otherInd.info['fitness1'] or ftn2 < otherInd.info['fitness2']) and addOrNot
            addOrNot = (addOrNot and
                        ((ftn1/(1+e) < otherInd.info['fitness1'] or ftn2/(1+e) < otherInd.info['fitness2'])
                        or (ftn1 == otherInd.info['fitness1'] and ftn2 == otherInd.info['fitness2']))
                        )

        if addOrNot:
            paretoPop.append(ind.copy())

    return paretoPop

def pareto_front_old(Pop):
    popFtn1 = sorted(Pop, key = lambda struct: (struct.info['fitness1'], struct.info['fitness2']))
    indFtn1 = popFtn1[0]
    popFtn2 = sorted(Pop, key = lambda struct: (struct.info['fitness2'], struct.info['fitness1']))
    indFtn2 = popFtn2[0]

    popFtn1 = popFtn1[:popFtn1.index(indFtn2)]
    popFtn2 = popFtn2[:popFtn2.index(indFtn1)]
    # list(set(enth_Pop).intersection(set(gap_Pop)))
    paretoPop = [indFtn1, indFtn2]
    cutFtn2 = indFtn1.info['fitness2']
    for ind in popFtn1[1:]:
        if ind.info['fitness2'] < cutFtn2:
            paretoPop.append(ind)
            gapCut = ind.info['fitness2']

    return paretoPop

def gauss_mut(parInd, sigma=0.5, cellCut=1, distCut=1):
    """
    sigma: Gauss distribution standard deviation
    distCut: coefficient of gauss distribution in atom mutation
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
        atGauss = np.array([random.gauss(0, sigma)*distCut for i in range(3)])
        at.position += atGauss*covalent_radii[atomic_numbers[at.symbol]]

    chdInd.wrap()
    chdInd.info = parInd.info.copy()
    # chdInd.info['Origin'] = 'Mutate'

    return chdInd



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

def crossover(parInd, bestInd, gamma = 0.2):
    chdInd = parInd.copy()

    parpos, bestpos = [ ind.get_scaled_positions() for ind in (parInd, bestInd)]
    chdpos = gamma*bestpos + (1- gamma)*parpos
    chdInd.set_scaled_positions(chdpos)

    parcell, bestcell = [ ind.get_cell() for ind in (parInd, bestInd)]
    chdcell = gamma*bestcell + (1- gamma)*parcell
    chdInd.set_cell(chdcell, scale_atoms=True)

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
    # logging.info("survival: %s Individual" %len(dupPop))
    return dupPop

def calc_dominators(Pop):

    # domPop = Pop[:]
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

def rou_select(rateList, selectNum=1):
    """Roulette selection, return a list of the selected index"""
    sumRate = sum(rateList)
    indices = []
    for i in range(selectNum):
        pick = random.uniform(0, sumRate)
        tmpSum = 0
        for index, val in enumerate(rateList):
            tmpSum += val
            if tmpSum >= pick:
                indices.append(index)
                break
    # logging.info("Error: cannot select an index by Roulette")
    # return None
    return indices

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


def check_dist(pop, threshold=0.7):
    checkPop = []
    for ind in pop:
    #    ase.io.write('checking.vasp', ind, format='vasp', direct=True, vasp5=True)
        if check_dist_individual(ind, threshold):
            checkPop.append(ind)

    return checkPop

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
                logging.debug("coef: {}".format(coef))
                logging.debug("numMols: {}".format(tryMolc.numMols))
                molC = tryMolc
                maxMolNum = tryMolc.numMols
        oriVol = ind.get_volume()
        oriCell = ind.get_cell()
        radius = np.array(molC.get_radius())
        # eleRad = covalent_radii[max(ind.get_atomic_numbers())]
        # radius += eleRad
        molVol = 4/3 * pi * np.power(radius, 3).sum()
        logging.debug("partition {}".format(molC.partition))
        logging.debug("oriVol: {}\tmolVol: {}".format(oriVol, molVol))
        if scale_cell and molVol > oriVol:
            tmpVol = 0.5*(molVol + oriVol)
            ratio = float((tmpVol/oriVol)**(1./3))
            molCell = oriCell*ratio
        else:
            molCell = oriCell
        ind.info['molDict'] = molC.to_dict()
        ind.info['molCell'] = molCell

    return molPop


def mol_gauss_mut(parInd, sigma=0.5, cellCut=1, distCut=1):
    """
    Gaussian mutation for molecule crystal.
    parInd should be a MolCryst object.
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
    chdCell = cellpar_to_cell(cellPar)
    chdInd.set_cell(chdCell, scale_atoms=False, scale_centers=True)


    atGauss = np.array([random.gauss(0, sigma)*distCut for i in range(3)])
    chdCenters = parInd.centers + atGauss
    chdInd.update_centers_and_rltPos(centers=chdCenters)
        # at.position += atGauss*covalent_radii[atomic_numbers[at.symbol]]

    chdInd.info = parInd.info.copy()
    # chdInd.info['Origin'] = 'Mutate'

    return chdInd

def mol_rotation(parInd):


    chdInd = parInd.copy()

    chdRltPos = []

    for pos in chdInd.rltPos:
        newPos = np.dot(pos, rand_rotMat())
        chdRltPos.append(newPos)

    chdInd.update_centers_and_rltPos(rltPos=chdRltPos)
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
            if 0.5*n < sclCenters[i, axis] < 0.5*(n+1):
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

def tournament(pop, num):
    smpPop = random.sample(pop, num)
    ind = sorted(smpPop, key=lambda x:x.info['dominators'])[0]
    return ind
