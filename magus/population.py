import numpy as np
import os
from scipy.spatial.distance import cdist, pdist
import itertools
import ase.io
from ase.data import covalent_radii,atomic_numbers
from ase.neighborlist import neighbor_list
from ase.atom import Atom
from .utils import *
import logging
from sklearn import cluster
from .descriptor import ZernikeFp
import copy
from .setfitness import set_fit_calcs

def set_ind(parameters):
    if parameters.calcType == 'fix':
        return FixInd(parameters)
    if parameters.calcType == 'var':
        return VarInd(parameters)

class Population:
    """
    a class of atoms population
    TODO __iter__
    """
    def __init__(self,parameters):
        self.parameters = parameters
        self.Individual = set_ind(parameters)
        self.fit_calcs = set_fit_calcs(parameters)

    def __call__(self,pop,name='temp',gen=None):
        newpop = self.__class__(self.parameters)
        pop = [self.Individual(ind) if ind.__class__.__name__ == 'Atoms' else ind for ind in pop]
        newpop.pop = pop
        newpop.name = name
        newpop.gen = gen
        for i,ind in enumerate(pop):
            ind.info['identity'] = [name, i]
        return newpop

    def __len__(self):
        return len(self.pop)

    def __add__(self,other):
        newPop = self.copy()
        for ind in other.pop:
            newPop.append(ind)
        return newPop

    def append(self,ind):
        if ind.__class__.__name__ == 'Atoms':
            ind = self.Individual(ind)
        for ind_ in self.pop:
            if ind == ind_:
                return False
        else:
            self.pop.append(ind)
            return True
    
    def extend(self,pop):
        for ind in pop:
            self.append(ind)

    def copy(self):
        newpop = [ind.copy() for ind in self.pop]
        return self(newpop,name=self.name,gen=self.gen)

    def save(self,filename=None,gen=None,savedir=None):
        filename = self.name if filename is None else filename
        gen = self.gen if gen is None else gen
        savedir = self.parameters.resultsDir if savedir is None else savedir
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        pop = []
        for ind in self.pop:
            atoms = ind.atoms.copy()
            pop.append(atoms)
        ase.io.write("{}/{}{}.traj".format(savedir,filename,gen),pop,format='traj')
        logging.debug("save {}{}.traj".format(filename,gen))

    @property
    def frames(self):
        pop = []
        for ind in self.pop:
            atoms = ind.atoms.copy()
            pop.append(atoms)
        return pop

    def calc_dominators(self):
        self.calc_fitness()
        domLen = len(self.pop)
        for ind1 in self.pop:
            dominators = -1 #number of individuals that dominate the current ind
            for ind2 in self.pop:
                for key in ind1.info['fitness']:
                    if ind1.info['fitness'][key] > ind2.info['fitness'][key]:
                        break
                else:
                    dominators += 1

            ind1.info['dominators'] = dominators
            ind1.info['MOGArank'] = dominators + 1
            ind1.info['sclDom'] = (dominators)/domLen

    def calc_fitness(self):
        for fit_calc in self.fit_calcs:
            fit_calc(self.pop)

    def del_duplicate(self):
        logging.info('del_duplicate {} begin'.format(self.name))
        newpop = []
        for ind1 in self.pop:
            for ind2 in newpop:
                if ind1==ind2:
                    break
            else:
                newpop.append(ind1)
        logging.info('del_duplicate {} finish'.format(self.name))
        self.pop = newpop

    def check(self):
        logging.info("check distance")
        checkpop = []
        for ind in self.pop:
            if ind.check():
                checkpop.append(ind)
        logging.info("check survival: {}".format(len(checkpop)))
        self.pop = checkpop

    def symmetrize_pop(self):
        pass

    def clustering(self, numClusters):
        """
        clustering by fingerprints
        TODO may not be a class method
        """
        pop = [ind.copy() for ind in self.pop]
        if numClusters >= len(pop):
            return np.arange(len(pop)),pop

        fpMat = np.array([ind.fingerprint for ind in pop])
        labels = cluster.KMeans(n_clusters=numClusters).fit_predict(fpMat)
        goodpop = [None]*numClusters
        for label, ind in zip(labels, pop):
            curBest = goodpop[label]
            if curBest:
                if ind.info['dominators'] < curBest.info['dominators']:
                    goodpop[label] = ind
            else:
                goodpop[label] = ind
        return labels, goodpop

    def get_volRatio(self):
        volRatios = [ind.get_volRatio() for ind in self.pop]
        return np.mean(volRatios)

    def select(self,n):
        if len(self) > n:
            self.pop = sorted(self.pop, key=lambda x:x.info['dominators'])[:n]
        
class Individual:
    def __init__(self,parameters):
        self.parameters = parameters
        Requirement=['formula','symbols']
        Default={'repairtryNum':10}
        checkParameters(self,parameters,Requirement,Default)

        #TODO add more comparators
        from ase.ga.standard_comparators import AtomsComparator
        self.comparator = AtomsComparator()

        #fingerprint
        cutoff = self.parameters.cutoff
        nmax = self.parameters.ZernikeNmax
        lmax = self.parameters.ZernikeLmax
        ncut = self.parameters.ZernikeNcut
        diag = self.parameters.ZernikeDiag
        elems = [atomic_numbers[element] for element in parameters.symbols]
        self.cf = ZernikeFp(cutoff, nmax, lmax, ncut, elems,diag=diag)

    def __eq__(self, obj):
        return self.comparator.looks_like(self.atoms,obj.atoms)

    def copy(self):
        newind = self.__class__(self.parameters)
        newind.atoms = self.atoms.copy()
        newind.info = copy.deepcopy(self.info)
        return newind

    def save(self, filename):
        if self.atoms:
            atoms = self.atoms.copy()
            atoms.set_calculator(None)
            atoms.info = self.info
            ase.io.write(filename,atoms)
        else:
            logging.debug('None')

    @property
    def fingerprint(self):
        if 'fingerprint' not in self.info:
            Efps = self.cf.get_all_fingerprints(self.atoms)[0]
            self.info['fingerprint'] = np.mean(Efps,axis=0)
        return self.info['fingerprint']

    def get_ball_volume(self):
        ballVol = 0
        for num in self.atoms.get_atomic_numbers():
            ballVol += 4*np.pi/3*(covalent_radii[num])**3
        self.ball_volume = ballVol
        return ballVol

    def get_volRatio(self):
        self.volRatio = self.atoms.get_volume()/self.get_ball_volume()
        return self.volRatio
        
    def check_cellpar(self,atoms=None):
        """
        check if cellpar reasonable
        TODO bond
        """
        return True
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()

        minLen = self.parameters.minLen
        maxLen = self.parameters.maxLen
        cellPar = a.get_cell_lengths_and_angles()
        if (minLen < cellPar).all() and (cellPar < maxLen).all():
            return True
        else:
            return False

    def check_distance(self,atoms=None):
        """
        The distance between the atoms should be larger than
        threshold * sumR(the sum of the covalent radii of the two
        corresponding atoms).
        """
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()

        threshold = self.parameters.dRatio  
        cell = a.get_cell()
        nums = a.get_atomic_numbers()
        unique_types = sorted(list(set(nums)))
        pos = a.get_positions()

        for nx, ny, nz in itertools.product([-1,0,1],repeat=3):
            displacement = np.array([nx, ny, nz])@cell
            pos_new = pos + displacement
            distances = cdist(pos, pos_new)

            if nx == 0 and ny == 0 and nz == 0:
                distances += 1e2 * np.identity(len(a))

            iterator = itertools.combinations_with_replacement(unique_types, 2)
            for type1, type2 in iterator:
                x1 = np.where(nums == type1)
                x2 = np.where(nums == type2)
                if np.min(distances[x1].T[x2]) < (covalent_radii[type1]+covalent_radii[type2])*threshold:
                    return False
        return True

    def check(self,atoms=None):
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        return self.check_cellpar(a) and self.check_distance(a)

    def sort(self):
        indices = []
        for s in self.symbols:
            indices.extend([i for i,atom in enumerate(self.atoms) if atom.symbol==s])
        self.atoms = self.atoms[indices]

    def merge_atoms(self, tolerance=0.3,):
        """
        TODO threshold
        if a pair of atoms are too close, merge them.
        """
        atoms = self.atoms
        cutoffs = [tolerance * covalent_radii[num] for num in atoms.get_atomic_numbers()]
        nl = neighbor_list("ij", atoms, cutoffs)
        indices = list(range(len(atoms)))
        exclude = []

        for i, j in zip(*nl):
            if i not in exclude and not j in exclude:
                exclude.append(np.random.choice([i,j]))

        if len(exclude) > 0:
            save = [index for index in indices if index not in exclude]
            mAts = atoms[save]
            mAts.info = atoms.info.copy()
        else:
            mAts = atoms
        self.atoms = mAts

    def repair_atoms(self):
        """
        sybls: a list of symbols
        toFrml: a list of formula after repair
        """
        if not self.needrepair:
            self.sort()
            return True
        atoms = self.atoms
        dRatio = self.parameters.dRatio
        syms = atoms.get_chemical_symbols()
        #nowFrml = Counter(atoms.get_chemical_symbols())
        targetFrml = self.get_targetFrml()
        if not targetFrml:
            self.atoms = None
            return False
        toadd, toremove = {} , {}
        for s in targetFrml:
            if syms.count(s) < targetFrml[s]:
                toadd[s] = targetFrml[s] - syms.count(s)
            elif syms.count(s) > targetFrml[s]:
                toremove[s] = syms.count(s) - targetFrml[s]
        repatoms = atoms.copy()
        #remove before add
        while toremove:
            del_symbol = np.random.choice(list(toremove.keys()))
            del_index = np.random.choice([atom.index for atom in repatoms if atom.symbol==del_symbol])
            if toadd:
                #if some symbols need to add, change symbol directly
                add_symbol = np.random.choice(list(toadd.keys()))
                repatoms[del_index].symbol = add_symbol
                toadd[add_symbol] -= 1
                if toadd[add_symbol] == 0:
                    toadd.pop(add_symbol)
            else:
                del repatoms[del_index]
            toremove[del_symbol] -= 1
            if toremove[del_symbol] == 0:
                toremove.pop(del_symbol)
        
        while toadd:
            add_symbol = np.random.choice(list(toadd.keys()))
            for _ in range(self.repairtryNum):
                # select a center atoms
                centerAt = repatoms[np.random.randint(0,len(repatoms)-1)]
                basicR = covalent_radii[centerAt.number] + covalent_radii[atomic_numbers[s]]
                # random position in spherical coordination
                radius = basicR * (dRatio + np.random.uniform(0,0.3))
                theta = np.random.uniform(0,np.pi)
                phi = np.random.uniform(0,2*np.pi)
                pos = centerAt.position + radius*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),np.cos(theta)])
                if check_new_atom_dist(repatoms, pos, add_symbol, dRatio):
                    addAt = Atom(symbol=add_symbol, position=pos)
                    repatoms.append(addAt)
                    break
            else:
                self.atoms = None
                return False
                
        self.sort()
        return True

class FixInd(Individual):
    def __call__(self,atoms):
        newind = self.__class__(self.parameters)
        newind.atoms = atoms
        newind.sort()
        newind.info = {'numOfFormula':int(round(len(atoms)/sum(self.formula)))}
        newind.info['fitness'] = {}
        return newind

    def needrepair(self):
        #check if atoms need repair
        Natoms = len(self.atoms)
        if Natoms < self.minAt or Natoms > self.maxAt:
            return False

        symbols = self.atoms.get_chemical_symbols()
        formula = np.array([symbols.count(s) for s in self.symbols])
        numFrml = int(round(Natoms/sum(self.formula)))
        targetFrml = numFrml*np.array(self.formula)
        return np.all(targetFrml == formula)

    def get_targetFrml(self):
        #TODO initial self.formula
        atoms = self.atoms
        Natoms = len(atoms)
        if self.parameters.minAt <= Natoms <= self.parameters.maxAt :
            numFrml = int(round(Natoms/sum(self.formula)))
        else:
            numFrml = int(round(0.5 * sum([ind.info['numOfFormula'] for ind in self.parents])))
        self.info['formula'] = self.formula
        self.info['numOfFormula'] = numFrml
        targetFrml = {s:numFrml*i for s,i in zip(self.symbols,self.formula)}
        return targetFrml

class VarInd(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.rank = np.linalg.matrix_rank(self.formula)
        self.invF = np.linalg.pinv(self.formula)
        checkParameters(self,parameters,[],{'fullEles':False})

    def __call__(self,atoms):
        newind = self.__class__(self.parameters)
        newind.atoms = atoms
        newind.info = {'numOfFormula':1}
        newind.info['fitness'] = {}
        return newind

    def needrepair(self):
        #check if atoms need repair
        Natoms = len(self.atoms)
        if Natoms < self.minAt or Natoms > self.maxAt:
            return False
        symbols = self.atoms.get_chemical_symbols()
        formula = [symbols.count(s) for s in self.symbols]
        rank = np.linalg.matrix_rank(np.concatenate((self.formula, [formula])))
        return rank == self.rank
    
    def get_targetFrml(self):
        #TODO initial self.formula
        #TODO var
        symbols = self.atoms.get_chemical_symbols()
        formula = [symbols.count(s) for s in self.symbols]
        coef = np.rint(np.dot(formula, self.invF)).astype(np.int)
        newFrml = np.dot(coef, self.formula).astype(np.int)
        bestFrml = newFrml.tolist()
        if self.parameters.minAt <= sum(bestFrml) <= self.parameters.maxAt:
            targetFrml = {s:i for s,i in zip(self.symbols,bestFrml)}
        else:
            targetFrml = None
        return targetFrml

    def check_full(self,atoms=None):
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        symbols = a.get_chemical_symbols()
        formula = [symbols.count(s) for s in self.symbols]
        return not self.fullEles or 0 not in formula 

    def check(self, atoms=None):
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        return super().check(atoms=a) and self.check_full(atoms=a)