import numpy as np
import os,re,itertools,random
from collections import Counter
from scipy.spatial.distance import cdist, pdist
import ase.io
from ase import Atoms
from ase.data import covalent_radii,atomic_numbers
from ase.neighborlist import neighbor_list
from ase.atom import Atom
import spglib
from .utils import *
import logging
from sklearn import cluster
from .descriptor import ZernikeFp
import copy
from .molecule import Molfilter
#TODO
# check seed?

log = logging.getLogger(__name__)


def set_ind(parameters):
    if parameters.calcType == 'fix':
        return FixInd(parameters)
    if parameters.calcType == 'var':
        return VarInd(parameters)

class Population:
    """
    a class of atoms population
    """
    def __init__(self,parameters):
        self.p = EmptyClass()
        Requirement=['resultsDir','calcType','symbols']
        Default={'chkSeed': False}
        checkParameters(self.p,parameters,Requirement,Default)
        self.Individual = set_ind(parameters)

    def __iter__(self):
        for i in self.pop:
            yield i

    def __getitem__(self,i):
        return self.pop[i]

    def __call__(self,pop,name='temp',gen=None):
        newPop = self.__new__(self.__class__)
        newPop.p = self.p
        pop = [self.Individual(ind) if ind.__class__.__name__ == 'Atoms' else ind for ind in pop]
        newPop.Individual = self.Individual
        newPop.fit_calcs = self.fit_calcs
        newPop.pop = pop
        log.debug('generate:Pop {} with {} ind'.format(name,len(pop)))
        newPop.name = name
        newPop.gen = gen
        for i,ind in enumerate(pop):
            ind.info['identity'] = [name, i]
            ind.Pop = self
        return newPop

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
        ind.info['identity'] = [self.name, len(self.pop)]
        self.pop.append(ind)
        return True
        #for ind_ in self.pop:
        #    if ind == ind_:
        #        return False
        #else:
        #    self.pop.append(ind)
        #    return True

    def extend(self, pop):
        for ind in pop:
            self.append(ind)

    def copy(self):
        newpop = [ind.copy() for ind in self.pop]
        return self(newpop,name=self.name,gen=self.gen)

    def save(self,filename=None,gen=None,savedir=None):
        filename = self.name if filename is None else filename
        gen = self.gen if gen is None else gen
        savedir = self.p.resultsDir if savedir is None else savedir
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        pop = []
        for ind in self.pop:
            atoms = ind.to_save()
            pop.append(atoms)
        ase.io.write("{}/{}{}.traj".format(savedir,filename,gen),pop,format='traj')
        log.debug("save {}{}.traj".format(filename,gen))

    @property
    def frames(self):
        pop = []
        for ind in self.pop:
            atoms = ind.atoms.copy()
            pop.append(atoms)
        return pop

    @property
    def all_frames(self):
        pop = []
        for ind in self.pop:
            atoms = ind.atoms.copy()
            if 'trajs' in atoms.info:
                traj = atoms.info['trajs'][-1]
                for frame in traj:
                    pop.append(read_atDict(frame))
            else:
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
            fit_calc.calc(self)

    def del_duplicate(self):
        log.info('del_duplicate {} begin, popsize:{}'.format(self.name,len(self.pop)))
        newpop = []
        # sort the pop before deleting duplicates
        self.pop = sorted(self.pop, key=lambda x:x.info['enthalpy'] if 'enthalpy' in x.info else 100)
        for ind1 in self.pop:
            #for ind2 in newpop:
            # compare enthalpies, save the ind with lowest enthalpy
            newLen = len(newpop)
            for n in range(newLen):
                ind2 = newpop[n]
                if ind1 == ind2:
                    if 'enthalpy' in ind1.info and 'enthalpy' in ind2.info:
                        if ind1.info['enthalpy'] < ind2.info['enthalpy']:
                            newpop[n] = ind1
                    break
            else:
                newpop.append(ind1)
        log.info('del_duplicate survival: {}'.format(len(newpop)))
        self.pop = newpop

    def check(self):
        log.info("check population {}, popsize:{}".format(self.name,len(self.pop)))
        checkpop = []
        for ind in self.pop:
            #log.debug("checking {}".format(ind.info['identity']))
            if not ind.need_check() or ind.check():
                checkpop.append(ind)
        log.info("check survival: {}".format(len(checkpop)))
        self.pop = checkpop

    def check_full(self):
        log.info("check_full population {}, popsize:{}".format(self.name,len(self.pop)))
        checkpop = []
        for ind in self.pop:
            if ind.check_full():
                checkpop.append(ind)
        log.info("check_full survival: {}".format(len(checkpop)))
        self.pop = checkpop

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

    def find_spg(self):
        for ind in self.pop:
            ind.find_spg()

    def add_symmetry(self):
        for ind in self.pop:
            ind.add_symmetry()

    def select(self,n):
        # self.calc_dominators()
        self.pop = sorted(self.pop, key=lambda x:x.info['dominators'])
        if len(self) > n:
            self.pop = self.pop[:n]

    def bestind(self):
        # self.calc_dominators()
        dominators = np.array([ind.info['dominators'] for ind in self.pop])
        best_i = np.where(dominators == np.min(dominators))[0]
        bestInds = [self.pop[i] for i in best_i]
        # Write generation of bestind
        for ind in bestInds:
            ind.info['gen'] = self.gen
        return  bestInds
        #return [self.pop[i] for i in best_i]

class Individual:
    def __init__(self,parameters):
        self.p = EmptyClass()
        Requirement=['formula','symbols','minAt','maxAt','symprec','molDetector','bondRatio','dRatio','diffE','diffV']
        Default={'repairtryNum':3, 'molMode':False, 'chkMol':False,
                 'minLattice':None, 'maxLattice':None, 'dRatio':0.7,
                 'addSym':False, 'chkSeed': True}
        checkParameters(self.p,parameters,Requirement,Default)
        # if self.p.addSym:
        #     checkParameters(self.p,parameters,[],{'symprec':0.01})

        #TODO add more comparators
        from .comparator import FingerprintComparator, Comparator
        from .compare.naive import NaiveComparator
        from .compare.bruteforce import ZurekComparator
        from .compare.base import OrGate, AndGate
        #self.comparator = FingerprintComparator()
        #self.comparator = Comparator(dE=self.p.diffE, dV=self.p.diffV)
        self.comparator = OrGate([NaiveComparator(dE=self.p.diffE, dV=self.p.diffV), ZurekComparator()])
        #fingerprint
        self.cf = ZernikeFp(parameters)

        if self.p.molMode:
            assert hasattr(parameters,'molList')
            self.inputMols = [Atoms(**molInfo) for molInfo in parameters.molList]
            self.molCounters = [Counter(inMol.get_chemical_symbols()) for inMol in self.inputMols]

        else:
            self.inputMols = []
            self.molCounters = []
        #Sometimes self.inputFormulas are wrong, so I use Counter instead. -hgao
        self.inputFormulas = []
        for mol in self.inputMols:
            s = []
            symbols = mol.get_chemical_symbols()
            unique_symbols = sorted(np.unique(self.p.symbols))
            for symbol in unique_symbols:
                s.append(symbol)
                n = self.p.symbols.count(symbol)
                if n > 1:
                    s.append(str(n))
            s = ''.join(s)
            self.inputFormulas.append(s)

        #log.debug('self.inputFormulas: {}'.format(self.inputFormulas))

    def __eq__(self, obj):
        return self.comparator.looks_like(self, obj)
        # return self.comparator.looks_like(self.atoms, obj.atoms)

    def copy(self):
        atoms = self.atoms.copy()
        newind = self(atoms)
        newind.info = copy.deepcopy(self.info)
        return newind

    def to_save(self):
        atoms = self.atoms.copy()
        atoms.set_calculator(None)
        # atoms.info = self.info
        for key, val in self.info.items():
            atoms.info[key] = val
        return atoms

    def fix_atoms_info(self):
        for key, val in self.info.items():
            self.atoms.info[key] = val

    @property
    def fingerprint(self):
        if 'fingerprint' not in self.info:
            Efps = self.cf.get_all_fingerprints(self.atoms)[0]
            self.info['fingerprint'] = Efps
        return self.info['fingerprint']

    def find_spg(self):
        atoms = self.atoms
        symprec = self.p.symprec
        spg = spglib.get_spacegroup(atoms, symprec)
        pattern = re.compile(r'\(.*\)')
        try:
            spg = pattern.search(spg).group()
            spg = int(spg[1:-1])
        except:
            spg = 1
        atoms.info['spg'] = spg
        priCell = spglib.find_primitive(atoms, symprec=symprec)
        if priCell:
            lattice, pos, numbers = priCell
            atoms.info['priNum'] = numbers
            atoms.info['priVol'] = abs(np.linalg.det(lattice))
        else:
            atoms.info['priNum'] = atoms.get_atomic_numbers()
            atoms.info['priVol'] = atoms.get_volume()
        self.atoms = atoms

    def add_symmetry(self):
        atoms = self.atoms
        symprec = self.p.symprec
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
        self.atoms = symAts

    def get_ball_volume(self):
        ballVol = 0
        for num in self.atoms.get_atomic_numbers():
            ballVol += 4*np.pi/3*(covalent_radii[num])**3
        self.ball_volume = ballVol
        return ballVol

    def get_volRatio(self):
        self.volRatio = self.atoms.get_volume()/self.get_ball_volume()
        return self.volRatio

    def need_check(self, atoms=None):
        return True
        """
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        return self.p.chkSeed or not a.info['origin'] == 'seed'
        """
        
    def check_cellpar(self,atoms=None):
        """
        check if cellpar reasonable
        TODO bond
        """
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()

        minLen = self.p.minLattice if self.p.minLattice else [0,0,0,45,45,45]
        maxLen = self.p.maxLattice if self.p.maxLattice else [100,100,100,135,135,135]
        #minLen = [0,0,0,45,45,45]
        #maxLen = [100,100,100,135,135,135]

        minLen,maxLen = np.array([minLen,maxLen])
        cellPar = a.cell.cellpar()

        cos_ = np.cos(cellPar[3:]/180*np.pi)
        sin_ = np.sin(cellPar[3:]/180*np.pi)
        X = np.sum(cos_**2)-2*cos_[0]*cos_[1]*cos_[2]
        angles = np.arccos(np.sqrt(X-cos_**2)/sin_)/np.pi*180

        return (minLen <= cellPar).all() and (cellPar <= maxLen).all() and (angles>=30).all()
        #return (minLen <= cellPar).all() and (cellPar <= maxLen).all()

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

        threshold = self.p.dRatio
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

    def check_mol(self,atoms=None):
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()

        molCryst = Molfilter(a, coef=self.p.bondRatio)
        for mol in molCryst:
            molCt = Counter(mol.symbols)
            if molCt not in self.molCounters:
            #if mol.symbol not in self.inputFormulas:
                return False
        return True

    def check(self,atoms=None):
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        check_cellpar = self.check_cellpar(a)
        check_distance = self.check_distance(a)
        check_formula = self.check_formula(a)
        check_mol = True
        if self.p.chkMol:
            check_mol = self.check_mol(a)
        if not check_cellpar:
            log.debug("Fail in check_cellpar")
        if not check_distance:
            log.debug("Fail in check_distance")
        if not check_mol:
            log.debug("Fail in check_mol")
        return check_cellpar and check_distance and check_mol and check_formula

    def sort(self):
        #indices = []
        #for s in self.p.symbols:
        #    indices.extend([i for i,atom in enumerate(self.atoms) if atom.symbol==s])
        numbers = self.atoms.get_atomic_numbers()
        indices = sorted(range(len(self.atoms)), key=lambda x:numbers[x])
        self.atoms = self.atoms[indices]

    def merge_atoms(self):
        """
        TODO threshold
        if a pair of atoms are too close, merge them.
        """
        atoms = self.atoms
        tolerance = self.p.dRatio
        cutoffs = [tolerance * covalent_radii[num] for num in atoms.get_atomic_numbers()]
        nl = neighbor_list("ij", atoms, cutoffs,max_nbins=10)
        indices = list(range(len(atoms)))
        exclude = []

        for i, j in zip(*nl):
            if i not in exclude and not j in exclude:
                exclude.append(random.choice([i,j]))

        if len(exclude) > 0:
            save = [index for index in indices if index not in exclude]
            if len(save) > 0: 
                mAts = atoms[save]
            else:
                mAts = Atoms(cell=atoms.get_cell(), pbc=atoms.get_pbc())
            mAts.info = atoms.info.copy()
        else:
            mAts = atoms
        self.atoms = mAts

    def repair_atoms(self):
        """
        sybls: a list of symbols
        toFrml: a list of formula after repair
        """
        #if not self.needrepair():
        if self.check_formula():
            self.sort()
            return True
        if len(self.atoms) == 0:
            self.atoms = None
            log.debug("Empty crystal after merging!")
            return False
        atoms = self.atoms
        dRatio = self.p.dRatio
        syms = atoms.get_chemical_symbols()
        #nowFrml = Counter(atoms.get_chemical_symbols())
        targetFrml = self.get_targetFrml()
        # log.debug("Target formula: {}".format(targetFrml))
        if not targetFrml:
            log.debug("Cannot get target formula: {}".format(targetFrml))
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
                remain_index = [i for i in range(len(repatoms)) if i != del_index]
                pos = repatoms.positions[del_index]
                if check_new_atom_dist(repatoms[remain_index], pos, add_symbol, dRatio):
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
            for _ in range(self.p.repairtryNum):
                # select a center atoms
                centerAt = repatoms[np.random.randint(0,len(repatoms))]
                basicR = covalent_radii[centerAt.number] + covalent_radii[atomic_numbers[add_symbol]]
                # random position in spherical coordination
                radius = basicR * (dRatio + np.random.uniform(0,0.3))
                theta = np.random.uniform(0,np.pi)
                phi = np.random.uniform(0,2*np.pi)
                pos = centerAt.position + radius*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),np.cos(theta)])
                if check_new_atom_dist(repatoms, pos, add_symbol, dRatio):
                    addAt = Atom(symbol=add_symbol, position=pos)
                    repatoms.append(addAt)
                    toadd[add_symbol] -= 1
                    if toadd[add_symbol] == 0:
                        toadd.pop(add_symbol)
                    break
            else:
                self.atoms = None
                return False
        if len(repatoms) > 0:
            self.atoms = repatoms
            self.sort()
            return True
        else:
            self.atoms = None
            return False

    def to_mol(self):
        """
        generate molecular crystal according to molDetector and bondRatio
        """
        self.molCryst = Molfilter(self.atoms, self.p.molDetector, self.p.bondRatio)

class FixInd(Individual):
    def __call__(self,atoms):
        newind = self.__new__(self.__class__)
        newind.p = self.p

        newind.comparator = self.comparator
        newind.cf = self.cf
        newind.inputMols = self.inputMols
        newind.molCounters = self.molCounters
        newind.inputFormulas = self.inputFormulas

        if atoms.__class__.__name__ == 'Molfilter':
            atoms = atoms.to_atoms()
        atoms.wrap()
        newind.atoms = atoms
        newind.sort()
        newind.info = {'numOfFormula':int(round(len(atoms)/sum(self.p.formula)))}
        newind.info['fitness'] = {}
        return newind

    #def needrepair(self):
    #    #check if atoms need repair
    def check_formula(self, atoms=None):
        # check if the current formual is right
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        Natoms = len(a)
        if Natoms < self.p.minAt or Natoms > self.p.maxAt:
            return False

        symbols = a.get_chemical_symbols()
        for s in symbols:
            if s not in self.p.symbols:
                return False
        formula = np.array([symbols.count(s) for s in self.p.symbols])
        #formula = get_formula(a, self.p.symbols)
        numFrml = int(round(Natoms/sum(self.p.formula)))
        targetFrml = numFrml*np.array(self.p.formula)
        return np.all(targetFrml == formula)
        # rank = np.linalg.matrix_rank(np.concatenate(([self.p.formula], [formula])))
        # return rank == 1


    def get_targetFrml(self):
        atoms = self.atoms
        Natoms = len(atoms)
        if self.p.minAt <= Natoms <= self.p.maxAt :
            numFrml = int(round(Natoms/sum(self.p.formula)))
        else:
            numFrml = int(round(np.mean([ind.info['numOfFormula'] for ind in self.parents])))
        self.info['formula'] = self.p.formula
        self.info['numOfFormula'] = numFrml
        targetFrml = {s:numFrml*i for s,i in zip(self.p.symbols,self.p.formula)}
        return targetFrml

class VarInd(Individual):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.rank = np.linalg.matrix_rank(self.p.formula)
        self.invF = np.linalg.pinv(self.p.formula)
        checkParameters(self.p,parameters,[],{'fullEles':False})

    def __call__(self,atoms):
        newind = self.__new__(self.__class__)
        newind.p = self.p

        newind.rank = self.rank
        newind.invF = self.invF
        newind.comparator = self.comparator
        newind.cf = self.cf
        newind.inputMols = self.inputMols
        newind.inputFormulas = self.inputFormulas

        if atoms.__class__.__name__ == 'Molfilter':
            atoms = atoms.to_atoms()
        atoms.wrap()
        newind.atoms = atoms
        newind.sort()
        newind.info = {'numOfFormula':1}
        newind.info['fitness'] = {}
        return newind

    #def needrepair(self):
    #    #check if atoms need repair
    def check_formula(self, atoms=None):
        # check if the current formual is right
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        Natoms = len(a)
        if Natoms < self.p.minAt or Natoms > self.p.maxAt:
            return False

        symbols = a.get_chemical_symbols()
        for s in symbols:
            if s not in self.p.symbols:
                return False
        formula = [symbols.count(s) for s in self.p.symbols]
        #formula = get_formula(a, self.p.symbols)
        rank = np.linalg.matrix_rank(np.concatenate((self.p.formula, [formula])))
        if rank == self.rank:
            coef = np.rint(np.dot(formula, self.invF)).astype(np.int)
            newFrml = np.dot(coef, self.p.formula).astype(np.int)
            return (coef>=0).all() and (newFrml == formula).all()
        else:
            return False
        

    def get_targetFrml(self):
        symbols = self.atoms.get_chemical_symbols()
        formula = [symbols.count(s) for s in self.p.symbols]
        coef = np.rint(np.dot(formula, self.invF)).astype(np.int)
        # make sure that all elements in coef >= 0
        coef[np.where(coef<0)] = 0 
        newFrml = np.dot(coef, self.p.formula).astype(np.int)
        bestFrml = newFrml.tolist()
        if self.p.minAt <= sum(bestFrml) <= self.p.maxAt:
            targetFrml = {s:i for s,i in zip(self.p.symbols,bestFrml)}
        else:
            targetFrml = None
        return targetFrml

    def check_full(self,atoms=None):
        # return True
        # """
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        symbols = a.get_chemical_symbols()
        symCount = Counter(symbols)
        formula = [symCount[s] for s in self.p.symbols]
        return not self.p.fullEles or 0 not in formula
        # """

    def check(self, atoms=None):
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()
        # return super().check(atoms=a) and self.check_full(atoms=a)
        return super().check(atoms=a)
