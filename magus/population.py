import numpy as np
from scipy.spatial.distance import cdist, pdist
import itertools
import ase.io
from ase.data import covalent_radii,atomic_numbers
from ase.neighborlist import neighbor_list
from ase.atom import Atom
from .utils import check_new_atom_dist,sort_elements
import logging
from sklearn import cluster
from .descriptor import ZernikeFp
class Population:
    """
    a class of atoms population
    TODO __iter__
    """
    def __init__(self,pop,parameters,name='temp'):
        self.pop = pop
        self.name = name
        for i,ind in enumerate(pop):
            ind.info['identity'] = [self.name, i]
        self.parameters = parameters
    
    def __len__(self):
        return len(self.pop)

    def append(self,ind):
        self.pop.append(ind)

    def get_all_fitness(self):
        pass

    def del_duplicate(self):
        pass

    def check(self):
        checkPop = []
        for ind in self.pop:
            if ind.check():
                checkPop.append(ind)
        self.pop = checkPop

    def symmetrize_pop(self):
        pass

    def clustering(self, numClusters):
        """
        clustering by fingerprints
        """
        if numClusters >= len(self.pop):
            return np.arange(len(self.pop))

        fpMat = np.array([ind.fingerprint for ind in self.pop])
        return cluster.KMeans(n_clusters=numClusters).fit_predict(fpMat)

class Individual:
    def __init__(self,atoms,parameters):
        self.atoms = atoms
        self.parameters = parameters
        self.formula = parameters.formula
        self.symbols = parameters.symbols
        self.info = {'numOfFormula':int(round(len(atoms)/sum(self.formula)))}

        #TODO add more comparators
        from ase.ga.standard_comparators import AtomsComparator
        self.comparator = AtomsComparator()

        #fingerprint
        cutoff = self.parameters.cutoff
        elems = [atomic_numbers[element] for element in parameters.symbols]
        nmax = self.parameters.ZernikeNmax
        lmax = self.parameters.ZernikeLmax
        ncut = self.parameters.ZernikeNcut
        diag = self.parameters.ZernikeDiag
        self.cf = ZernikeFp(cutoff, nmax, lmax, ncut, elems,diag=diag)


    def __eq__(self, obj):
        return self.comparator.looks_like(self.atoms,obj.atoms)

    def save(self, filename):
        if self.atoms:
            atoms = self.atoms.copy()
            atoms.set_calculator(None)
            atoms.info = self.info
            ase.io.write(filename,atoms)
        else:
            logging.debug('None')

    def new(self,atoms):
        newatoms = atoms.copy()
        parameters = self.parameters
        return Individual(newatoms,parameters)

    @property
    def fingerprint(self):
        if 'fingerprint' not in self.info:
            Efps = self.cf.get_all_fingerprints(self.atoms)[0]
            self.info['fingerprint'] = np.mean(Efps,axis=0)
        return self.info['fingerprint']

    @property
    def fitness(self):
        if 'fitness' not in self.info:
            self.info['fitness'] = self.info['energy']
        return self.info['fitness']

    def check_cellpar(self,atoms=None):
        """
        check if cellpar reasonable
        """
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

    def repair_atoms(self):
        """
        sybls: a list of symbols
        toFrml: a list of formula after repair
        """

        atoms = self.atoms
        dRatio = self.parameters.dRatio
        syms = atoms.get_chemical_symbols()
        #nowFrml = Counter(atoms.get_chemical_symbols())
        targetFrml = self.get_targetFrml()
        toadd, toremove = {} , {}
        for s in targetFrml:
            if syms.count(s) < targetFrml[s]:
                toadd[s] = targetFrml[s] - syms.count(s)
            elif syms.count(s) > targetFrml[s]:
                toremove[s] = syms.count(s) - targetFrml[s]
        logging.debug('sysm:{}\ntarget:{}\ntoadd:{}\ntoremove:{}'.format(syms,targetFrml,toadd,toremove))
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
                logging.debug('del:{}'.format(del_index))
                del repatoms[del_index]
                logging.debug('atoms number:{}'.format(len(repatoms)))
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
                
        self.atoms = sort_elements(repatoms)
        return True

