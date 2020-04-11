"""
Base module for all operators that create offspring.
steal from ase.ga
"""
import numpy as np
import logging
from ase import Atoms
from ase.geometry import cell_to_cellpar,cellpar_to_cell,get_duplicate_atoms
from ase.neighborlist import NeighborList
from ase.data import covalent_radii,chemical_symbols
from .population import Population
from .molecule import Molfilter
import ase.io
from .utils import *
class OffspringCreator:
    def __init__(self,tryNum=10):
        self.tryNum = tryNum
        self.descriptor = type(self).__name__

    def get_new_individual(self):
        pass

class Mutation(OffspringCreator):
    def __init__(self, tryNum=10):
        self.optype = 'Mutation'
        super().__init__(tryNum=tryNum)

    def mutate(self,ind):
        raise NotImplementedError(self.descriptor)

    def get_new_individual(self,ind,chkMol=False):
        for _ in range(self.tryNum):
            newind = self.mutate(ind)
            if newind is None:
                continue
            newind.parents = [ind]
            if not chkMol:
                newind.merge_atoms()
                # if not newind.check_distance():
                #     logging.debug("Too close atoms in merged ind!")
                if newind.repair_atoms():
                    break
            else:
                #if not newind.needrepair() and newind.check_mol():
                if newind.check_formula() and newind.check_mol():
                    break
        else:
            logging.debug('fail {} in {}'.format(self.descriptor,ind.info['identity']))
            return None

        # remove some parent infomation
        rmkeys = ['enthalpy', 'spg', 'priVol', 'priNum', 'ehull']
        for k in rmkeys:
            if k in newind.atoms.info.keys():
                del newind.atoms.info[k]
        if hasattr(newind, 'molCryst'):
            delattr(newind, 'molCryst')

        newind.info['parents'] = [ind.info['identity']]
        newind.info['parentE'] = ind.info['enthalpy']
        newind.info['pardom'] = ind.info['dominators']
        newind.info['origin'] = self.descriptor
        newind.info['symbols'] = ind.p.symbols
        newind.info['formula'] = get_formula(newind.atoms, newind.info['symbols'])
        newind.fix_atoms_info()

        return newind

class Crossover(OffspringCreator):
    def __init__(self, tryNum=10):
        self.optype = 'Crossover'
        super().__init__(tryNum=tryNum)

    def cross(self,ind):
        raise NotImplementedError(self.descriptor)

    def get_new_individual(self,parents,chkMol=False):
        f,m = parents
        for _ in range(self.tryNum):
            newind = self.cross(f,m)
            if newind is None:
                continue
            newind.parents = [f,m]
            if not chkMol:
                newind.merge_atoms()
                if newind.repair_atoms():
                    break
            else:
                #if not newind.needrepair() and newind.check_mol():
                if newind.check_formula() and newind.check_mol():
                    break
        else:
            logging.debug('fail {} between {} and {}'.format(self.descriptor,f.info['identity'],m.info['identity']))
            return None

        # remove some parent infomation
        rmkeys = ['enthalpy', 'spg', 'priVol', 'priNum', 'ehull']
        for k in rmkeys:
            if k in newind.atoms.info.keys():
                del newind.atoms.info[k]

        newind.info['parents'] = [f.info['identity'],m.info['identity']]
        newind.info['parentE'] = 0.5*(f.info['enthalpy']+m.info['enthalpy'])
        newind.info['pardom'] = 0.5*(f.info['dominators']+m.info['dominators'])
        newind.info['origin'] = self.descriptor
        newind.info['symbols'] = f.p.symbols
        newind.info['formula'] = get_formula(newind.atoms, newind.info['symbols'])
        newind.fix_atoms_info()

        return newind
class SoftMutation(Mutation):
    """
    Mutates the structure by displacing it along the lowest (nonzero)
    frequency modes found by vibrational analysis, as in:

    * `Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632`__

      __ https://dx.doi.org/10.1016/j.cpc.2010.06.007

    """
    def __init__(self, calculator, bounds=[0.5,2.0],tryNum=10):
        self.bounds = bounds
        self.calc = calculator
        super().__init__(tryNum=tryNum)

    def _get_hessian(self, atoms, dx):
        N = len(atoms)
        pos = atoms.get_positions()
        hessian = np.zeros([3*N,3*N])
        for i in range(N):
            for j in range(3):
                pos_ = np.copy(pos)
                pos_[i,j] += dx
                atoms.set_positions(pos_)
                f1 = atoms.get_forces().flatten()

                pos_[i,j] -= 2*dx
                atoms.set_positions(pos_)
                f2 = atoms.get_forces().flatten()
                hessian[3*i+j] = (f1 - f2)/(2 * dx)
        atoms.set_positions(pos)
        hessian = -0.5*(hessian + hessian.T)
        return hessian

    def _get_modes(self, atoms, dx=0.02, k=2, massweighing=False):
        hessian = self._get_hessian(atoms, dx)
        if massweighing:
            m = np.array([np.repeat(atoms.get_masses()**-0.5, 3)])
            hessian *= (m * m.T)
        eigvals, eigvecs = np.linalg.eigh(hessian)
        modes = {eigval: eigvecs[:, i] for i, eigval in enumerate(eigvals)}
        keys = np.array(sorted(modes))
        ekeys = np.e**(-k*keys)
        ekeys[:3] = 0
        p = ekeys/np.sum(ekeys)
        key = np.random.choice(keys,p=p)
        mode = modes[key].reshape(-1,3)
        return mode

    def mutate(self, ind):
        atoms = ind.atoms.copy()
        atoms.set_calculator(self.calc)

        if ind.p.molDetector != 0:
            atoms = ind.molCryst
        pos = atoms.get_positions()
        mode = self._get_modes(atoms)
        largest_norm = np.max(np.apply_along_axis(np.linalg.norm, 1, mode))
        amplitude = np.random.uniform(*self.bounds)/largest_norm
        direction = np.random.choice([-1,1])
        pos_new = pos + direction * amplitude * mode
        atoms.set_positions(pos_new)
        atoms.wrap()
        return ind(atoms)

class PermMutation(Mutation):
    def __init__(self, fracSwaps=0.5,tryNum=10):
        """
        fracSwaps -- max ratio of atoms exchange
        """
        self.fracSwaps = fracSwaps
        super().__init__(tryNum=tryNum)

    def mutate(self, ind):
        fracSwaps = self.fracSwaps
        atoms = ind.atoms.copy()

        if ind.p.molDetector != 0:
            atoms = ind.molCryst

        maxSwaps = int(fracSwaps*len(atoms))
        if maxSwaps < 2:
            maxSwaps = 2
        numSwaps = np.random.randint(1, maxSwaps)

        symbols = atoms.get_chemical_symbols()
        symList = list(set(symbols))
        if len(symList)<2:
            return None

        indices = list(range(len(atoms)))
        n = 0
        while n < numSwaps:
            s1, s2 = np.random.choice(symList, 2)
            if s1 == s2 and s1 in chemical_symbols:
                continue
            s1list = [index for index in indices if atoms[index].symbol==s1]
            s2list = [index for index in indices if atoms[index].symbol==s2]
            if len(s1list)==0 or len(s2list)==0:
                n += 1
                continue
            i = np.random.choice(s1list)
            j = np.random.choice(s2list)
            if i == j:
                n += 1
                continue
            atoms[i].position,atoms[j].position = atoms[j].position,atoms[i].position
            indices.remove(i)
            indices.remove(j)
            n += 1
        return ind(atoms)

class LatticeMutation(Mutation):
    def __init__(self, sigma=0.05, cellCut=1,tryNum=10):
        """
        sigma: Gauss distribution standard deviation
        cellCut: coefficient of gauss distribution in cell mutation
        """
        self.sigma = sigma
        self.cellCut = cellCut
        super().__init__(tryNum=tryNum)

    def mutate(self,ind):
        sigma = self.sigma
        cellCut = self.cellCut
        atoms = ind.atoms.copy()
        oldCell = atoms.get_cell()

        latGauss = np.random.normal(0, sigma,6) *cellCut
        strain = np.array([
            [1+latGauss[0], latGauss[1]/2, latGauss[2]/2],
            [latGauss[1]/2, 1+latGauss[3], latGauss[4]/2],
            [latGauss[2]/2, latGauss[4]/2, 1+latGauss[5]]
            ])
        newCell = oldCell@strain
        ratio = atoms.get_volume()/np.abs(np.linalg.det(newCell))
        cellPar = cell_to_cellpar(newCell)
        cellPar[:3] = [length*ratio**(1/3) for length in cellPar[:3]]

        if ind.p.molDetector != 0:
            atoms = ind.molCryst
            atoms.set_cell(cellpar_to_cell(cellPar))
        else:
            atoms.set_cell(cellPar, scale_atoms=True)
            positions = atoms.get_positions()
            atGauss = np.random.normal(0,sigma,[len(atoms),3])/sigma
            radius = covalent_radii[atoms.get_atomic_numbers()][:,np.newaxis]
            positions += atGauss * radius
            atoms.set_positions(positions)

        return ind(atoms)


class SlipMutation(Mutation):
    def __init__(self, cut=0.5, randRange=[0.5, 2],tryNum=10):
        self.cut = cut
        self.randRange = randRange
        super().__init__(tryNum=tryNum)

    def mutate(self, ind):
        '''
        from MUSE
        '''
        cut = self.cut
        atoms = ind.atoms.copy()

        if ind.p.molDetector != 0:
            atoms = ind.molCryst


        scl_pos = atoms.get_scaled_positions()
        axis = list(range(3))
        np.random.shuffle(axis)

        z = np.where(scl_pos[:,axis[0]] > cut)
        scl_pos[z,axis[1]] += np.random.uniform(*self.randRange)
        scl_pos[z,axis[2]] += np.random.uniform(*self.randRange)
        atoms.set_scaled_positions(scl_pos)

        return ind(atoms)

class RippleMutation(Mutation):
    def __init__(self, rho=0.1, mu=1, eta=1,tryNum=10):
        self.rho = rho
        self.mu = mu
        self.eta = eta
        super().__init__(tryNum=tryNum)

    def mutate(self,ind):
        '''
        from XtalOpt
        '''
        atoms = ind.atoms.copy()

        if ind.p.molDetector != 0:
            atoms = ind.molCryst

        scl_pos = atoms.get_scaled_positions()
        axis = list(range(3))
        np.random.shuffle(axis)

        scl_pos[:, axis[0]] += self.rho*\
            np.cos(2*np.pi*self.mu*scl_pos[:,axis[1]] + np.random.uniform(0,2*np.pi,len(atoms)))*\
            np.cos(2*np.pi*self.eta*scl_pos[:,axis[2]] + np.random.uniform(0,2*np.pi,len(atoms)))

        atoms.set_scaled_positions(scl_pos)

        return ind(atoms)

class RotateMutation(Mutation):
    def __init__(self, p=0.5,tryNum=10):
        self.p = p
        super().__init__(tryNum=tryNum)

    def mutate(self,ind):
        atoms = ind.atoms.copy()
        # atoms = Molfilter(atoms)
        atoms = ind.molCryst
        for mol in atoms:
            if len(mol)>1 and np.random.rand() < self.p:
                phi, theta, psi = np.random.uniform(-1,1,3)*np.pi*2
                mol.rotate(phi,theta,psi)
        return ind(atoms)

class FormulaMutation(Mutation):
    def __init__(self, symbols, p1=0.5, p2=0.2, tryNum=10):
        self.p1 = p1
        self.p2 = p2
        self.symbols = symbols
        super().__init__(tryNum=tryNum)

    def mutate(self,ind):
        """
        Randomly change symbols, only used for variable formula search
        and unavailable for molecular crystals (chkMol should be False).
        """
        atoms = ind.atoms.copy()
        Nat = len(atoms)
        symbols = self.symbols
        #symList = list(set(symbols))
        rmInds = []
        for i, atom in enumerate(atoms):
            if np.random.rand() < self.p1:
                otherSym = [s for s in symbols if s != atom.symbol]
                atom.symbol = str(np.random.choice(otherSym))
                # Delete atoms randomly
                if np.random.rand() < self.p2:
                    rmInds.append(i)
        saveInds = [j for j in range(Nat) if j not in rmInds]
        if len(saveInds) > 0:
            return ind(atoms[saveInds])
        else:
            return None

class CutAndSplicePairing(Crossover):
    """ A cut and splice operator for bulk structures.

    For more information, see also:

    * `Glass, Oganov, Hansen, Comp. Phys. Comm. 175 (2006) 713-720`__

      __ https://doi.org/10.1016/j.cpc.2006.07.020

    * `Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387`__

      __ https://doi.org/10.1016/j.cpc.2010.07.048
    """
    def __init__(self, cutDisp=0):
        self.cutDisp = cutDisp
        super().__init__()

    def cross(self, ind1, ind2):
        """
        cut two cells to get a new cell
        """
        atoms1,atoms2,ratio1,ratio2 = match_lattice(ind1.atoms,ind2.atoms)
        cutCell = (atoms1.get_cell()+atoms2.get_cell())*0.5
        cutCell[2] = (atoms1.get_cell()[2]/ratio1+atoms2.get_cell()[2]/ratio2)*0.5
        cutVol = (atoms1.get_volume()/ratio1+atoms2.get_volume()/ratio2)*0.5
        cutCellPar = cell_to_cellpar(cutCell)
        ratio = cutVol/abs(np.linalg.det(cutCell))
        if ratio > 1:
            cutCellPar[:3] = [length*ratio**(1/3) for length in cutCellPar[:3]]

        cutAtoms = Atoms(cell=cutCellPar,pbc = True,)

        if ind1.p.molDetector != 0:
            atoms1 = ind1.molCryst
            atoms2 = ind2.molCryst
            cutAtoms = Molfilter(cutAtoms, ind1.p.molDetector, ind1.p.bondRatio)


        scaled_positions = []
        cutPos = [0, 0.5+self.cutDisp*np.random.uniform(-0.5, 0.5), 1]

        for n, atoms in enumerate([atoms1, atoms2]):
            spositions = atoms.get_scaled_positions()
            for i,atom in enumerate(atoms):
                if cutPos[n] <= spositions[i, 2] < cutPos[n+1]:
                    cutAtoms.append(atom)
                    scaled_positions.append(spositions[i])
        if len(scaled_positions) == 0:
            return None

            #raise RuntimeError('No atoms in the new cell')
        cutAtoms.set_scaled_positions(scaled_positions)

        if ind1.p.molDetector != 0:
            cutAtoms = cutAtoms.to_atoms()

        cutInd = ind1(cutAtoms)
        cutInd.parents = [ind1 ,ind2]
        return cutInd

class ReplaceBallPairing(Crossover):
    """
    TODO rotate the replace ball
    how to rotate the ball to make energy lower
    """
    def __init__(self, cutrange=[1,2]):
        self.cutrange = cutrange
        super().__init__()

    def cross(self, ind1, ind2):
        """replace some atoms in a ball
        """
        cutR = np.random.uniform(*self.cutrange)
        atoms1,atoms2 = ind1.atoms.copy(),ind2.atoms.copy()
        i,j = np.random.randint(len(atoms1)),np.random.randint(len(atoms2))
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
        cutInd = ind1(newatoms)
        cutInd.parents = [ind1 ,ind2]
        return newatoms

class PopGenerator:
    def __init__(self,numlist,oplist,parameters):
        self.oplist = oplist
        self.numlist = numlist
        self.p = EmptyClass()
        Requirement = ['popSize','saveGood','molDetector', 'calcType']
        Default = {'chkMol': False,'addSym': False,'randFrac': 0.0}
        checkParameters(self.p,parameters,Requirement,Default)

    def get_pairs(self, Pop, crossNum ,clusterNum, tryNum=50,k=0.3):
        pairs = []
        labels,_ = Pop.clustering(clusterNum)
        fail = 0
        while len(pairs) < crossNum and fail < tryNum:
            label = np.random.choice(np.unique(labels))
            subpop = [ind for j,ind in enumerate(Pop.pop) if labels[j] == label]
            if len(subpop) < 2:
                fail+=1
                continue

            dom = np.array([ind.info['dominators'] for ind in subpop])
            edom = np.exp(-k*dom)
            p = edom/np.sum(edom)
            pair = tuple(np.random.choice(subpop,2,False,p=p))
            if pair in pairs:
                fail+=1
                continue
            pairs.append(pair)
        return pairs

    def get_inds(self,Pop,mutateNum,k=0.3):
        dom = np.array([ind.info['dominators'] for ind in Pop.pop])
        edom = np.exp(-k*dom)
        p = edom/np.sum(edom)
        # mutateNum = min(mutateNum,len(Pop))
        if mutateNum > len(Pop):
            return np.random.choice(Pop.pop,mutateNum,True,p=p)
        else:
            return np.random.choice(Pop.pop,mutateNum,False,p=p)

    def generate(self,Pop,saveGood):
        # calculate dominators before checking formula
        Pop.calc_dominators()
        if self.p.calcType == 'var':
            Pop.check_full()
        assert len(Pop) >= saveGood, \
            "saveGood should be shorter than length of curPop!"
        #TODO move addsym to ind
        if self.p.addSym:
            Pop.add_symmetry()
        newPop = Pop([],'initpop',Pop.gen+1)
        for op,num in zip(self.oplist,self.numlist):
            if num == 0:
                continue
            logging.debug('name:{} num:{}'.format(op.descriptor,num))
            if op.optype == 'Mutation':
                mutate_inds = self.get_inds(Pop,num)
                for i,ind in enumerate(mutate_inds):
                    if self.p.molDetector != 0 and not hasattr(newind, 'molCryst'):
                        ind.to_mol()
                    newind = op.get_new_individual(ind, chkMol=self.p.chkMol)
                    if newind:
                        newPop.append(newind)
            elif op.optype == 'Crossover':
                cross_pairs = self.get_pairs(Pop,num,saveGood)
                for i,parents in enumerate(cross_pairs):
                    if self.p.molDetector != 0:
                        for ind in parents:
                            if not hasattr(ind, 'molCryst'):
                                ind.to_mol()
                    newind = op.get_new_individual(parents,chkMol=self.p.chkMol)
                    if newind:
                        newPop.append(newind)
            logging.debug("popsize after {}: {}".format(op.descriptor, len(newPop)))

        if self.p.calcType == 'var':
            newPop.check_full()
        #newPop.save('testnew')
        newPop.check()
        return newPop

    def select(self,Pop,num,k=0.3):
        if num < len(Pop):
            pardom = np.array([ind.info['pardom'] for ind in Pop.pop])
            edom = np.e**(-k*pardom)
            p = edom/np.sum(edom)
            Pop.pop = list(np.random.choice(Pop.pop,num,False,p=p))
            return Pop
        else:
            return Pop

    def next_Pop(self,Pop):
        saveGood = self.p.saveGood
        popSize = int(self.p.popSize*(1-self.p.randFrac))
        newPop = self.generate(Pop,saveGood)
        return self.select(newPop,popSize)

class MLselect(PopGenerator):
    def __init__(self, numlist, oplist, calc,parameters):
        super().__init__(numlist, oplist, parameters)
        self.calc = calc

    def select(self,Pop,num,k=0.3):
        predictE = []
        if num < len(Pop):
            for ind in Pop:
                ind.atoms.set_calculator(self.calc)
                ind.info['predictE'] = ind.atoms.get_potential_energy()
                predictE.append(ind.info['predictE'])
                ind.atoms.set_calculator(None)

            dom = np.argsort(predictE)
            edom = np.exp(-k*dom)
            p = edom/np.sum(edom)
            Pop.pop = np.random.choice(Pop.pop,num,False,p=p)
            return Pop
        else:
            return Pop


if __name__ == '__main__':
    from .population import Individual
    from .readparm import read_parameters
    from .utils import EmptyClass
    from ase.calculators.emt import EMT
    from ase.calculators.lj import LennardJones
    import ase.io
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="print debug information", action='store_true', default=False)
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(message)s")
        logging.info('Debug mode')
    else:
        logging.basicConfig(filename='log.txt', level=logging.INFO, format="%(message)s")
    parameters = read_parameters('input.yaml')
    p = EmptyClass()
    for key, val in parameters.items():
        setattr(p, key, val)
    pop_ = []
    traj = ase.io.read('good.traj',':')
    for ind in traj:
        pop_.append(Individual(ind,p))

    pop = Population(pop_,p)

    calc = LennardJones()
    soft = SoftMutation(calc)
    cutandsplice = CutAndSplicePairing()
    perm = PermMutation()
    lattice = LatticeMutation()
    ripple = RippleMutation()
    slip = SlipMutation()

    oplist = [soft,cutandsplice,perm,lattice,ripple,slip]
    numlist = [10,10,10,10,10,10]
    popgen = PopGenerator(numlist,oplist,p)
    newpop = popgen.next_pop(pop)
    newpop.save('new.traj')
