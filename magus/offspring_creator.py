"""
Base module for all operators that create offspring.
steal from ase.ga
"""
import numpy as np
import logging
from ase import Atoms
from ase.geometry import cell_to_cellpar,cellpar_to_cell,get_duplicate_atoms
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
from .population import Population
from .renew import match_lattice
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

    def get_new_individual(self,ind):
        for _ in range(self.tryNum):
            newind = self.mutate(ind)
            if newind is None:
                continue
            newind.parents = [ind]
            newind.merge_atoms()
            if newind.repair_atoms():
                break
        else:
            logging.debug('fail {} in {}'.format(self.descriptor,ind.info['identity']))
            return None

        # remove some parent infomation
        rmkeys = ['enthalpy', 'spg', 'priVol', 'priNum', 'ehull']
        for k in rmkeys:
            if k in newind.atoms.info.keys():
                del newind.atoms.info[k]

        newind.info['parents'] = [ind.info['identity']]
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

    def get_new_individual(self,parents):
        f,m = parents
        for _ in range(self.tryNum):
            newind = self.cross(f,m)
            if newind is None:
                continue
            newind.parents = [f,m]
            newind.merge_atoms()
            if newind.repair_atoms():
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

    As in the reference above, the next-lowest mode is used if the
    structure has already been softmutated along the current-lowest
    mode.

    Parameters:

    bounds: list
            Lower and upper limits (in Angstrom) for the largest
            atomic displacement in the structure. For a given mode,
            the algorithm starts at zero amplitude and increases
            it until either blmin is violated or the largest
            displacement exceeds the provided upper bound).
            If the largest displacement in the resulting structure
            is lower than the provided lower bound, the mutant is
            considered too similar to the parent and None is
            returned.
    """

    def __init__(self, calculator, bounds=[0.5, 2.0],tryNum=10):
        self.bounds = bounds
        self.calc = calculator
        super().__init__(tryNum=tryNum)

    def _get_hessian(self, atoms, dx):
        """
        Returns the Hessian matrix d2E/dxi/dxj using a first-order
        central difference scheme with displacements dx.
        """
        N = len(atoms)
        pos = atoms.get_positions()
        hessian = np.zeros((3 * N, 3 * N))

        for i in range(3 * N):
            row = np.zeros(3 * N)
            for direction in [-1, 1]:
                disp = np.zeros(3)
                disp[i % 3] = direction * dx
                pos_disp = np.copy(pos)
                pos_disp[i // 3] += disp
                atoms.set_positions(pos_disp)
                f = atoms.get_forces()
                row += -1 * direction * f.flatten()

            row /= (2. * dx)
            hessian[i] = row

        hessian += np.copy(hessian).T
        hessian *= 0.5
        atoms.set_positions(pos)

        return hessian

    def _calculate_normal_modes(self, atoms, dx=0.02, massweighing=False):
        """Performs the vibrational analysis."""
        hessian = self._get_hessian(atoms, dx)
        if massweighing:
            m = np.array([np.repeat(atoms.get_masses()**-0.5, 3)])
            hessian *= (m * m.T)

        eigvals, eigvecs = np.linalg.eigh(hessian)
        modes = {eigval: eigvecs[:, i] for i, eigval in enumerate(eigvals)}
        return modes

    def mutate(self, ind):
        """ Does the actual mutation. """
        a = ind.atoms.copy()
        a.set_calculator(self.calc)

        pos = a.get_positions()
        modes = self._calculate_normal_modes(a)

        # Select the mode along which we want to move the atoms;
        # The first 3 translational modes as well as previously
        # applied modes are discarded.

        keys = np.array(sorted(modes))
        index = 3

        key = keys[index]
        mode = modes[key].reshape(np.shape(pos))

        # Find a suitable amplitude for translation along the mode;
        # at every trial amplitude both positive and negative
        # directions are tried.

        mutant = ind.atoms.copy()
        amplitude = 0.
        increment = 0.1
        direction = 1
        largest_norm = np.max(np.apply_along_axis(np.linalg.norm, 1, mode))

        while amplitude * largest_norm < self.bounds[1]:
            pos_new = pos + direction * amplitude * mode
            mutant.set_positions(pos_new)
            mutant.wrap()
            too_close = ind.check_distance(mutant)
            if too_close:
                amplitude -= increment
                pos_new = pos + direction * amplitude * mode
                mutant.set_positions(pos_new)
                mutant.wrap()
                break

            if direction == 1:
                direction = -1
            else:
                direction = 1
                amplitude += increment

        if amplitude * largest_norm < self.bounds[0]:
            return None

        return ind(mutant)

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

        if ind.p.is_mol:
            atoms = Molfilter(atoms)

        maxSwaps = int(fracSwaps*len(atoms))
        if maxSwaps == 0:
            maxSwaps = 1
        numSwaps = np.random.randint(1, maxSwaps)

        symbols = atoms.get_chemical_symbols()
        symList = list(set(symbols))
        if len(symList)<2:
            return None

        indices = list(range(len(atoms)))
        for _ in range(numSwaps):
            s1, s2 = np.random.choice(symList, 2, replace = False)
            s1list = [index for index in indices if atoms[index].symbol==s1]
            s2list = [index for index in indices if atoms[index].symbol==s2]
            if len(s1list)==0 or len(s2list)==0:
                break
            i = np.random.choice(s1list)
            j = np.random.choice(s2list)
            atoms[i].position,atoms[j].position = atoms[j].position,atoms[i].position
            indices.remove(i)
            indices.remove(j)
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

        if ind.p.is_mol:
            atoms = Molfilter(atoms)

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

        if ind.p.is_mol:
            atoms = Molfilter(atoms)


        scl_pos = atoms.get_scaled_positions()
        axis = list(range(3))
        np.random.shuffle(axis)

        z = np.where(scl_pos[:,axis[0]] > cut)
        scl_pos[z,axis[1]] += np.random.uniform(*self.randRange)
        scl_pos[z,axis[2]] += np.random.uniform(*self.randRange)
        atoms.set_scaled_positions(scl_pos)

        return ind(atoms)

class RippleMutation(Mutation):
    def __init__(self, rho=0.3, mu=2, eta=1,tryNum=10):
        self.rho = rho
        self.mu = mu
        self.eta = eta
        super().__init__(tryNum=tryNum)

    def mutate(self,ind):
        '''
        from XtalOpt
        '''
        atoms = ind.atoms.copy()

        if ind.p.is_mol:
            atoms = Molfilter(atoms)

        scl_pos = atoms.get_scaled_positions()
        axis = list(range(3))
        np.random.shuffle(axis)

        scl_pos[:, axis[0]] += self.rho*\
            np.cos(2*np.pi*self.mu*scl_pos[:,axis[1]] + np.random.uniform(0,2*np.pi,len(atoms)))*\
            np.cos(2*np.pi*self.eta*scl_pos[:,axis[2]] + np.random.uniform(0,2*np.pi,len(atoms)))

        atoms.set_scaled_positions(scl_pos)

        return ind(atoms)


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

        if ind1.p.is_mol:
            atoms1 = Molfilter(atoms1)
            atoms2 = Molfilter(atoms2)
            cutAtoms =  Molfilter(cutAtoms)


        scaled_positions = []
        cutPos = [0, 0.5+self.cutDisp*np.random.uniform(-0.5, 0.5), 1]

        for n, atoms in enumerate([atoms1, atoms2]):
            spositions = atoms.get_scaled_positions()
            for i,atom in enumerate(atoms):
                if cutPos[n] <= spositions[i, 2] < cutPos[n+1]:
                    cutAtoms.append(atom)
                    scaled_positions.append(spositions[i])
        cutAtoms.set_scaled_positions(scaled_positions)
        if len(cutAtoms) == 0:
            raise RuntimeError('No atoms in the new cell')

        if ind1.p.is_mol:
            cutAtoms =  cutAtoms.to_atoms()

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
        Requirement = ['popSize','saveGood']
        Default = {}
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
            edom = np.e**(-k*dom)
            p = edom/np.sum(edom)
            pair = tuple(np.random.choice(subpop,2,False,p=p))
            if pair in pairs:
                fail+=1
                continue
            pairs.append(pair)
        return pairs

    def get_inds(self,Pop,mutateNum,k=0.3):
        dom = np.array([ind.info['dominators'] for ind in Pop.pop])
        edom = np.e**(-k*dom)
        p = edom/np.sum(edom)
        mutateNum = min(mutateNum,len(Pop))
        return np.random.choice(Pop.pop,mutateNum,False,p=p)

    def generate(self,Pop,saveGood):
        assert len(Pop) >= saveGood, \
            "saveGood should be shorter than length of curPop!"
        Pop.calc_dominators()
        #TODO move addsym to ind
        if Pop[0].p.addSym:
            Pop.add_symmetry()
        newPop = Pop([],'initpop',Pop.gen+1)
        for op,num in zip(self.oplist,self.numlist):
            logging.debug('name:{} num:{}'.format(op.descriptor,num))
            if op.optype == 'Mutation':
                mutate_inds = self.get_inds(Pop,num)
                for i,ind in enumerate(mutate_inds):
                    newind = op.get_new_individual(ind)
                    if newind:
                        newPop.append(newind)
            elif op.optype == 'Crossover':
                cross_pairs = self.get_pairs(Pop,num,saveGood)
                for i,parents in enumerate(cross_pairs):
                    newind = op.get_new_individual(parents)
                    if newind:
                        newPop.append(newind)
            logging.debug("popsize after {}: {}".format(op.descriptor, len(newPop)))

        newPop.check()
        return newPop

    def select(self,Pop,num,k=0.3):
        if num < len(Pop):
            pardom = np.array([ind.info['pardom'] for ind in Pop.pop])
            edom = np.e**(-k*pardom)
            p = edom/np.sum(edom)
            Pop.pop = np.random.choice(Pop.pop,num,False,p=p)
            return Pop
        else:
            return Pop

    def next_Pop(self,Pop):
        saveGood = self.p.saveGood
        popSize = self.p.popSize
        newPop = self.generate(Pop,saveGood)
        return self.select(newPop,popSize)



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
