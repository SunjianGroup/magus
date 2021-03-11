"""
Base module for all operators that create offspring.
steal from ase.ga
"""
import numpy as np
import logging, copy
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
        logging.debug('success {} in {}'.format(self.descriptor,ind.info['identity']))
        # remove some parent infomation
        rmkeys = ['enthalpy', 'spg', 'priVol', 'priNum', 'ehull', 'energy','forces']
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

        logging.debug('success {} between {} and {}'.format(self.descriptor,f.info['identity'],m.info['identity']))
        # remove some parent infomation
        rmkeys = ['enthalpy', 'spg', 'priVol', 'priNum', 'ehull','energy','forces']
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
            atoms = copy.deepcopy(ind.molCryst)
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
            #atoms = ind.molCryst
            atoms = copy.deepcopy(ind.molCryst)

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
    def __init__(self, sigma=0.1, cellCut=1,tryNum=10):
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
        for i in range(6):
            gau = latGauss[i]
            if gau >= 1 or gau <= -1:
                latGauss[i] = sigma    
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
            #atoms = ind.molCryst
            atoms = copy.deepcopy(ind.molCryst)
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
            #atoms = ind.molCryst
            atoms = copy.deepcopy(ind.molCryst)


        scl_pos = atoms.get_scaled_positions()
        axis = list(range(3))
        np.random.shuffle(axis)

        z = np.where(scl_pos[:,axis[0]] > cut)
        scl_pos[z,axis[1]] += np.random.uniform(*self.randRange)
        scl_pos[z,axis[2]] += np.random.uniform(*self.randRange)
        atoms.set_scaled_positions(scl_pos)

        return ind(atoms)

class LyrSlipMutation(Mutation):
    def __init__(self, cut=0.2, randRange=[0, 1],tryNum=10):
        self.cut = cut
        self.randRange = randRange
        super().__init__(tryNum=tryNum)
        import math
        
    def mutate(self, ind):
        """
        slip of one layer.
        """
        cut = self.cut
        atoms = ind.atoms.copy()
        from .reconstruct import LayerIdentifier
        layers = LayerIdentifier(atoms, prec = self.cut)
        chosenlayer = layers[np.random.choice(len(layers))]
        direction = np.random.uniform(0,2*math.pi)
        trans = [math.cos(direction), math.sin(direction),0]

        pos = atoms.get_positions().copy()
        pos[chosenlayer, :] += np.dot(np.array(trans)*np.random.uniform(*self.randRange), atoms.get_cell())

        atoms.set_positions(pos)
        atoms.wrap()
        return ind(atoms)

from magus.reconstruct import resetLattice
class SymLyrMutation(Mutation):
    def __init__(self, tryNum=10, symprec = 1e-4):
        super().__init__(tryNum=tryNum)
        self.symprec = symprec
    
    def mirrorsym(self, atoms, rot):
        #TODO: remove the self.threshold below
        #self.threshold = 0.5
        ats = atoms.copy()
        axis = atoms.get_cell().copy()
        axis_1 = np.linalg.inv(axis)
        
        #1. calculate the mirror line.
        #For the mirror line in x^y plane and goes through (0,0), its k, i.e., y/x must be a fix number.
        #for mirror matrix[[A,B], [C,D]], k =[ C*x0 + (1+D)*y0]/ [ (1+A)*x0 + B*y0 ] independent to x0, y0. 
        A, B, C, D, k = *(1.0*rot[:2, :2].flatten()), 0
        if C==0 and 1+D == 0:
            k = 0
        elif 1+A == 0 and B ==0:
            k = None
        else:
            #x0, y0 = 1, -(1+A)/B + 1            ...so it is randomly chosen by me...
            k =  (C + (1+D)*( 1 -(1+A)/B ) ) / B if not B==0 else C / (1+A)

        #2. If the mirror line goes through the cell itself, reset it. 
        #Replicate it to get a huge triangle with mirror line and two of cell vectors.
        if not ( (k is None) or k <= 0):
            scell = resetLattice(atoms = ats,expandsize= (4,1,1))
            slattice = ats.get_cell() * np.reshape([-1]*3 + [1]*6, (3,3))
            ats = scell.get(slattice)

        cell = ats.get_cell()
        ats = ats * (2,2,1)
        ats.set_cell(cell)
        ats.translate([-np.sum(ats.get_cell()[:2], axis = 0)]*len(ats))
        index = [i for i, p in enumerate(np.dot(ats.get_positions(), axis_1)) if ((p[1] - k * p[0] >= 0) if not k is None else (p[0] >= 0))]
        
        ats = ats[index]
        rats = ats.copy()
        """
        outats = rats.copy()
        outats.set_cell(ats.get_cell()[:]*np.reshape([2]*6+[1]*3, (3,3)))
        outats.translate([np.sum(ats.get_cell()[:2], axis = 0)]*len(outats))
        ase.io.write('rats1.vasp', outats, format = 'vasp', vasp5=1)
        """
        cpos = np.array([np.dot(np.dot(rot, p), axis) for p in np.dot(ats.get_positions(), axis_1)])
        index = [i for i, p in enumerate(cpos) if math.sqrt(np.sum([x**2 for x in p - ats[i].position])) >= 2*self.threshold* covalent_radii[ats[i].number] ]
        ats = ats[index] 
        ats.set_positions(cpos[index])
        
        rats += ats
        """
        outats = rats.copy()
        outats.set_cell(rats.get_cell()[:]*np.reshape([2]*6+[1]*3, (3,3)))
        outats.translate([np.sum(ats.get_cell()[:2], axis = 0)]*len(outats))
        ase.io.write('rats2.vasp', outats, format = 'vasp', vasp5=1)
        """
        return resetLattice(atoms=rats, expandsize=(1,1,1)).get(atoms.get_cell()[:], neworigin = -np.mean(atoms.get_cell()[:2], axis = 0) )

    def axisrotatesym(self, atoms, rot, mult):
        #TODO: remove the self.threshold below
        #self.threshold = 0.5
        ats = atoms.copy()
        axis = atoms.get_cell().copy()
        axis_1 = np.linalg.inv(axis)
        
        _, _, c, _, _, gamma = ats.get_cell_lengths_and_angles()
        if not np.round(gamma*mult) == 360:
            if mult == 2:
                ats = ats * (2,1,1)
                ats.set_cell(axis)
                ats.translate([-ats.get_cell()[0]]*len(ats))
            else:
                scell = resetLattice(atoms = ats,expandsize= (4,4,1))
                slattice = (ats.get_cell()[:]).copy()

                #here we rotate slattice_a @mult degrees to get a new slattice_b. For sym '3', '4', '6', lattice_a must equals lattice_b.
                #The rotate matrix is borrowed from <cluster.cpp> and now I forget how to calculate it. 
                r1, r2, r3, x, y, z  = *slattice[2]/c, *slattice[0]
                cosOmega, sinOmega=math.cos(2*math.pi/mult), math.sin(2*math.pi/mult)
                slattice[1] = [x*(r1*r1*(1-cosOmega)+cosOmega)+y*(r1*r2*(1-cosOmega)-r3*sinOmega)+z*(r1*r3*(1-cosOmega)+r2*sinOmega), 
                    x*(r1*r2*(1-cosOmega)+r3*sinOmega)+y*(r2*r2*(1-cosOmega)+cosOmega)+z*(r2*r3*(1-cosOmega)-r1*sinOmega), 
                    x*(r1*r3*(1-cosOmega)-r2*sinOmega)+y*(r2*r3*(1-cosOmega)+r1*sinOmega)+z*(r3*r3*(1-cosOmega)+cosOmega) ]

                ats = scell.get(slattice)
                
                #print(ats.get_cell_lengths_and_angles())
        rats = ats.copy()
        #ase.io.write('rats.vasp', rats, format = 'vasp', vasp5=1)
        index = [i for i in range(len(ats)) if sqrt(np.sum([x**2 for x in ats[i].position])) < 2* self.threshold* covalent_radii[ats[i].number]]
        if len(index):
            del ats[index]

        for i in range(mult-1):
            newats = ats.copy()
            newats.set_positions([np.dot(np.dot(rot, p), axis) for p in np.dot(newats.get_positions(), axis_1)])
            rats += newats
            ats = newats.copy()
            """
            outatoms = rats.copy()
            outatoms.set_cell(outatoms.get_cell()[:]*3)
            outatoms.translate(-np.mean(outatoms.get_cell()[:], axis = 0))
            ase.io.write('rats{}.vasp'.format(i), outatoms, format = 'vasp', vasp5=1)
            """
        return resetLattice(atoms=rats, expandsize=(1,1,1)).get(atoms.get_cell()[:], neworigin = -np.mean(atoms.get_cell()[:2], axis = 0) )


    def mutate(self, ind):
        self.threshold = ind.p.dRatio
        """
        re_shape the layer according to its substrate symmetry. 
        For z_axis independent '2', 'm', '4', '3', '6' symmetry only.
        """
        substrate_sym = ind.substrate_sym(symprec = self.symprec)
        r, trans, mult = substrate_sym[np.random.choice(len(substrate_sym))]
        atoms = ind.atoms.copy()
        atoms.translate([-trans] * len(atoms))
        atoms.wrap()

        if mult == 'm':
            atoms = self.mirrorsym(atoms, r)
        else:
            atoms = self.axisrotatesym(atoms, r, mult)
        
        atoms.translate([trans] *len(atoms))
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

        if ind.p.molDetector != 0:
            #atoms = ind.molCryst
            atoms = copy.deepcopy(ind.molCryst)

        scl_pos = atoms.get_scaled_positions()
        axis = list(range(3))
        np.random.shuffle(axis)

        scl_pos[:, axis[0]] += self.rho*\
            np.cos(2*np.pi*self.mu*scl_pos[:,axis[1]] + np.random.uniform(0,2*np.pi,len(atoms)))*\
            np.cos(2*np.pi*self.eta*scl_pos[:,axis[2]] + np.random.uniform(0,2*np.pi,len(atoms)))

        atoms.set_scaled_positions(scl_pos)

        return ind(atoms)

class RotateMutation(Mutation):
    def __init__(self, p=1,tryNum=10):
        self.p = p
        super().__init__(tryNum=tryNum)

    def mutate(self,ind):
        atoms = ind.atoms.copy()
        # atoms = Molfilter(atoms)
        #atoms = ind.molCryst
        atoms = copy.deepcopy(ind.molCryst)
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

class RattleMutation(Mutation):
    """Class to perform rattle mutations on structures.
    Modified from GOFEE

    Rattles a number of randomly selected atoms within a sphere 
    of radius 'rattle_range' of their original positions.
    
    Parameters:
    
    p: float
        possibility of rattle

    rattle_range: float
        The maximum distance within witch to rattle the
        atoms. Atoms are rattled uniformly within a sphere of this
        radius.  
    """
    def __init__(self, p=0.5, rattle_range=3, dRatio=0.7,tryNum=10):
        self.p = p
        self.rattle_range = rattle_range
        self.dRatio = dRatio
        super().__init__(tryNum=tryNum)

    def mutate(self, ind):
        """ Rattles atoms one at a time within a sphere of radius self.rattle_range.
        """
        atoms = ind.atoms.copy()
        Natoms = len(atoms)
        for i,atom in enumerate(atoms):
            if np.random.rand() < self.p:
                newatoms = atoms.copy()
                del newatoms[i]
                pos,symbol = atoms[i].position,atoms[i].symbol
                for _ in range(200):
                    r = self.rattle_range * np.random.rand()**(1/3)
                    theta = np.random.uniform(0,np.pi)
                    phi = np.random.uniform(0,2*np.pi)
                    newpos = pos + r*np.array([np.sin(theta)*np.cos(phi), 
                                          np.sin(theta)*np.sin(phi),
                                          np.cos(theta)])
                    if check_new_atom_dist(newatoms, pos, symbol, self.dRatio):
                        atoms[i].position = newpos
                        break
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

        if ind1.p.molDetector != 0:
            atoms1 = copy.deepcopy(ind1.molCryst)
            atoms2 = copy.deepcopy(ind2.molCryst)
            cutAtoms = Molfilter(cutAtoms, ind1.p.molDetector, ind1.p.bondRatio)


        scaled_positions = []
        cutPos = [0, 0.5+self.cutDisp*np.random.uniform(-0.5, 0.5), 1]
        axis = np.random.choice([0,1,2])

        for n, atoms in enumerate([atoms1, atoms2]):
            spositions = atoms.get_scaled_positions()
            for i,atom in enumerate(atoms):
                if cutPos[n] <= spositions[i, axis] < cutPos[n+1]:
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

    def clustering(self, clusterNum):
        Pop = self.Pop
        labels,_ = Pop.clustering(clusterNum)
        uqLabels = list(sorted(np.unique(labels)))
        subpops = []
        for label in uqLabels:
            subpop = [ind for j,ind in enumerate(Pop.pop) if labels[j] == label]
            subpops.append(subpop)

        self.uqLabels = uqLabels
        self.subpops = subpops
    def get_pairs(self, Pop, crossNum ,clusterNum, tryNum=50,k=0.3):
        pairs = []
        labels,_ = Pop.clustering(clusterNum)
        fail = 0
        while len(pairs) < crossNum and fail < tryNum:
            #label = np.random.choice(self.uqLabels)
            #subpop = self.subpops[label]
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
        #Pop = self.Pop
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

        #remove bulk_layer and relaxable_layer before crossover and mutation
        if self.p.calcType=='rcs':
            Pop = Pop.copy()
            Pop.removebulk_relaxable_vacuum()
        if self.p.calcType=='clus':
            Pop.randrotate()
        if self.p.calcType == 'var':
            Pop.check_full()
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
                    #if self.p.molDetector != 0 and not hasattr(newind, 'molCryst'):
                    if self.p.molDetector != 0:
                        if not hasattr(ind, 'molCryst'):
                            ind.to_mol()
                    newind = op.get_new_individual(ind, chkMol=self.p.chkMol)
                    if newind:
                        newPop.append(newind)
            elif op.optype == 'Crossover':
                cross_pairs = self.get_pairs(Pop,num,saveGood)
                #cross_pairs = self.get_pairs(Pop,num)
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
        if self.p.calcType=='rcs':
            newPop.addbulk_relaxable_vacuum()
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

    oplist = [soft,cutandsplice,lattice,perm,ripple,slip]
    numlist = [10,10,10,10,10,10]
    popgen = PopGenerator(numlist,oplist,p)
    newpop = popgen.next_pop(pop)
    newpop.save('new.traj')
