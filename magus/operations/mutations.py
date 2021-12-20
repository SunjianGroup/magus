import math, copy
import numpy as np
from spglib import get_symmetry_dataset
from collections import Counter
from ase import Atom 
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.data import covalent_radii,chemical_symbols
from magus.utils import *
from magus.populations.individuals import to_target_formula
from .base import Mutation


__all__ = [
    'SoftMutation', 'PermMutation', 'LatticeMutation', 'RippleMutation', 'SlipMutation',
    'RotateMutation', 'RattleMutation', 'FormulaMutation', 
    'LyrSlipMutation', 'LyrSymMutation', 'ShellMutation', 'CluSymMutation',
    ]


class SoftMutation(Mutation):
    """
    Mutates the structure by displacing it along the lowest (nonzero)
    frequency modes found by vibrational analysis, as in:

    * `Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632`__

      __ https://dx.doi.org/10.1016/j.cpc.2010.06.007

    """
    parm = {'tryNum':10 ,'bounds': [0.5,2.0]}
    def __init__(self, calculator = None, **kwargs):
        self.calc = calculator
        super().__init__(**kwargs)

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
    """
    frac_swaps -- max ratio of atoms exchange
    """
    Default = {'tryNum': 50, 'frac_swaps': 0.5}

    def mutate_bulk(self, ind):
        atoms = ind.for_heredity()
        num_swaps = np.random.randint(1, min(int(self.frac_swaps * len(atoms)), 2))
        unique_symbols = np.unique([atom.symbol for atom in atoms]) # or use get_chemical_symbol?
        if len(unique_symbols) < 2:
            return None
        for _ in range(num_swaps):
            s1, s2 = np.random.choice(unique_symbols, 2, replace=False)
            s1_list = [i for i in range(len(atoms)) if atoms[i].symbol == s1]
            s2_list = [i for i in range(len(atoms)) if atoms[i].symbol == s2]
            i = np.random.choice(s1_list)
            j = np.random.choice(s2_list)
            atoms[i].position, atoms[j].position = atoms[j].position, atoms[i].position
        return ind.__class__(atoms)


class LatticeMutation(Mutation):
    """
    sigma: Gauss distribution standard deviation
    cell_cut: coefficient of gauss distribution in cell mutation
    keep_volume: whether to keep the volume unchange
    """
    Default = {'tryNum': 50, 'sigma': 0.1, 'cell_cut': 1, 'keep_volume': True}

    def mutate_bulk(self, ind):
        atoms = ind.for_heredity()
        strain = np.clip(np.random.normal(0, self.sigma, 6), -self.sigma, self.sigma) * self.cell_cut
        strain = np.array([
            [1 + strain[0], strain[1] / 2, strain[2] / 2],
            [strain[1] / 2, 1 + strain[3], strain[4] / 2],
            [strain[2] / 2, strain[4] / 2, 1 + strain[5]],
            ])
        new_cell = ind.get_cell() @ strain
        if self.keep_volume:
            ratio = ind.get_volume() / np.abs(np.linalg.det(new_cell))
            cellpar = cell_to_cellpar(new_cell)
            cellpar[:3] = [length * ratio ** (1/3) for length in cellpar[:3]]
            new_cell = cellpar_to_cell(cellpar)

        atoms.set_cell(new_cell, scale_atoms=True)
        positions = atoms.get_positions() + np.random.normal(0, 1, [len(atoms), 3])
        atoms.set_positions(positions)
        return ind.__class__(atoms)


class SlipMutation(Mutation):
    Default = {'tryNum':50, 'cut': 0.5, 'randRange': [0.5, 2]}

    def mutate_bulk(self, ind):
        atoms = ind.for_heredity()
        scl_pos = atoms.get_scaled_positions()
        axis = list(range(3))
        np.random.shuffle(axis)

        z = np.where(scl_pos[:, axis[0]] > self.cut)
        scl_pos[z,axis[1]] += np.random.uniform(*self.randRange)
        scl_pos[z,axis[2]] += np.random.uniform(*self.randRange)
        atoms.set_scaled_positions(scl_pos)
        return ind.__class__(atoms)


##################################
# XtalOpt: An open-source evolutionary algorithm for crystal structure prediction. 
#   Computer Physics Communications 182, 372–387 (2011).
##################################
class RippleMutation(Mutation):

    Default = {'tryNum': 50, 'rho': 0.3, 'mu': 2, 'eta': 1}

    def mutate_bulk(self, ind):
        atoms = ind.for_heredity()
        scl_pos = atoms.get_scaled_positions()
        axis = list(range(3))
        np.random.shuffle(axis)

        phase1 = np.cos(2 * np.pi * self.mu  * scl_pos[:, axis[1]] + np.random.uniform(0, 2 * np.pi))
        phase2 = np.cos(2 * np.pi * self.eta * scl_pos[:, axis[2]] + np.random.uniform(0, 2 * np.pi))
        scl_pos[:, axis[0]] += self.rho * phase1 * phase2

        atoms.set_scaled_positions(scl_pos)
        return ind.__class__(atoms)


class RotateMutation(Mutation):
    Default = {'tryNum': 50, 'p': 1}

    def mutate_bulk(self, ind):
        assert ind.mol_detector > 0
        atoms = ind.for_heredity()
        for mol in atoms:
            if len(mol) > 1 and np.random.rand() < self.p:
                phi, theta, psi = np.random.uniform(-1, 1, 3) * np.pi * 2
                mol.rotate(phi, theta, psi)
        return ind.__class__(atoms)


# TODO: how to apply in mol
class FormulaMutation(Mutation):
    Default = {'tryNum': 10, 'n_candidate': 5}

    def mutate(self, ind):
        candidate = ind.get_target_formula(n=self.n_candidate)
        if len(candidate) > 1:
            target_formula = candidate[np.random.randint(1, self.n_candidate)]
            atoms = to_target_formula(ind, target_formula, ind.distance_dict)
            if len(atoms) > 0:
                return ind.__class__(atoms)


# TODO: keep symmetry
class RattleMutation(Mutation):
    """
    Rattles atoms one at a time within a sphere of radius self.rattle_range.
    p: possibility of rattle
    rattle_range: The maximum distance within witch to rattle the atoms. 
                  Atoms are rattled uniformly within a sphere of this radius.  
    """
    Default = {'tryNum':50, 'p': 0.25, 'rattle_range': 4, 'dRatio':0.7, 'keep_sym': False, 'symprec': 1e-1}

    #random movement around pos.
    def rattle(self, pos):
        r = self.rattle_range * np.random.rand()**(1/3)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        newpos = pos + r * np.array([np.sin(theta) * np.cos(phi), 
                                     np.sin(theta) * np.sin(phi),
                                     np.cos(theta)])
        return newpos

    def mutate_normal(self, ind):
        atoms = ind.for_heredity()
        for i in range(len(atoms)):
            if np.random.rand() < self.p:
                atoms[i].position = self.rattle(atoms[i].position)
        return ind.__class__(atoms)

    def mutate_bulk(self, ind):
        ind = self.mutate_normal(ind) if not self.keep_sym else self.mutate_sym(ind)
        return ind

    def mutate_sym(self, atoms):
        sym_ds = get_symmetry_dataset(atoms, self.symprec)
        eq_at = sym_ds['equivalent_atoms']
        
        for key in list(Counter(eq_at).keys()):
            eq = np.where(eq_at == key)[0]

            if np.random.rand() < 1 - (1-self.p)**len(eq):
                newatoms = atoms.copy()
                del newatoms[eq]
                pos,symbol = atoms[key].position,atoms[key].symbol
                for _ in range(200):
                    _p_ = self.rattle(pos)
                    
                    newpos = np.dot(_p_, np.linalg.inv(atoms.get_cell()))
                    newpos = np.dot(sym_ds['rotations'], newpos) + sym_ds['translations']
    
                    for i, _ in enumerate(newpos):
                        for j, _ in enumerate(np.where(atoms.get_pbc())[0]):
                            newpos[i][j] += -int(newpos[i][j]) if newpos[i][j] >= 0 else -int(newpos[i][j]) +1

                    newpos = np.dot(newpos, atoms.get_cell())

                    for _pos_ in newpos:
                        if check_new_atom_dist(newatoms, _pos_, symbol, self.dRatio):
                            newatoms += Atom(position=_pos_, symbol=symbol)
                        else:
                            break
                    else:
                        for i, index in enumerate(eq):
                            atoms[index].position = newpos[i]
                        else:
                            for j in range(i, len(newpos)):
                                atoms += Atom(position=newpos[j], symbol=symbol)
                            #It is only a temp solution. Relying on ind.merge() to solve the following problem.
                        break

        return atoms

# from .reconstruct import weightenCluster
class ShellMutation(Mutation):
    """
    Original proposed by Lepeshkin et al. in J. Phys. Chem. Lett. 2019, 10, 102−106
    Mutation (6)/(7), aiming to add/remove atom i of a cluster with probability pi proportional to maxi∈s[Oi]−Oi,
    def Exp_j = exp(-(r_ij-R_i-R_j)/d); Oi = sum_j (Exp_j) / max_j(Exp_j)
    d is the empirically determined parameter set to be 0.23.
    """
    parm = {'tryNum':10, 'd':0.23}
    
    def mutate(self,ind, addatom = True, addfrml = None):
        
        atoms = ind.atoms.copy()
        i = weightenCluster(self.d).choseAtom(ind)
        
        if not addatom:
            del atoms[i]
        else:
            if addfrml is None:
                addfrml = {atoms[0].number: 1}

            for _ in range(self.tryNum):
                if addfrml:
                    #borrowed from Individual.repair_atoms
                    atomnum = list(addfrml.keys())[0]
                    basicR = covalent_radii[atoms[i].number] + covalent_radii[atomnum]
                    # random position in spherical coordination
                    radius = basicR * (ind.p.dRatio + np.random.uniform(0,0.3))
                    theta = np.random.uniform(0,np.pi)
                    phi = np.random.uniform(0,2*np.pi)
                    pos = atoms[i].position + radius*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),np.cos(theta)])
                    
                    atoms.append(Atom(symbol = atomnum, position=pos))
                    
                    for jth in range(len(atoms)-1):
                        if atoms.get_distance(len(atoms)-1, jth) < ind.p.dRatio * basicR:
                            del atoms[-1]
                            break 
                    else:
                        addfrml[atomnum] -=1
                        if addfrml[atomnum] == 0 :
                            del addfrml[atomnum]
                else:
                    break

        return ind(atoms)


class LyrSlipMutation(Mutation):
    parm = {'tryNum':10, 'cut':0.2, 'randRange':[0, 1]}

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


# from magus.reconstruct import resetLattice
class LyrSymMutation(Mutation):
    parm = {'tryNum':10, 'symprec': 1e-4}
    
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
        atoms.translate([-np.dot(trans, atoms.get_cell())] * len(atoms))
        atoms.wrap()

        if mult == 'm':
            atoms = self.mirrorsym(atoms, r)
        else:
            atoms = self.axisrotatesym(atoms, r, mult)
        
        atoms.translate([np.dot(trans, atoms.get_cell())] * len(atoms))
        return ind(atoms)


class CluSymMutation(LyrSymMutation):
    """
    maybe it is not a good mutation schedule but it was widely used in earlier papers for cluster prediction, such as
        Rata et al, Phys. Rev. Lett. 85, 546 (2000) 'piece reflection'
        Schönborn et al, j. chem. phys 130, 144108 (2009) 'twinning mutation' 
    I put it here for it is very easy to implement with codes we have now.
    And since population is randrotated before mutation, maybe it doesnot matter if 'm' and '2'_axis is specified.  
    """

    def mutate(self, ind):

        self.threshold = ind.p.dRatio
        COU = np.array([0.5, 0.5, 0])
        sym = [(np.array([[-1,0,0], [0,-1,0], [0,0,1]]), 2), (np.array([[1,0,0], [0,-1,0], [0,0,1]]), 'm')] 
        r, mult = sym[np.random.choice([0,1])]

        atoms = ind.atoms.copy()
        atoms.translate([-np.dot(COU, atoms.get_cell())] * len(atoms))
        atoms.set_pbc(True)
        atoms.wrap()

        if mult == 'm':
            atoms = self.mirrorsym(atoms, r)
        else:
            atoms = self.axisrotatesym(atoms, r, mult)
        
        atoms.wrap()
        
        return ind(atoms)

