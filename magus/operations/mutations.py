import copy
import numpy as np
from collections import Counter
import ase
from ase import Atom 
from magus.utils import *
from magus.populations.individuals import to_target_formula
from .base import Mutation
from ase.ga.soft_mutation import BondElectroNegativityModel


__all__ = [
    'SoftMutation', 'PermMutation', 'LatticeMutation', 'RippleMutation', 'SlipMutation',
    'RotateMutation', 'RattleMutation', 'FormulaMutation', 
    ]


class SoftMutation(Mutation):

    """
    Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632
    https://dx.doi.org/10.1016/j.cpc.2010.06.007

    """
    Default = {'tryNum': 50, 'bounds': [0.5,2.0]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_hessian(self, atoms, dx):
        N = len(atoms)
        pos = atoms.get_positions()
        hessian = np.zeros([3*N,3*N])
        for i in range(N):
            for j in range(3):
                pos_ = np.copy(pos)
                pos_[i, j] += dx
                atoms.set_positions(pos_)
                f1 = atoms.get_forces().flatten()

                pos_[i, j] -= 2 * dx
                atoms.set_positions(pos_)
                f2 = atoms.get_forces().flatten()
                hessian[3 * i + j] = (f1 - f2) / (2 * dx)
        atoms.set_positions(pos)
        hessian = -0.5*(hessian + hessian.T)
        return hessian

    def _get_modes(self, atoms, dx=0.02, k=2):
        hessian = self._get_hessian(atoms, dx)
        eigvals, eigvecs = np.linalg.eigh(hessian)
        modes = {eigval: eigvecs[:, i] for i, eigval in enumerate(eigvals)}
        keys = np.array(sorted(modes))
        ekeys = np.e**(-k * keys)
        ekeys[:3] = 0
        p = ekeys / np.sum(ekeys)
        key = np.random.choice(keys, p=p)
        mode = modes[key].reshape(-1, 3)
        return mode

    def mutate_bulk(self, ind):
        atoms = ind.for_heredity()
        atoms.set_calculator(BondElectroNegativityModel(atoms))
        pos = atoms.get_positions()
        mode = self._get_modes(atoms)
        largest_norm = np.max(np.apply_along_axis(np.linalg.norm, 1, mode))
        amplitude = np.random.uniform(*self.bounds) / largest_norm
        direction = np.random.choice([-1, 1])
        pos_new = pos + direction * amplitude * mode
        atoms.set_positions(pos_new)
        atoms.wrap()
        return ind.__class__(atoms)


class PermMutation(Mutation):
    """
    frac_swaps -- max ratio of atoms exchange
    """
    Default = {'tryNum': 50, 'frac_swaps': 0.5}

    @staticmethod
    def permutate(atoms, num_swaps, unique_symbols):
        for _ in range(num_swaps):
            s1, s2 = np.random.choice(unique_symbols, 2, replace=False)
            s1_list = [i for i in range(len(atoms)) if atoms[i].symbol == s1]
            s2_list = [i for i in range(len(atoms)) if atoms[i].symbol == s2]
            i = np.random.choice(s1_list)
            j = np.random.choice(s2_list)
            atoms[i].position, atoms[j].position = atoms[j].position, atoms[i].position
        return atoms

    def mutate_bulk(self, ind):
        atoms = ind.for_heredity()
        num_swaps = np.random.randint(1, max(int(self.frac_swaps * len(atoms)), 2))
        unique_symbols = np.unique([atom.symbol for atom in atoms]) # or use get_chemical_symbol?
        if len(unique_symbols) < 2:
            return None
        atoms = self.permutate(atoms, num_swaps, unique_symbols)
        return ind.__class__(atoms)

    def mutate_layer(self, ind):
        atoms = ind.for_heredity()
        num_swaps = np.random.randint(1, max(int(self.frac_swaps * len(atoms)), 2))
        unique_symbols = np.unique([atom.symbol for atom in atoms]) # or use get_chemical_symbol?
        if len(unique_symbols) < 2:
            return None
        atoms = self.permutate(atoms, num_swaps, unique_symbols)
        atoms = ind.add_vacuum(atoms, ind.vacuum_thickness)
        return atoms

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
        if self.keep_volume:
            strain /= np.linalg.det(strain)
        new_cell = atoms.get_cell() @ strain
        atoms.set_cell(new_cell, scale_atoms=True)
        # positions = atoms.get_positions() + np.random.normal(0, 1, [len(atoms), 3])
        # atoms.set_positions(positions)
        return ind.__class__(atoms)

    def mutate_layer(self, ind):
        atoms = ind.for_heredity()
        strain = np.clip(np.random.normal(0, self.sigma, 6), -self.sigma, self.sigma) * self.cell_cut
        strain = np.array([
            [1 + strain[0], strain[1] / 2, 0],
            [strain[1] / 2, 1 + strain[3], 0],
            [            0,             0, 1],
            ])
        if self.keep_volume:
            strain /= np.linalg.det(strain)
        new_cell = atoms.get_cell() @ strain
        atoms.set_cell(new_cell, scale_atoms=True)
        atoms = ind.add_vacuum(atoms, ind.vacuum_thickness)
        return atoms


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

    def mutate_layer(self, ind):
        atoms = ind.for_heredity()
        scl_pos = atoms.get_scaled_positions()
        axis = list(range(2))
        np.random.shuffle(axis)

        z = np.where(scl_pos[:, axis[0]] > self.cut)
        scl_pos[z, axis[1]] += np.random.uniform(*self.randRange)
        atoms.set_scaled_positions(scl_pos)
        atoms = ind.add_vacuum(atoms, ind.vacuum_thickness)
        return atoms


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
    
    def mutate_layer(self, ind):
        atoms = ind.for_heredity()
        scl_pos = atoms.get_scaled_positions()
        axis = list(range(2))
        np.random.shuffle(axis)

        phase = np.cos(2 * np.pi * self.mu  * scl_pos[:, axis[1]] + np.random.uniform(0, 2 * np.pi))
        scl_pos[:, axis[0]] += self.rho * phase

        atoms = ind.add_vacuum(atoms, ind.vacuum_thickness)
        return atoms


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
            target_formula = candidate[np.random.randint(1, len(candidate))]
            atoms = to_target_formula(ind, target_formula, ind.distance_dict)
            if len(atoms) > 0:
                return ind.__class__(atoms)

#random movement around origin positions of inds.
class RattleMutation(Mutation):
    """
    Rattles atoms one at a time within a sphere of radius self.rattle_range.
    p: possibility of rattle
    rattle_range: The maximum distance within witch to rattle the atoms. 
                  Atoms are rattled uniformly within a sphere of this radius.  
    """
    Default = {'tryNum':50, 'p': 0.25, 'rattle_range': 1.0, 'd_ratio':0.7, 'keep_sym': None, 'symprec': 1e-1}

    @staticmethod
    def rattle(atoms, indexs, movemodes):
        for index, movemode in zip(indexs, movemodes):
            r, theta, phi = movemode
            atoms[index].position += r * np.array([
                np.sin(theta) * np.cos(phi), 
                np.sin(theta) * np.sin(phi),
                np.cos(theta)])

        return atoms

    def mutate_p1(self, ind):

        atoms = ind.for_heredity()

        indexs = [i for i in range(len(atoms)) if np.random.rand() < self.p]
        movemodes = [ [self.rattle_range * np.random.rand()**(1/3),
                                np.random.uniform(0, np.pi),
                                np.random.uniform(0, 2*np.pi)
                                ]
                                for _ in indexs]

        atoms = self.rattle(atoms, indexs, movemodes)
        return ind.__class__(atoms)

    def mutate_layer(self, ind):
        atoms = ind.for_heredity()

        indexs = [i for i in range(len(atoms)) if np.random.rand() < self.p]
        movemodes = [ [self.rattle_range * np.random.rand()**(1/3),
                                np.pi /2,
                                np.random.uniform(0, 2*np.pi)
                                ]
                                for _ in indexs]

        atoms = self.rattle(atoms, indexs, movemodes)
        atoms = ind.add_vacuum(atoms, ind.vacuum_thickness)
        return atoms

    def mutate_sym(self, ind):
        """
        Mutation that keeps symmetry. Three methods are considered in
            Xuecheng Shao, et al, J. Chem. Phys. 156, 014105 (2022),
        Namely:
            [i] mutate spacegroup
            [ii] keep spg and mutate combinations
            [iii] mutate and keep spg and combinations
        
        HERE [ii] and [iii] are implied. (For purpose[i], does generating a new structure is ok?)
        """
        from ..reconstruct.utils import sym_rattle
        
        atoms = ind.for_heredity()

        method = self.keep_sym if self.keep_sym in ['keep_spg', 'keep_comb'] else \
                                        np.random.choice(['keep_spg', 'keep_comb'])    
        
        mutate_sym_ = getattr(sym_rattle, method)
        new_atoms = mutate_sym_(atoms, symprec = self.symprec, trynum = self.tryNum,
                                mutate_rate = self.p, rattle_range = self.rattle_range, d_ratio = self.d_ratio)
        return ind.__class__(new_atoms)

    def mutate_bulk(self, ind):
        ind = self.mutate_p1(ind) if (self.keep_sym is None) else self.mutate_sym(ind)
        return ind


