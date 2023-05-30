from ..operations.base import Mutation
from .utils import weightenCluster, resetLattice
from ase.data import covalent_radii
import math
import numpy as np
from ase import Atom, Atoms
import logging

log = logging.getLogger(__name__)

__all__ = ['ShellMutation', 'SymMutation']

#class CutAndSplicePairing
def cross_surface_cutandsplice(instance, ind1, ind2):
    cut_disp = instance.cut_disp

    axis = np.random.choice([0, 1])
    atoms1 = ind1.for_heredity()
    atoms2 = ind2.for_heredity()
    
    atoms1.set_scaled_positions(atoms1.get_scaled_positions() + np.random.rand(3))
    atoms2.set_scaled_positions(atoms2.get_scaled_positions() + np.random.rand(3))

    cut_cellpar = atoms1.get_cell()
    
    cut_atoms = atoms1.__class__(Atoms(cell=cut_cellpar, pbc=ind1.pbc))

    scaled_positions = []
    cut_position = [0, 0.5 + cut_disp * np.random.uniform(-0.5, 0.5), 1]

    for n, atoms in enumerate([atoms1, atoms2]):
        spositions = atoms.get_scaled_positions()
        for i, atom in enumerate(atoms):
            if cut_position[n] <= spositions[i, axis] < cut_position[n+1]:
                cut_atoms.append(atom)
                scaled_positions.append(spositions[i])
    if len(scaled_positions) == 0:
        return None
    
    cut_atoms.set_scaled_positions(scaled_positions)
    return ind1.__class__(cut_atoms)



class ShellMutation(Mutation):
    """
    Original proposed by Lepeshkin et al. in J. Phys. Chem. Lett. 2019, 10, 102-106
    Mutation (6)/(7), aiming to add/remove atom i of a cluster with probability pi proportional to maxi∈s[Oi]-Oi,
    def Exp_j = exp(-(r_ij-R_i-R_j)/d); Oi = sum_j (Exp_j) / max_j(Exp_j)
    d is the empirically determined parameter set to be 0.23.
    """
    Default = {'tryNum':10, 'd':0.23}
    
    def mutate_surface(self,ind, addatom = True, addfrml = None):
        
        atoms = ind.for_heredity()
        i = weightenCluster(self.d).choseAtom(atoms)
        
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
                    radius = basicR * (ind.d_ratio + np.random.uniform(0,0.3))
                    theta = np.random.uniform(0,np.pi)
                    phi = np.random.uniform(0,2*np.pi)
                    pos = atoms[i].position + radius*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi),np.cos(theta)])
                    
                    atoms.append(Atom(symbol = atomnum, position=pos))
                    
                    for jth in range(len(atoms)-1):
                        if atoms.get_distance(len(atoms)-1, jth) < ind.d_ratio * basicR:
                            del atoms[-1]
                            break 
                    else:
                        addfrml[atomnum] -=1
                        if addfrml[atomnum] == 0 :
                            del addfrml[atomnum]
                else:
                    break

        return ind.__class__(atoms)

    def mutate_cluster(self,ind, addatom = True, addfrml = None):
        return self.mutate_surface(self,ind, addatom = True, addfrml = None)

#class SlipMutation
from .utils import LayerIdentifier
def mutate_surface_slip(self, ind):
    """
    slip of one layer.
    """
    atoms = ind.for_heredity()

    layers = LayerIdentifier(atoms, prec = self.cut)
    chosenlayer = layers[np.random.choice(len(layers))]
    direction = np.random.uniform(0,2*math.pi)
    trans = [math.cos(direction), math.sin(direction),0]

    pos = atoms.get_positions().copy()
    pos[chosenlayer, :] += np.dot(np.array(trans)*np.random.uniform(*self.randRange), atoms.get_cell())

    atoms.set_positions(pos)
    atoms.wrap()
    return ind.__class__(atoms)


class SymMutation(Mutation):
    Default = {'tryNum':50, 'symprec': 1e-4}
    
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
        index = [i for i in range(len(ats)) if math.sqrt(np.sum([x**2 for x in ats[i].position])) < 2* self.threshold* covalent_radii[ats[i].number]]
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


    def mutate_bulk(self, ind):
        self.threshold = ind.d_ratio
        """
        re_shape the layer according to its substrate symmetry. 
        For z_axis independent '2', 'm', '4', '3', '6' symmetry only.
        """
        substrate_sym = ind.substrate_sym(symprec = self.symprec)
        r, trans, mult = substrate_sym[np.random.choice(len(substrate_sym))]
        atoms = ind.for_heredity()
        atoms.translate([-np.dot(trans, atoms.get_cell())] * len(atoms))
        atoms.wrap()

        if mult == 'm':
            atoms = self.mirrorsym(atoms, r)
        else:
            atoms = self.axisrotatesym(atoms, r, mult)
        
        atoms.translate([np.dot(trans, atoms.get_cell())] * len(atoms))
        return ind.__class__(atoms)


    """
    maybe it is not a good mutation schedule but it was widely used in earlier papers for cluster prediction, such as
        Rata et al, Phys. Rev. Lett. 85, 546 (2000) 'piece reflection'
        Schönborn et al, j. chem. phys 130, 144108 (2009) 'twinning mutation' 
    I put it here for it is very easy to implement with codes we have now.
    And since population is randrotated before mutation, maybe it doesnot matter if 'm' and '2'_axis is specified.  
    """

    def mutate_cluster(self, ind):

        self.threshold = ind.d_ratio
        COU = np.array([0.5, 0.5, 0])
        sym = [(np.array([[-1,0,0], [0,-1,0], [0,0,1]]), 2), (np.array([[1,0,0], [0,-1,0], [0,0,1]]), 'm')] 
        r, mult = sym[np.random.choice([0,1])]

        atoms = ind.for_heredity()
        atoms.translate([-np.dot(COU, atoms.get_cell())] * len(atoms))
        atoms.set_pbc(True)
        atoms.wrap()

        if mult == 'm':
            atoms = self.mirrorsym(atoms, r)
        else:
            atoms = self.axisrotatesym(atoms, r, mult)
        
        atoms.wrap()
        
        return ind.__class__(atoms)
    

from ..operations import remove_end

rcs_op_list = [ShellMutation, SymMutation]
rcs_op_dict = {remove_end(op.__name__): op for op in rcs_op_list}

def GA_interface():
    from ..operations.crossovers import CutAndSplicePairing
    from ..operations.mutations import SlipMutation
    setattr(CutAndSplicePairing, "cross_surface", cross_surface_cutandsplice)
    setattr(CutAndSplicePairing, "cross_cluster", cross_surface_cutandsplice)
    setattr(SlipMutation, "mutate_surface", mutate_surface_slip)
