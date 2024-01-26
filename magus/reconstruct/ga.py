from ..operations.base import Mutation
from .utils import weightenCluster, resetLattice
from ase.data import covalent_radii
import math
import numpy as np
from ase import Atom, Atoms
import logging
from itertools import combinations
import ase.io

log = logging.getLogger(__name__)

__all__ = ['ShellMutation', 'SymMutation']

#class CutAndSplicePairing
def cross_surface_cutandsplice(instance, ind1, ind2):
    cut_disp = instance.cut_disp

    axis = np.random.choice([0, 1])
    atoms1 = ind1.for_heredity()
    atoms2 = ind2.for_heredity()
    atoms1.set_scaled_positions(np.modf(atoms1.get_scaled_positions() + np.random.rand(3))[0])
    atoms2.set_scaled_positions(np.modf(atoms2.get_scaled_positions() + np.random.rand(3))[0])

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
        return self.mutate_surface(ind, addatom = addatom, addfrml = addfrml)

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

import spglib
from ase.ga.ofp_comparator import OFPComparator

class SymMutation(Mutation):
    Default = {'tryNum':50, 'symprec': 1e-4}

    @staticmethod
    def __fix_hex_cell_to_tri_form(multiplity, atoms, evats = None):
        #fix the acquired cell rotated by '3' or '6' symmetry.
        
        #WTF '3' cannot be fixed this way, the symmetry will be broken.
        #'6' is not tested because DDL is close and my testing system luckily have no 6 symmetry. 
        #My dear fellows if you want to use this function pls test it first! GOOD LUCK ;)
        
        fixatoms = Atoms(cell = atoms.get_cell(), pbc = 1)
        fixevats = []

        #if not multiplity in [3, 6]:
        #    return fixatoms, fixevats

        return fixatoms, fixevats

        if evats is None:
            evats = list(range(0, len(atoms)))
        """
        the main idea is: I have a cell part, \2/3\ for example, and cellpar gamma is 120
        I rotated this cell counterclockwise 120 degrees for two times and got a hexagon
         -----
        /1\2/3\
        \4/5\6/
         -----
        Then I fix the bottom-right and top-left regions which have no atoms
        -------                     --------
        \/1\2/3\        to          \6/1\2/3\
         \4/5\6/\                     \4/5\6/1\
          --------                     --------
        """
        assert abs(atoms.cell.cellpar()[-1] - 120)<1, 'change to cell with gamma 120 before calling __fix_hex_cell_to_tri_form: {}'.format(atoms.cell.cellpar()[-1])
        
        #bottom-right equals P1, top-left equals P6
        sp = atoms.get_scaled_positions(wrap = True)
        p1_index = [i for i in range(len(atoms)) if 0.5 <= sp[i][1] < 1 and 0<= sp[i][0]< 0.5]
        p6_index = [i for i in range(len(atoms)) if 0.5 <= sp[i][0] < 1 and 0<= sp[i][1]< 0.5]
        new_sp1 = sp[p1_index] + [0.5,-0.5,0]
        new_sp6 = sp[p6_index] + [-0.5, 0.5,0]

        for p_index, new_sp in zip([p1_index, p6_index], [new_sp1, new_sp6]):
            fixevats.extend(p_index)
            new_part = atoms[p_index].copy()
            new_part.set_scaled_positions(new_sp)
            fixatoms +=  new_part
        
        fixatoms.wrap()  

        #ase.io.write('beforefix.vasp', atoms,vasp5=1)
        #ase.io.write('afterfix.vasp', atoms+fixatoms,vasp5=1)

        return fixatoms, fixevats
    
    @staticmethod
    def __get_an_atom_with_least_local_order__(atoms, multiplity, evats):
        #atoms with higher local order is favored, which means better energy
        atoms = atoms.copy()
        atoms.wrap()
        
        local_orders = OFPComparator().get_local_orders(atoms)
        
        i = 0
        while i<len(atoms):            
            at_with_least_local_order = np.argsort(local_orders)[i]

            i +=1 
            if at_with_least_local_order >= len(atoms):
                #sometimes len(local_order) > len(atoms)
                #it seems like an interesting bug and idk what caused that and i cannot fix it either lmao
                continue

            if len(np.where(np.array(evats) == evats[at_with_least_local_order])[0]) == multiplity:
                return at_with_least_local_order
            
        return -1
    
    def __share_method__(self, atoms, rot, mult, distance_dict = {}, possible_symbols_numlist = None):
         
        ats = atoms.copy()
        if ats.cell.cellpar()[-1] < 89.9:
            ats = resetLattice(atoms=ats,expandsize=(2,2,1)).get(np.dot(np.diag(-1,1,1), atoms.get_cell()[:]))

        axis = atoms.get_cell().copy()
        axis_1 = np.linalg.inv(axis)

        ats.translate([-np.sum(ats.get_cell()[:2], axis = 0)/2]*len(ats))
        sps = np.dot(ats.get_positions(), axis_1)

        func_get_part = getattr(self, 'sym_{}_part'.format(mult))
        ats = ats[func_get_part(rot, sps)]
        #ase.io.write('rat1.vasp', ats, format = 'vasp', vasp5=1)
        multiplity = int(mult) if not mult == 'm' else 2

        rats = ats.copy()
        for _ in range(0,multiplity-1):
            rpos = np.array([np.dot(np.dot(rot, p), axis) for p in np.dot(ats.get_positions(), axis_1)])
            ats.set_positions(rpos)
            rats += ats

        rats.pbc = 1
        #ase.io.write('rat2.vasp', rats, format = 'vasp', vasp5=1)
        
        Trats, evats = self.merge_evats_until_check_distance_is_true(rats, multiplity, distance_dict)
        #print('rat4', spglib.get_spacegroup(Trats, 0.1))
        #ase.io.write('rat4.vasp', Trats, vasp5=1)

        Trats = self.remove_evats_until_check_formula_is_true(Trats, evats, multiplity, possible_symbols_numlist)
        

        if Trats is None:
            #print('failed check formula')
            return None
        #print('rat5', spglib.get_spacegroup(Trats, 0.1))
        #ase.io.write('rat5.vasp',Trats, vasp5=1)   

        rats = resetLattice(atoms=Trats,expandsize=(4,4,1)).get(atoms.get_cell()[:])    #, neworigin = np.average(atoms.get_cell()[:2], axis = 0) )
        #print('fin', spglib.get_spacegroup(rats, 0.1))
        #ase.io.write('rat6.vasp',rats, vasp5=1)

        return rats 


    def merge_evats_until_check_distance_is_true(self, rats, multiplity, distance_dict = {}):
        LratsN = int(len(rats)/multiplity)
        evats = []
        for _ in range(0, multiplity):
            evats += list(range(0,LratsN))

        fixatoms, fixevats = self.__fix_hex_cell_to_tri_form(multiplity, rats, evats=evats)
        

        to_merge = [] 
        rats += fixatoms
        for x in range(0, LratsN):
            x_prime_with_x = [x + LratsN*ii for ii in range(0, multiplity)] 
            x_prime_fix = list(np.where(np.array(fixevats) == x)[0] + LratsN * multiplity)
            if np.any([rats.get_distance(xx,yy,mic=True) < distance_dict[(rats[xx].symbol, rats[yy].symbol)] for xx,yy in combinations(x_prime_with_x + x_prime_fix, 2)]):
                # merge atoms 
                rats[x].position = np.average([rats[xx].position for xx in x_prime_with_x], axis=0) 
                to_merge.extend(x_prime_with_x[1:] + x_prime_fix)
        
        #outrats = rats.copy()
        #ase.io.write('rat3.vasp', outrats, vasp5=1)
        evats.extend(fixevats)

        check_distance = False
        while check_distance == False:
            for x,y in combinations([i for i in range(0,len(rats)) if not i in to_merge], 2):
                if rats.get_distance(x,y,mic=True) < distance_dict[(rats[x].symbol, rats[y].symbol)]:       # merge atoms 
                    y_prime_with_y = np.where(np.array(evats) == evats[y])[0]
                    to_merge.extend(y_prime_with_y)

                    break
            else:
                check_distance = True
            
        Trats = rats[[x for x in range(len(rats)) if not x in to_merge]]
        evats = [value for x, value in enumerate(evats) if not x in to_merge]
        
        return Trats, evats


    def remove_evats_until_check_formula_is_true(self, Trats, evats, multiplity, possible_symbols_numlist = {}):

        for key in Trats.symbols.formula.count().keys() | possible_symbols_numlist.keys():
            if key not in Trats.symbols.formula.count().keys():
                return None
            if Trats.symbols.formula.count()[key] < np.min(possible_symbols_numlist[key]):
                return None
            while Trats.symbols.formula.count()[key] not in possible_symbols_numlist[key]:
                gomul = True
                if (Trats.symbols.formula.count()[key] - np.min(possible_symbols_numlist[key]) ) > multiplity and gomul:

                    x = self.__get_an_atom_with_least_local_order__(Trats, multiplity, evats)
                    if not x == -1:
                        x_prime_with_x = np.where(np.array(evats) ==evats[x])[0]
                        to_merge = x_prime_with_x
                    else:
                        gomul = False

                else:
                    x = self.__get_an_atom_with_least_local_order__(Trats, 1, evats)
                    if x == -1:
                        return None
                    to_merge = [x]
                Trats = Trats[[x for x in range(len(Trats)) if not x in to_merge]]
                evats = [value for x, value in enumerate(evats) if not x in to_merge]

        return Trats
    
    
    def sym_m_part(self, rot, sps):
        
        #calculate the mirror line.
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

        #if np.mean(OFPComparator().get_local_orders(ats[index1])) > np.mean(OFPComparator().get_local_orders(ats[index2])):
        #maybe choosing the part with more atoms is better? ##HIGHER LOCAL ORDER?
        
        indexes = [
            [i for i, p in enumerate(sps) if ((p[1] - k * p[0] >= 0) if not k is None else (p[0] >= 0))],
            [i for i, p in enumerate(sps) if ((p[1] - k * p[0] <= 0) if not k is None else (p[0] <= 0))]
        ]
        
        return indexes[np.argmax([len(index) for index in indexes])]
        
        
    def sym_2_part(self, rot, sps):

        indexes = [
            [i for i, p in enumerate(sps) if p[1]>= 0],
            [i for i, p in enumerate(sps) if p[1]<= 0],
            [i for i, p in enumerate(sps) if p[0]<= 0],
            [i for i, p in enumerate(sps) if p[0]<= 0]
        ]
        return indexes[np.argmax([len(index) for index in indexes])]
    
    def sym_4_part(self, rot, sps):

        indexes = [
            [i for i, p in enumerate(sps) if p[0]>= 0 and p[1]>=0],
            [i for i, p in enumerate(sps) if p[0]>= 0 and p[1]<=0],
            [i for i, p in enumerate(sps) if p[0]<= 0 and p[1]>=0],
            [i for i, p in enumerate(sps) if p[0]<= 0 and p[1]<=0]
        ]
        return indexes[np.argmax([len(index) for index in indexes])]


    def sym_3_part(self, rot, sps):
        indexes = [
            [i for i, p in enumerate(sps) if p[0]>= 0 and p[1]>=0],
            [i for i, p in enumerate(sps) if p[0]<= 0 and p[0]<=p[1]<=p[0]+0.5],
            [i for i, p in enumerate(sps) if p[1]<= 0 and p[0]-0.5<=p[1]<=p[0]],
        ]
        return indexes[np.argmax([len(index) for index in indexes])]
    
    def sym_6_part(self, rot, sps):
        indexes = [
            [i for i, p in enumerate(sps) if 0<=p[1]<=p[0]],
            [i for i, p in enumerate(sps) if 0<=p[0]<=p[1]],
            [i for i, p in enumerate(sps) if p[1]>=0 and p[0]<= 0 and p[0]<=p[1]<=p[0]+0.5],
            [i for i, p in enumerate(sps) if p[1]<=0 and p[0]<= 0 and p[0]<=p[1]<=p[0]+0.5],
            [i for i, p in enumerate(sps) if p[0]<=0 and p[1]<= 0 and p[0]-0.5<=p[1]<=p[0]],
            [i for i, p in enumerate(sps) if p[0]>=0 and p[1]<= 0 and p[0]-0.5<=p[1]<=p[0]]
        ]
        return indexes[np.argmax([len(index) for index in indexes])]
           
    def mutate_bulk(self, ind):

        """
        re_shape the layer according to its substrate symmetry. 
        For z_axis independent '2', 'm', '4', '3', '6' symmetry only.
        """
        substrate_sym = ind.substrate_sym(symprec = self.symprec)
        #substrate_sym = [ss for ss in substrate_sym if ss[-1] == 3]
        r, trans, mult = substrate_sym[np.random.choice(len(substrate_sym))]

        trans[2] = 0.0
        atoms = ind.for_heredity()
        atoms.pbc = [True, True, False]
    
        #ase.io.write('for_heredity.vasp', atoms, vasp5=1)
        #print(r, trans, mult)
        atoms.translate([-np.dot(trans, atoms.get_cell())] * len(atoms))
        atoms.wrap()
        try:
            possible_symbols_numlist = dict(zip(ind.symbol_list, ind.symbol_numlist_pool["{},{}".format(*ind.info['size'])].T))
        except:
            possible_symbols_numlist = {s:[value] for s,value in atoms.symbols.formula.count().items()}

        atoms = self.__share_method__(atoms, r, mult, ind.distance_dict, possible_symbols_numlist)
                
        if atoms is None:
            return None
        
        atoms.translate([np.dot(trans, atoms.get_cell())] * len(atoms))
        #ase.io.write('get_heredity.vasp', atoms, vasp5=1)

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
