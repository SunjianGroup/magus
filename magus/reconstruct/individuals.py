import numpy as np
from ase.io import read
from ..populations.individuals import Individual
from ..utils import check_parameters
from ..populations.populations import Population
from ase.data import covalent_radii,atomic_numbers

from .utils import FixAtoms, modify_fixatoms, FixAtomsZ
from ase import Atoms
try:
    from ase.spacegroup.symmetrize import FixSymmetry
except:
    from ase.constraints import FixSymmetry

import logging
import ase.io
#from .molecule import Molfilter
from ase import neighborlist, Atoms
from scipy import sparse

from .fitness import ErcsFitness
from ..fitness import fit_dict
import math
import spglib


log = logging.getLogger(__name__)


class RcsPopulation(Population):
    @classmethod
    def set_parameters(cls, **parameters):
        cls.all_parameters = parameters
        Requirement = ['results_dir', 'pop_size', 'symbols', 'formula', 'units']
        Default = {'check_seed': False, 'spg_miner':{}}
        check_parameters(cls, parameters, Requirement, Default)
        
        cls.atoms_generator = parameters['atoms_generator']
        parameters['symbol_numlist_pool'] = cls.atoms_generator.symbol_numlist_pool

        if parameters['structureType'] == 'surface':
            Ind = Surface
        elif parameters['structureType'] == 'cluster':
            Ind = Cluster
        elif parameters['structureType'] == 'adclus':
            Ind = AdClus
        elif parameters['structureType'] == 'interface':
            Ind = Interface
        Ind.set_parameters(**parameters)
        cls.Ind = Ind

        rcs_fit_dict = {**fit_dict,
            'Ercs': ErcsFitness,
        }
        fitness_calculator = []
        if 'Fitness' in parameters:
            for fitness in parameters['Fitness']:
                fitness_calculator.append(rcs_fit_dict[fitness](parameters))
        elif parameters['structureType'] == 'surface' or parameters['structureType'] == 'interface':
            fitness_calculator.append(rcs_fit_dict['Ercs'](parameters))
        elif parameters['structureType'] == 'cluster' or parameters['structureType'] == 'adclus':
            fitness_calculator.append(rcs_fit_dict['Enthalpy'](parameters))

        cls.fit_calcs = fitness_calculator
        return 

    def fill_up_with_random(self, targetLen = None):
        n_random = (targetLen - len(self)) if not targetLen is None else (self.pop_size - len(self)) 
        add_frames = self.atoms_generator.generate_pop(n_random)

        for ind in add_frames:
            self.append(self.Ind(ind))

    def calc_dominators(self):
        #separate different formula
        log.debug("calculating dominators...")
        self.calc_fitness()
        domLen = len(self.pop)
        for ind1 in self.pop:
            dominators = -1 #number of individuals that dominate the current ind
            for ind2 in self.pop:
                if len(ind1) == len(ind2):
                    for key in ind1.info['fitness']:
                        if ind1.info['fitness'][key] > ind2.info['fitness'][key]:
                            break
                    else:
                        dominators += 1
                        
            ind1.info['dominators'] = dominators
            ind1.info['MOGArank'] = dominators + 1
            ind1.info['sclDom'] = (dominators)/domLen


def get_newcell(atoms, thickness):
    # we multiply c by a ratio so the thickness of vacuum is fixed for non orthorhombic cell
    ratio = thickness * np.linalg.norm(atoms.cell.reciprocal()[2]) + 1
    newcell = atoms.get_cell()
    newcell[2] *= ratio
    return newcell

def add_extra_layer(atoms, layer):
    """
    add extra layer to atoms
    atoms is always at the end
    """
    thickness = 1 / np.linalg.norm(atoms.cell.reciprocal()[2])
    n_top = len(atoms)
    newatoms = layer.copy()  # use layer as basic atoms, so the constrains will ramain

    newatoms.set_cell(get_newcell(newatoms, thickness))
    newatoms.extend(atoms)
    newatoms.positions[-n_top:, 2] += thickness
    return newatoms


def add_vacuum_layer(atoms, thickness):
    newatoms = atoms.copy()
    newatoms.set_cell(get_newcell(atoms, thickness))
    return newatoms


class Surface(Individual):
    """'slices_file'= 'Ref/layerslices.traj'
    in order of bulk, buffer(optional)"""

    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        cls.symbol_list = list(set(cls.symbol_list))
        Default = {
            'refE': None, 
            'vacuum_thickness': 10,
            'buffer': True,
            'fixbulk':True,
            'slices_file': 'Ref/layerslices.traj',
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list]
            }
        check_parameters(cls, parameters, [], Default)
        cls.slices = read(cls.slices_file, index = ':')
        
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])
        
        if cls.refE is not None:
            cls.show_refE()

    @classmethod
    def show_refE(cls):
        
        log.info("---- Default reference energy: ------")
        log.info(f"compound   : {''.join([f'{s}{n} ' for s, n in sorted(cls.refE['compound'].items())])}")
        log.info(f"compoundE  : {cls.refE['compoundE']}")
        log.info(f"substrate  : {''.join([f'{s}{n} ' for s, n in sorted(cls.refE['substrate'].items())])}")
        log.info(f"substrateE : {cls.refE['substrateE']}")
        log.info(f"adEs       :")
        for vkey, vvalue in cls.refE['adEs'].items():
            log.info(f"             {vkey.ljust(2)}: {vvalue}")
        log.info("-------------------------------------")

    def __init__(self, *args, **kwargs):
        """???
        if 'symbols' in kwargs:
            if isinstance(kwargs['symbols'], Molfilter):
                kwargs['symbols'] = kwargs['symbols'].to_atoms()
        if len(args) > 0:
            if isinstance(args[0], Molfilter):
                args = list(args)
                args[0] = args[0].to_atoms()
        """
        self.full_ele = False
        super().__init__(*args, **kwargs)
        if 'check_cell' in self.check_list:
            self.check_list.remove('check_cell')
        #if 'check_spg' not in self.check_list:
        #    self.check_list.append('check_spg')
        
        if 'size' not in self.info:
            self.info['size'] = [1,1]
        self.bulk_layer = self.slices[0] * (*self.info['size'], 1)
        if self.buffer:
            self.buffer_layer = self.slices[1] * (*self.info['size'], 1)
        self.set_pbc([True, True, False])

        modify_fixatoms()

    
    def __eq__(self, obj):
        atom1 = self.for_heredity()
        obj_is_list = isinstance(obj, list)
        if obj_is_list:
            atoms2 = [_obj.for_heredity() for _obj in obj]
        else:
            atoms2 = obj.for_heredity()
        comparator_returned_value = self.comparator.looks_like(atom1, atoms2)

        # copy 'compare_info' into self

        def update_compare_info(at0, atc):
            if 'compare_info' in atc.info:
                at0.info['compare_info'] = {}
                at0.info['compare_info'].update(atc.info['compare_info'])

        update_compare_info(self, atom1)
        
        if obj_is_list:
            for i, _ in enumerate(obj):
                update_compare_info(obj[i], atoms2[i])
        else:
            if isinstance(atoms2, list):
                atoms2 = atoms2[0]
            update_compare_info(obj, atoms2)
        
        return comparator_returned_value
    


    def for_heredity(self):
        atoms = self.copy()
        if 'n_top' in atoms.info:
            atoms = self.get_top_layer(atoms)
            del atoms.info['n_top']
        return atoms

    def substrate_sym(self, symprec = 1e-4):
        
        if not hasattr(self, "substrate_symmetry"):
            if self.buffer:
                substrate = self.add_extra_layer('bulk',add=1, atoms=self.buffer_layer) 
            else:
                substrate = self.bulk_layer
            
            symdataset = spglib.get_symmetry_dataset((substrate.cell, substrate.get_scaled_positions(), substrate.numbers), symprec= symprec)
            self.substrate_symmetry = list(zip(symdataset['rotations'], symdataset['translations']))
            self.substrate_symmetry = [s for s in self.substrate_symmetry if ( (s[0]==s[0]*np.reshape([1,1,0]*2+[0]*3, (3,3))+ np.reshape([0]*8+[1], (3,3))).all()  and not (s[0]==np.eye(3)).all())]
            cell = substrate.get_cell()[:]

            #some simple tricks for '2', 'm', '4', '3', '6' symmetry. No sym_matrix containing 'z'_axis transformation are included.
            for i, s in enumerate(self.substrate_symmetry):
                r = s[0]
                sample = np.array([*np.random.uniform(1,2,2), 0])
                sample_prime = np.dot(r, sample)
                cart, cart_prime = np.dot(sample, cell), np.dot(sample_prime, cell)
                length = np.sum([i**2 for i in cart])
                #angle = math.acos(np.dot(cart, cart_prime)/ length) / math.pi * 180
                angle = math.acos(np.round(np.dot(cart, cart_prime)/ length, 4)) / math.pi * 180

                if not (np.round(angle) == np.array([180, 90, 120, 60])).any():
                    self.substrate_symmetry[i] = self.substrate_symmetry[i]+('m',)
                else:
                    self.substrate_symmetry[i] = self.substrate_symmetry[i]+(int(np.round(360.0/angle)),)

            #self.substrate_symmetry is a list of tuple (r, t, multiplity), in which multiplity = 2 for '2', 'm', others = self.rotaterank.

        return self.substrate_symmetry
    
    def add_substrate(self, atoms = None):
        ats = atoms.copy() if not atoms is None else self.copy()
        ats = self.add_extra_layer('buffer',add=1, atoms=ats) 
        ats = self.add_extra_layer('bulk',add=1, atoms=ats) 
        ats = self.add_vacuum(add=1, atoms=ats)
        return ats

    def for_calculate(self):
        """
        std_para = spglib.standardize_cell((self.cell, self.get_scaled_positions(), self.numbers), symprec=0.1, to_primitive=False)
        std_cell = Atoms(cell=std_para[0], scaled_positions=std_para[1], numbers=std_para[2], pbc = True)
        std_cell.info['size'] = self.info['size']
        std_cell = self.__class__(std_cell)
        std_cell.set_cell(self.get_cell(),scale_atoms = True)
        self = std_cell
        """
        atoms = self.copy()
        if 'n_top' not in atoms.info:
            atoms = self.add_substrate()
            #atoms.set_constraint(FixAtomsZ(indices=range(0, len(atoms))))
            atoms.info['n_top'] = len(self)   # record the n_top so extra layers can easily removed by atoms[-n_top:]
        """
        else:
            #This is purposely for reseting substrate
            self = self.for_heredity()
            self = self.add_substrate()
            atoms = self

            atoms.info['n_top'] = len(self)
        """
        return atoms

    def get_top_layer(self, atoms):
        ats = atoms.copy() 
        ats = self.set_substrate(ats, self.bulk_layer, add=-1) 
        if self.buffer:
            ats = self.set_substrate(ats, self.buffer_layer, add=-1) 
        ats.constraints = [c for c in ats.constraints if not isinstance(c, FixSymmetry)]

        ats = self.set_vacuum(ats, -1 *self.vacuum_thickness)
        return ats

    @staticmethod
    def set_substrate(atoms, substrate, add = 1):
        atoms_top = atoms.copy() 
        atoms_bottom=substrate.copy()

        newcell=atoms_top.get_cell()
        newcell[2]+=atoms_bottom.get_cell()[2] * add
        atoms_top.set_cell(newcell)

        #for interfaces: what if cell of atoms_top and atoms_bottom slightly dismatch?
        atoms_bottom.set_cell([*newcell[:2], atoms_bottom.get_cell()[2]], scale_atoms = True)

        trans=[atoms_bottom.get_cell()[2]* add]*len(atoms_top)
        atoms_top.translate(trans)

        if add == 1:
            # addextralayer
            atn = len(atoms_top)
            atoms_top+=atoms_bottom
            constraints = atoms_bottom.constraints
            for c in constraints:
                c.index += atn
            atoms_top.constraints += constraints
            #    if c in atoms_top.constraints:
            #        c.index = np.append(c.index, atoms_top.constraints.index)

            #atoms_top.set_constraint(constraints)

        else :
            # rmextralayer

            #pos=atoms_top.get_scaled_positions(wrap=False)
            #del atoms_top[[atom.index for atom in atoms_top if pos[atom.index][2]<0]]
            
            vertical_dis = atoms_top.get_scaled_positions(wrap=False)[ : , 2 ].copy()
            indices = sorted(range(len(atoms_top)), key=lambda x:vertical_dis[x])
            indices = indices[ len(atoms_bottom) :  ]
            if atoms_top.constraints:
                del atoms_top.constraints
            atoms_top = atoms_top[indices]
        
        return atoms_top

    def add_extra_layer(self, type, atoms = None, add = 1):
        
        #add: addextralayer if add = 1; else rmextralayer 
        atoms_top = atoms or self

        if type=='buffer':
            if not self.buffer:
                return atoms_top
            else:
                extratoms = self.buffer_layer
                sp = extratoms.get_scaled_positions()[:,2]
                index = [i for i in range( 0, len(extratoms) )if sp[i] <0.5]
                c = FixAtoms(indices=index, adjust_force = self.fixbulk)
                #c = FixAtoms(indices=range( 0, len(extratoms) ), adjust_force = self.fixbulk)
                extratoms.set_constraint(c)

        elif type=='bulk':
            extratoms = self.bulk_layer
            
            c = FixAtoms(indices=range( 0, len(extratoms) ), adjust_force = self.fixbulk) 
            extratoms.set_constraint(c)
        
        atoms_top = self.set_substrate(atoms_top, extratoms, add)
                
        if atoms is None:
            self = atoms_top 

        return atoms_top 

    @staticmethod
    def set_vacuum(atoms, vacuum):
        newatoms = atoms.copy()

        ratio = 1.0*vacuum/newatoms.get_cell_lengths_and_angles()[2]
        newcell = newatoms.get_cell()
        newcell[2]*=2*ratio+1
        trans=[newatoms.get_cell()[2]*ratio]*len(newatoms)

        newatoms.set_cell(newcell)
        newatoms.translate(trans)
        return newatoms

    def add_vacuum(self, add = 1, atoms = None):

        vacuum=self.vacuum_thickness*add
        newatoms = atoms or self
        newatoms = Surface.set_vacuum(newatoms, vacuum)
        
        if atoms is None:
            self = newatoms 
        return newatoms 

    def check_sym(self, atoms =None, p = 0.7):
        """
        a = atoms or self
        if spglib.get_spacegroup((a.cell, a.get_scaled_positions(), a.numbers), self.symprec) == 'P1 (1)':
            if np.random.rand() < p:
                return False
        """
        return True

    def check_formula(self, atoms=None):
        atoms = atoms or self
        if 'n_top' in atoms.info:
            return True

        symbols_numlist = np.array(self.numlist)
        for possible_symbols_numlist in self.symbol_numlist_pool["{},{}".format(*self.info['size'])]:
            if np.all(symbols_numlist == possible_symbols_numlist):
                return True
        else:
            return False

    def get_target_formula(self, n=1):
        standard = self.symbol_numlist_pool["{},{}".format(*self.info['size'])].T
        
        target_formula = {}
        for i, s in enumerate(self.symbol_list):
            #if self.symbols.formula.count(s) in standard[i]:
            #    target_formula[s] = self.symbols.formula.count()[s]
            #else:
                #seek for a most close number
                _nows = self.symbols.formula.count()[s] if s in self.symbols.formula.count() else 0
                abdifference = np.abs(np.array(standard[i]) - _nows)
                target_formula[s] = standard[i][np.argmin(abdifference)]

        return [target_formula]

    @property
    def fingerprint(self):
        if 'fingerprint' not in self.info:
            atoms = self.get_top_layer(self) if 'n_top' in self.info else self
            self.info['fingerprint'] = self.fp_calc.get_all_fingerprints(atoms)[0]
        return self.info['fingerprint']
    

from ..generators.gensym import symbols_0d
from ..populations.individuals import check_new_atom
from ase import Atom

# TODO weighten
def to_target_formula(atoms, target_formula, distance_dict, max_n_try=10): 
    symbols = atoms.get_chemical_symbols()
    toadd, toremove = {}, {}
    for s in target_formula:
        if symbols.count(s) < target_formula[s]:
            toadd[s] = target_formula[s] - symbols.count(s)
        elif symbols.count(s) > target_formula[s]:
            toremove[s] = symbols.count(s) - target_formula[s]
    rep_atoms = atoms.copy()

    #remove before add
    while toremove:

        del_symbol = np.random.choice(list(toremove.keys()))
        del_index = np.random.choice([atom.index for atom in rep_atoms if atom.symbol == del_symbol])
        if toadd:
            #if some symbols need to add, change symbol directly
            add_symbol = np.random.choice(list(toadd.keys()))
            remain_index = [i for i in range(len(rep_atoms)) if i != del_index]
            pos = rep_atoms.positions[del_index]
            if check_new_atom(rep_atoms[remain_index], pos, add_symbol, distance_dict):
                rep_atoms[del_index].symbol = add_symbol
                toadd[add_symbol] -= 1
                if toadd[add_symbol] == 0:
                    toadd.pop(add_symbol)
            else:
                del rep_atoms[del_index]
        else:
            del rep_atoms[del_index]
        toremove[del_symbol] -= 1
        if toremove[del_symbol] == 0:
            toremove.pop(del_symbol)
    while toadd:
        add_symbol = np.random.choice(list(toadd.keys()))
        for _ in range(max(max_n_try, int(len(rep_atoms)/3))):
            # select a center atoms
            mean_p = np.average(rep_atoms.positions, axis=0)
            d = np.sqrt([np.sum([x**2 for x in p-mean_p]) for p in rep_atoms.positions])
            index = np.argsort(d)[math.floor(len(d)/5):]
            if len(index) < 1:
                continue
            center_atom = rep_atoms[np.random.choice(index)]
            basic_r = distance_dict[(center_atom.symbol, add_symbol)]
            radius = basic_r * (1 + np.random.uniform(0, 0.3))
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            new_pos = center_atom.position + radius * np.array([np.sin(theta) * np.cos(phi), 
                                                                np.sin(theta) * np.sin(phi),
                                                                np.cos(theta)])
            if check_new_atom(rep_atoms, new_pos, add_symbol, distance_dict):
                rep_atoms.append(Atom(symbol=add_symbol, position=new_pos))
                toadd[add_symbol] -= 1
                if toadd[add_symbol] == 0:
                    toadd.pop(add_symbol)
                break
        else:
            return Atoms()
    return rep_atoms


class Cluster(Individual):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Default = {
            'vacuum_thickness': 10, 
            'cutoff': 1.0,
            'weighten': True,
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list]}
        check_parameters(cls, parameters, [], Default)
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])

        cls.pointgroup_symbols = {symbol:index+1 for index,symbol in enumerate(symbols_0d)}
        
    #TODO: Molecule Molfilter????
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_pbc([0,0,0])
        if not self.info['origin'] == 'seed' and not self.check_seed:
            self.check_list.append('check_connection')

    @property
    def bounding_sphere(self):
        positions = self.get_positions()
        center = np.mean(positions, axis=0)
        return math.sqrt(np.max([np.sum([x**2 for x in (p - center)]) for p in positions]))
    
    @staticmethod
    def randrotate(atoms):
        new_atoms = atoms.copy()

        theta = np.random.uniform(0,2*np.pi)
        phi = np.random.uniform(0,np.pi)
        angle = np.random.uniform(0,180)
        new_atoms.rotate(angle, v=[np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], center='COU', rotate_cell=False)
        
        return Cluster.reset_center(new_atoms)

    @staticmethod
    def reset_center(atoms, set_lattice = None):
        new_atoms = atoms.copy()

        pos = new_atoms.get_positions().copy()
        newlattice = np.zeros((3,3))
        if set_lattice is None:
            for i in range(3):
                ref = sorted(pos[:,i])
                newlattice[i][i] = ref[-1] - ref[0] + Cluster.vacuum_thickness

        else:
            newlattice = set_lattice

        trans = [np.mean(newlattice, axis = 0) - np.mean(pos, axis = 0)]*len(new_atoms)
        new_atoms.translate(trans)
        new_atoms.set_cell (newlattice)
        return new_atoms
    
    @property
    def volume_ratio(self):
        #cell = self.atoms.get_cell_lengths_and_angles()[:3] 
        #cell -= np.array([self.p.vacuum]*3)
        #self.volRatio = cell[0]*cell[1]*cell[2]/self.get_ball_volume()
        return 4/3 * math.pi * self.bounding_sphere**3 / self.ball_volume

    @staticmethod
    def _connecty_(atoms):
        cutOff = np.array(neighborlist.natural_cutoffs(atoms))*Cluster.cutoff
        neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
        neighborList.update(atoms)
        matrix = neighborList.get_connectivity_matrix()
        return sparse.csgraph.connected_components(matrix)

    def check_connection(self,atoms=None):
        atoms = atoms or self
        n_components, _ = self._connecty_(atoms)
        return False if n_components > 1 else True

    def repair_atoms(self):
        try:
            return self._repair_atoms()
        except:
            return False
    
    def _repair_atoms(self):
        n_components, component_list = self._connecty_(self)
        if n_components ==1:
            self.merge_atoms()         # merge atoms too close before repair it

            if len(self) == 0:
                log.debug("Empty crystal after merging!")
                return False
            for target_formula in self.get_target_formula():
                rep_atoms = to_target_formula(self, target_formula, self.distance_dict, self.n_repair_try)
                if len(rep_atoms) > 0:
                    self.__init__(rep_atoms)
                    self.sort()
                    return True
            else:
                return False
        else:
            log.debug("By repair_atoms: attempts to make cluster unite again!")
            oldatoms = self.copy()
            for _ in range(3):
                a = self
                originpos = a.get_positions().copy()
                originc = np.mean(originpos, axis = 0)
                for subclus in range(n_components):
                    sc = [i for i in range(len(a)) if component_list[i] == subclus]
                    randcenter = [np.random.uniform(0, self.bounding_sphere/5), np.random.uniform(0,math.pi), np.random.uniform(0,2*math.pi)]
                    randcenter = randcenter[0] * np.array([math.sin(randcenter[1])*math.cos(randcenter[2]),math.sin(randcenter[1])*math.sin(randcenter[2]),math.cos(randcenter[1])])
                    originpos[sc] += ( originc - np.mean(originpos[sc], axis = 0) + randcenter)
                a.set_positions(originpos)
                
                ith =0
                while ith < len(a)-1:
                    dises = a.get_distances(ith, range(ith+1, len(a))) 
                    to_merge = [i+ith+1 for i in range(len(dises)) if dises[i] < self.d_ratio*(covalent_radii[a[ith].number]+covalent_radii[a[i+ith+1].number])]
                    if len(to_merge):
                        lucky_index = np.random.choice(to_merge)
                        newpos = (a[ith].position + a[lucky_index].position)/2
                        a[lucky_index].position = newpos
                        del a[ith]
                    else:
                        ith +=1

                if super().repair_atoms():   #weighten=self.p.weighten):
                    return True
                else:
                    self = oldatoms.copy()
            log.debug("repair_atoms failed...")
            self = None 
            return False
    
    def find_spg(self):
        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
        symprec = self.symprec
        molecule = Molecule(self.symbols,self.get_positions())
        
        spg = PointGroupAnalyzer(molecule, symprec).sch_symbol

        if spg in symbols_0d:
            spg = self.pointgroup_symbols[spg]
        else:
            spg = 1
        
        self.info['spg'] = spg
        self.info['priNum'] = self.get_atomic_numbers()
        self.info['priVol'] = 4/3 * math.pi * self.bounding_sphere**3

    def remove_vacuum(self, atoms = None):
        ats = atoms or self
        ats = ats.copy()

        ats.positions -= self.vacuum_thickness
        ats.cell -= np.eye(3)*self.vacuum_thickness
        return ats

    def for_heredity(self):
        atoms = Cluster.randrotate(self)
        return self.remove_vacuum(atoms)

    def for_calculate(self):
        return Cluster.reset_center(self)
 
class AdClus(Cluster):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Cluster.set_parameters(**parameters)
        
        Default = {
            'substrate': 'substrate.vasp', 
            'dist_clus2surface':2, 
            'size':[1,1]
            }
        check_parameters(cls,parameters, Requirement=[], Default=Default )
        
        cls._substratefile_ = ase.io.read(cls.substrate)*(*cls.size, 1)
        c = FixAtoms(indices=range(0, len(cls._substratefile_) ))
        cls._substratefile_.set_constraint(c)

        parameters['set_lattice'] =  cls._substratefile_.get_cell()[:]
        cls.set_lattice = parameters['set_lattice']

    
    #Cell should be the exact cell of substrate! 
    @staticmethod
    def absorb(cluster):
        cluster = cluster.copy()

        ref = AdClus._substratefile_
        cluspos = cluster.get_positions()
        cluster = Cluster.reset_center(cluster, set_lattice=AdClus.set_lattice)
        cluster.set_cell(ref.get_cell())
        
        trans = [[*(np.mean(ref.get_cell()[:2], axis = 0)[:2] - np.mean(cluspos[:,:2], axis = 0)),-np.min(cluspos[:,2]) +np.max(ref.get_positions()[:, 2]) +AdClus.dist_clus2surface]]*len(cluster)
        cluster.translate(trans)
        
        cluster += ref
        return cluster

    def __init__(self, *args, **kwargs):
        """???
        if 'symbols' in kwargs:
            if isinstance(kwargs['symbols'], Molfilter):
                kwargs['symbols'] = kwargs['symbols'].to_atoms()
        if len(args) > 0:
            if isinstance(args[0], Molfilter):
                args = list(args)
                args[0] = args[0].to_atoms()
        """
        super().__init__(*args, **kwargs)

        if len(self) > len(self._substratefile_):
            #Maybe it is a weird if condition, but I think it's enough to tell if the substrate is in "atoms"
            if 'check_formula' in self.check_list:
                self.check_list.remove('check_formula')
        self.set_pbc(False)

    def get_top_layer(self, atoms):
        len_clus = len(atoms) - len(self._substratefile_)
        cluster_index = np.argpartition(self.get_scaled_positions()[:,2], -len_clus)[-len_clus:]
        return atoms[cluster_index]

    def check_connection(self, atoms=None):
        atoms = atoms or self
        atoms = atoms.copy()
        if len(atoms) > len(self._substratefile_):
            atoms = self.get_top_layer(atoms)

        return super().check_connection(atoms)

    def for_heredity(self):
        atoms = self.copy()
        if len(atoms) > len(self._substratefile_):
            atoms = self.get_top_layer(atoms)

        c = Cluster.reset_center(atoms)

        return c

    def for_calculate(self):
        atoms = self.copy()
        if not len(atoms) > len(self._substratefile_):
            atoms =  AdClus.absorb(self)
        
        return atoms

    def get_target_formula(self, n=1):
        x = [{s: np.random.choice(self.symbol_numlist_pool[i]) for i,s in enumerate(self.symbol_list)}]
        return x
    
    @property
    def fingerprint(self):
        if 'fingerprint' not in self.info:
            atoms = self.get_top_layer(self) if len(self) > len(self._substratefile_) else self
            self.info['fingerprint'] = self.fp_calc.get_all_fingerprints(atoms)[0]
        return self.info['fingerprint']
    
class Interface(Surface):
    """'slices_file'= 'Ref/layerslices.traj'"""

    @classmethod
    def set_parameters(cls, **parameters):
        Individual.set_parameters(**parameters)
        Default = {
            'refE': None, 
            'vacuum_thickness': 10,
            'slices_file': 'Ref/layerslices.traj',
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list],
            'buffer': True,
            'fixbulk':True,
            }
        check_parameters(cls, parameters, [], Default)
        cls.slices = read(cls.slices_file, index = ':')
        
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])        

    def __init__(self, *args, **kwargs):
        """???
        if 'symbols' in kwargs:
            if isinstance(kwargs['symbols'], Molfilter):
                kwargs['symbols'] = kwargs['symbols'].to_atoms()
        if len(args) > 0:
            if isinstance(args[0], Molfilter):
                args = list(args)
                args[0] = args[0].to_atoms()
        """
        self.full_ele = False
        Individual.__init__(self, *args, **kwargs)
        if 'check_cell' in self.check_list:
            self.check_list.remove('check_cell')
        self.set_pbc([True, True, False])
        modify_fixatoms()

        #I used 'size' instead of name 'trans' here for convenience or I should also change GA to keep this info. -YH
        if 'size' not in self.info:
            self.info['size'] = [0,0]
        self.bulk_layers = [self.slices[0].copy(), self.slices[1].copy()]

        self.bulk_layers[0].set_scaled_positions(self.bulk_layers[0].get_scaled_positions() - [*self.info['size'],0])
        self.bulk_layers[0].wrap()

        if self.buffer:
            self.buffer_layers = [self.slices[2].copy(), self.slices[3].copy()]
            self.buffer_layers[0].set_scaled_positions(self.buffer_layers[0].get_scaled_positions() - [*self.info['size'],0])
            self.buffer_layers[0].wrap()

        
    #add substrate in both top and bottom
    def add_substrate(self, atoms = None):
        ats = atoms.copy() if not atoms is None else self.copy()

        for i in [0,1]:
            if self.buffer:
                self.buffer_layer = self.buffer_layers[i]
            self.bulk_layer = self.bulk_layers[i]
            ats = self.add_extra_layer('buffer',add=1, atoms=ats)

            ats = self.add_extra_layer('bulk',add=1, atoms=ats)

            sp = ats.get_scaled_positions()
            sp[:,2] = 1.0-sp[:,2]
            ats.set_scaled_positions(sp)

        ats = self.add_vacuum(add=1, atoms=ats)
        return ats
    
    #remove substrate in both top and bottom
    def get_top_layer(self, atoms):
        ats = atoms.copy() 
        for i in [0,1]:
            self.bulk_layer = self.bulk_layers[i]
            ats = self.set_substrate(ats, self.bulk_layer, add=-1) 
            if self.buffer:
                self.buffer_layer = self.buffer_layers[i]
                ats = self.set_substrate(ats, self.buffer_layer, add=-1) 
            sp = ats.get_scaled_positions()
            sp[:,2] = 1.0-sp[:,2]
            ats.set_scaled_positions(sp)
            
        ats = self.set_vacuum(ats, -1 *self.vacuum_thickness)
        return ats
    
    def check(self):
        return True

