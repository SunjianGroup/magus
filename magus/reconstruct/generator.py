from ..generators.random import SPGGenerator
import numpy as np
import math, os, ase.io
from .utils import check_distance, cutcell, match_symmetry, resetLattice
from ..utils import check_parameters
import logging
from .individuals import Surface
from ase.geometry import cell_to_cellpar,cellpar_to_cell
import spglib
import itertools


log = logging.getLogger(__name__)

class ClusterSPGGenerator(SPGGenerator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Default = {'vacuum_thickness':10}
        check_parameters(self, parameters, [], Default)
        self.dimension = 0
        self.spacegroup = [spg for spg in self.spacegroup if spg <= 56]

    #For cluster genertor, generates atom positions lies in distance (from origin) range of (minLattice[0], maxLattice[0])
    def get_min_lattice(self, numlist):
        return [3*self.d_ratio*np.mean(self.radius)]*3 + [60,60,60] if self.min_lattice[0] < 0 else self.min_lattice
        
    def get_max_lattice(self, numlist):
        mean_volume = self.get_volume(numlist)[0] *2
        mean_volume *= 0.7              #???
        max_lattice = [(mean_volume / (4/3 * math.pi))**(1.0/3)]*3 + [120,120,120] 

        return max_lattice if self.max_lattice[0] < 0 else self.max_lattice

    def get_generate_parm(self, spg, numlist):
        d = super().get_generate_parm(spg, numlist)
        d.update({
            'dimension': 0,
            'random_swap_axis': False,
            'vacuum': self.vacuum_thickness,
            })
        return d

from ..operations import RattleMutation
class randwalk:
    def __init__(self, r = 2, d_ratio = 0.5, attempts = 100, **kwargs):
        self.attempts = attempts
        self.rattle_range = r
        self.d_ratio = d_ratio
        self.base_p = 1.2        #???

    def generate(self, atoms):
        atoms = atoms.copy()

        #weight: Atoms (i)closer to 'top' surface; (ii)less in atomic number; are easier to move
        vertical_p = atoms.get_scaled_positions()[:, 2]
        vertical_p /= np.linalg.norm(vertical_p)

        mass_p = atoms.get_atomic_numbers()
        mass_p = 1 - mass_p/np.linalg.norm(mass_p)

        self.weight = [ math.sqrt(vp * mp) for vp, mp in zip(vertical_p, mass_p)]

        for _ in range(self.attempts):
            new_atoms = self.mutate(atoms)
            if check_distance(new_atoms, self.d_ratio):
                return True, new_atoms
        else:
            return False, None

    def mutate(self, atoms):
        
        indexs, movemodes = [], []

        for i, p in enumerate(self.weight):
            if np.random.rand() < p * self.base_p:
                indexs.append(i)
                movemodes.append([self.rattle_range*p, 
                                                np.random.uniform(0, np.pi),
                                                np.random.uniform(0, 2*np.pi)])
    
        return RattleMutation.rattle(atoms, indexs, movemodes)

def split_formula(modifier):
    expand = []

    for item in modifier:
        if isinstance(item, int):
            expand.append(item)
        elif isinstance(item, str):
            if '~' not in item:
                raise Exception ("wrong format of formula")
            s1, s2 = item.split('~')
            s1, s2 = int(s1), int(s2)
            expand.extend(list(range(s1, s2+1)))

    return expand

def formula_add(fA, fB):
    for _s_ in fB.keys():
        if _s_ in fA.keys():
            fA[_s_] = np.array(fA[_s_]) + np.array(fB[_s_])
        else:
            fA[_s_] = np.array(fB[_s_])
    return fA

def formula_minus(fA, fB):
    for _s_ in fB.keys():
        assert _s_ in fA.keys(), "error in formula_minus, symbol {} not in formula {}".format(_s_, fA)
        fA[_s_] = np.array(fA[_s_]) - np.array(fB[_s_])
    for _s_ in list(fA.keys()):
        fA[_s_] = np.array(fA[_s_])
        fA[_s_] = fA[_s_][np.where(fA[_s_] > 0)]
        if len(fA[_s_]) == 0:
            del fA[_s_]

    return fA

class SurfaceGenerator(SPGGenerator):
    """
    Main features of a reconstruct generator:
        #1. random walk of the surface atoms
        #2. 2d group symmetry generated structures.
    """
    @staticmethod
    def __cutcell__( bulk_file = None,
                                bulk_layernum = None, buffer_layernum = None, rcs_layernum = None,
                                cutslices = None,
                                direction = None, rotate = 0, matrix = None, extra_c =1.0, 
                                addH = False, pcell = True, rcs_x = [1], rcs_y = [1] ,
                                refDir = '', refSlab = '', slices_file = '',
                                **kwargs):
        
        for key in [bulk_file, bulk_layernum, buffer_layernum, rcs_layernum]:
            assert not (key is None), "{} is needed to build a slab".format(key)
        
        if not os.path.exists(refDir):
            os.mkdir(refDir)

        #1. split layers into [bulk, buffer, rcs]
        originatoms = ase.io.read(bulk_file)
        originatoms.pbc = True

        # in some situations, build a supercell rather than a 1x1 cell.
        xy = [1,1]
        if np.any(np.array([*rcs_x, *rcs_y])%1 > 0) or pcell == False:
            xy= [rcs_x[0], rcs_y[0]]

        lyrnums = [bulk_layernum, buffer_layernum, rcs_layernum]
        cutcell(originatoms, lyrnums, totslices = cutslices, vacuum = extra_c, addH = addH ,direction= direction,
                xy = xy, rotate = rotate, pcell = pcell ,matrix = matrix)
        
        #2. get refslab to calculate refE
        ase.io.write(refSlab, ase.io.read(bulk_file), format = 'traj')
        
        return

    def _init_lattice_(self):

        setlattice = []
        if self.buffer:
            # mode = 'reconstruct'
            self.ref_layer = self.input_layers[2]
            setlattice = self.ref_layer.get_cell_lengths_and_angles().copy()
            vertical_dis = self.ref_layer.get_scaled_positions()[:,2].copy()
            setlattice[2] *=(np.max(vertical_dis) - np.min(vertical_dis) )#+ 1.0 / setlattice[2])
        else:
            # mode = 'add atoms'
            self.randwalk_ratio = 0
            self.ref_layer = self.input_layers[1].copy()
            lattice = self.ref_layer.get_cell().copy()
            lattice [2]/= self.slabinfo['buffer_layernum']
            self.ref_layer.set_cell(lattice)
            setlattice = list(cell_to_cellpar(lattice))

        setlattice = np.round(setlattice, 3)
        setlattice[3:] = [i if np.round(i) != 60 else 120.0 for i in setlattice[3:]]
        self.symtype = 'hex' if 120 in np.round(setlattice[3:]) else 'orth'
        if self.slabinfo['pcell'] == False:
            self.symtype = 'c-cell'
            
        self.reflattice = list(setlattice).copy()

        return

    def __init__(self, **parameters):
        super().__init__(**parameters)

        #i dont think it makes sense to support user-define. change it here if u really want to. 
        refDir, refSlab, slices_file = 'Ref', 'Ref/refslab.traj', 'Ref/layerslices.traj'
        
        Default = { 
            'randwalk_range': 0.5, 
            'randwalk_ratio': 0.3,
            'rcs_x': [1], 
            'rcs_y': [1],  
            'buffer': True,
            'rcs_formula': None,
            'spg_type': 'plane',
            'molMode':False,            #???
            }

        self.slabinfo = {
            'bulk_file': None, 
            'cutslices': 2,
            'bulk_layernum': 3, 
            'buffer_layernum': 3, 
            'rcs_layernum': 2, 
            'direction': None, 
            'rotate': 0, 
            'matrix': None, 
            'extra_c': 1.0,
            'addH': False, 
            'pcell': True,
            }
        self.slabinfo.update(parameters['slabinfo'])

        self.modification = {
            'adsorb': {},
            'clean': {},
            'defect': {},
            }
        if 'modification' in parameters:
            self.modification.update(parameters['modification'])
        
        check_parameters(self, parameters, [], Default)
        
        self.dimension = 2

        #If no input_layers, cut the bulk into slices.
        self.slices_file = os.path.join(self.all_parameters['workDir'], slices_file)
        refDir, refSlab = os.path.join(self.all_parameters['workDir'], refDir), os.path.join(self.all_parameters['workDir'], refSlab)
        
        if os.path.exists(refDir) and os.path.exists(refSlab) and os.path.exists(slices_file):
            log.info("Used layerslices in {}.".format(refDir))
        else:
            self.__cutcell__(**self.slabinfo, rcs_x = self.rcs_x, rcs_y = self.rcs_y, refDir = refDir, refSlab = refSlab, slices_file = slices_file)
        
        self.input_layers = ase.io.read(self.slices_file, index = ':') 
        assert len(self.input_layers) == (2 + self.buffer), "SurfaceGenerator: len(input_layers) must be {} {} buffer layer.".format(2+self.buffer, 'with' if self.buffer else 'without')

        self._init_lattice_()

        self.symbol_list = [s for s in self.symbols]
        self.get_default_formula_pool()


        if self.spg_type == 'plane':
            self.choice = 0
            self.spacegroup = [spg for spg in self.spacegroup if spg <= 17]
        elif self.spg_type == 'layer':
            self.choice = 1
            self.spacegroup = [spg for spg in self.spacegroup if spg <= 80]
        self._choice_ = self.choice
    
    def afterprocessing(self, atoms, *args, size = [1,1], origin = 'random', **kwargs):
        atoms.info['symbols'] = list(set(atoms.get_chemical_symbols()))
        atoms.info['parentE'] = 0.
        atoms.info['size'] = size
        atoms.info['origin'] = origin

        return atoms

    def _randwalk_(self, ind):
        label, atoms = randwalk(r = self.randwalk_range, dratio = self.d_ratio, attempts = self.max_attempts).generate(ind)

        return label, atoms
        

    @staticmethod
    def match_symmetry_plane(extraind, bottomind): 
        rots = []
        trs = []
        for ind in list([bottomind, extraind]):
            sym = spglib.get_symmetry_dataset(ind,symprec=1.0)
            if not sym:
                sym = spglib.get_symmetry_dataset(ind)
            if not sym:
                return False, extraind
            rots.append(sym['rotations'])
            trs.append(sym['translations'])

        m = match_symmetry(*zip(rots, trs), z_axis_only = True)
        if not m.has_shared_sym:
            return False, extraind
        _dis_, rot = m.get()
        #_dis_, rot = match_symmetry(*zip(rots, trs)).get() 
        _dis_[2] = 0
        _dis_ = np.dot(-_dis_, extraind.get_cell())

        extraind.translate([_dis_]*len(extraind))
        return True, extraind

    def get_spg(self, kind, grouptype):
        if grouptype == 'layergroup':
            cstar = [1, 2, 22, 26, 35, 36, 47, 48, 10, 13, 18]
            if kind == 'hex':
                #sym = 'c*', 'p6*', 'p3*', 'p-6*', 'p-3*' 
                return cstar + range(65, 81) 
            elif kind == 'c-cell':
                return cstar
            else:
                return list(range(1, 65))
        elif grouptype == 'planegroup':
            cstar = [1, 2, 5, 9]
            if kind == 'hex':
                return cstar + list(range(13, 18))
            elif kind == 'c-cell':
                return cstar
            else:
                return list(range(1, 13))

    def reset_rind_lattice(self, atoms, _x, _y, layersub = None):

        refcell = (self.ref_layer * (_x, _y, 1)).get_cell_lengths_and_angles()
        cell = atoms.get_cell_lengths_and_angles()

        if not np.allclose(cell[:2], refcell[:2], atol=0.1):
            return False, None
        if not np.allclose(cell[3:], refcell[3:], atol=0.5):
            #'hex' lattice
            if np.round(refcell[-1] + cell[-1] )==180:
                atoms = resetLattice(atoms = atoms.copy(), expandsize = (4,1,1)).get(np.dot(np.diag([-1, 1, 1]), atoms.get_cell() ))

            else:
                return False, None
        atoms.set_cell(np.dot(atoms.get_cell(), np.diag([1,1, refcell[2]/cell[2]])))

        refcell_ = (self.ref_layer * (_x, _y, 1)).get_cell()
        atoms.set_cell(refcell_, scale_atoms = True)

        #axis z:
        """
        if layersub is None:
            layersub = self.input_layers[1] *(_x, _y, 1)  
        else:
            layersub.set_cell(np.dot(np.diag([1,1,np.max(layersub.get_scaled_positions()[:, 2])]), layersub.get_cell()))
        spa = atoms.get_scaled_positions(wrap = False)
        sps = layersub.get_scaled_positions(wrap = False)
        latticesub = layersub.get_cell_lengths_and_angles()
        
        scaledp_matrix = np.zeros([len(atoms), len(layersub)])
        
        for i, a in enumerate(atoms):
            for j, l in enumerate(layersub):
                mindis = (covalent_radii[a.number]+covalent_radii[l.number])*self.d_ratio
                dx, dy, _ = spa[i] - sps[j]
                x2y2 = np.sum([x**2 for x in [dx + dy*math.cos(latticesub[-1]), dy*math.sin(latticesub[-1])]])
                minz = mindis**2 - x2y2
                scaledp_matrix[i][j] = sps[j][2]*latticesub[2] if minz < 0 else sps[j][2]*latticesub[2] + math.sqrt(minz)
                scaledp_matrix[i][j] -= spa[i][2] * refcell[2] + latticesub[2] 
        scaledp_matrix /= refcell[2]
        
        dz = max(np.min(scaledp_matrix), -np.min(spa[:, 2]))
        #dz = np.min(scaledp_matrix)
        #dz = max(np.min(scaledp_matrix), -np.min(spa[:, 2]))
        """
        dz = -1.5         #dz = 1.5 Ang from bottom to substrate top, more flexible later 
        atoms.translate([ -dz/refcell[2]* atoms.get_cell()[2]]* len(atoms))
        
        return True, atoms

        
    def reset_generator_lattice(self, _x, _y, spg):
        symtype = 'default'
        if self.symtype == 'hex':
            if (self.choice == 0 and spg < 13) or (self.choice == 1 and spg < 65):
                #for hex-lattice, 'a' must equal 'b'
                if self.reflattice[0] == self.reflattice[1] and _x == _y:    
                    symtype = 'hex'

        if symtype == 'hex' or self.symtype == 'c-cell':
            #self.GetConventional = False
            self.p_pri = 1
        elif symtype == 'default': 
            #self.GetConventional = True
            self.p_pri = 0
            
        self.min_lattice = list(self.reflattice *np.array([_x, _y]+[1]*4))
        self.max_lattice = self.min_lattice
        self.min_volume = np.linalg.det(cellpar_to_cell(self.min_lattice))
        return symtype
    
    def get_min_lattice(self, numlist):
        return self.min_lattice
    def get_max_lattice(self, numlist):
        return self.max_lattice
    def get_volume(self, numlist):
        return self.min_volume, self.min_volume
    
    def generate_ind(self, spg, formula):
        self.symbols = np.array([s for s in formula.keys() if formula[s] > 0])
        numlist = np.array([formula[s] for s in self.symbols])
        return super().generate_ind(spg, numlist, np.random.choice(self.n_split))

    def generate_random_walk_ind(self, _x, _y):
        ind = self.ref_layer * (_x , _y, 1)
        ind = ind.copy()
        add, keep, rm = self._atom_to_modify_    
        add, rm = {s: np.random.choice(add[s]) for s in add.keys()}, {s: np.random.choice(rm[s]) for s in rm.keys()}
        if rm:
            for symbol in rm:
                while rm[symbol] > 0:
                    eq_at = dict(zip(range(len(ind)), spglib.get_symmetry_dataset(ind,1e-2)['equivalent_atoms']))
                    indices = [atom.index for atom in ind if atom.symbol == symbol]
                    lucky_atom_to_rm = eq_at[np.random.choice(indices)]
                    eq_ats_with_him = np.array([i for i in eq_at if eq_at[i] == lucky_atom_to_rm])
                    size = np.random.choice(range(1,np.min([rm[symbol] , len(eq_ats_with_him)])+1))
                    _to_del = np.random.choice(eq_ats_with_him, size =size, replace=False)
                    rm[symbol] -= len(_to_del)
                    del ind[_to_del]
        
        if add:
            spg = np.random.choice(self.spacegroup) if self.choice == 0 else np.random.choice(self.get_spg(self.symtype, 'planegroup'))                
            self.choice = 0
            self.reset_generator_lattice(_x, _y, spg)

            label,extraind = self.generate_ind(spg, add)
            if label:
                botp = np.max(ind.get_scaled_positions()[:,2]) + np.random.choice(range(5,20))/100
                label, extraind = self.reset_rind_lattice(extraind, _x, _y, layersub = ind.copy())
            if label:              
                label, extraind = self.match_symmetry_plane(extraind, Surface.set_substrate(ind, self.input_layers[0]*(_x, _y, 1))) 
                hight = np.random.uniform(np.mean(ind.get_scaled_positions()[:, 2]), np.max(ind.get_scaled_positions()[:, 2]) + 1/(extraind.get_cell()[2]))
                extraind.translate([hight * extraind.get_cell()[2]]*len(extraind))
                #extraind.translate([np.mean(ind.get_scaled_positions()[:, 2]) + np.max(ind.get_scaled_positions()[:, 2])/2 * extraind.get_cell()[2]]*len(extraind))
                ind += extraind
                
            if not label:
                return label, None
        
        return self._randwalk_(ind)

    def generate_pop(self, n_pop, format_filter=None, *args, **kwargs):
        build_pop = []

        #Source 1. random walk
        while n_pop*self.randwalk_ratio > len(build_pop):
            for _ in range(self.max_n_try):
                _x = np.random.choice(self.rcs_x)
                _y = np.random.choice(self.rcs_y)

                label, ind = self.generate_random_walk_ind(_x, _y)

                if label:
                    self.afterprocessing(ind, origin = 'rand.randmove', size= [_x, _y])
                    ref = self.ref_layer * (_x , _y, 1)
                    ind.set_cell(ref.get_cell().copy(), scale_atoms=True)

                    build_pop.append(ind)
                    break
            else:
                break  

        #Source 2. Random layer
        while n_pop > len(build_pop):
            for _ in range(self.max_n_try):
            
                spg = np.random.choice(self.spacegroup)
                _x = np.random.choice(self.rcs_x)
                _y = np.random.choice(self.rcs_y)

                formula_pool = self.formula_pool_["{},{}".format(_x,_y)]
                rand_formula = formula_pool[np.random.randint(len(formula_pool))]
                    
                self.choice = self._choice_
                self.reset_generator_lattice(_x,_y, spg)

                #log.debug("random layer of formula {} with chosen spg = {}".format(rand_formula,spg))
                label,ind = self.generate_ind(spg, dict(zip(self.symbol_list, rand_formula)))
                #print(ind.get_all_distances(mic=True))
                if label:
                    label, ind = self.reset_rind_lattice(ind, _x, _y)

                if label:
                    label, ind = self.match_symmetry_plane(ind, Surface.set_vacuum(Surface.set_substrate(ind, self.input_layers[0]* (_x, _y, 1)), 10))

                if label:
                    self.afterprocessing(ind, origin='rand.symmgen', size=[_x, _y])
                    build_pop.append(ind)
                    break
            else:
                break

        return build_pop

    def refine_modification_list(self):
        symbols = self.modification['adsorb'].keys() | self.modification['clean'].keys() | self.modification['defect'].keys()
        add, keep, rm = {s:[] for s in symbols}, {s:[] for s in symbols}, {s:[] for s in symbols}

        for s in self.modification['adsorb']:
            n_list = split_formula(self.modification['adsorb'][s])
            add[s].extend([n for n in n_list if n > 0])
            keep[s].extend([n for n in n_list if n == 0])
            rm[s].extend([-1* n for n in n_list if n < 0])

        for s in self.modification['clean']:
            keep[s].extend([n for n in n_list if n == 0])

        for s in self.modification['defect']:
            n_list = split_formula(self.modification['defect'][s])
            rm[s].extend([n for n in n_list if n > 0])
            keep[s].extend([n for n in n_list if n == 0])
            add[s].extend([-1* n for n in n_list if n < 0])    

        for d in add, keep, rm:
            for s in list(d.keys()):
                d[s] = list(set(d[s]))
                if len(d[s]) ==0:
                    del d[s]
    
        self._atom_to_modify_ = add, keep, rm


    @property
    def symbol_numlist_pool(self):
        return self.formula_pool_
  
    def get_default_formula_pool(self):
        #rcs_formula: [ [], [], [] ] with len(symbols)
        #different expressions of same formula on top of Si(for example) layer [10] can be expressed by
        #(i) formula: [8~12]; (ii) formula: [8,9,10,11,12]
        #or 
        #(iii) adsorb: {'Si': [-2, -1, 0, +1, +2]} or (iv) adsorb: {'Si': [-2~2]}
        #add defects before adsorption.
        if hasattr(self, "formula_pool_"):
            return self.formula_pool_

        self.formula_pool_ = {}
        
        if self.rcs_formula:
            self.cal_formula_typeI()
        else:
            self.cal_formula_typeIII()

        return self.formula_pool_

    @property
    def ref_formula(self):
        if not hasattr(self, "ref_formula_"):
            if self.buffer:
                ref_symbols = self.ref_layer.get_chemical_symbols()
                self.ref_formula_ = {_s_: np.array([ref_symbols.count(_s_)]) for _s_ in set(ref_symbols)}
            else:
               self.ref_formula_ = {}
        return self.ref_formula_
    
    @staticmethod
    def combination_in_rcs_formula(rcs_formula):
        formula_pool = []
        for f in itertools.product(*rcs_formula):
            formula_pool.append(np.array(f))
        return np.array(formula_pool)

    #TYPE I/II. BY self.rcs_formula
    def cal_formula_typeI(self):
    
        for i, f in enumerate(self.rcs_formula):
            self.rcs_formula[i] = np.array(split_formula(f))

        formula_pool = self.combination_in_rcs_formula(self.rcs_formula)
        
        for x in self.rcs_x:
            for y in self.rcs_y:
                self.formula_pool_["{},{}".format(x,y)] = formula_pool
                differ = formula_add({s:-1*np.array(f) for s,f in zip(self.symbol_list, self.rcs_formula)}, {s: self.ref_formula[s]*x*y for s in self.ref_formula.keys()})
                self.update(self.modification['adsorb'], differ)
                
        self.refine_modification_list() 
        return self.formula_pool_

    #TYPE III/IV. BY self.modification
    def cal_formula_typeIII(self):
        self.refine_modification_list() 
        add, keep, rm = self._atom_to_modify_
        for x in self.rcs_x:
            for y in self.rcs_y:
                fxy = {s: self.ref_formula[s]*x*y for s in self.ref_formula.keys()}
                f = formula_add(fxy, keep)
                self.update(f, formula_add(fxy, add))
                self.update(f, formula_minus(fxy, rm))
                
                self.rcs_formula = []
                for s in self.symbol_list:
                    n = list(set(f[s])) if s in f.keys() else [0]
                    self.rcs_formula.append(np.array(n))
                
                self.formula_pool_["{},{}".format(x,y)] = self.combination_in_rcs_formula(self.rcs_formula)

    @staticmethod
    def update(dictA, dictB):
        for key in dictB:
            dictA[key] = dictB[key] if not key in dictA else list(set(list(dictA[key]) + list(dictB[key])))


from .utils import InterfaceMatcher, matrix_match
from ase import Atoms

class InterfaceGenerator(SPGGenerator):

    @staticmethod
    def __matchcell__(*args, **kwargs):
        im = InterfaceMatcher(*args, **kwargs)
        im.match_result(verbose=True, save_intermediate='inter_ml.npy')
        return
    
    def __generate_substrates__(self, refDir):
        if os.path.exists(self.slices_file):
            log.info("Used layerslices in {}.".format(self.slices_file))
            self.input_layers = ase.io.read(self.slices_file, index = ':') 
            return
        
        if os.path.exists(self.slabinfo['traj_file']) and os.path.exists(self.slabinfo['matrix_file']):
            log.info("Used match result from {}, {}".format(self.slabinfo['traj_file'], self.slabinfo['matrix_file']))
        else:
            if not os.path.exists(refDir):
                os.mkdir(refDir)
            self.__matchcell__(**self.slabinfo)
            
        self.ml = matrix_match.load_match_list(self.slabinfo['matrix_file'])[self.match_order_index]
        #ml = ['id-A', 'id-B','hkl-A', 'hkl-B', 'matrix-A', 'matrix-B', 'match-fit']
        id_a, id_b = self.ml[0:2]
        layers = InterfaceMatcher.get_id_from_slices(id_a, id_b, self.slabinfo['traj_file'], buffer=self.buffer)
        ma, mb = self.ml[4:6]

        a_b_order = None

        for i,layer in enumerate(layers):
            m = ma if i%2==0 else mb
            layers[i] = InterfaceMatcher.matrix_times_cell(layer, m)
            c = layers[i].cell.cellpar()
            a_b_order = a_b_order if not a_b_order is None else c

            if np.sum(np.abs([a_b_order[:2] - c[:2]])) >  np.sum(np.abs([a_b_order[:2] - [c[1], c[0]]])):
                #swap axis a,b of layers[i]
                sp = layers[i].get_scaled_positions()
                layers[i].cell[:2] = layers[i].cell[1::-1]
                sp[:,:2] = sp[:,1::-1]
                layers[i].set_scaled_positions(sp)
                

            layers[i] = InterfaceMatcher.rotate_c2z_and_a2x(layers[i])

        ca = layers[0].cell.cellpar()
        cb = layers[1].cell.cellpar()
        self.input_layers = layers
        ase.io.write(self.slices_file, layers)

        s = "Saved matcher {}-th from match_list '{}':\n".format(self.match_order_index, self.slabinfo['matrix_file'])
        s += "-------------------\n"

        s += "{}: {} @ {}\n".format("id".ljust(15, ' '), *self.ml[:2])
        s += "{}: {}\n                 {}\n".format("hkl".ljust(15, ' '), self.ml[2], self.ml[3])
        s += "{}: {}\n                 {}\n".format("matrix".ljust(15, ' '), str(self.ml[4]).replace('\n', ','), str(self.ml[4]).replace('\n', ','))
        s += "{}: {}\n                 {}\n".format("[a,b,gamma]".ljust(15, ' '), np.around([*ca[0:2], ca[-1]],3), np.around([*cb[0:2], cb[-1]],3))
        s += "{}: {}\n".format("match fitness".ljust(15, ' '), self.ml[-1])
        s += "-------------------"
        log.info(s)
        return 
        
    def __init__(self, **parameters):
        super().__init__(**parameters)
        refDir = 'Ref'
        Default = {
            'vacuum_thickness':10, 
            'buffer': True, 
            'slices_file': refDir + '/layerslices.traj',
            'match_order_index': 0,
            }
        self.slabinfo = {
            'bulk_a': None,
            'bulk_b': None, 
            'range_hkl': [-5,6],
            'range_matrix': [-3,4], 
            'range_area': [0., 100.], 
            'range_a': [0,13.],
            'range_ang': [45.,135.],
            'bulk_layernum': 2, 
            'buffer_layernum': 1, 
            'rcs_layernum': 1, 
            'cutslices': 3,
            'thread_para': 50,
            'addH': True,
            'range_substrate_thickness': [12, 10.], 
            'traj_file': refDir + '/match_file.traj',
            'matrix_file': refDir + '/match_file.npy',
            }
        
        self.slabinfo.update(parameters['slabinfo'])
        self.slabinfo.update({
            'bulk_a': ase.io.read(self.slabinfo['bulk_a']),
            'bulk_b': ase.io.read(self.slabinfo['bulk_b']),
        })
        check_parameters(self, parameters, [], Default)

        self.__generate_substrates__(refDir)

    @property
    def formula_pool(self):
        return np.array([1])


    def generate_ind(self, translation = [0,1], vertical_distance = 1.0):
        #remove most top vacuum
        for i in [-1, -2]:
            rcs = self.input_layers[i]
            old_cell = rcs.get_cell()
            old_cell[2] *= np.max(rcs.get_scaled_positions()[:,2])
            self.input_layers[i].set_cell(old_cell)

        rcs_b, rcs_a = self.input_layers[-1], self.input_layers[-2]
        la, lb, lc, _, _, gamma = (rcs_b.cell.cellpar() + rcs_a.cell.cellpar())/2
        lc = lc*2 + vertical_distance
        gamma = gamma / 180 * math.pi
        new_cell = np.array([[la,0,0],[lb*math.cos(gamma), lb*math.sin(gamma),0],[0,0,lc]])

        sp_b, sp_a = rcs_b.get_scaled_positions(), rcs_a.get_scaled_positions()
        c_ratio_a = rcs_a.cell.cellpar()[2] / lc
        c_ratio_dis = vertical_distance / lc
        sp_a[:,2] = sp_a[:,2]* c_ratio_a
        sp_a[:, 0:2] -= translation

        sp_b[:,2] =  1- sp_b[:,2] * (1 - c_ratio_a - c_ratio_dis)
        
        nums_b, nums_a = rcs_b.get_atomic_numbers(), rcs_a.get_atomic_numbers()

        atoms = Atoms(cell = new_cell, 
                      scaled_positions = np.concatenate((sp_b, sp_a), axis=0), 
                      numbers = np.concatenate((nums_b, nums_a), axis=0))   
        
        atoms.info['size'] = translation
        atoms.wrap()
        return True, atoms

    def generate_pop(self, n_pop, format_filter=None, *args, **kwargs):
        build_pop = []

        #Source 1. random walk
        while n_pop > len(build_pop):
            for _ in range(self.max_n_try):
                
                label, ind = self.generate_ind(translation = np.random.uniform(0,1,2), vertical_distance = 1.0)
                if label:
                    self.afterprocessing(ind, origin = 'random')
                    build_pop.append(ind)
                    break
            else:
                break  

        return build_pop