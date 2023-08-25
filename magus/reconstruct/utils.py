import numpy as np
import ase.io
from ase.data import covalent_radii
import logging
from ase import Atoms, Atom
from collections import Counter
import math
import spglib
from ase.neighborlist import neighbor_list
import traceback

log = logging.getLogger(__name__)

def check_distance(atoms, d_ratio):
    i_indices = neighbor_list('i', atoms, d_ratio, max_nbins=100.0)
    return len(i_indices) == 0

class resetLattice:
    def __init__(self, atoms=None, expandsize = (16,16,16)):
        if atoms:
            #1. build a very large supercell
            supercell = atoms * expandsize
            supercell.translate( [np.sum( np.dot( np.diag([-int((s-1)/2) for s in expandsize]), atoms.get_cell()), axis = 0)] *len(supercell))
            self.supercell = supercell

    def expand(self, size):
        supercell = self.supercell
        supercell = supercell * [2*i if not i==0 else 1 for i in size]
        supercell.translate( [-np.sum(np.dot( np.diag(size), supercell.get_cell()), axis = 0)] *len(supercell))

    def InCell(self, pos):
        for i in range(3):
            if pos[i]>=1 or pos[i]<0:
                return False 
        return True

    def refinepos(self, pos):
        for i in range(len(pos)):
            for j in range(3):
                if abs(pos[i][j]-1)< 1e-4:
                    pos[i][j] = 1        
                if abs(pos[i][j])<1e-4:
                    pos[i][j]=0
        return pos

    def get(self, newcell, neworigin = None):
        supercell = self.supercell
        if not neworigin is None:
            supercell.translate([[-1*i for i in neworigin]] * len(supercell))
        supercell.set_cell(newcell)
        pos =supercell.get_scaled_positions(wrap=False).copy()
        pos = self.refinepos(pos)
        index = [i for i in range(len(pos)) if self.InCell(pos[i])==True]
        if len(index)>0:
            return supercell[index].copy()
        else:
            log.warning("err in resetLattice: no atoms in newcell")
            return Atoms(cell=supercell.get_cell())

from ase.geometry import cell_to_cellpar
from math import gcd
class cutcell:
    def __init__(self,originstruct, layernums, totslices= None, vacuum = 1.0, addH = False, direction=[0,0,1], 
        xy = [1,1], rotate = 0, pcell = True, 
        matrix = None, 
        save_file = 'Ref/layerslices.traj', range_length_c = [0.,15.,]):
        """
        @parameters:
        [auto, but be given is strongly suggested] totslices: layer number of originstruct
        layernums: layer number of [bulk, buffer, rcs_region]
        [*aborted] startpos:  start position of bulk layer, default =0.9 to keep atoms which are very close to z=1(0?)s.
        vacuum: [in Ang] vacuum to add to rcs_layer  
        addH: change the most bottom layer of atoms to hydrogen
        direction: miller indices[orth cells] / bravais-miller indices[hex cells]
        + wood's notation:
           xy: (1, 1)
           rotate: [not implied yet] angle of R.
           pcell: if False, get c(nxn) cell
        + matrix notation:
           matrix: 2x2 matrix. For matrix notations.
        save_file: save cut slices to save_file
        range_length_c[1]: (max) sometimes periodic surface cell have very large length c, but we only need the top several layers of it
        range_length_c[0]: (min) sometimes you want slabs thicker.
        """
        self.atoms = originstruct.copy()
        self.totslices, self.layernums = totslices, layernums
        #1. if direction is in bravais-miller indices, turn to miller indices
        if len(direction) == 4:
            direction = self.tomillerindex(direction)
            log.debug('changed bravais-miller indices direction to miller index = {}'.format(direction))
        
        #2. get surface of conventional cell
        self.range_length_c = range_length_c
        newcell = self.get_ccell(direction)
        
        #3. get primitive surface vector and startpos
        surface_vector = self.get_pcell(self.atoms, newcell, self.totslices)

        #4. get surface cell! if matrix notation/wood's notation exists, expand cell.
        ## matrix notation only or wood's notation only 
        if matrix:
            surface_vector = self.matrix_notation(matrix, surface_vector)
        else:
            surface_vector = self.wood_notation(xy, rotate, pcell, surface_vector)
        ##4.5. maybe it's better to expand supercell too!
        self.supercell.expand((1, 1, 0))

        #5. cutcell!
        self.cut_traj = self.cut(self.layernums, self.totslices, surface_vector, vacuum, addH)

        if not save_file is None:
            log.info("save cutslices into file {}".format(save_file))
            ase.io.write(save_file, self.cut_traj,format='traj')
    
    def tomillerindex(self, direction):
        """
        [UVW] <-> [uvtw]
        u = (2U - V) / 3            v = (2V - U) / 3
        t = -(u+v) = -(U+V) / 3
        w = W
        """
        assert direction[2] == -direction[0] - direction[1], "for bravais-miller indices [uvtw], t must eqs -(u+v)"
        miller = np.array([direction[0]*2 + direction[1], direction[0] + 2*direction[1], direction[-1]])
        return miller / gcd(gcd(miller[0], miller[1]), miller[2])

    def get_ccell(self, direction):
        #Step A: analyze direction
        #newcell_c = direction                                               #warning: for orth-cells only. 
        rx, ry, rz = [i if not i==0 else 1e+10 for i in direction]
            
        points = [[1/rx, 0, 0], [0, 1/ry, 0], [0, 0, 1/rz]]             #cross points at axis. 1/h, 1/k, 1/l
        newcell = []
        left = []
        for i, point in enumerate(points):
            if np.allclose(point, [0,0,0]):
                newcell_a = [1 if j==i else 0 for j in range(3)]
                newcell.append(newcell_a)
            else:
                left.append(point)
        i = 1
        while len(newcell) < 2:
            newcell_a = np.array(left[i]) - np.array(left[0])
            newcell.append(newcell_a)
            i+=1

        def norm(cell):
            #normalize to integers
            #trick: multiply [7,9,11,13,17,19] = 2909907 to avoid cases like cell = [0.33333,1,0]

            for i, c in enumerate(cell):
                cell[i] *=2909907000
            return np.array(cell)/gcd(int(np.round(cell[0])), int(np.round(cell[1])), int(np.round(cell[2])))

        for i, _ in enumerate(newcell):
            newcell[i] = norm(newcell[i])

        #newcell.append(newcell_c)

        #Step B: dot product of direction x cell
        newcell = np.dot(np.array(newcell), self.atoms.get_cell())
        newcell_c = np.cross(*newcell)

        newc = norm(np.dot(newcell_c, np.linalg.inv(self.atoms.get_cell())))
        newcell = np.array([*newcell, np.dot(newc, self.atoms.get_cell())])

        #in case some direction cannot be simplified and gets too large in c
        if np.linalg.norm( newcell[2]) > self.range_length_c[1]:
            newcell[2] = newcell[2] /  np.linalg.norm( newcell[2])*self.range_length_c[1]
            #and thus totslices cannot be larger than sum(layernums of [bulk, buffer, rcs_region])
            if self.totslices < np.sum(self.layernums):
                _lm = self.layernums.copy()
                self.layernums = np.round(self.layernums / np.sum(self.layernums) *self.totslices)
                for n in [0,1,2]:
                    if self.layernums[n] ==0 and _lm[n] > 0:
                        self.layernums[n] = 1
                while np.sum(self.layernums) > self.totslices:
                    self.layernums[np.argmax(self.layernums)] -=1
                log.warning('Cannot get periodic surface cell for its c is larger than max_substrate_thickness, ' +
                            'thus totslices cannot be larger than sum(layerslices). It has been automatically changed ' +
                            'from {} to {} (totslices = {})'.format(_lm, self.layernums, self.totslices) )

        else:
            self.adjust_layernums(np.linalg.norm( newcell[2]))

        log.debug("cutcell with conventional surface vector\n{}".format(np.dot(newcell, np.linalg.inv(self.atoms.get_cell()))))

        return newcell

    def adjust_layernums(self, lattice_length_c):
        _lm = self.layernums.copy()
        if lattice_length_c / self.totslices * np.sum(self.layernums) < self.range_length_c[0]:
            while lattice_length_c / self.totslices * np.sum(self.layernums) < self.range_length_c[0]:
                self.layernums[0] +=1
            log.warning('The given layernums generate a slab thiner than min_substrate_thickness, ' +
                            'thus has been automatically changed ' +
                            'from {} to {} (totslices = {})'.format(_lm, self.layernums, self.totslices) )
        return
    
    def get_pcell(self, atoms, newcell, totslices):
        #TODO: if slab is not complete, expand expandsize. Time cost may increase.
        self.supercell = resetLattice(atoms, expandsize = (16,16,16))
        newlattice = self.supercell.get(newcell)
        allayers = None
        if totslices is None:
            for totslices in range(8, 1,-1):
                try:
                    allayers = LayerIdentifier(newlattice, prec = 0.5/totslices, n_clusters = totslices +1, lprec = 0.4/totslices)
                    if len(allayers) == totslices or len(allayers) == totslices+1:
                        break
                except:
                    pass
            log.warning("Number of total layers is not given. Used auto-detected value {}.".format(totslices))
            self.totslices = totslices
        else:
            allayers = LayerIdentifier(newlattice, prec = 0.5/totslices, n_clusters = totslices +1, lprec = 0.4/totslices)

        onelayer = newlattice[allayers[0]]
        startpos = np.max(newlattice.get_scaled_positions()[:,2]) + 0.01 
        startpos = startpos - int(startpos)

        if len(allayers) == totslices + 1:

            onelayer = newlattice[allayers[1]]
            startpos = np.max(newlattice[allayers[-2]].get_scaled_positions()[:,2]) + 0.01
            startpos = startpos - int(startpos)

        onelayer.set_cell(onelayer.get_cell()[:] * np.reshape([1]*6 + [2.33]*3, (3,3)))
        #print(startpos)
        self.startpos = startpos

        surface_vector = spglib.get_symmetry_dataset(onelayer,symprec = 1e-4)['primitive_lattice']
        abcc, abcp = cell_to_cellpar(onelayer.get_cell()[:])[:3], cell_to_cellpar(surface_vector)[:3]
        axisc = np.where(np.abs(abcp-abcc[2]) < 1e-4)[0]
        assert len(axisc) ==1, "cannot match primitive lattice with origin cell, primitive abc = {} while origin abc = {}".format(abcp, abcc)
        if not axisc[0] ==2:
            surface_vector[[axisc[0], 2]] = surface_vector[[2, axisc[0]]]

        surface_vector[2] = newcell[2]
        
        log.debug("primitive surface vector\n{}".format(np.dot(surface_vector, np.linalg.inv(atoms.get_cell()))))
        return surface_vector

    def matrix_notation(self, matrix, surface_vector):
        surface_vector[:2] = np.dot(matrix, surface_vector[:2])
        log.debug("changed by matrix notation\n{}".format(np.dot(surface_vector, np.linalg.inv(self.atoms.get_cell()))))
        return surface_vector

    def wood_notation(self, xy, rotate, pcell, surface_vector):
        #TODO: add rotate
        
        #here modify some sqrt(2)/ sqrt(3) xy.
        for i, x in enumerate(xy):
            if abs(x - 1.4) < 0.1:
                xy[i] = math.sqrt(2)
            elif abs(x - 1.6) < 0.1:
                xy[i] = math.sqrt(3)

        surface_vector[:2] = np.dot(np.diag([*xy, 1]), surface_vector)[:2]
        if not pcell:
            surface_vector[:2] = np.array([surface_vector[0] + surface_vector[1], surface_vector[0] - surface_vector[1]])/2
        log.debug("changed by wood's notation\n{}".format(np.dot(surface_vector, np.linalg.inv(self.atoms.get_cell()))))
        return surface_vector

    def cut(self, layernums, totslices, surface_vector, vacuum, addH):
        #5. expand unit surface cell on z direction
        bot, mid, top = layernums[0], layernums[1], layernums[2]
        slicepos = np.array([0, bot, bot + mid,  bot + mid + top])/totslices
        slicepos = slicepos + np.array([self.startpos]*4)
        #log.info("cutslice = {}".format(slicepos)) 

        #6. build bulk, buffer, rcs layer slices 
        pop= []
        if slicepos[-1]==slicepos[-2]:
            slicepos = slicepos[:-1]
            log.info("warning: rcs layer have no atoms. Change mode to adatoms.")  

        for i in range(1, len(slicepos)):

            cell = surface_vector.copy()
            cell[2] = surface_vector[2] * (slicepos[i]-slicepos[i-1])
            origin = (slicepos[i-1] if i==1 else slicepos[i-1]-slicepos[i-2]) * surface_vector[2]
            
            layerslice = self.supercell.get(cell, neworigin = origin)
            if len(layerslice):
                layerslice=layerslice[layerslice.numbers.argsort()]
            else:
                slicename = ['bulk', 'buffer', 'reconstruct']
                log.warning("No atom in {} layer!! Please check the layers before next step.".format(slicename[i-1]))

            pop.append(layerslice)

        #add extravacuum to rcs_layer  
        if len(pop)==3:        
            cell = pop[2].get_cell()
            cell[2]*= ( 1.0 + vacuum/pop[2].cell.cellpar()[2])
            pop[2].set_cell(cell)
        
        #vasp does not work if the triple product of the basis vectors is negative. Make a check.
        cell = pop[0].get_cell()
        if np.dot(np.cross(*cell[:2]), cell[2]) < 0:
            for i, ind in enumerate(pop):
                cell = ind.get_cell()
                cell[[0,1]] = cell[[1,0]]
                pop[i].set_cell(cell, scale_atoms = True)

        #7. add hydrogen
        if addH:
            nl = ase.neighborlist.NeighborList(np.array(ase.neighborlist.natural_cutoffs(pop[0]))*1.1, self_interaction=False, bothways=True)
            pop[0].set_pbc([1,1,0])
            sp = pop[0].get_scaled_positions(wrap = True)
            sp[:, :2] = np.array(list(map(lambda x: (0 if abs(x-1)<1e-2 else x), sp[:,:2].flatten()))).reshape(-1,2)
            nl.update(pop[0])

            to_del = []
            sp_h = []

            for i in range(len(sp)):
                indexs, offsets = nl.get_neighbors(i)

                not_bottom = np.any([sp[j][2] < sp[i][2] - 0.05 for j in indexs])

                if not_bottom:
                    continue
                to_del.append(i)
                for j, ost in zip(indexs, offsets):
                    bond_scale =  (covalent_radii[pop[0][j].number] + covalent_radii[1]) \
                                        / (covalent_radii[pop[0][i].number] + covalent_radii[pop[0][j].number])
                    sp_h.append( sp[j] - bond_scale * (sp[j] + ost - sp[i] ))

            sp_h = np.unique(sp_h, axis = 0)
            del pop[0][to_del]
            pop[0] += Atoms(numbers = [1]*len(sp_h), cell = pop[0].get_cell(), scaled_positions = sp_h)
        
        return pop

import prettytable as pt
import os, time
import multiprocessing
import itertools 
import fcntl

def warpper1(func, queue, setargs):
    args, kwargs = setargs
    queue.put(func(*args, **kwargs))
    return 0

def warpper2(func, queue, setargs):
    queue.put(func(*setargs))
    return 0

class matrix_match:
    """
    match lattice matrix MA(2x2) with lattice matrix MB(2x2).
    i.e,              R * A * MA ~ B * MB 
        R * A * MA * (MB^-1) ~ B 
        
    in which A, B are component of integers; R is rotational matrix which means |R|=1
    """
    @staticmethod
    def parallelize(func, args, num_threads):

        indexlist = [[] for _ in range(num_threads)]
        for i, _ in enumerate(args):
            indexlist[i%len(indexlist)].append(i)
        results = []

        if np.any([type(a) is dict for a in args[0]]):
            warpper = warpper1
        else:
            warpper = warpper2

        for ith, arg in enumerate(indexlist[0]):
            #print(ith, "out of", len(indexlist[0]))
            process_pool =  []
            Queue = multiprocessing.Queue()
            for thread in range(num_threads):
                if ith < len(indexlist[thread]):
                    _arg = args[indexlist[thread][ith]]
                    process = multiprocessing.Process(target=warpper, args=(func, Queue, _arg))
                    process_pool.append(process)

            for i in range(len(process_pool)):
                process_pool[i].start()

            #for i in range(len(process_pool)):
            #    process_pool[i].join()

            #use queue.get for blocking instead of process.join
            for i in range(len(process_pool)):
                results.append(Queue.get())

        return results
    
    @staticmethod
    def cell_to_cellpar_d2(cell):
        """
        cell matrix(2x2) to (a, b, gamma)
        """
        a, b = np.linalg.norm(cell, axis=1)
        _acos = np.dot(*cell)/a/b
        #avoid sometimes 1.00000001 
        gamma = math.acos(np.round(_acos, 4))        #/math.pi*180
        return max(a,b), min(a, b), gamma
    
    @staticmethod
    def cellpar_to_cell_d2(a, b, theta):
        """
        (a, b, theta) to [[a,0],[bx,by]]
        """
        ux = a
        vx = b * math.cos(theta)
        vy = b * math.sin(theta)
        return ux, vx, vy
    
    def good_transMatrixes(self, cellname, cellm, r = [-3,4], range_a = [0, 25.], range_ang = [45., 135.], range_area = [0., 100.]):
        listA = getattr(self, "list{}".format(cellname))

        for c in itertools.product(*[sorted(range(*r), key=lambda x:abs(x))]*4):
            A = np.array(c).reshape(2,2)
            if np.linalg.det(A) ==0:
                continue

            a, b, ang = matrix_match.cell_to_cellpar_d2(np.dot(A, cellm))
            uax, vax, vay = matrix_match.cellpar_to_cell_d2(a, b, ang)
            if range_a[0] < a < range_a[1] and range_ang[0] < ang < range_ang[1] and  range_area[0] < uax*vay < range_area[1]:
                listA.append([A, uax, vax, vay])

    def fitness(self, LA, LB):
        A, uax, vax, vay = LA
        B, ubx, vbx, vby = LB

        _exx = (ubx-uax)/uax
        _eyy = (vby-vay)/vay
        _2_exy = vbx/vay - ubx*vax/uax/vay
        fitness = np.round(abs(_exx) + abs(_eyy) + abs(_2_exy), 5)
        return [A, B, fitness]
    
    def match(self, ma, mb, r = [-3,4], range_a = [0, 25.], range_ang = [45., 135.], range_area = [0., 100.], 
              num_threads = 1, verbose = False,  save_intermediate = None, **info):

        best_fit = [np.eye(2), np.eye(2),1e+5]
        #second_best_fit = [np.eye(2), np.eye(2),1e+5]
        range_ang = np.array(range_ang) /180 * math.pi

        self.listA, self.listB = [], []
        self.good_transMatrixes('A', ma, r = r, range_a = range_a, range_ang = range_ang, range_area = range_area)
        self.good_transMatrixes('B', mb, r = r, range_a = range_a, range_ang = range_ang, range_area = range_area)

        args = list(itertools.product(self.listA, self.listB))
        if len(args):
            fits = self.parallelize(self.fitness, args, min(num_threads, len(self.listA)*len(self.listB)))
        else:
            fits = []

        for f in fits:
            if f[-1] < best_fit[-1]:
                #second_best_fit = best_fit
                best_fit = f
                
        res = [info['id_a'], info['id_b'], info['hkl_a'], info['hkl_b'], *best_fit]
        if not save_intermediate is None:
            self.save_match_list([res], save_intermediate)

        if verbose:
            print("{}\tdone match A{} and B{}".format(time.asctime(time.localtime()), info['hkl_a'], info['hkl_b']))
        return res
    
    #match_list = ['id-A', 'id-B','hkl-A', 'hkl-B', 'matrix-A', 'matrix-B', 'match-fit']
    @staticmethod
    def save_match_list(match_list, file_name):
        array = []
        for ml in match_list:
            array.append(np.array([ml[0], ml[1], *ml[2], *ml[3], *ml[4].flatten(),*ml[5].flatten(),ml[6]]))
        with open(file_name, 'ab') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            np.save(f, np.array(array))
            fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def load_intermediate(file_name):
        match_list = []
        with open(file_name, 'rb') as f:
            while True:
                try:
                    a = np.load(f)[0]
                    match_list.append([int(a[0]), int(a[1]), a[2:5], a[5:8], np.array(a[8:12]).reshape(2,2), np.array(a[12:16]).reshape(2,2), a[16]])
                except ValueError:         #npyio.py"Cannot load file containing pickled data when allow_pickle=False"
                    break
        return match_list
    
    @staticmethod
    def load_match_list(file_name):
        match_list = []
        array = np.load(file_name)
        for a in array:
            match_list.append([int(a[0]), int(a[1]), a[2:5], a[5:8], np.array(a[8:12]).reshape(2,2), np.array(a[12:16]).reshape(2,2), a[16]])
        return match_list

class InterfaceMatcher:
    def __init__(self, bulk_a, bulk_b, range_hkl = [-5,6], range_matrix = [-4,5], 
                 range_a = [0., 15.], range_ang = [45., 135.], range_area = [0., 100.],
                 range_substrate_thickness = 15., 
                 bulk_layernum = 3, buffer_layernum= 1, rcs_layernum =1, cutslices = None, addH =True,
                 tol = 1000, traj_file = 'match_file.traj', matrix_file = 'match_file.npy', thread_para = 1, verbose = False):
        
        self.lattice_a, self.lattice_b = bulk_a, bulk_b
        self.range_hkl, self.range_matrix = range_hkl, range_matrix
        self.range_a, self.range_ang, self.range_area = range_a, range_ang, range_area
        self.range_substrate_thickness =  range_substrate_thickness 
        self.layer_nums, self.cutslices = [bulk_layernum , buffer_layernum, rcs_layernum], cutslices
        self.tol, self.addH = tol, addH
        self.traj_file, self.matrix_file = traj_file, matrix_file
        self.thread_para = thread_para
        self.verbose = verbose

    @staticmethod
    def rotate_c2z_and_a2x(cell):
        """
        rotate lattice so that direction of cell_c is along z axis and cell_a is along a axis.
        """
        cell = cell.copy()
        a,b,c, _, _, gamma = cell.cell.cellpar()
        gamma = gamma/180*math.pi
        new_cell = np.array([[a,0,0],[b*math.cos(gamma), b*math.sin(gamma), 0],[0,0,c]])
        cell.set_cell(new_cell, scale_atoms = True)
        return cell
    
    @staticmethod
    def is_miller_index(h,k,l):

        direction = []
        for index in [h,k,l]:
            if not index == 0:
                direction.append(index)
        direction = np.array(direction)
        if len(direction) == 0:  #hkl = (0,0,0)
            return False
        if np.sum(direction < 0) >1 or (len(direction)==1 and direction[0]< 0):     #hkl=(-1,0,0)
            return False
        if not math.gcd(*[x for x in direction]) == 1:    #hkl=(2,2,2)
            return False
        """
        d = direction / math.lcm(np.abs(direction))
        for xd in d:                                                    
            if not abs(1/xd - int(1/xd)) < 1e-2:
                return False
        """
        return True
    
    def in_miller_list(self, cell_name, hkl_1, miller_list):
        if not hasattr(self, "sym_{}".format(cell_name)):
            lattice = getattr(self, "lattice_{}".format(cell_name))
            sym = spglib.get_symmetry_dataset(lattice,1e-4)['rotations']
            setattr(self, "sym_{}".format(cell_name), sym)
        
        for ml in miller_list:
            if cell_name == ml[0][0]:
                hkl_2 = ml[0][1:]
                for r in getattr(self, "sym_{}".format(cell_name)):
                    if np.all(np.dot(r, hkl_2) == hkl_1):
                        return True
            
        return False
    
    @staticmethod
    def matrix_times_cell(atoms, matrix):
        """matrix: (2x2)
        """
        m = np.eye(3)
        m[0:2,0:2] = matrix
        new_lattice = np.dot(m, atoms.get_cell())
        new_atoms = resetLattice(atoms).get(new_lattice)
        return new_atoms
    
    @staticmethod
    def get_id_from_slices(id_a, id_b, traj_file, buffer = True):
        ids = [id_a, id_b]
        layers = [ase.io.read(traj_file, index = ids[0]), ase.io.read(traj_file, index = ids[1])]
        if buffer:
            new_layers = [1,2]            #add buffer, rcs
        else:
            new_layers = [1]                #add rcs

        for x in new_layers:
            buffer_layers = [ase.io.read(traj_file, index = ids[0] + x), ase.io.read(traj_file, index = ids[1] + x)]
            for i,b in enumerate(buffer_layers):
                for k in ['name', 'h','k','l']:
                    assert b.info[k] == layers[i].info[k], "id {} of traj_file isnot buffer of bulk (id {}) : {}, {}".format(    
                                                                                                ids[i], ids[i+1], b.info, layers[i].info ) 
            layers.extend(buffer_layers)
    
        return layers

    
    def generate_cutcell(self, cell_name, h,k,l, verbose = False, **kwargs):
        lattice = getattr(self, "lattice_{}".format(cell_name))
        if verbose:
            print("cutcell for cell {}({})".format(cell_name, [h,k,l]))

        cell_a = []
        try:
            c = cutcell(lattice, self.layer_nums, direction=[h,k,l], **kwargs).cut_traj
            traj_info = {'name': cell_name, 'h':h, 'k':k, 'l':l}
            types = ['bulk', 'buffer', 'rcs']
            for i in range(len(c)):
                ci = Atoms(numbers = c[i].get_atomic_numbers(), positions = c[i].get_positions(), cell = c[i].get_cell())
                ci = self.rotate_c2z_and_a2x(ci)
                for k in traj_info:
                    ci.info[k] = traj_info[k]
                ci.info['type'] = types[i]
                cell_a.append(ci)
        except Exception:
            log.warning("failed cutcell for cell {} ({}, {}):\n{}".format(cell_name, (h,k,l), kwargs, traceback.format_exc()))
        
        return cell_a

    def cut_cell(self, verbose = False):
        if os.path.exists(self.traj_file):
            self.traj_pop = ase.io.read(self.traj_file, index = ':')
        else:
            self.traj_pop, miller_list = [], []
            cut_cell_kwargs =  {"totslices": self.cutslices, "vacuum":10., "save_file":None, 
                                "range_length_c":self.range_substrate_thickness, "verbose": verbose, "addH": self.addH}
            '''
            for hkl in itertools.product(*[range(*self.range_hkl)]*3):
                h, k, l = hkl
                #remove unreasonable miller indexes
                if not self.is_miller_index(h,k,l):
                    continue
                else:
                    if not self.in_miller_list('a', [h,k,l], miller_list):
                        miller_list.append([("a",h,k,l), cut_cell_kwargs])
                    if not self.in_miller_list('b', [h,k,l], miller_list):
                        miller_list.append([("b",h,k,l), cut_cell_kwargs])
            '''
            miller_list.append([("a",1,0,0), cut_cell_kwargs])
            miller_list.append([("b",1,1,1), cut_cell_kwargs])

            if len(miller_list):
                for p in matrix_match.parallelize(self.generate_cutcell, miller_list, min(self.thread_para, len(miller_list))):
                    self.traj_pop.extend(p)
            
            ase.io.write(self.traj_file, self.traj_pop)

    def best_match(self, *args, **kwargs):
        return matrix_match().match(*args, **kwargs)

    def match_result(self, verbose = False, save_intermediate = None):
        self.cut_cell(verbose=verbose)
        self.match_list, pop_list = [], []

        if os.path.exists(save_intermediate):
            self.match_list = matrix_match.load_intermediate(save_intermediate)
            matched_id_list = [ml[0:2] for ml in self.match_list]
            log.warning("Used stored matrix match file {}. \n".format(save_intermediate))
        else:
            matched_id_list = []

        for i in range(0, len(self.traj_pop)):
            if not self.traj_pop[i].info['type'] == "bulk":
                continue 
            if self.traj_pop[i].info['name'] == 'b':
                continue
            cell_a = self.traj_pop[i].get_cell()
            
            for j in range(0, len(self.traj_pop)):
                if not self.traj_pop[j].info['type'] == "bulk":
                    continue
                if self.traj_pop[j].info['name'] == 'a':
                    continue
                cell_b = self.traj_pop[j].get_cell()
                if [i,j] in matched_id_list:
                    continue

                pop_list.append([(cell_a[0:2, 0:2], cell_b[0:2, 0:2]), 
                                {"r": self.range_matrix, "range_a": self.range_a, "range_ang": self.range_ang, "range_area": self.range_area,
                                'num_threads': self.thread_para, "verbose": verbose, "save_intermediate": save_intermediate,
                                'id_a':i, 'id_b':j, 
                                'hkl_a':np.array([self.traj_pop[i].info["h"], self.traj_pop[i].info["k"],self.traj_pop[i].info["l"]]), 
                                'hkl_b':np.array([self.traj_pop[j].info["h"], self.traj_pop[j].info["k"],self.traj_pop[j].info["l"]])}
                                ])
        if len(pop_list):        
            self.match_list = matrix_match.parallelize(self.best_match, pop_list, self.thread_para)

        self.match_list.sort(key=lambda a:a[-1])

        if self.matrix_file:
            if os.path.exists(self.matrix_file):
                os.remove(self.matrix_file)
            matrix_match.save_match_list(self.match_list, self.matrix_file)
        if log.level <=20:
            s = self.ml_to_string(self.match_list, self.traj_pop)
            log.warning(s)


    @staticmethod
    def ml_to_string(match_list, traj_pop):
        table = pt.PrettyTable()
        table.field_names = ['id-A', 'id-B','hkl-A', 'hkl-B', 'matrix-A', 'matrix-B', 'match-fit', 'cell-A', 'cell-B']
        for i in range(len(match_list)):
            cell_a = np.array(matrix_match.cell_to_cellpar_d2(np.dot(match_list[i][4], 
                                                            traj_pop[match_list[i][0]].get_cell()[0:2,0:2])))
            cell_a[-1] = cell_a[-1] / math.pi * 180

            cell_b = np.array(matrix_match.cell_to_cellpar_d2(np.dot(match_list[i][5], 
                                                            traj_pop[match_list[i][1]].get_cell()[0:2,0:2])))
            cell_b[-1] = cell_b[-1] / math.pi * 180
            
            table.add_row([*(match_list[i][:6]),
                        np.round(match_list[i][6],3),
                        np.round(cell_a,3),
                        np.round(cell_b,3)
                        ])
            
        return table.__str__()

from scipy.spatial import ConvexHull
class RCSPhaseDiagram:
    '''
    construct a convex hull of Eo(delta_n).
    for a binary AaBb compound, (eg. TiO2, A=Ti, a=1, B=O, b=2), which could be reduced to A(a/b)B form (i.e., TiO2 to Ti0.5O), 
    define Eo = E_slab - numB*E_ref, [E_ref = energy of unit A(a/b)B]
    define delta_n = numA - numB *(a/b)
    E_surface = Eo - delta_n*potentialA
    * Q. Zhu, L. Li, A. R. Oganov, and P. B. Allen, Phys. Rev. B 87, 195317 (2013).
    * https://doi.org/10.1103/PhysRevB.87.195317

    some codes modified from ase.phasediagram in this part.
    '''
    def __init__(self, references):
        '''
        reference must be a list of tuple (delta_n, Eo).
        binary AaBb compound only, due to the definition of delta_n.
        '''
        
        self.references = references

        self.points = np.zeros((len(self.references), 2))
        for s, ref in enumerate(self.references):
            self.points[s] = np.array(ref)
        
        hull = ConvexHull(self.points)

        # Find relevant simplices:
        ok = hull.equations[:, -2] < 0
        self.simplices = hull.simplices[ok]

    def decompose(self, delta_n):

        # Find coordinates within each simplex:
        X = self.points[self.simplices, 0] - delta_n

        # Find the simplex with positive coordinates that sum to
        # less than one:
        eps = 1e-15
        for i, Y in enumerate(X):
            try:
                x =  -Y[0] / (Y[1] - Y[0])
            except:
                pass
            if x > -eps and x < 1 + eps:
                break
        else:
            assert False, X

        indices = self.simplices[i]
        points = self.points[indices]

        coefs = [1 - x , x]
        energy = np.dot(coefs, points[:, -1])

        return energy, indices, np.array(coefs)


from ase.constraints import FixAtoms    #, IndexedConstraint
"""
modify FixAtoms class in ASE. choose if delete 'force' for fixed atoms.
ASE version = 3.22.1
"""

def modify_fixatoms():
    if hasattr(FixAtoms, 'init'):
        pass
    else:
        setattr(FixAtoms, "init", FixAtoms.__init__)
        setattr(FixAtoms, "__init__", fa_init_)
        setattr(FixAtoms, "change_force", FixAtoms.adjust_forces)
        setattr(FixAtoms, "adjust_forces", adjust_forces)

def fa_init_(cls, indices=None, mask=None, adjust_force = True):
    cls.__setattr__("adj_f", adjust_force)
    """ to use later
    IndexedConstraint().__init__(indices, mask)
    """
    cls.init(indices, mask)

def adjust_forces(cls, atoms, forces):
    if cls.adj_f:
        cls.change_force(atoms, forces)
    else:
        pass


from sklearn import cluster
def LayerIdentifier(ind, prec = 0.2, n_clusters = 4, lprec = 0.05):    
    """
    clustering by position[z]
    """
    pos = ind.get_scaled_positions()[:,2].copy()
    pos = np.array([[p,0] for p in pos])
    n, layers, kmeans = n_clusters, [], None
    #print("pos = {}".format(pos))
    for n in range(n_clusters, 1, -1):
        #print("n = {}".format(n))
        try:
            #I put a try-exception here because of situations of thin layers (4-)
            kmeans = cluster.KMeans(n_clusters=n, n_init='auto').fit(pos)
        except:
            continue
        centers = kmeans.cluster_centers_.copy()[:,0]
        centers = [(centers[i], i) for i in range(len(centers))]
        centers.sort(key = lambda c:c[0])
        layerid = {j[1]: i for (i,j) in enumerate(centers)}
        #print("centers = {}".format(centers))
        #print("layerid = {}".format(layerid))
        for i in range(1,len(centers)):
            if centers[i][0] - centers[i-1][0] < prec:
                break
        else:
            layers = [ [] for i in range(n)]
            labels = kmeans.labels_
            for i, a in enumerate(labels):
                layers[layerid[a]].append(i)
            #print("layers = {}".format(layers))
            for i in range(1, n):
                #print("layers{} = {}".format(i-1, pos[layers[i-1], 0]))
                #print("layers{} = {}".format(i, pos[layers[i], 0]))
                if np.min(pos[layers[i], 0]) - np.max(pos[layers[i-1], 0]) < lprec:
                    break
            else:
                break
    else:
        layers = [list(range(len(ind)))]

    return layers



class match_symmetry:
    """
    For two slices with symmetry: 
    R*x1 + T1 = x11;  (slice 1)
    R*x2 + T2 = x22;  (slice 2), which (x1, x11) are equivalent atoms and (x2, x22) are equivalent atoms, 
    suppose we translate slice 2 for distance xT to match its symmetry with slice 1, namely x2 = x1 + xT ; R*(x1 + xT) + T2 = x11 + xT
    and we get
    (R - I) *xT = T1 - T2               (*1)
    sometimes for slice 1 and slice 2,  
    R1*x1 + T1 = x11;  (slice 1)
    R2*x2 + T2 = x22;  (slice 2)
    another transform matrix Tr is need to transform R2 to R1; 
        for example, if slice1 has x_z plane as mirror plane [R1 = [1,0,0], [0,-1,0],[0,0,1] ]
        and slice2 has y_z plane as mirror plane [R2 = [-1,0,0],[0,1,0],[0,0,1] ]
        and the transform matrix of R2 to R1 satisfies Tr*R2 = R1.
    in that case, a rotation of crystal and reset the basis are needed. 
    
    """
    def __init__(self, sym1 = (None,None), sym2 = (None, None), z_axis_only = False):
        self.sgm = []
        self.sortedranks = []
        
        sym1, sym2 = self.removetransym(sym1), self.removetransym(sym2)
        self.r1, self.t1 = sym1
        self.r2, self.t2 = sym2
        #print("self.r1 = {}".format(self.r1))
        #print("self.r2 = {}".format(self.r2))
        #print("self.t1 = {}".format(self.t1))
        #print("self.t2 = {}".format(self.t2))
                    
        for i, r1 in enumerate(self.r1):
            for j, r2 in enumerate(self.r2):
                #label, trans = self.issametype(r1, r2)
                label = (r1 == r2).all()
                trans = np.eye(3)
                if label:
                    self.sgm.append((r1 - np.eye(3), self.t1[i]- np.dot(self.t2[j], trans), trans))
        #print("sgm preprocess = {} ".format(self.sgm))
        if z_axis_only:
            self.sgm = [m for m in self.sgm if (m[0] == m[0]*np.reshape([1,1,0]*2+[0]*3, (3,3))).all()]

        #print("sgm z_only = {} ".format(self.sgm))
        #sgm: list of tuples of (R-I, T1-T2, Tr)
        self.sortrank()
    
    @property    
    def has_shared_sym(self):
        return len(self.sgm)>0

    def removetransym(self, sym):
        #to remove simple translate symmetry. For cells generate by subcell* (_x, _y, _z) 
        #    and some translation symmerty in spacegroup.
        
        #1. remove simple R=I matrix
        transymmeryDB = []
        rot, tr = sym
        #print(np.where( ((rot ==np.eye(3)).all(axis = 1)).all(axis = 1)  ==True))
        index = np.where( ((rot ==np.eye(3)).all(axis = 1)).all(axis = 1)  ==True) [0]
        if len(index) >1:
            #assert np.allclose(tr[0], np.array([0,0,0]), rtol=0, atol=0.01) == True 'first translation matrix must be [0,0,0]'
            transymmeryDB = (tr[index])[1:] - (tr[index])[0]
            transymmeryDB = np.array([tt - int(tt) if tt >= 0 else tt - int(tt) + 1 for _i_ in range(len(transymmeryDB)) for tt in list(transymmeryDB[_i_])])
            transymmeryDB = np.reshape(transymmeryDB, (-1,3))
            #print("transDB: {}\nend".format(transymmeryDB))
        
        keep = [i for i in range(len(rot)) if i not in list(index)]
        rot, tr = rot[keep], tr[keep]

        if len(index)  == 1:
            return rot, tr
        
        #2. remove R+T symmetry couldn't obtained by re-set of axis to be R

        keep = [i for i in range(len(rot)) if np.linalg.matrix_rank(rot[i] - np.eye(3)) >= np.linalg.matrix_rank(np.c_[rot[i] - np.eye(3),tr[i]])]
        #for i in range(len(rot)):
            #print("rotrank = {}".format(np.linalg.matrix_rank(rot[i] - np.eye(3))))
            #print("trank = {}, {}".format(np.linalg.matrix_rank(np.c_[rot[i] - np.eye(3),tr[i]]), np.c_[rot[i] - np.eye(3),tr[i]]))
        rot, tr = rot[keep], tr[keep]
        #print("rot = {}".format(rot))
        #print("tr = {}".format(tr))
        #3. choose a lucky r to represent all of the equivalent r.
        uniquer, uniquei = np.unique(rot, axis=0, return_index=True)

        if len(uniquer) == len(rot):
            return rot, tr

        to_del = []
        for j, r in (zip(uniquei, uniquer)):
            _to_del = np.where((r == rot).all(axis = 1).all(axis = 1))[0]
            #print("_to_del = {}".format(_to_del))
            for d in _to_del:
                if d==j :
                    continue
                t = tr[j] - tr[d]
                #print(t)
                t = np.array([tt - int(tt) if tt >= 0 else tt - int(tt) + 1 for tt in list(t)])
                #print(t)
                #print([np.allclose(db, t, rtol=0, atol=0.01) for db in transymmeryDB])
                if np.array([np.allclose(db, t, rtol=0, atol=0.01) for db in transymmeryDB]).any() ==True:
                    to_del.append(d)

        keep = [i for i in range(len(rot)) if i not in to_del] 
        #print('index = {}'.format(index))
        #print('todel = {}'.format(to_del))
        #print('keep = {}'.format(keep))
        return rot[keep], tr[keep]    

    def issametype(self, r1, r2):
        if (r1 == np.eye(3)).all():
            return (False, None)
        if (r2 == np.eye(3)).all():
            return (False, None)
        r = np.dot(r1, np.linalg.inv(r2))
        return (True, r) if np.linalg.det(r) ==1 and (r[2] == np.array([0,0,1])).all() else (False, None)

    def sortrank(self):
        sgm = self.sgm
        #print(sgm)
        ranks = [np.linalg.matrix_rank(r) for r, _, _ in sgm]
        #print(ranks)
        for i,rank in enumerate(ranks):
            rot, _, _ = sgm[i]
            if rank == 3:
                ranks[i] = sgm[i] + (rank, list(range(0,3)))
            else:
                axis = np.array([0,1,2])
                for a in axis:
                    r = np.eye(3)
                    sureindex = np.where(axis!=a)[0] if rank ==2 else np.array([a])

                    r [sureindex] = rot[sureindex]
                    if not np.linalg.det(r) ==0:
                        ranks[i] = sgm[i] + (rank, sureindex) if isinstance(ranks[i], np.int64) else ranks[i] + (sureindex, )

        #ranks: list of tuples of (R-I, T1-T2, Tr, rank(R-I), sureindexA, sureindexB...) in which [0:3] are a copy of self.sgm
        #sortedranks is a 2D list to sort "ranks" with rank(R-I). The first dimention stands for the rank, 
        #    for example, sortedranks[0] always has no contents, for no matrix's rank is 0 in our situation; 
        #    all members in ranks with the third item == 1 are placed in sortedranks[1], etc.
        #Why for this: 
        #    We can't directly solve equation (*1) with numpy, for rank(R-I) in most case is lower than 3.
        #    Consider the same example in the intro, a slice with x_z plane as mirror plane [R1 = [1,0,0], [0,-1,0],[0,0,1] ]
        #    and R-I = [0,0,0], [0,-2,0],[0,0,0], (rank = 1), which means the slice can move freely along x, z axis without breaking its symmetry.
        #    we need to know which "index" is free in R-I leading to the zero rank, i.e. not "sureindex".
        #    Then we try to get a combination of R-I s if there are more than one R matrix for slice 1 and 2,
        #    get more "sureindex" and minimum the freedom of xT for a higher symmetry.
        #    Why for sureindexA, sureindexB... :
        #        sometimes we could only know the constraint is x=y rather than make sure one of them. 
        #        Consider if the normal vector of the mirror plane is [1,-1,0], and we could say the sureindex could be x or y. 

        trs = np.array([x[3] for x in ranks])
        for rank in range(0,4):
            index = np.where(trs == rank)[0]
            self.sortedranks.append([ranks[i] for i in index])
        #print("self.sortedranks ={}".format(self.sortedranks) )
        
        self.availrank = [rank for rank in range(0,4) if len(self.sortedranks[rank])]
        return 
    
    def getachoice(self, trynum = 5):
        availrank = self.availrank.copy()
        nowrank = 0
        choice = []
        rotmatrix = []
        #print("availrank ={}".format(availrank) )
        for _ in range(0,trynum):
            nowrank = 0
            choice = []
            nowindex = []
            for _ in range(0,trynum):
                r = np.random.choice(availrank)
                c = np.random.choice(range(0, len(self.sortedranks[r])))
                if len(rotmatrix):
                    if not (rotmatrix == self.sortedranks[r][c][2]).all():
                        continue
                else:
                    rotmatrix = self.sortedranks[r][c][2]
                #print("self.sortedranks[r][c] {}".format(self.sortedranks[r][c]))
                index = [j for i, j in enumerate(self.sortedranks[r][c]) if i >=4]
                #print("index = {}".format(index))
                for i in index:
                    #print("i = {} nowindex = {}".format(i, nowindex))
                    _index_ = list(nowindex) + list(i)
                    if len(_index_) == len(list(set(_index_))):
                        
                        nowrank += r
                        availrank = [rank for rank in availrank if rank <= 3-nowrank]
                        nowindex.extend(i)
                        #print("choice = {}, nowindex = {}".format(choice, self.sortedranks[r][c][0:3] + (i, )))
                        choice.append(self.sortedranks[r][c][0:3] + (i, ))
                        if nowrank ==3 or len(availrank) == 0 :
                            return (True, choice)
        return (True, choice) if len(choice) else (False, None)    
    
    def get(self):

        if self.has_shared_sym:
            #print("self.sgm = {}".format(self.sgm))
            label, choice = self.getachoice()
            #print("choice = {}".format(choice))
            if label:
                trans = np.zeros(3)
                #trans = np.random.uniform(0,1,size=3)
                for c in choice:
                    index = c[3]
                    trans[index] = np.dot(np.linalg.inv(c[0][index][:,index]), c[1][index])
                return trans, choice[0][2]
                
        return (np.array([np.random.uniform(0,1), np.random.uniform(0,1),0]), np.eye(3))


class weightenCluster:
    def __init__(self, d = 0.23):
        self.d = d
    
    def Exp_j(self, ith, jth):
        return math.exp(-(self.atoms.get_distance(ith, jth) - self.radii[ith] - self.radii[jth]) / self.d)
    
    def choseAtom(self, ind, atomnum = -1):
        if isinstance(ind, Atoms):
            self.atoms = ind.copy()
        else:
            self.atoms = ind.atoms.copy()
            
        self.radii = [covalent_radii[atom.number] for atom in self.atoms]

        O = np.zeros(len(self.atoms))
        for i, _ in enumerate (self.atoms):
            Exp = np.array([self.Exp_j(i, j) for j in range(len(self.atoms)) if not i == j])
            O[i] = np.sum(Exp) / np.max(Exp)
        
        probability = np.max(O) - O
        if not atomnum == -1:
            probability = np.array([p if self.atoms[i].number == atomnum else 0 for i, p in enumerate(probability)])
        probability = probability / np.sum(probability)
        a = np.random.random()
        #print('probability = {}, a = {}'.format(probability, a))
        
        for i, _ in enumerate(probability):
            a -= probability[i]
            if a < 0 :
                break
        #print('i = {}, prange {} ~ {}'.format(i, np.sum(probability[:i]), np.sum(probability[:i+1])))
        return i

class ClusComparator:
    def __init__(self, tolerance = 0.1):
        self.tolerance = tolerance
    def distance(self, vector1, vector2):
        raise NotImplementedError()
    def fingervector(self, ind):
        raise NotImplementedError()

    def looks_like(self, aInd, bInd):
        
        for ind in [aInd, bInd]:
            if 'spg' not in ind.atoms.info:
                ind.find_spg()
        a,b = aInd.atoms,bInd.atoms
        
        if a.info['spg'] != b.info['spg']:
            return False
        if Counter(a.info['priNum']) != Counter(b.info['priNum']):
            return False

        vector1, vector2 = aInd.fingervector, bInd.fingervector
        distance = self.distance(vector1, vector2)
        #print('distance = {}'.format(distance))
        if distance > self.tolerance:
            return False

        return True

class OverlapMatrixComparator(ClusComparator):
    """
    Borrowed from Sadeghi et al, J. Chem. Phys. 139, 184118 (2013) https://doi.org/10.1063/1.4828704 ;
    J. Chem. Phys. 144, 034203 (2016) https://doi.org/10.1063/1.4940026
    """
    def __init__(self, orbital= 's', tolerance = 1e-4, width = 1.0):
        super().__init__(tolerance = tolerance)
        self.orbital = orbital
        #orbital: overlap orbital, could be 's' <s only> or 'p' <s and p>
        
        self.width = width
        #width: Gaussian width αi 
        #inversely proportional to the square of the covalent radius of atom i, namely, ai = self.width/radius**2

    def distance(self, vector1, vector2):
    #Euclidean distance between vector1 and vector2
        v = vector1 - vector2
        return math.sqrt(np.dot(v, v)/ len(v)) 

    def S(self, ind):
        N, S = len(ind), None
        if self.orbital == 's':
            S = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if j < i:
                        S[i][j] = S[j][i]
                    else:
                        S[i][j] = self.OverlapM(ind[i], ind[j])

        elif self.orbital == 'p':
            S = np.zeros((4*N, 4*N))
            for i in range(0, N):
                for j in range(i, N):
                    subS = self.OverlapM(ind[i], ind[j])
                    #si, sj = i, j
                    S[i][j] = subS[0][0]
                    S[j][i] = S[i][j]
                    #pxi, pyi, pzi = N+i, 2*N+i, 3*N+i 
                    for x in range(1,4):
                        S[i][x*N+j] = subS[0][x]
                        S[j][x*N+i] = subS[x][0]
                    for x1 in range(1,4):
                        for x2 in range(1,4):
                            S[x1*N+i][x2*N+j] = subS[x1][x2]
                            S[x1*N+j][x2*N+i] = subS[x1][x2]
        
        return S
                    
    def fingervector(self, ind):
        vector = np.linalg.eig(self.S(ind))[0]
        return np.array(sorted(vector, reverse = True))

    def OverlapM(self, ati, atj):
        ri_rj = ati.position - atj.position
        rij2 = np.sum([r**2 for r in ri_rj])
        ai, aj =  self.width / np.array([covalent_radii[ati.number]**2, covalent_radii[atj.number]**2])
        #ai, aj =  self.width * np.array([covalent_radii[ati.number], covalent_radii[atj.number]])  ???????
        N = ai*aj/(ai+aj)
        Sij = (2*N/math.sqrt(ai*aj))**1.5 * math.exp(-N*rij2)
        if self.orbital == 's':
            #ss_ij = Sij
            return Sij
        
        def ps_ij(x):
            #x equals {0,1,2}, stands for {px, py, pz}
            return -2*N/math.sqrt(ai)*((ri_rj)[x]) *Sij
        def ps_ji(x):
            return 2*N/math.sqrt(aj)*((ri_rj)[x]) *Sij
        def pp_ij(x1, x2):
            delta = 1 if x1==x2 else 0
            return 2*N/math.sqrt(ai*aj)*Sij*( delta-2*N*((ri_rj)[x1])*((ri_rj)[x2]) )

        """
        returns a overlap matrix S of ati, atj 
        i\j     s        px      py      pz
        s   _si-sj_| ___ si-pj_____ 
        px            |     
        py    pi-sj |          pi-pj
        pz            |

        In this matrix, pi-sj != si-pj, but pi-pj = pj-pi.
        """
        S = np.zeros((4,4))
        S[0][0] = Sij
        for x in range(1,4):
            S[0][x] = ps_ji(x-1)
            S[x][0] = ps_ij(x-1)

        for x1 in range(1,4):
            for x2 in range(1,4):
                if x2 < x1:
                    S[x1][x2] = S[x2][x1]
                else:
                    S[x1][x2] = pp_ij(x1-1, x2-1)

        return S

    def looks_like(self,aInd,bInd):
        return super().looks_like(aInd, bInd)   

class OganovComparator(ClusComparator):
    """
    Borrowed from Lyakhov et al, Computer Physics Communications 184 (2013) 1172–1182 https://doi.org/10.1016/j.cpc.2012.12.009 ;
    J. Chem. Phys. 130, 104504 (2009) https://doi.org/10.1063/1.3079326

    2 atom species only???
    """
    def __init__(self, tolerance = 0.1, width = 0.075, delta = 0.05, dimComp = 630, maxR = 15):
        super().__init__(tolerance = tolerance)
        self.width = width
        self.delta = delta
        self.dimComp = dimComp
        self.maxR = maxR
        #for clusters, maxR being cluster's bounding_sphere is okay. 15 is for extended systems.
    
    #Gaussian-smeared delta-function; parameter *10 comes from https://doi.org/10.1016/j.jcp.2016.06.014
    def f_delta(self, x, x0):
        if x < x0 - self.width * 10 or x > x0 + self.width * 10:
            return 0
        else:
            return math.exp(-(x - x0)**2 / self.width **2 /2) / self.width / math.sqrt(2*math.pi)

    def F_AB(self, ind):
        symbols = list(set(ind.get_chemical_symbols()))
        A, B = None, None
        if len(symbols) >1:
            A = [i for i, atom in enumerate(ind) if atom.symbol == symbols[0][0] ]
            B = [[i for i, atom in enumerate(ind) if atom.symbol == symbols[0][1] ]]*len(A)
        else:
            #ind = ind.copy()
            #a = Atom(symbol=atomic_numbers[symbols[0][0]] +1, position=np.mean(ind.get_positions(), axis=0))
            #ind += a
            #A = [len(ind)-1]
            #B = [i for i in range(0, len(ind)-1)]
            A = [i for i in range(0, len(ind))]
            B = [[j for j in range(0, len(ind)) if not j==i] for i in A]
                
        
        def f_ab_R(R):
            F = 0
            for i, ai in enumerate(A):
                f = 0
                for bj in B[i]:
                    Rij = ind.get_distance(ai, bj)
                    f += self.f_delta(R, Rij) / (Rij**2)
                F += f/4/math.pi / self.delta / len(B)
            return F / len(A)
        
        return f_ab_R
    
    def fingervector(self, ind):
        f = self.F_AB(ind)
        step = self.maxR / self.dimComp
        return np.array([f(k*step) for k in range(1, self.dimComp+1)])

    def distance(self, vector1, vector2):
    #cosine distance of vector1, vector2
        return 0.5*(1- np.dot(vector1, vector2)/math.sqrt(np.dot(vector1, vector1) * np.dot(vector2, vector2)))

    def looks_like(self,aInd,bInd):
        return super().looks_like(aInd, bInd)

#from ..generators.gensym import wyckoff_positions_3d

class sym_rattle:

    @staticmethod
    def read_ds(spacegroup, wyckoff_label):
        #static const vector< vector<double> > wyck [spg-1] = [num_of_wyckpos, label, matrix, mul, uni]
        data = wyckoff_positions_3d[spacegroup-1]

        index = (ord(wyckoff_label) -97 ) *15 +2
        wyck_matrix = np.array(data[index:index+12])
        return wyck_matrix.reshape(3,4)
        """
        num_of_wyck = data[0]
        data = np.array(data[1:]).reshape(num_of_wyck, 15)
        ds = [ {chr(line[0]): [
                        np.array(line[1:13]).reshape(3,4),
                        line[13],
                        bool(line[14])
                    ]
                    } for line in data]
        return ds
        """
    @staticmethod
    def _share_method_(atoms, func_get_all_position, symprec, trynum, mutate_rate, rattle_range, d_ratio):
        atoms = atoms.copy()
        sym_ds = spglib.get_symmetry_dataset(atoms, symprec)

        equivalent_atoms = sym_ds['equivalent_atoms']
        rotations, translations = sym_ds['rotations'], sym_ds['translations']
        spg = sym_ds['number']
        wyckoff_labels = sym_ds['wyckoffs']

        for key in list(Counter(equivalent_atoms).keys()):

            eq = np.where(equivalent_atoms == key)[0]

            if np.random.rand() < 1 - (1-mutate_rate)**len(eq):
                newatoms = atoms.copy()
                
                pos, label = atoms[key].position, wyckoff_labels[key]

                for _ in range(trynum):
                    r, theta, phi = rattle_range * np.random.rand()**(1/3), \
                                            np.random.uniform(0, np.pi), \
                                            np.random.uniform(0, 2*np.pi)
                    _p_ = pos + r * np.array([np.sin(theta) * np.cos(phi), 
                                                                        np.sin(theta) * np.sin(phi),
                                                                        np.cos(theta)])
                    _sp_ = np.dot(_p_, np.linalg.inv(atoms.get_cell()))

                    new_cartpos = func_get_all_position( _sp_, (atoms.get_cell(), atoms.get_pbc()), \
                                                                                        (rotations, translations, spg, label), \
                                                                                        len(eq))
                    
                    if new_cartpos is None:
                        continue

                    newatoms.positions[eq] = new_cartpos
                    
                    if check_distance(newatoms, d_ratio) and \
                                    (not spglib.get_spacegroup(newatoms, symprec) == 'P1 (1)'):
                        #print("new_spg", spglib.get_spacegroup(newatoms, symprec))
                        atoms = newatoms.copy()
                        break
        return atoms

    @staticmethod
    def merge_close(_sp_, cellinfo, symmetry, target_length, *args):
        cell, pbc = cellinfo
        rotations, translations, spg, wyckoff_label = symmetry

        new_sp = np.dot(rotations, _sp_) + translations

        for i, _ in enumerate(new_sp):
            for j, _ in enumerate(np.where(pbc)[0]):
                new_sp[i][j] += -int(new_sp[i][j]) if new_sp[i][j] >= 0 else -int(new_sp[i][j]) +1
        
        new_cartpos = np.dot(new_sp, cell)

        if not len(new_cartpos) == target_length:            
            new_cartpos = symposmerge(new_cartpos, target_length).merge_pos()
            
        return new_cartpos


    @staticmethod
    def keep_spg(atoms, symprec = 0.1, trynum = 20, mutate_rate = 0.25, rattle_range = 4, d_ratio = 0.5):
        """
        (a). find equivalent atoms
        (b). mutate a unique(*non equivalent to any mutated atom) atom
        (c). use rotations, translations to get its equivalent atoms to keep symmetry
        (d). dealing with different multiplicity number: merge close atoms. **
                **Andriy O. Lyakhova et al, Computer Physics Communications 184 (2013) 1172-1182
        """
        
        func_get_all_position = sym_rattle.merge_close
        return sym_rattle._share_method_(atoms, func_get_all_position, symprec, trynum, mutate_rate, rattle_range, d_ratio)
                    
    @staticmethod
    def use_wyck(_sp_, cellinfo, symmetry, target_length, *args):
        cell, pbc = cellinfo
        rotations, translations, spg, wyckoff_label = symmetry
        wyck_matrix = sym_rattle.read_ds(spg, wyckoff_label)
        
        _sp_ = np.dot(wyck_matrix[:, :3], _sp_) 

        new_sp = np.dot(rotations, _sp_) + translations
        new_sp = np.unique(new_sp, axis = 0)

        return np.dot(new_sp, cell) if  len(new_sp) == target_length else None

    @staticmethod
    def keep_comb(atoms, symprec = 0.1, trynum = 10, mutate_rate = 0.25, rattle_range = 4, d_ratio = 0.5):
        """
        (a). find wyckoff_labels
        (b). mutate but keep this wyckoff position
        (c). calculate all equivalent atoms
        """
        func_get_all_position = sym_rattle.use_wyck
        
        return sym_rattle._share_method_(atoms, func_get_all_position, symprec, trynum, mutate_rate, rattle_range, d_ratio)
        

class symposmerge:
    def __init__(self, positions, length, dmprec = 4):
        self.positions = positions
        self.length = length
        self.prec = dmprec
        self.dmupdate()

    #update distance matrix 
    def dmupdate(self):
        l = len(self.positions)
        self.distance_matrix = np.full((l, l), 0.)
        for i in range(l):
            for j in range(i+1, l):
                self.distance_matrix[i][j] = np.round(math.sqrt(np.sum([x**2 for x in self.positions[i] - self.positions[j]])), self.prec)
        self.sorted_dis = np.unique(np.sort(self.distance_matrix.flatten()))
        #print("updatedm = {}".format(self.distance_matrix))
    
    #return index of [i,j] whose i-j distance eqs sorted_dis[mindis]
    def where(self, mindis = 1):
        pair_index = np.nonzero(self.distance_matrix == self.sorted_dis[mindis])
        #print("distance eq {}th mindis, {}: {}".format(mindis, self.sorted_dis[mindis], [[i, j] for i,j in zip(pair_index[0], pair_index[1])]))
        return [[i, j] for i,j in zip(pair_index[0], pair_index[1])]
    
    
    def merge_pos(self):
        positions = self.positions
        #print("merge_pos, {} to {}".format(positions, self.length))
        while len(positions) > self.length:
            to_merge = []
            for mindis in range(1, len(self.sorted_dis)):
                m = self.where(mindis)
                if len(m) > 1:
                    to_merge = m
                    break
            else:
                mindis = 1
                to_merge = self.where(1)

            #print("to_merge {}".format(to_merge))

            if len(np.unique(np.array(to_merge))) < len(np.array(to_merge).flatten()):
                _m, originindex = [], []
                for m0 in to_merge:
                    for index, pairs in enumerate(_m):
                        share = [(i in pairs) for i in m0]
                        if np.any(share):
                            _m[index] =  np.unique([*pairs, *m0])
                            originindex[index].append(m0)
                            break
                    else:
                        _m.append(m0)
                        originindex.append([m0])
                
                for index, pairs in enumerate(_m):
                    if not len(pairs) == len(originindex[index]) and not len(pairs) ==2:
                        for ith,p in enumerate(pairs):
                            if np.all([p in x for x in originindex[index]]):
                                _m[index] = [x for x in _m[index] if not x==p]

                to_merge = _m

            merged_pos = np.array([np.mean(positions[m], axis=0) for m in to_merge])

            to_merge = [x for m in to_merge for x in m ]
            positions = np.delete(positions, to_merge, 0)
            positions = np.append(positions, merged_pos, axis=0)


            self.positions = positions.copy()
            self.dmupdate()
        
        return positions if len(positions) == self.length else None
        

if __name__ == '__main__':
    
    import ase.io

    Si = ase.io.read('Si.cif')
    Quartz = ase.io.read('Quartz.cif')
    Cristobalite = ase.io.read("Cristobalite.cif")
    Tridymite = ase.io.read("Tridymite.cif")


    im = InterfaceMatcher(Si, Tridymite, range_hkl = [-5,6], range_matrix = [-3,4], 
                          range_area=[0., 100.], range_a=[0,13.],range_ang=[45.,135.],
                          thread_para = 50)
    matchmodes  = im.match_result(verbose=True, save_intermediate='inter_ml.npy')


    #ma = cutcell(Quartz, [1,1,0], direction=[-2,1,0], totslices=1).cut_traj[0]
    #ase.io.write("100.vasp", ma, vasp5=1)
    #from ase.build import surface
    #s1 = surface(Quartz, (-2, 1, 0), 2)
    #s1.center(vacuum=10, axis=2)
    #ase.io.write('quartz.vasp', s1, vasp5=1)