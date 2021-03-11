from . import moveatom
import numpy as np
import ase.io
import spglib as spg 
from ase.data import atomic_numbers, covalent_radii
from .utils import sort_elements
import logging
import copy

def str_(l):
    l=str(l)
    return l[l.find('[')+1:l.find(']')]

class move:
    def __init__(self, originpositions, shift, atomnumber, atomname, lattice, spacegroup = 1, eq_atoms=None):
        self.info=moveatom.Info(np.random.randint(1000))
        for i in range(len(atomnumber)):
            pos=np.array(originpositions[i]).flatten()
            self.info.AppendAtoms(int(atomnumber[i]),atomname[i],covalent_radii[atomic_numbers[atomname[i]]],pos)   #(atomsnumber, atomname, atomradius, atomspos)
            self.info.AppendShift(np.array(shift[i]))  #shift for the atoms above
            if(eq_atoms):
                self.info.EquivalentAtoms(np.array(1.0*eq_atoms[i]))
        self.lattice=lattice.flatten()
        self.info.SetLattice(np.array(self.lattice))
        self.info.spacegroup = spacegroup
        self.info.resultsize=1
        
    def Rotate(self, centeratomtype, centeratompos, clustersize, targetatomtype, threshold, maxattempts):
        self.info.Rotate(centeratomtype, centeratompos, clustersize, targetatomtype, threshold, maxattempts)
        self.info.WritePoscar()
        '''
        for i in range(self.info.resultsize):
            #print(self.info.GetPosition(i))
        '''
        return

    def Shift(self, threshold, maxattempts):
        label = self.info.Shift(threshold, maxattempts)
        self.info.WritePoscar()
        '''
        for i in range(self.info.resultsize):
            #print(self.info.GetPosition(i))
        '''
        return label

    def WritePoscar(self):
        self.info.WritePoscar()
        return

    def GetPos(self):
        return self.info.GetPosition(0)

#m=move()
#m.Shift(0.5, 100)
#m.Rotate(1, 1, 6, 0, 0.5, 100)

class reconstruct:
    def __init__(self, moverange, originlayer, threshold, maxattempts=100):
        self.originlayer=originlayer
        self.range=moverange
        self.threshold = threshold
        self.maxattempts = maxattempts

        self.atoms = sort_elements(self.originlayer)
        self.atoms.set_cell(self.atoms.get_cell_lengths_and_angles().copy(), scale_atoms=True)
        
        self.pos=[]
        self.atomname=[]
        self.atomnum=[]
        self.shifts=[]
        self.eq_atoms=[]

    def reconstr(self):
        self.lattice=np.array(self.atoms.get_cell())
        atomicnum=self.atoms.numbers
        atomname=self.atoms.get_chemical_symbols()
        #dataset = spglib.get_symmetry_dataset(self.atoms, symprec=1e-2)
        #self.spacegroup = dataset['number']
        #transmatrix = dataset['transformation_matrix']
        #shiftarray = dataset['origin_shift']
        #eq_atoms=dataset['equivalent_atoms']

        unique, t=np.unique(atomicnum, return_index=True)
        for i in range(len(unique)):
            atomnum=np.sum(atomicnum==unique[i])
            self.atomname.append(atomname[t[i]])

            index = range(t[i], t[i]+atomnum)
            atom_part = self.atoms[index].copy()
            #eqatom = eq_atoms[index].copy()
            #eqatom -= t[i]

            #index = sorted(range(0,atomnum), key=lambda x: eqatom[x] )
            #atom_part = atom_part[index]
            #eqatom = eqatom[index]

            atompos=atom_part.get_scaled_positions().copy()
            #for i in range(len(atompos)):
                #atompos[i] = np.dot(transmatrix, atompos[i]) + shiftarray

            shift1=atompos[:,2].copy()
            shift1=np.array(shift1).flatten()
            shift1*=self.range

            self.shifts.append(shift1)
            self.atomnum.append(atomnum)

            self.pos.append(atompos)
            #self.eq_atoms.append(eqatom)

        m=move(self.pos, self.shifts, self.atomnum, self.atomname, self.lattice)  #, self.spacegroup, self.eq_atoms)
            
        label = m.Shift(self.threshold, self.maxattempts)
        if label:
            self.positions=m.GetPos()
            self.positions=np.array(self.positions).reshape(sum(self.atomnum),3)

            #for i in range(len(self.positions)):
                #self.positions[i] = np.dot(np.linalg.inv(transmatrix), (self.positions[i]-shiftarray))
        
            return label, self.positions
        else:
            return label, None

    def WritePoscar(self, filename):
        f=open(filename,'w')
        f.write("filename\n")
        f.write('1 \n')
        for i in range(3):
            f.write(str_(self.lattice[i])+'\n')

        for name in self.atomname:
            f.write(name+'  ')

        f.write('\n')
        f.write(str_(np.array(self.atomnum)))
        f.write('\nDirect\n') 
        po=self.positions
        for i in range(len(po)):
            f.write(str_(po[i])+'\n')
        f.close()
        return

class resetLattice:
    def __init__(self, atoms=None, expandsize = (8,8,8)):
        if atoms:
            #1. build a very large supercell
            supercell = atoms * expandsize
            supercell.translate( [np.sum( np.dot( np.diag([-int((s-1)/2) for s in expandsize]), atoms.get_cell()), axis = 0)] *len(supercell))
            self.supercell = supercell
    
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
        assert len(index)>0, "err in resetLattice: no atoms in newcell"
        return supercell[index].copy()

from ase.geometry import cell_to_cellpar
class cutcell:
    def __init__(self,originstruct, layernums, totslices= None, direction=[0,0,1], rotate = 0, vacuum = 1.0):
        """
        [auto, but be given is strongly suggested] totslices: layer number of originstruct
        layernums: layer number of [bulk, relaxable, rcs_region]
        [*aborted] startpos:  start position of bulk layer, default =0.9 to keep atoms which are very close to z=1(0?)s.
        direction: miller indices
        rotate: [not implied yet] angle of R.
        vacuum: [in Ang] vacuum to add to rcs_layer  
        """
        originatoms = originstruct.copy()
        atoms=originatoms.copy()

        #TODO: add rotate
        
        #2. get surface of conventional cell
        rx, ry, rz = direction[0], direction[1], direction[2]            
            
        points = [[rx, 0, 0], [0, ry, 0], [0, 0, rz]]
        newcell_c = np.array([rx, ry, rz])
        newcell = []
        left = []
        for i, point in enumerate(points):
            if point == [0,0,0]:
                newcell_a = [1 if j==i else 0 for j in range(3)]
                newcell.append(newcell_a)
            else:
                left.append(point)
        i = 1
        while len(newcell) < 2:
            newcell_a = np.array(left[i]) - np.array(left[0])
            newcell.append(newcell_a)
            i+=1
        newcell.append(newcell_c)
        newcell = np.array(newcell)            
        newcell = np.dot(newcell, atoms.get_cell())
        
        #3. get primitive surface vector and startpos
        supercell = resetLattice(atoms)
        newlattice = supercell.get(newcell)
        allayers = None
        if totslices is None:
            for totslices in range(8, 1,-1):
                try:
                    allayers = LayerIdentifier(newlattice, prec = 0.5/totslices, n_clusters = totslices +1, lprec = 0.4/totslices)
                    if len(allayers) == totslices or len(allayers) == totslices+1:
                        break
                except:
                    pass
            logging.warning("Number of total layers is not given. Used auto-detected value {}.".format(totslices))
        else:
            allayers = LayerIdentifier(newlattice, prec = 0.5/totslices, n_clusters = totslices +1, lprec = 0.4/totslices)

        onelayer = newlattice[allayers[0]]
        startpos = np.max(newlattice.get_scaled_positions()[:,2]) + 0.1 
        startpos = startpos - int(startpos)

        if len(allayers) == totslices + 1:

            onelayer = newlattice[allayers[1]]
            startpos = np.max(newlattice[allayers[-2]].get_scaled_positions()[:,2]) + 0.01
            startpos = startpos - int(startpos)
        
        onelayer.set_cell(onelayer.get_cell()[:] * np.reshape([1]*6 + [2.33]*3, (3,3)))
        #print(startpos)
        surface_vector = spg.get_symmetry_dataset(onelayer,symprec = 1e-4)['primitive_lattice']
        abcc, abcp = cell_to_cellpar(onelayer.get_cell()[:])[:3], cell_to_cellpar(surface_vector)[:3]
        axisc = np.where(abcp == abcc[2])[0]
        assert len(axisc) ==1, "cannot match primitive lattice with origin cell, primitive abc = {} while origin abc = {}".format(abcp, abcc)
        if not axisc[0] ==2:
            surface_vector[[axisc[0], 2]] = surface_vector[[2, axisc[0]]]
 

        #4. get surface cell!
        #print(layernums)
        #5. expand unit surface cell on z direction
        bot, mid, top = layernums[0], layernums[1], layernums[2]
        slicepos = np.array([0, bot, bot + mid,  bot + mid + top])/totslices
        slicepos = slicepos + np.array([startpos]*4)
        logging.info("cutslice = {}".format(slicepos)) 

        #6. build bulk, relaxable, rcs layer slices 
        pop= []
        if slicepos[-1]==slicepos[-2]:
            slicepos = slicepos[:-1]
            logging.info("warning: rcs layer have no atoms. Change mode to adatoms.")  

        for i in range(1, len(slicepos)):

            cell = surface_vector.copy()
            cell[2] = newcell[2] * (slicepos[i]-slicepos[i-1])
            origin = (slicepos[i-1] if i==1 else slicepos[i-1]-slicepos[i-2]) * newcell[2]

            layerslice = supercell.get(cell, neworigin = origin)

            if len(layerslice)==0:
                slicename = ['bulk', 'relaxable', 'reconstruct']
                raise Exception("No atom in {} layer".format(slicename[i-1]))

            layerslice=sort_elements(layerslice)
            pop.append(layerslice)

        #add extravacuum to rcs_layer  
        if len(pop)==3:        
            cell = pop[2].get_cell()
            cell[2]*= ( 1.0 + vacuum/pop[2].get_cell_lengths_and_angles()[2])
            pop[2].set_cell(cell)
        
        logging.info("save cutslices into file layerslices.traj")
        ase.io.write("Ref/layerslices.traj",pop,format='traj')


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


from ase.constraints import FixAtoms
class fixatoms(FixAtoms):
    """
    modified from FixAtoms class in ASE. Don't delete 'force' for fixed atoms.
    """
    def adjust_forces(self, atoms, forces):
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
        kmeans = cluster.KMeans(n_clusters=n).fit(pos)
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

if __name__ == '__main__':
    t=reconstruct(0.8, ase.io.read("POSCAR_3.vasp",format='vasp'), 0.8,2 )
    t.reconstr()
    t.WritePoscar("result.vasp")
