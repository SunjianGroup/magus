import numpy as np
try:
    from . import GenerateNew
except:
    import GenerateNew
from ase.data import atomic_numbers, covalent_radii
from ase import Atoms,build
from ase.spacegroup import Spacegroup 
from spglib import get_symmetry_dataset
from ase.geometry import cellpar_to_cell,cell_to_cellpar
from scipy.spatial.distance import cdist, pdist
import ase,ase.io
import copy
import logging
from .utils import *
from .reconstruct import reconstruct, cutcell, match_symmetry, resetLattice
from .population import RcsInd
import math


log = logging.getLogger(__name__)


class Generator:
    def __init__(self,parameters):
        self.p = EmptyClass()
        Requirement=['symbols','formula','minAt','maxAt','spgs','dRatio','fixCell','setCellPar', 'bondRatio', 'molMode']
        Default={'threshold':1.0,'maxAttempts':50,'method':1, 'p_pri':0.,
        'volRatio':1.5,'maxtryNum':100,'minLattice':None,'maxLattice':None, 'dimension':3, 'choice': 0 }
        checkParameters(self.p,parameters,Requirement,Default)
        radius = [float(covalent_radii[atomic_numbers[atom]]) for atom in self.p.symbols]
        checkParameters(self.p,parameters,[],{'radius':radius})

    def update_volume_ratio(self, volume_ratio):
        log.info("change volRatio from {} to {}".format(self.p.volRatio, volume_ratio))
        self.p.volRatio = volume_ratio

    def get_swap(self):
        M = np.array([
            [[1,0,0],[0,1,0],[0,0,1]],
            [[0,1,0],[1,0,0],[0,0,1]],
            [[0,1,0],[0,0,1],[1,0,0]],
            [[1,0,0],[0,0,1],[0,1,0]],
            [[0,0,1],[1,0,0],[0,1,0]],
            [[0,0,1],[0,1,0],[1,0,0]]])
        return M[np.random.randint(6)] if self.p.dimension ==3 else M[0]

    def getVolumeandLattice(self,numlist):
        # Recalculate atomic radius, considering the change of radius in molecular crystal mode
        Volume = np.sum(4*np.pi/3*np.array(self.p.radius)**3*np.array(numlist))*self.p.volRatio
        minVolume = Volume*0.5
        maxVolume = Volume*1.5
        minLattice= [2*np.max(self.p.radius)]*3+[60]*3
        # maxLattice= [maxVolume/2/np.max(self.p.radius)]*3+[120]*3
        maxLattice= [maxVolume**(1./3)]*3+[120]*3
        if self.p.minLattice:
            minLattice = self.p.minLattice
            minVolume = np.linalg.det(cellpar_to_cell(minLattice)) 
        if self.p.maxLattice:
            maxLattice = self.p.maxLattice
            maxVolume = np.linalg.det(cellpar_to_cell(maxLattice)) 
        if self.p.fixCell:
            minLattice = self.p.setCellPar
            minVolume = np.linalg.det(cellpar_to_cell(minLattice))
            maxLattice = [l+0.01 for l in minLattice]
            maxVolume = np.linalg.det(cellpar_to_cell(maxLattice))
        return minVolume,maxVolume,minLattice,maxLattice

    def Generate_ind(self, spg, numlist):
        spg=int(spg)
        numType = len(numlist)
        generator = GenerateNew.Info()
        generator.spg = spg
        generator.spgnumber = 1
        generator.maxAttempts = self.p.maxAttempts
        generator.dimension = self.p.dimension
        
        if hasattr(self.p, 'vacuum'):
            generator.vacuum = self.p.vacuum
        if hasattr(self.p, 'choice'):
            generator.choice = self.p.choice
        
        if self.p.molMode:
            generator.threshold=self.p.bondRatio
        else:
            generator.threshold=self.p.dRatio
        generator.method=self.p.method
        generator.forceMostGeneralWyckPos=False
        generator.UselocalCellTrans = 'y'
        generator.GetConventional = True if np.random.rand() > self.p.p_pri else False

        minVolume,maxVolume,minLattice,maxLattice=self.getVolumeandLattice(numlist)
        # TODO should be encapsulated into HanYu code
        #only fit for 3D conditions !
        swap_matrix = self.get_swap() 
        minLattice = np.kron(np.array([[1,0],[0,1]]), swap_matrix) @ minLattice
        maxLattice = np.kron(np.array([[1,0],[0,1]]), swap_matrix) @ maxLattice

        generator.minVolume = minVolume
        generator.maxVolume = maxVolume
        generator.SetLatticeMins(minLattice[0], minLattice[1], minLattice[2], minLattice[3], minLattice[4], minLattice[5])
        generator.SetLatticeMaxes(maxLattice[0], maxLattice[1], maxLattice[2], maxLattice[3], maxLattice[4], maxLattice[5])
        numbers=[]
        for i in range(numType):
            if numlist[i] > 0:
                if self.p.molMode:
                    mole = self.p.inputMols[i]
                    if len(mole) > 1:
                        radius = np.array([covalent_radii[atomic_numbers[atom.symbol]] for atom in mole])
                        radius = np.array(list({}.fromkeys(radius).keys()))
                        positions = mole.positions.reshape(-1)
                        symbols = mole.get_chemical_symbols()
                        uni_symbols = list({}.fromkeys(symbols).keys())
                        assert len(uni_symbols)<5 
                        #TODO char array
                        namearray = [str(_s) for _s in uni_symbols]
                        numinfo = np.array([symbols.count(s) for s in uni_symbols],dtype=float)

                        symprec = self.p.symprec
                        generator.threshold_mol = self.p.threshold_mol
                        
                        generator.AppendMoles(int(numlist[i]),mole.get_chemical_formula()\
                            ,radius, positions, numinfo, namearray, symprec)

                        number = sum([num for num in [[atomic_numbers[s]]*int(n)*numlist[i] \
                            for s,n in zip(uni_symbols,numinfo)]],[])
                        numbers.extend(number)
                    else:
                        symbol = mole.get_chemical_symbols()[0]
                        radius = covalent_radii[atomic_numbers[symbol]]
                        generator.AppendAtoms(int(numlist[i]), symbol, radius, False)
                        numbers.extend([atomic_numbers[symbol]]*numlist[i])
                else:
                    generator.AppendAtoms(int(numlist[i]), str(i), self.p.radius[i], False)
                    numbers.extend([atomic_numbers[self.p.symbols[i]]]*numlist[i])
        label = generator.Generate(np.random.randint(1000))

        if label:
            cell = generator.GetLattice(0)
            cell = np.reshape(cell, (3,3))
            cell_ = np.linalg.inv(swap_matrix) @ cell
            Q, L = np.linalg.qr(cell_.T)
            scaled_positions = generator.GetPosition(0)
            scaled_positions = np.reshape(scaled_positions, (-1, 3))
            positions = scaled_positions @ cell @ Q
            if np.linalg.det(L) < 0:
                L[2, 2] *= -1
                positions[:, 2] *= -1
            atoms = ase.Atoms(cell=L.T, positions=positions, numbers=numbers, pbc=1)
            atoms.wrap(pbc=[1, 1, 1])
            atoms = build.sort(atoms)
            return label, atoms
        else:
            return label, None

class BaseGenerator(Generator):
    def __init__(self,parameters):
        super().__init__(parameters)
        minFrml = int(np.ceil(self.p.minAt/sum(self.p.formula)))
        maxFrml = int(self.p.maxAt/sum(self.p.formula))
        self.p.numFrml = list(range(minFrml, maxFrml + 1))

    def afterprocessing(self,ind,nfm):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = self.p.formula
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = 'random'
        return ind

    def Generate_pop(self,popSize,initpop=False):
        buildPop = []
        tryNum=0
        while tryNum<self.p.maxtryNum*popSize and popSize > len(buildPop):
            nfm = np.random.choice(self.p.numFrml)
            spg = np.random.choice(self.p.spgs)
            numlist=np.array(self.p.formula)*nfm
            label,ind = self.Generate_ind(spg,numlist)
            if label:
                self.afterprocessing(ind,nfm)
                buildPop.append(ind)
            else:
                tryNum+=1

        # Allow P1 structure
        if popSize > len(buildPop):
            for _ in range(popSize - len(buildPop)):
                nfm = np.random.choice(self.p.numFrml)
                spg = np.random.choice(self.p.spgs)
                numlist=np.array(self.p.formula)*nfm
                label,ind = self.Generate_ind(spg,numlist)
                if label:
                    self.afterprocessing(ind,nfm)
                    buildPop.append(ind)
                else:
                    label,ind = self.Generate_ind(1,numlist)
                    if label:
                        self.afterprocessing(ind,nfm)
                        buildPop.append(ind)
        return buildPop
class CellsplitGenerator(Generator):
    pass
    
class LayerGenerator(BaseGenerator):
    def __init__(self, p):
        super().__init__(p)
        super().checkParameters(Requirement=['cmax'])

        self.cmax=p.cmax
        self.d=p.d if hasattr(p,'d') else 15
        self.minLen=p.minLen if hasattr(p,'minLen') else [2*np.max(self.p.radius)]*3
        amax=(self.maxVolume*np.max(self.p.numFrml)/self.cmax)**(1./2)
        self.maxLen=p.maxLen if hasattr(p,'maxLen') else [amax,amax,self.cmax]

    def addVacuumlayer(self,ind):
        c=ind.get_cell()[2]
        c_=ind.get_reciprocal_cell()[2]
        k=self.d*np.linalg.norm(c_)/np.dot(c,c_)+1
        ind.cell[2]*=k

    def afterprocessing(self,ind,nfm):
        super().afterprocessing(ind,nfm)
        #self.addVacuumlayer(ind)

class MoleculeGenerator(Generator):
    def __init__(self,parameters):
        super().__init__(parameters)
        Requirement=['inputMols','molFormula','numFrml']
        Default = {'molMode':True, 'symprec':0.1, 'threshold_mol': 1.0}
        checkParameters(self.p,parameters,Requirement,Default)
        radius = [get_radius(mol) for mol in self.p.inputMols]
        self.volume = [sum([4*np.pi/3*(covalent_radii[num])**3
            for num in mol.get_atomic_numbers()])
            for mol in self.p.inputMols ]
        checkParameters(self.p,parameters,[],{'radius':radius})

    def getVolumeandLattice(self,numlist):
        # Recalculate atomic radius, considering the change of radius in molecular crystal mode
        Volume = np.sum(self.volume*np.array(numlist))*self.p.volRatio
        minVolume = Volume*0.5
        maxVolume = Volume*1.5
        minLattice= [2*np.max(self.p.radius)]*3+[60]*3
        # maxLattice= [maxVolume/2/np.max(self.p.radius)]*3+[120]*3
        maxLattice= [maxVolume**(1./3)]*3+[120]*3
        if self.p.minLattice:
            minLattice = self.p.minLattice
            minVolume = np.linalg.det(cellpar_to_cell(minLattice)) 
        if self.p.maxLattice:
            maxLattice = self.p.maxLattice
            maxVolume = np.linalg.det(cellpar_to_cell(maxLattice)) 
        if self.p.fixCell:
            minLattice = self.p.setCellPar
            minVolume = np.linalg.det(cellpar_to_cell(minLattice))
            maxLattice = [l+0.01 for l in minLattice]
            maxVolume = np.linalg.det(cellpar_to_cell(maxLattice))
        return minVolume,maxVolume,minLattice,maxLattice

    def afterprocessing(self,ind,nfm):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = self.p.formula
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = 'random'
        return ind

    def Generate_pop(self,popSize,initpop=False):
        buildPop = []
        tryNum=0
        while tryNum<self.p.maxtryNum*popSize and popSize > len(buildPop):
            nfm = np.random.choice(self.p.numFrml)
            spg = np.random.choice(self.p.spgs)
            numlist=np.array(self.p.molFormula)*nfm
            label,ind = self.Generate_ind(spg,numlist)
            if label:
                self.afterprocessing(ind,nfm)
                buildPop.append(ind)
            else:
                tryNum+=1

        # Allow P1 structure
        if popSize > len(buildPop):
            for _ in range(popSize - len(buildPop)):
                nfm = np.random.choice(self.p.numFrml)
                spg = np.random.choice(self.p.spgs)
                numlist=np.array(self.p.molFormula)*nfm
                label,ind = self.Generate_ind(spg,numlist)
                if label:
                    self.afterprocessing(ind,nfm)
                    buildPop.append(ind)
                else:
                    label,ind = self.Generate_ind(1,numlist)
                    if label:
                        self.afterprocessing(ind,nfm)
                        buildPop.append(ind)
        return buildPop


class VarGenerator(Generator):
    def __init__(self,parameters):
        super().__init__(parameters)
        Requirement=['minAt','maxAt']
        Default={'fullEles':True,'eleSize':1}
        checkParameters(self.p,parameters,Requirement,Default)
        # self.projection_matrix=np.dot(self.p.formula.T,np.linalg.pinv(self.p.formula.T))
        self.p.invFrml = np.linalg.pinv(self.p.formula).tolist()

    def afterprocessing(self,ind,numlist,nfm):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = numlist
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = 'random'
        return ind

    def Generate_pop(self,popSize,initpop=False):
        buildPop = []
        for i in range(popSize):
            for j in range(self.p.maxtryNum):
                numAt = np.random.randint(self.p.minAt, self.p.maxAt+1)
                numlist = np.random.rand(len(self.p.symbols))
                numlist *= numAt/np.sum(numlist)
                nfm = np.rint(np.dot(numlist,self.p.invFrml)).astype(np.int)
                #if (nfm<0).any():
                #    continue
                nfm[np.where(nfm<0)] = 0
                numlist = np.dot(nfm,self.p.formula)
                # numlist = np.rint(np.dot(self.projection_matrix,numlist)).astype(np.int)
                if np.sum(numlist) < self.p.minAt or np.sum(numlist) > self.p.maxAt or (self.p.fullEles and 0 in numlist) or np.sum(numlist<0)>0:
                    continue

                spg = np.random.choice(self.p.spgs)

                label,ind = self.Generate_ind(spg,numlist)
                if label:
                    self.afterprocessing(ind,numlist,nfm)
                    buildPop.append(ind)
                    break
                else:
                    continue

        # Generate simple substance in variable mode
        if initpop:
            for n,symbol in enumerate(self.p.symbols):
                for i in range(self.p.eleSize):
                    for j in range(self.p.maxtryNum):
                        numAt = np.random.randint(self.p.minAt, self.p.maxAt+1)
                        numlist = [0]*len(self.p.symbols)
                        numlist[n] = numAt
                        spg = np.random.choice(self.p.spgs)

                        label,ind = self.Generate_ind(spg,numlist)
                        if label:
                            self.afterprocessing(ind,numlist,nfm)
                            buildPop.append(ind)
                            break
                        else:
                            continue
        return buildPop



def read_seeds(seed_file):
    seedPop = []
    if not os.path.exists(seed_file):
        return []
    if 'traj' in seed_file:
        readPop = ase.io.read(seed_file, index=':', format='traj')
    elif 'POSCARS' in seed_file:
        readPop = ase.io.read(seed_file, index=':', format='vasp-xdatcar')
    else:
        try:
            readPop = ase.io.read(seed_file, index=':')
        except:
            raise Exception("unknown file format: {}".format(seed_file))
    seedPop = readPop
    for i, ind in enumerate(seedPop):
        ind.info['origin'] = 'seed'
    return seedPop

def read_ref(bulkFile):
    bulk = ase.io.read(bulkFile, index =':', format='traj')
    if len(bulk) >0:
        log.info("Reading Refslab ...")
    
    from ase.constraints import FixAtoms

    for ind in bulk:
        c = FixAtoms(indices=range(len(ind) ))
        ind.set_constraint(c)

    return bulk
    
class ReconstructGenerator():
    def __init__(self,parameters):
        para_t = EmptyClass()
        Requirement=['layerfile']
        Default={'cutslices': None, 'bulk_layernum':3, 'range':0.5, 'relaxable_layernum':3, 'rcs_layernum':2, 'randratio':0.5,
        'rcs_x':[1], 'rcs_y':[1], 'direction': None, 'rotate': 0, 'matrix': None, 'extra_c':1.0, 'addH': False, 'pcell': True,
        'dimension':2, 'choice':0, 'molMode':False }

        checkParameters(para_t, parameters, Requirement,Default)
        
        if os.path.exists("Ref") and os.path.exists("Ref/refslab.traj") and os.path.exists("Ref/layerslices.traj"):
            log.info("Used layerslices in Ref.")
        else:
            if not os.path.exists("Ref"):
                os.mkdir('Ref')
            #here starts to get Ref/refslab to calculate refE
            ase.io.write("Ref/refslab.traj", ase.io.read(para_t.layerfile), format = 'traj')
            #here starts to split layers into [bulk, relaxable, rcs]
            originatoms = ase.io.read(para_t.layerfile)
            originatoms.pbc = True
            xy = [1,1]

            if np.any(np.array([*para_t.rcs_x, *para_t.rcs_y])%1 > 0) or para_t.pcell == False:
                xy= [para_t.rcs_x[0], para_t.rcs_y[0]]

            layernums = [para_t.bulk_layernum, para_t.relaxable_layernum, para_t.rcs_layernum]
            cutcell(originatoms, layernums, totslices = para_t.cutslices, vacuum = para_t.extra_c, addH = para_t.addH ,direction= para_t.direction,
                    xy = xy, rotate = para_t.rotate, pcell = para_t.pcell ,matrix = para_t.matrix)
            #layer split ends here
            from .entrypoints.getslab import getslab
            slab = getslab(slabfile = None)
            slab.info['size'] = xy
            ase.io.write("Ref/refslab.traj", slab, append = True)

        self.range=para_t.range
        
        self.ind=RcsInd(parameters)

        #here get new parameters for self.generator 
        _parameters = copy.deepcopy(parameters)
        _parameters.attach(para_t)
        self.layerslices = ase.io.read("Ref/layerslices.traj", index=':', format='traj')
        
        setlattice = []
        if len(self.layerslices)==3:
            #mode = 'reconstruct'
            self.ref = self.layerslices[2]
            setlattice = self.ref.get_cell_lengths_and_angles().copy()
            vertical_dis = self.ref.get_scaled_positions()[:,2].copy()
            setlattice[2] *=(np.max(vertical_dis) - np.min(vertical_dis) )#+ 1.0 / setlattice[2])

        else:
            #mode = 'add atoms'
            para_t.randratio = 0
            self.ref = self.layerslices[1].copy()
            lattice = self.ref.get_cell().copy()
            lattice [2]/= para_t.relaxable_layernum
            self.ref.set_cell(lattice)
            setlattice = list(cell_to_cellpar(lattice))

        setlattice = np.round(setlattice, 3)
        setlattice[3:] = [i if np.round(i) != 60 else 120.0 for i in setlattice[3:]]
        self.symtype = 'hex' if 120 in np.round(setlattice[3:]) else 'orth'
        if para_t.pcell == False:
            self.symtype = 'c-cell'
            
        self.reflattice = list(setlattice).copy()
        target = self.ind.get_targetFrml()
        _symbol = [s for s in target]
        requirement = {'minLattice': list(setlattice), 'maxLattice':list(setlattice), 'symbols':_symbol}

        for key in requirement:
            if not hasattr(_parameters, key):
                setattr(_parameters,key,requirement[key])
            else:
                if getattr(_parameters,key) == requirement[key]:
                    pass
                else:
                    log.info("warning: change user defined {} to {} to match rcs layer".format(key, requirement[key]))
                    setattr(_parameters,key,requirement[key])
        
        if para_t.molMode == False:
            self.rcs_generator =Generator(_parameters)  
        else: 
            self.rcs_generator= MoleculeGenerator(_parameters)
            para_t.randratio = 0

        self.rcs_generator.p.choice =_parameters.choice
        #got a generator! next put all parm together except changed ones

        self.p = EmptyClass()
        self.p.attach(para_t)
        self.p.attach(self.rcs_generator.p)

        origindefault={'symbols':parameters.symbols}
        origindefault['minLattice'] = parameters.minLattice if hasattr(parameters, 'minLattice') else None
        origindefault['maxLattice'] = parameters.maxLattice if hasattr(parameters, 'maxLattice') else None

        for key in origindefault:
            if not hasattr(self.p, key):
                pass
            else:
                setattr(self.p,key,origindefault[key])
        
        #some other settings
        minFrml = int(np.ceil(self.p.minAt/sum(self.p.formula)))
        maxFrml = int(self.p.maxAt/sum(self.p.formula))
        self.p.numFrml = list(range(minFrml, maxFrml + 1))
        self.threshold = self.p.dRatio
        self.maxAttempts = 100

    def afterprocessing(self,ind,nfm, origin, size):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = self.p.formula
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = origin
        ind.info['size'] = size

        return ind
        
    def update_volume_ratio(self, volume_ratio):
        pass
        #return self.rcs_generator.update_volume_ratio(volume_ratio)
    def Generate_ind(self,spg,numlist):
        return self.rcs_generator.Generate_ind(spg,numlist)

    def reconstruct(self, ind):

        c=reconstruct(self.range, ind.copy(), self.threshold, self.maxAttempts)
        label, pos=c.reconstr()
        numbers=[]
        if label:
            for i in range(len(c.atomnum)):
                numbers.extend([atomic_numbers[c.atomname[i]]]*c.atomnum[i])
            cell=c.lattice
            pos=np.dot(pos,cell)
            atoms = ase.Atoms(cell=cell, positions=pos, numbers=numbers, pbc=1)
            
            return label, atoms
        else:
            return label, None

    def match_symmetry_plane(self, extraind, bottomind): 
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

        refcell = (self.ref * (_x, _y, 1)).get_cell_lengths_and_angles()
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

        refcell_ = (self.ref * (_x, _y, 1)).get_cell()
        atoms.set_cell(refcell_, scale_atoms = True)

        #axis z:
        if layersub is None:
            layersub = self.layerslices[1] *(_x, _y, 1)  
        else:
            layersub.set_cell(np.dot(np.diag([1,1,np.max(layersub.get_scaled_positions()[:, 2])]), layersub.get_cell()))
        spa = atoms.get_scaled_positions(wrap = False)
        sps = layersub.get_scaled_positions(wrap = False)
        latticesub = layersub.get_cell_lengths_and_angles()
        
        scaledp_matrix = np.zeros([len(atoms), len(layersub)])
        for i, a in enumerate(atoms):
            for j, l in enumerate(layersub):
                mindis = (covalent_radii[a.number]+covalent_radii[l.number])*self.p.threshold 
                dx, dy, _ = spa[i] - sps[j]
                x2y2 = np.sum([x**2 for x in [dx + dy*math.cos(latticesub[-1]), dy*math.sin(latticesub[-1])]])
                minz = mindis**2 - x2y2
                scaledp_matrix[i][j] = sps[j][2]*latticesub[2] if minz < 0 else sps[j][2]*latticesub[2] + math.sqrt(minz)
                scaledp_matrix[i][j] -= spa[i][2] * refcell[2] + latticesub[2] 
        scaledp_matrix /= refcell[2]
        dz = max(np.min(scaledp_matrix), -np.min(spa[:, 2]))
        dz = -2
        #dz = max(np.min(scaledp_matrix), -np.min(spa[:, 2]))
        atoms.translate([ -dz/refcell[2]* atoms.get_cell()[2]]* len(atoms))
        return True, atoms

        
    def reset_generator_lattice(self, _x, _y, spg):
        symtype = 'default'
        if self.symtype == 'hex':
            if (self.rcs_generator.p.choice == 0 and spg < 13) or (self.rcs_generator.p.choice == 1 and spg < 65):
                #for hex-lattice, 'a' must equal 'b'
                if self.reflattice[0] == self.reflattice[1] and _x == _y:    
                    symtype = 'hex'

        if symtype == 'hex' or self.symtype == 'c-cell':
            #self.rcs_generator.p.GetConventional = False
            self.rcs_generator.p.p_pri = 1
        elif symtype == 'default': 
            #self.rcs_generator.p.GetConventional = True
            self.rcs_generator.p.p_pri = 0
            
        self.rcs_generator.p.minLattice = list(self.reflattice *np.array([_x, _y]+[1]*4))
        self.rcs_generator.p.maxLattice = self.rcs_generator.p.minLattice
        return symtype

    def Generate_pop(self,popSize,initpop=False, inspg = None):
        buildPop = []
        tryNum=0

        while tryNum<self.p.maxtryNum*popSize and popSize*self.p.randratio > len(buildPop):
            nfm = np.random.choice(self.p.numFrml)
            spg = 1
            _x = np.random.choice(self.p.rcs_x)
            _y = np.random.choice(self.p.rcs_y)

            ind = self.ref * (_x , _y, 1)
            add, rm = self.ind.AtomToModify()
            
            if rm:
                for symbol in rm:
                    while rm[symbol] > 0:
                        eq_at = dict(zip(range(len(ind)), get_symmetry_dataset(ind,1e-2)['equivalent_atoms']))
                        indices = [atom.index for atom in ind if atom.symbol == symbol]
                        lucky_atom_to_rm = eq_at[np.random.choice(indices)]
                        eq_ats_with_him = np.array([i for i in eq_at if eq_at[i] == lucky_atom_to_rm])
                        size = np.random.choice(range(1,np.min([rm[symbol] , len(eq_ats_with_him)])+1))
                        _to_del = np.random.choice(eq_ats_with_him, size =size, replace=False)
                        rm[symbol] -= len(_to_del)
                        del ind[_to_del]
            

            if add:
                numlist = np.array([add[s] for s in self.rcs_generator.p.symbols])
                spg = np.random.choice(self.p.spgs) if self.p.choice == 0 else np.random.choice(self.get_spg(self.symtype, 'planegroup'))                
                self.rcs_generator.p.choice = 0
                self.reset_generator_lattice(_x, _y, spg)

                label,extraind = self.rcs_generator.Generate_ind(spg,numlist)
                if label:
                    botp = np.max(ind.get_scaled_positions()[:,2]) + np.random.choice(range(5,20))/100
                    label, extraind = self.reset_rind_lattice(extraind, _x, _y, layersub = ind.copy())
                if label:
                    ind.info['size'] = [_x, _y]
                    bottom = self.ind(ind)
                    label, extraind = self.match_symmetry_plane(extraind, bottom.addbulk_relaxable_vacuum()) 
                    hight = np.random.uniform(np.mean(ind.get_scaled_positions()[:, 2]), np.max(ind.get_scaled_positions()[:, 2]) + 1/(extraind.get_cell()[2]))
                    extraind.translate([hight * extraind.get_cell()[2]]*len(extraind))
                    #extraind.translate([np.mean(ind.get_scaled_positions()[:, 2]) + np.max(ind.get_scaled_positions()[:, 2])/2 * extraind.get_cell()[2]]*len(extraind))
                    ind += extraind
                    
                if not label:
                    tryNum+=1
                    continue
            
            label,ind = self.reconstruct(ind)
            if label:
                self.afterprocessing(ind,nfm,'rand.randmove', [_x, _y])
                ref = self.ref * (_x , _y, 1)
                ind.set_cell(ref.get_cell().copy(), scale_atoms=True)
                ind.info['size'] = [_x, _y]
                ind = self.ind.addbulk_relaxable_vacuum(atoms = ind)
                buildPop.append(ind)
                
            else:
                tryNum+=1

        #add random structure
        while tryNum<self.p.maxtryNum*popSize and popSize > len(buildPop):
            
            #spg = np.random.choice(self.p.spgs)
            spg = inspg if inspg else np.random.choice(self.p.spgs)
            nfm = 1
            _x = np.random.choice(self.p.rcs_x)
            _y = np.random.choice(self.p.rcs_y)

            target = self.ind.get_targetFrml(_x , _y)
            numlist = np.array([target[s] for s in self.rcs_generator.p.symbols]) if not self.p.molMode else self.p.molFormula

            self.reset_generator_lattice(_x,_y, spg)
            self.rcs_generator.p.choice = self.p.choice
            #log.debug("formula {} of number {} with chosen spg = {}".format(self.rcs_generator.p.symbols, numlist,spg))
            #log.debug("with maxlattice = {}".format(self.rcs_generator.p.maxLattice))
            label,ind = self.rcs_generator.Generate_ind(spg,numlist)
            if label:
                label, ind = self.reset_rind_lattice(ind, _x, _y)

            if label:
                _bot_ = (self.layerslices[1] * (_x, _y, 1)).copy()
                _bot_.info['size'] = [_x, _y]
                
                label, ind = self.match_symmetry_plane(ind, self.ind.addvacuum(add = 1, atoms = self.ind.addextralayer('bulk', atoms=_bot_, add = 1)))
            if label:
                self.afterprocessing(ind,nfm,'rand.symmgen', [_x, _y])
                ind = self.ind.addbulk_relaxable_vacuum(atoms = ind)
                buildPop.append(ind)
            if not label:
                tryNum+=1

        return buildPop


class ClusterGenerator(BaseGenerator):
    def __init__(self,parameters):
        super().__init__(parameters)
        Default = {'vacuum':10}
        checkParameters(self.p,parameters, [], Default)
        self.p.dimension = 0

    def afterprocessing(self,ind,nfm):
        super().afterprocessing(ind,nfm)

    def getVolumeandLattice(self,numlist):
        #For cluster genertor, generates atom positions lies in distance (from origin) range of (minLattice[0], maxLattice[0])
        atomicR = [float(covalent_radii[atomic_numbers[atom]]) for atom in self.p.symbols]
        Volume = np.sum(4*np.pi/3*np.array(atomicR)**3*np.array(numlist))*self.p.volRatio
        minVolume = Volume*0.2
        maxVolume = Volume*0.8
        minLattice = [3*self.p.dRatio*np.mean(atomicR)]*3 + [60,60,60] if not self.p.minLattice else self.p.minLattice
        maxLattice = [ ( Volume / (4/3 * math.pi))**(1.0/3)]*3 + [120,120,120] if not self.p.maxLattice else self.p.maxLattice

        return minVolume,maxVolume,minLattice,maxLattice

    def Generate_pop(self,popSize,initpop=False):
        pop =  super().Generate_pop(popSize,initpop)
        for ind in pop:
            ind.set_pbc([0,0,0])
        return pop

from .population import AdClusInd
class AdClusterGenerator(ClusterGenerator):
    def __init__(self,parameters):
        super().__init__(parameters)
        self.ind = AdClusInd(parameters)

    def Generate_pop(self,popSize,initpop=False):
        pop =  super().Generate_pop(popSize,initpop)
        for i, ind in enumerate(pop):
            c = self.ind(ind)
            c.absorb()
            pop[i] = c.atoms.copy()
            nfm = 1
            self.afterprocessing(pop[i], nfm)
        return pop
        
#test
if __name__ == '__main__':
    class EmptyClass:
        def __init__(self):
            pass
    import ase.io
    p=EmptyClass()
    Requirement=['symbols','formula','numFrml']
    p.symbols=['C','H','O','N']
    p.formula=np.array([1,4,1,2])
    p.numFrml=[1]
    p.volRatio=2

    g=BaseGenerator(p)
    buildind=g.Generate_pop(10)
    ase.io.write('a.traj',buildind)
