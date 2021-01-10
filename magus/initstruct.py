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
from .utils import *
from .reconstruct import reconstruct, cutcell
from .population import RcsInd
import math


class Generator:
    def __init__(self,parameters):
        self.p = EmptyClass()
        Requirement=['symbols','formula','minAt','maxAt','spgs','dRatio','fixCell','setCellPar', 'bondRatio', 'molMode']
        Default={'threshold':1.0,'maxAttempts':50,'method':1,
        'volRatio':1.5,'maxtryNum':100,'minLattice':None,'maxLattice':None, 'dimension':3}
        checkParameters(self.p,parameters,Requirement,Default)
        radius = [float(covalent_radii[atomic_numbers[atom]]) for atom in self.p.symbols]
        checkParameters(self.p,parameters,[],{'radius':radius})

    def updatevolRatio(self,volRatio):
        self.p.volRatio=volRatio
        logging.debug("new volRatio: {}".format(self.p.volRatio))

    def getVolumeandLattice(self,numlist):
        # Recalculate atomic radius, considering the change of radius in molecular crystal mode
        atomicR = [float(covalent_radii[atomic_numbers[atom]]) for atom in self.p.symbols]
        Volume = np.sum(4*np.pi/3*np.array(atomicR)**3*np.array(numlist))*self.p.volRatio
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

    def Generate_ind(self,spg,numlist):
        spg=int(spg)
        numType = len(numlist)
        generator = GenerateNew.Info()
        generator.spg = spg
        generator.spgnumber = 1
        generator.maxAttempts = self.p.maxAttempts
        generator.dimension = self.p.dimension
        try:
            generator.vacuum = self.p.vacuum
        except:
            pass

        try:
            generator.choice = self.p.choice
        except:
            pass

        if self.p.molMode:
            generator.threshold=self.p.bondRatio
        else:
            generator.threshold=self.p.dRatio
        generator.method=self.p.method
        generator.forceMostGeneralWyckPos=False
        generator.UselocalCellTrans = 'y'
        generator.GetConventional = True

        minVolume,maxVolume,minLattice,maxLattice=self.getVolumeandLattice(numlist)
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

        label = generator.PreGenerate(np.random.randint(1000))

        if label:
            cell = generator.GetLattice(0)
            cell = np.reshape(cell, (3,3))
            positions = generator.GetPosition(0)
            positions = np.reshape(positions, (-1, 3))
            positions = np.dot(positions,cell)
            atoms = ase.Atoms(cell=cell, positions=positions, numbers=numbers, pbc=1)
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
        checkParameters(self.p,parameters,[],{'radius':radius})

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



def read_seeds(parameters, seedFile, goodSeed=False):
    seedPop = []
    setSym = parameters.symbols
    setFrml = parameters.formula
    minAt = parameters.minAt
    maxAt = parameters.maxAt
    calcType = parameters.calcType

    if os.path.exists(seedFile):
        if goodSeed:
            readPop = ase.io.read(seedFile, index=':', format='traj')
        else:
            readPop = ase.io.read(seedFile, index=':', format='traj')
        if len(readPop) > 0:
            logging.info("Reading Seeds ...")
            
        seedPop = readPop
        #seedPop = read_bare_atoms(readPop, setSym, setFrml, minAt, maxAt, calcType)
        #for ind in seedPop:
            #if goodSeed:
                #ind.info['origin'] = 'goodseed'
            #else:
                #ind.info['origin'] = 'seed'
    return seedPop

def read_ref(bulkFile):
    bulk = ase.io.read(bulkFile, index =':', format='traj')
    if len(bulk) >0:
        logging.info("Reading Refslab ...")
    
    from ase.constraints import FixAtoms

    for ind in bulk:
        c = FixAtoms(indices=range(len(ind) ))
        ind.set_constraint(c)

    return bulk
    
class ReconstructGenerator():
    def __init__(self,parameters):
        para_t = EmptyClass()
        Requirement=['layerfile','cutslices']
        Default={'bulk_layernum':3, 'range':0.5, 'relaxable_layernum':3, 'rcs_layernum':2, 'startpos': 0.9,
        'rcs_x':[1], 'rcs_y':[1], 'SymbolsToAdd': None, 'AtomsToAdd': None, 'direction': None, 'rotate': 0,
        'dimension':2, 'choice':0 }

        checkParameters(para_t, parameters, Requirement,Default)
        #here starts to get Ref/refslab to calculate refE
        if os.path.exists("Ref/refslab.traj"):
            logging.info("Used layerslices in Ref.")
            pass
        else:
            ase.io.write("Ref/refslab.traj", ase.io.read(para_t.layerfile), format = 'traj')

        
        #here starts to split layers into [bulk, relaxable, rcs]
        if os.path.exists("Ref/layerslices.traj"):
            pass
        else:
            originatoms = ase.io.read(para_t.layerfile)
            layernums = [para_t.bulk_layernum, para_t.relaxable_layernum, para_t.rcs_layernum]
            cutcell(originatoms, para_t.cutslices, layernums, para_t.startpos, para_t.direction, para_t.rotate)

        #layer split ends here    

        self.range=para_t.range
        
        self.ind=RcsInd(parameters)

        #here get new parameters for self.generator 
        _parameters = copy.deepcopy(parameters)
        _parameters.attach(para_t)
        self.layerslices = ase.io.read("Ref/layerslices.traj", index=':', format='traj')
        
        setlattice = []
        if len(self.layerslices)==3:
            self.ref = self.layerslices[2]
            vertical_dis = self.ref.get_scaled_positions()[:,2].copy()
            mincell = self.ref.get_cell().copy()
            mincell[2] *= (np.max(vertical_dis) - np.min(vertical_dis))*1.2
            setlattice = list(cell_to_cellpar(mincell))
        else:
            self.ref = self.layerslices[1].copy()
            lattice = self.ref.get_cell().copy()
            lattice [2]/= para_t.relaxable_layernum
            self.ref.set_cell(lattice)
            setlattice = list(cell_to_cellpar(lattice))

        setlattice = [np.round(setlattice[i], 3) for i in range(0,6)]
        if np.round(setlattice[5],0)==60:
            setlattice[5]= 120.0 

        self.reflattice = setlattice.copy()
        target = self.ind.get_targetFrml()
        _symbol = [s for s in target]
        requirement = {'minLattice': setlattice, 'maxLattice':setlattice, 'symbols':_symbol}

        for key in requirement:
            if not hasattr(_parameters, key):
                setattr(_parameters,key,requirement[key])
            else:
                if getattr(_parameters,key) == requirement[key]:
                    pass
                else:
                    logging.info("warning: change user defined {} to {} to match rcs layer".format(key, requirement[key]))
                    setattr(_parameters,key,requirement[key])

        self.rcs_generator =Generator(_parameters)
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

    def updatevolRatio(self,volRatio):
        return self.rcs_generator.updatevolRatio(volRatio)
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

    def rand_displacement(self, extraind, bottomind, trynum = 50):
        for _ in range(0,trynum):
            _extra_ = extraind.copy()
            _dire_ = np.random.uniform(0,2*math.pi)
            _dis_ = np.dot(np.array([math.cos(_dire_), math.sin(_dire_),0])*np.random.uniform(0,0.2), _extra_.get_cell())
            
            _extra_.translate([_dis_]*len(_extra_))
            _extra_ += bottomind
            if spglib.get_spacegroup(_extra_, symprec = 0.2) !='P1 (1)':
                extraind.translate([_dis_]*len(extraind))
                return extraind
        else:
            extraind.translate([_dis_]*len(extraind))
            return extraind

    def reset_generator_lattice(self, _x, _y):
        self.rcs_generator.p.minLattice[0] = self.reflattice[0]*_x
        self.rcs_generator.p.minLattice[1] = self.reflattice[1]*_y
        self.rcs_generator.p.maxLattice = self.rcs_generator.p.minLattice

    def Generate_pop(self,popSize,initpop=False):
        buildPop = []
        tryNum=0

        while tryNum<self.p.maxtryNum*popSize and popSize/2 > len(buildPop):
            nfm = np.random.choice(self.p.numFrml)
            spg = 1
            _x = np.random.choice(self.p.rcs_x)
            _y = np.random.choice(self.p.rcs_y)

            ind = self.ref * (_x , _y, 1)
            add, rm = self.ind.AtomToModify()

            if rm:
                eq_at = dict(zip(range(len(ind)), get_symmetry_dataset(ind,1e-2)['equivalent_atoms']))
                to_del = []
                for symbol in rm:
                    while rm[symbol] > 0:
                        indices = [atom.index for atom in ind if atom.symbol == symbol]
                        lucky_atom_to_rm = eq_at[np.random.choice(indices)]
                        eq_ats_with_him = np.array([i for i in eq_at if eq_at[i] == lucky_atom_to_rm])
                        size = np.random.choice(range(1,np.min([rm[symbol] , len(eq_ats_with_him)])+1))
                        _to_del = np.random.choice(eq_ats_with_him, size =size, replace=False)
                        _to_del = [i for i in _to_del if i not in to_del]
                        to_del .extend(_to_del)
                        rm[symbol] -= len(_to_del)
                del ind[to_del]

            if add:
                numlist = np.array([add[s] for s in self.rcs_generator.p.symbols])
                self.reset_generator_lattice(_x, _y)
                #logging.debug("formula {} of number {} with chosen spg = {}".format(self.rcs_generator.p.symbols, numlist,spg))
                #logging.debug("with maxlattice = {}".format(self.rcs_generator.p.maxLattice))
                spg = np.random.choice(self.p.spgs) if self.p.choice == 0 else np.random.choice(list(range(13,18)) + list([1,2]))
                self.rcs_generator.p.choice = 0

                label,extraind = self.rcs_generator.Generate_ind(spg,numlist)
                if label:
                    extraind.set_cell(ind.get_cell().copy(), scale_atoms=True)
                    dis = np.max(ind.get_scaled_positions()[:,2]) - np.min(extraind.get_scaled_positions()[:,2]) + (np.random.choice(range(5,20))/100)
                    extraind.translate([dis * ind.get_cell()[2] ]*len(extraind))
                    extraind = self.rand_displacement(extraind, ind + (self.layerslices[0] + self.layerslices[1]) * (_x , _y, 1))
                    ind += extraind
                else:
                    tryNum+=1
                    continue
            
            label,ind = self.reconstruct(ind)
            if label:
                self.afterprocessing(ind,nfm,'rand.randmove', [_x, _y])
                ref = self.ref * (_x , _y, 1)
                ind.set_cell(ref.get_cell().copy(), scale_atoms=True)
                ind = self.ind(ind)
                ind = ind.addbulk_relaxable_vacuum()
                buildPop.append(ind.atoms)
                
            else:
                tryNum+=1

        #add random structure
        while tryNum<self.p.maxtryNum*popSize and popSize > len(buildPop):
            
            spg = np.random.choice(self.p.spgs)
            nfm = np.random.choice(self.p.numFrml)
            _x = np.random.choice(self.p.rcs_x)
            _y = np.random.choice(self.p.rcs_y)

            target = self.ind.get_targetFrml(_x , _y)
            numlist = np.array([target[s] for s in self.rcs_generator.p.symbols])

            self.reset_generator_lattice(_x,_y)
            self.rcs_generator.p.choice = self.p.choice
            #logging.debug("formula {} of number {} with chosen spg = {}".format(self.rcs_generator.p.symbols, numlist,spg))
            #logging.debug("with maxlattice = {}".format(self.rcs_generator.p.maxLattice))
            label,ind = self.rcs_generator.Generate_ind(spg,numlist)
            
            if label:
                ref = self.ref * (_x, _y, 1)
                #first set cell to stdcell(ref)
                ind.set_cell(ref.get_cell_lengths_and_angles().copy())
                #then change to true cell(ref)
                ind.set_cell(ref.get_cell().copy(), scale_atoms=True)
                
                vertical_dis = ref.get_scaled_positions()[:,2].copy()
                distance = np.average(vertical_dis) if self.p.dimension==2 and self.p.choice==0 else np.min(vertical_dis)
                layer_vertical_dis = ind.get_scaled_positions()[:,2].copy()
                layerbottom = np.min(layer_vertical_dis)

                ind.translate([ ref.get_cell()[2]*distance-ind.get_cell()[2]*layerbottom ]*len(ind))
                ind = self.rand_displacement(ind, (self.layerslices[0] + self.layerslices[1]) * (_x , _y, 1))
                self.afterprocessing(ind,nfm,'rand.symmgen', [_x, _y])
                ind= self.ind(ind)
                ind = ind.addbulk_relaxable_vacuum()
                buildPop.append(ind.atoms)
            else:
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
        minVolume,maxVolume,minLattice,maxLattice = super().getVolumeandLattice(numlist)
        return minVolume,maxVolume, list(np.array(minLattice)/2), list(np.array(maxLattice)/2)

    def Generate_pop(self,popSize,initpop=False):
        pop =  super().Generate_pop(popSize,initpop)
        for ind in pop:
            ind.set_pbc([0,0,0])
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
