from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell
import ase.io
import math, os, yaml, logging, copy
from functools import reduce
import numpy as np
from .initstruct import BaseGenerator,read_seeds,VarGenerator,MoleculeGenerator, ReconstructGenerator, ClusterGenerator
from .utils import *
from .queuemanage import JobManager
from .population import Population
from .fitness import fit_dict
from .calculators import get_calculator
from .offspring_creator import *
###############################

class magusParameters:
    def __init__(self,inputFile):
        with open(inputFile) as f:
            p_dict = yaml.load(f)
        self.p_dict = p_dict
        self.p_dict['workDir'] = os.getcwd()
        p = EmptyClass()
        p.workDir = os.getcwd()
        p.resultsDir = os.path.join(p.workDir,'results')
        p.calcDir = os.path.join(p.workDir,'calcFold')
        p.mlDir = os.path.join(p.workDir,'mlFold')
        for key, val in p_dict.items():
            setattr(p, key, val)

        Requirement = ['calcType','MainCalculator','popSize','numGen','saveGood','symbols']
        Default = {
            'spacegroup':list(range(1, 231)),
            'DFTRelax': True,
            'initSize':p.popSize,
            'goodSize':p.popSize,
            'molMode':False,
            'mlRelax':False,
            'symprec': 0.1,
            'bondRatio': 1.15,
            'eleSize': 0,
            'fullEles': False,
            'setAlgo': 'ea',
            'volRatio': 2,
            'dRatio': 0.7,
            'molDetector': 0,
            'fixCell': False,
            'setCellPar': None,
            'tourRatio': 0.1,
            'Algo': 'EA',
            'mlpredict': False,
            'useml': False,
            'addSym': False,
            'randFrac': 0.2,
            'chkMol': False,
            'chkSeed': True,
            'goodSeed': False,
            'goodSeedFile': '',
            'maxDataset': 500,
            'diffE': 0.01,
            'diffV': 0.05,
            #'ratNum': 0,
            'comparator': 'zurek',
        }
        if p.calcType=='rcs':
            log = logging.getLogger(__name__)
            log.info("rcs mode: used layerfile '{}'".format(p.layerfile))
            
        checkParameters(p,p,Requirement,Default)

        # p.initSize = p.popSize
        expandSpg = []
        for item in p.spacegroup:
            if isinstance(item, int):
                if 1 <= item <= 230:
                    expandSpg.append(item)
            if isinstance(item, str):
                assert '-' in item, 'Please check the format of spacegroup'
                s1, s2 = item.split('-')
                s1, s2 = int(s1), int(s2)
                assert 1 <= s1 < s2 <= 230, 'Please check the format of spacegroup'
                expandSpg.extend(list(range(s1, s2+1)))
        p.spgs = expandSpg
        
        if p.calcType=='rcs':
            p.originlayer=p.workDir+'/'+p.layerfile
            
        if p.molMode:
            from ase import build
            assert hasattr(p,'molFile'), 'Please define molFile'
            assert hasattr(p,'molFormula'), 'Please define molFormula'
            mols = [build.sort(ase.io.read("{}/{}".format(p.workDir, f), format='xyz')) for f in p.molFile]

            for mol in mols:
                assert not mol.pbc.any(), "Please provide a molecule ranther than a periodic system!"
            molSymbols = set(reduce(lambda x,y: x+y, [ats.get_chemical_symbols() for ats in mols]))
            assert molSymbols == set(p.symbols), 'Please check the compositions of molecules'
            if p.molType == 'fix':
                molFrmls = np.array([get_formula(mol, p.symbols) for mol in mols])
                p.formula = np.dot(p.molFormula, molFrmls).tolist()
            p.molList = [{'numbers': ats.get_atomic_numbers(),
                        'positions': ats.get_positions()}
                        for ats in mols]
            p.molNum = len(p.molFile)
            p.inputMols = [Atoms(**molInfo) for molInfo in p.molList]
            minFrml = int(np.ceil(p.minAt/sum(p.formula)))
            maxFrml = int(p.maxAt/sum(p.formula))
            p.numFrml = list(range(minFrml, maxFrml + 1))
        if p.chkMol:
            assert p.molDetector==1, "If you want to check molecules, molDetector should be 1."
        self.parameters = p

    def get_AtomsGenerator(self):
        if not hasattr(self,'AtomsGenerator'):
            if self.parameters.calcType == 'fix':
                if self.parameters.molMode:
                    AtomsGenerator = MoleculeGenerator(self.parameters)
                else:
                    AtomsGenerator = BaseGenerator(self.parameters)
            elif self.parameters.calcType == 'var':
                if self.parameters.molMode:
                    raise Exception("Ni deng hui , zhe ge hai mei jia ne")
                else:
                    AtomsGenerator = VarGenerator(self.parameters)

            elif self.parameters.calcType == 'rcs':
                AtomsGenerator = ReconstructGenerator(self.parameters)
            elif self.parameters.calcType == 'clus':
                AtomsGenerator = ClusterGenerator(self.parameters)
                
            else:
                raise Exception("Undefined calcType '{}'".format(self.parameters.calcType))
            self.AtomsGenerator = AtomsGenerator
            self.parameters.attach(AtomsGenerator.p)
        return self.AtomsGenerator

    def get_PopGenerator(self):
        if not hasattr(self,'PopGenerator'):
            #here's a suggestion. For crossovers, name it with 'xxPairing'; for mutations, name it with 'xxMutation'.
            #Modifying parms with input.yaml:
            #-OffspringCreator:
            #--xxNum [number of mutations]
            #--xx:
            #---parmA, parmB...
            _applied_operations_ = [CutAndSplicePairing, ReplaceBallPairing, 
                                  SoftMutation, PermMutation, LatticeMutation, RippleMutation, SlipMutation,
                                  RotateMutation, RattleMutation, FormulaMutation, 
                                  LyrSlipMutation, LyrSymMutation, ShellMutation, CluSymMutation]
            operations = {}
            op_nums = {}
            inputparm = getattr(self.parameters, 'OffspringCreator') if hasattr(self.parameters, 'OffspringCreator') else {}
            for methods in _applied_operations_:
                method_name = methods.__name__.lower()
                keyname = method_name[:-7] if method_name[-1]=='g' else method_name[:-8]
                if keyname in inputparm:
                    _parm_ = inputparm[keyname]
                    operations[keyname] = methods(**_parm_, symbols = self.parameters.symbols, dRatio = self.parameters.dRatio)
                else:
                    operations[keyname] = methods(symbols = self.parameters.symbols, dRatio = self.parameters.dRatio)
                op_nums[keyname] = 0
            
            #here's the special one 'softmutation'. Not tested yet. 
            #operations['soft'] = SoftMutation(calculator = self.get_MainCalculator().calcs[-1], bounds=[1.0,2.5])

            num = 3*int((1-self.parameters.randFrac)*self.parameters.popSize/8)+1
            
            for key in ['cutandsplice', 'slip', 'lattice', 'ripple', 'rattle']:
                op_nums[key] = num
            
            if len(self.parameters.symbols) > 1:
                op_nums['perm'] = num 
            if self.parameters.molDetector != 0:
                op_nums['rotate'] = num
            if not self.parameters.chkMol and self.parameters.calcType=='var':
                op_nums['formula'] = num

            if self.parameters.calcType=='rcs':
                op_nums['lattce'] = 0
                op_nums['formula'] = num if not self.parameters.chkMol and len(self.parameters.symbols) > 1 else 0
                op_nums['lyrslip'] = num
                op_nums['lyrsym'] = num
                
            if self.parameters.calcType=='clus':
                op_nums['slip'] = 0
                #op_nums['soft'] = num
                op_nums['shell'], op_nums['clusym'] = [num]*2
                operations['ripple'] = RippleMutation(rho=0.05)
                operations['rattle'] = RattleMutation(p=0.25,rattle_range=0.8,dRatio=self.parameters.dRatio)

            for key in list(op_nums.keys()):
                inputkey = key + 'Num'
                if inputkey in inputparm:
                    op_nums[key] = inputparm[inputkey]
                #compatible with old format settings
                if hasattr(self.parameters, inputkey):
                    op_nums[key] = getattr(self.parameters, inputkey)

            #print(op_nums)

            if self.parameters.Algo == 'EA':
                if self.parameters.mlpredict:
                    assert self.parameters.useml, "'useml' must be True"
                    calc = self.MLCalculator.calc
                    self.PopGenerator = MLselect(op_nums,operations,calc,self.parameters)
                else:
                    self.PopGenerator = PopGenerator(op_nums,operations,self.parameters)
            self.parameters.attach(self.PopGenerator.p)
        return self.PopGenerator

    def get_MLCalculator(self):
        if not hasattr(self, 'MLCalculator'):
            if 'MLCalculator' in self.p_dict:
                p_dict = copy.deepcopy(self.p_dict)
                p_dict.update(p_dict['MLCalculator'])
                p_dict['query_calculator'] = self.get_MainCalculator()
                self.MLCalculator = get_calculator(p_dict)   
            else:
                raise Exception('No ML Calculator!')
        return self.MLCalculator

    def get_MainCalculator(self):
        if not hasattr(self,'MainCalculator'):
            p_dict = copy.deepcopy(self.p_dict)
            p_dict.update(p_dict['MainCalculator'])
            self.MainCalculator = get_calculator(p_dict)
        return self.MainCalculator

    def get_FitnessCalculator(self):
        if not hasattr(self,'FitnessCalculator'):
            self.FitnessCalculator = []
            if hasattr(self.parameters, 'Fitness'):
                for fitness in self.parameters.Fitness:
                    self.FitnessCalculator.append(fit_dict[fitness])
            elif self.parameters.calcType == 'fix':
                self.FitnessCalculator.append(fit_dict['Enthalpy'])
            elif self.parameters.calcType == 'var':
                self.FitnessCalculator.append(fit_dict['Ehull'])
            elif self.parameters.calcType == 'rcs':
                self.FitnessCalculator.append(fit_dict['Eo'])
            elif self.parameters.calcType == 'clus':
                self.FitnessCalculator.append(fit_dict['Enthalpy'])

        return self.FitnessCalculator

    def get_Population(self):
        if not hasattr(self,'Population'):
            self.Population = Population(self.parameters)
            self.Population.fit_calcs = self.get_FitnessCalculator()
            self.parameters.attach(self.Population.p)
        return self.Population
