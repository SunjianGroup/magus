from __future__ import print_function, division
from ase.data import atomic_numbers
from ase.geometry import cellpar_to_cell
import ase.io
import math
import os
import yaml
import logging
from functools import reduce
import numpy as np
import copy
from .localopt import *
from .initstruct import BaseGenerator,read_seeds,VarGenerator,MoleculeGenerator, ReconstructGenerator
from .writeresults import write_dataset, write_results, write_traj
from .utils import *
from .machinelearning import LRmodel,GPRmodel,BayesLRmodel,pytorchGPRmodel
from .queuemanage import JobManager
from .population import Population
#ML module
#from .machinelearning import LRmodel
from .offspring_creator import *
###############################

class magusParameters:
    def __init__(self,inputFile):
        with open(inputFile) as f:
            p_dict = yaml.load(f)
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
            'initSize':p.popSize,
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
        }
        if p.calcType=='rcs':
            logging.info("rcs mode: \nlayerfile= "+p.layerfile)
            
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
                
            else:
                raise Exception("Undefined calcType '{}'".format(self.parameters.calcType))
            self.AtomsGenerator = AtomsGenerator
            self.parameters.attach(AtomsGenerator.p)
        return self.AtomsGenerator

    def get_PopGenerator(self):
        if not hasattr(self,'PopGenerator'):
            cutandsplice = CutAndSplicePairing()
            perm = PermMutation()
            lattice = LatticeMutation()
            ripple = RippleMutation()
            slip = SlipMutation()
            rot = RotateMutation()
            rattle = RattleMutation(p=0.25,rattle_range=4,dRatio=1)
            form = FormulaMutation(symbols=self.parameters.symbols)
            num = 3*int((1-self.parameters.randFrac)*self.parameters.popSize/8)+1
            Requirement = []
            cutNum,slipNum,latNum,ripNum,ratNum = [num]*5
            permNum = num if len(self.parameters.symbols) > 1 else 0
            rotNum = num if self.parameters.molDetector != 0 else 0
            #rotNum = num if self.parameters.molMode else 0
            formNum = num if not self.parameters.chkMol and self.parameters.calcType=='var' else 0

            if self.parameters.calcType=='rcs':
                latNum = 0
                formNum = num if not self.parameters.chkMol and len(self.parameters.symbols) > 1 else 0

            """
            if self.parameters.useml:
                self.get_MLCalculator()
                soft = SoftMutation(self.MLCalculator.calc)
                softNum = num
            else:
                soft = None
                softNum = 0
            """
            soft = None
            softNum = 0
            Default = {'cutNum':cutNum,'permNum': permNum, 'rotNum': rotNum,
                'slipNum': slipNum,'latNum': latNum, 'ripNum': ripNum, 'softNum':softNum, 
                'formNum': formNum,'ratNum':ratNum}
            checkParameters(self.parameters,self.parameters,Requirement,Default)
            numlist = [
                self.parameters.cutNum,
                self.parameters.permNum,
                self.parameters.latNum,
                self.parameters.ripNum,
                self.parameters.slipNum,
                self.parameters.rotNum,
                self.parameters.softNum,
                self.parameters.formNum,
                self.parameters.ratNum,
                ]
            oplist = [cutandsplice,perm,lattice,ripple,slip,rot,soft,form,rattle]
            if self.parameters.Algo == 'EA':
                if self.parameters.mlpredict:
                    assert self.parameters.useml, "'useml' must be True"
                    calc = self.MLCalculator.calc
                    self.PopGenerator = MLselect(numlist,oplist,calc,self.parameters)
                else:
                    self.PopGenerator = PopGenerator(numlist,oplist,self.parameters)
            self.parameters.attach(self.PopGenerator.p)
        return self.PopGenerator

    def get_MLCalculator(self):
        if not hasattr(self,'MLCalculator'):
            if self.parameters.useml:
                checkParameters(self.parameters,self.parameters,[],{'mlmodel':'GPR'})
                if self.parameters.mlmodel == 'LR':
                    self.MLCalculator = LRmodel(self.parameters)
                elif self.parameters.mlmodel == 'GPR':
                    self.MLCalculator = GPRmodel(self.parameters)
                elif self.parameters.mlmodel == 'GPR-torch':
                    self.MLCalculator = pytorchGPRmodel(self.parameters)
                elif self.parameters.mlmodel == 'BayesLR':
                    self.MLCalculator = BayesLRmodel(self.parameters)
            else:
                self.MLCalculator = None
            self.parameters.MLCalculator = self.MLCalculator.p
        return self.MLCalculator

    def get_MainCalculator(self):
        if not hasattr(self,'MainCalculator'):
            p = copy.deepcopy(self.parameters)
            for key, val in self.parameters.MainCalculator.items():
                setattr(p, key, val)
            checkParameters(p,p,['calculator'],{})
            if p.calculator == 'vasp':
                MainCalculator = VaspCalculator(p)
            elif p.calculator == 'lj':
                MainCalculator = LJCalculator(p)
            elif p.calculator == 'emt':
                MainCalculator = EMTCalculator(p)
            elif p.calculator == 'gulp':
                MainCalculator = GULPCalculator(p)
            elif p.calculator == 'xtb':
                MainCalculator = XTBCalculator(p)
            elif p.calculator == 'quip':
                MainCalculator = QUIPCalculator(p)
            elif p.calculator == 'lammps':
                MainCalculator = LammpsCalculator(p)
            else:
                raise Exception("Undefined calculator '{}'".format(p.calculator))
            self.MainCalculator = MainCalculator
            self.parameters.MainCalculator = self.MainCalculator.p
        return self.MainCalculator

    def get_Population(self):
        if not hasattr(self,'Population'):
            self.Population = Population(self.parameters)
            self.parameters.attach(self.Population.p)
        return self.Population

if __name__ == '__main__':
    parm = read_parameters('input.yaml')
    # print(parm['numFrml'])

