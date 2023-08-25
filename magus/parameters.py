import os, yaml, copy
from collections import defaultdict
from .populations import get_population
from .calculators import get_calculator
from .generators import get_random_generator, get_ga_generator
try:
    from .reconstruct.rcs_interface import rcs_type_list, rcs_interface
except:
    import traceback, warnings
    warnings.warn("Failed to load module for systems <clusters, surfaces, interfaces>:\n {}".format(traceback.format_exc()) +
                  "\nThis warning above can be ignored if the mentioned systems are not targets, elsewise should be fixed.\n" )
    rcs_type_list = []

#@Singleton
class magusParameters:
    def __init__(self, file):
        p_dict = defaultdict(int)
        if isinstance(file, dict):
            p_dict.update(file)
        elif isinstance(file, str):
            with open(file) as f:
                p_dict.update(yaml.load(f, Loader=yaml.FullLoader))
        p_dict['workDir']    = os.getcwd()
        p_dict['resultsDir'] = os.path.join(p_dict['workDir'], 'results')
        p_dict['calcDir']    = os.path.join(p_dict['workDir'], 'calcFold')
        p_dict['mlDir']      = os.path.join(p_dict['workDir'], 'mlFold')
        # not check here
        # Requirement = ['MainCalculator', 'popSize', 'numGen', 'saveGood', 'symbols']
        # for key in Requirement:
        #     if key not in p_dict:
        #         raise Exception('{} is not given'.format(key))
        Default = {
            'formulaType': 'fix', 
            'structureType': 'bulk',
            'spacegroup': list(range(1, 231)),
            'DFTRelax': False,
            'initSize': p_dict['popSize'],
            'goodSize': p_dict['popSize'],
            'molMode': False,
            'mlRelax': False,
            'symprec': 0.1,
            'bondRatio': 1.15,
            'eleSize': 0,
            'volRatio': 2,
            'dRatio': 0.7,
            'molDetector': 0,
            'addSym': True,
            'randRatio': 0.2,
            'chkMol': False,
            'chkSeed': True,
            'diffE': 0.01,
            'diffV': 0.05,
            'comparator': 'nepdes',
            'fp_calc': 'zernike',
            'n_cluster': p_dict['saveGood'],
            'autoOpRatio': False,
            'autoRandomRatio': False,
        }
        for key in Default:
            if key not in p_dict:
                p_dict[key] = Default[key]

        # translate spg such as 5-10 to list
        spg = []
        if not isinstance(p_dict['spacegroup'], list):
            p_dict['spacegroup'] = [p_dict['spacegroup']]
        for item in p_dict['spacegroup']:
            if isinstance(item, int):
                if 1 <= item <= 230:
                    spg.append(item)
            if isinstance(item, str):
                assert '-' in item, 'Please check the format of spacegroup'
                s1, s2 = item.split('-')
                s1, s2 = int(s1), int(s2)
                assert 1 <= s1 < s2 <= 230, 'Please check the format of spacegroup'
                spg.extend(list(range(s1, s2+1)))
        p_dict['spacegroup'] = spg

        if p_dict['chkMol']:
            assert p_dict['molDetector'] > 0, "If you want to check molecules, molDetector should be 1."

        self.p_dict = p_dict
        
        #This is interface to surface reconstruction, feel free to delete if not needed ;P
        if p_dict['structureType'] in rcs_type_list:
            rcs_interface(self)

    @property
    def RandomGenerator(self):
        if not hasattr(self, 'RandomGenerator_'):
            self.RandomGenerator_ = get_random_generator(self.p_dict)
        return self.RandomGenerator_

    @property
    def NextPopGenerator(self):
        if not hasattr(self, 'NextPopGenerator_'):
            self.NextPopGenerator_ = get_ga_generator(self.p_dict)
        return self.NextPopGenerator_

    @property
    def MLCalculator(self):
        if not hasattr(self, 'MLCalculator_'):
            if 'MLCalculator' in self.p_dict:
                p_dict = copy.deepcopy(self.p_dict)
                p_dict.update(p_dict['MLCalculator'])
                p_dict['query_calculator'] = self.MainCalculator
                self.MLCalculator_ = get_calculator(p_dict)   
            else:
                raise Exception('No ML Calculator!')
        return self.MLCalculator_

    @property
    def MainCalculator(self):
        if not hasattr(self,'MainCalculator_'):
            p_dict = copy.deepcopy(self.p_dict)
            p_dict.update(p_dict['MainCalculator'])
            self.MainCalculator_ = get_calculator(p_dict)
        return self.MainCalculator_

    @property
    def Population(self):
        if not hasattr(self,'Population_'):
            p_dict = copy.deepcopy(self.p_dict)
            p_dict['atoms_generator'] = self.RandomGenerator
            p_dict['units'] = self.RandomGenerator.units
            self.Population_ = get_population(p_dict)
        return self.Population_
