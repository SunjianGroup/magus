import os, yaml, copy
from collections import defaultdict
from .populations import get_population
from .calculators import get_calculator
from .generators import get_random_generator, get_ga_generator
from .parmbase import Parmbase
try:
    from .reconstruct.rcs_interface import rcs_type_list, rcs_interface
except:
    rcs_type_list = []

#@Singleton
class magusParameters(Parmbase):
    __help_list = ["default"]
    __default = {
        'formulaType        //type of formula, choose from fix or var': 'fix', 
        'structureType      //structure type, choose from bulk, layer, confined_bulk, cluster, surface': 'bulk',
        'spacegroup     //spacegroup to generate random structures': "[1-230]",
        'DFTRelax': False,
        'initSize       //size of first population': "=popSize",
        'goodSize       //number of good indivials per generation': '=popSize',
        'molMode            //search molecule clusters': False,
        'mlRelax            //use Machine learning relaxation': False,
        'symprec            //tolerance for symmetry finding': 0.1,
        'bondRatio              //limitation to detect clusters': 1.15,
        'eleSize                //used in variable composition mode, control how many boundary structures are generated': 0,
        'volRatio       //cell_volume/SUM(atom_ball_volume) when generating structures (around this number)': 2,
        'dRatio         //distance between each pair of two atoms in the structure is\n' + '                 '\
                                                            'not less than (radius1+radius2)*d_ratio': 0.7,
        'molDetector        //methods to detect mol, choose from 1 and 2. See\n' + '                 '\
                        'Hao Gao, Junjie Wang, Zhaopeng Guo, Jian Sun, "Determining dimensionalities and multiplicities\n' + '                 '\
                        'of crystal nets" npj Comput. Mater. 6, 143 (2020) [doi.org/10.1016/j.fmre.2021.06.005]\n' + '                 '\
                        'for more details.': 0,
        'addSym     //whether to add symmetry before crossover and mutation': True,
        'randRatio  //ratio of new generated random structures in next generation': 0.2,
        'chkMol     //use mol dectector': False,
        'chkSeed        //check seeds': True,
        'diffE          //energy difference to determin structure duplicates': 0.01,
        'diffV          //volume difference to determin structure duplicates': 0.05,
        'comparator     //comparator, type magus checkpack to see which comparators you have.': 'nepdes',
        'fp_calc        //fingerprints, type magus checkpack to see which fingerprint method you have.': 'nepdes',
        'n_cluster      //number of good individuals per generation': '=saveGood',
        'autoOpRatio        //automantic GA operation ratio': False,
        'autoRandomRatio        //automantic random structure generation ratio': False,
        }
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

        Default = self.transform(self.__default)
        #avoid trouble in exporting help for static __default. 
        Default.update({
            'spacegroup': list(range(1, 231)),
            'initSize': p_dict['popSize'],
            'goodSize': p_dict['popSize'],
            'n_cluster': p_dict['saveGood'],
        })
        
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
