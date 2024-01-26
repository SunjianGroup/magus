import numpy as np
from magus.phasediagram import PhaseDiagram
import abc
import magus.xrdutils as xrdutils
import logging
log = logging.getLogger(__name__)

class FitnessCalculator(abc.ABC):
    def __init__(self, parameters) -> None:    
        pass

    @abc.abstractmethod
    def calc(self, Pop):
        pass


class EnthalpyFitness(FitnessCalculator):
    def calc(self, pop):
        for ind in pop:
            ind.info['fitness']['enthalpy'] = -ind.info['enthalpy']


class GapFitness(FitnessCalculator):
    def __init__(self, parameters) -> None:
        self.target_gap = parameters['targetGap']

    def calc(self, pop):
        for ind in pop:
            ind.info['fitness']['gap'] = -abs(ind.info['direct_gap'] - self.target_gap) \
                                         -abs(ind.info['indirect_gap'] - ind.info['direct_gap']) 


class EhullFitness(FitnessCalculator):
    def __init__(self, parameters) -> None:
        super().__init__(parameters)
        self.boundary = parameters['units']

    def calc(self, pop):
        pd = PhaseDiagram(pop, self.boundary)
        for ind in pop:
            ehull = ind.info['enthalpy'] - pd.decompose(ind)
            if ehull < 1e-4:
                ehull = 0
            ind.info['ehull'] = ehull
            ind.info['fitness']['ehull'] = -ehull

class XrdFitness(FitnessCalculator):
    def __init__(self, parameters):
        self.wave_length = parameters['waveLength'] # in Angstrom
        self.match_tolerence = 2
        if 'matchTol' in parameters:
            self.match_tolerence = parameters['matchTol']
        self.target_peaks = np.array(parameters['targetXrd'],dtype='float')
        self.two_theta_range = [ max(min(self.target_peaks[0])-2,0),
                                 min(max(self.target_peaks[0])+2,180)]
        
    def calc(self,pop):
        for ind in pop:
            xrd = xrdutils.XrdStructure(ind,self.wave_length,self.two_theta_range)
            ind.info['fitness']['XRD'] = -xrdutils.loss(xrd.getpeakdata().T,self.target_peaks,self.match_tolerence)
import math, os

class AgeFitness(FitnessCalculator):
    def __init__(self, parameters) -> None:
        self.age_scale = parameters['age_scale']
        self.anti_seeds = parameters['ANTISEED']
        self.type = parameters['type']
        self.anti_seeds['structs'] = []

    def refresh(self):
        try:
            self.anti_seeds['structs'] = ase.io.read(self.anti_seeds['file'], format='traj', index = ':')
        except:
            self.anti_seeds['structs'] = []
        
    def calc(self, pop):
        for ind in pop:
            cur_n_gen = pop.gen if not pop.gen == '' else 1
            born_n_gen = int((ind.info['identity'].split('-')[0]) [4:] )
            if self.type == 'age':
                ind.info['fitness']['age'] = -ind.info['enthalpy'] - self.age_fit(cur_n_gen - born_n_gen)
            elif self.type == 'antiseeds':
                #print('calc antiseed of enthalpy ', ind.info['enthalpy'])
                ind.info['fitness']['age'] = -ind.info['enthalpy'] - self.calc_anti_seed(ind)
                ind.info['fitness']['enthalpy'] = -ind.info['enthalpy'] 
                

    def age_fit(self, age):
        favor_age, scale_parm = self.age_scale
        if age < favor_age:
            return 0.0
        else:
            return scale_parm* (age - favor_age)

    def calc_anti_seed(self, ind):

        summary = 0
        Wa = self.anti_seeds['W']
        SIGMAa2 = self.anti_seeds['SIGMA']**2
        #print('Wa', Wa, 'SIGMA2', SIGMAa2)

        for a in self.anti_seeds['structs']:
            Dia2 = np.average([x**2 for x in a.fingerprint - ind.fingerprint])
            summary += Wa * math.exp(- Dia2 /2 / SIGMAa2 )
            #if (Wa * math.exp(- Dia2 /2 / SIGMAa2 ) > 1e-3):
            #print('dia2', Dia2, 'sum', Wa * math.exp(- Dia2 /2 / SIGMAa2 ), "\t", a.info['enthalpy'], ind.info["enthalpy"])
        #print('summary', summary)
        return summary 
fit_dict = {
    'Enthalpy': EnthalpyFitness,
    'Ehull': EhullFitness,
    'Gap': GapFitness,
    'XRD': XrdFitness,
    'Age': AgeFitness,
    }

def get_fitness_calculator(p_dict):
    fitness_calculator = []
    if 'Fitness' in p_dict:
        for fitness in p_dict['Fitness']:
            fitness_calculator.append(fit_dict[fitness](p_dict))
    elif p_dict['formulaType'] == 'fix':
        fitness_calculator.append(fit_dict['Enthalpy'](p_dict))
    elif p_dict['formulaType'] == 'var':
        fitness_calculator.append(fit_dict['Ehull'](p_dict))
    return fitness_calculator
