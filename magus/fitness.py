import numpy as np
from magus.phasediagram import PhaseDiagram
import abc
import magus.xrdutils as xrdutils


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

fit_dict = {
    'Enthalpy': EnthalpyFitness,
    'Ehull': EhullFitness,
    'Gap': GapFitness,
    'XRD': XrdFitness,
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
