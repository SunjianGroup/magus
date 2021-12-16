import numpy as np
from ase.phasediagram import PhaseDiagram
import abc
from .reconstruct import RCSPhaseDiagram


class FitnessCalculator(abc.ABC):
    @abc.abstractmethod
    def calc(self, Pop):
        pass


class EnthalpyFitness(FitnessCalculator):
    def calc(self, pop):
        for ind in pop:
            ind.info['fitness']['enthalpy'] = -ind.info['enthalpy']


class EhullFitness(FitnessCalculator):
    def calc(self, pop):
        name = [ind.get_chemical_formula() for ind in pop]
        enth = [ind.info['enthalpy']*len(ind) for ind in pop]
        refs = list(zip(name, enth))
        symbols = pop.symbols
        # To make sure that the phase diagram can be constructed, we add elements with high energies.
        for sym in symbols:
            refs.append((sym, 100))
        pd = PhaseDiagram(refs, verbose=False)
        for ind in pop:
            refE = pd.decompose(ind.get_chemical_formula())[0]
            ehull = ind.info['enthalpy'] - refE/len(ind)
            if ehull < 1e-4:
                ehull = 0
            ind.info['ehull'] = ehull
            ind.info['fitness']['ehull'] = -ehull

class ErcsFitness(FitnessCalculator):
    def calc(self, Pop):
        
        if len(Pop.Individual.layerslices) == 3:
            return self.Eo(Pop)
        elif len(Pop.Individual.layerslices) == 2:
            return self.Eform(Pop)
        
    def Eo(self, Pop):
    # modified from var_fitness
        symbols = Pop.Individual.p.symbols

        #to remove H atom in bulk layer
        if symbols[0] =='H': 
            symbols = symbols[1:]

        pop = Pop.pop
        mark = 'fix'

        if len(symbols) >1 and Pop.Individual.p.AtomsToAdd:
            for atomnum in Pop.Individual.p.AtomsToAdd:
                if len(atomnum)>1:
                    mark = 'var'
                    break

        compound = Pop.Individual.p.refE['compound']
        compoundE = Pop.Individual.p.refE['compoundE']
            
        if mark == 'fix':
            refE_perAtom  = compoundE/np.sum([compound[s] for s in compound])

            for ind in pop:
                scale = 1.0 / ind.info['size'][0] / ind.info['size'][1]
                surfaceE = (ind.atoms.info['energy']-refE_perAtom*len(ind.atoms))*scale
                ind.info['Eo'] = surfaceE
                ind.info['enthalpy'] = ind.atoms.info['enthalpy']
                ind.info['fitness']['Eo'] = -surfaceE

        else:

            refE_perUnit = compoundE / compound[symbols[1]]
            ref_num0 =  1.0*compound[symbols[0]] / compound[symbols[1]]
            '''
            define Eo = E_slab - numB*E_ref, [E_ref = energy of unit A(a/b)B]
            define delta_n = numA - numB *(a/b)
            '''
            delta_n = []
            Eo = []
            for ind in pop:
                scale = 1.0 / ind.info['size'][0] / ind.info['size'][1]
                symbol, formula = symbols_and_formula(ind.atoms)
                frml = {s:i for s,i in zip(symbol, formula)}
                delta_n.append( (frml [symbols[0]] - frml[symbols[1]]*ref_num0) *scale)
                Eo.append((ind.atoms.info['energy'] -frml[symbols[1]]*refE_perUnit)*scale)

            refs = list(zip(delta_n, Eo))
            # To make sure that the phase diagram can be constructed, we add elements with high energies.
            refs.append((-ref_num0, 100))
            refs.append((1, 100))
            pd = RCSPhaseDiagram(refs)
            for i in range(len(pop)):
                refEo = pd.decompose(delta_n[i])[0]
                ehull =  Eo[i] - refEo
                if ehull < 1e-4:
                    ehull = 0
                pop[i].atoms.info['ehull'] = ehull
                pop[i].info['ehull'] = ehull
                pop[i].info['enthalpy'] = pop[i].atoms.info['enthalpy']
                pop[i].info['fitness']['ehull'] = -ehull
                pop[i].info['Eo'] = Eo[i]

    def Eform(self, Pop):
    #define E_form = E_total - E_ideal - sum_x (nxux)
    #Lu et al, Carbon 159 (2020) 9-15, https://doi.org/10.1016/j.carbon.2019.12.003
        uxdict = Pop.Individual.p.refE['adEs']
        for ind in Pop:
            ind.info['enthalpy'] = ind.atoms.info['enthalpy']
            symbol, formula = symbols_and_formula(ind.atoms)
            frml = {s:i for s,i in zip(symbol, formula)}
            Eform = ind.atoms.info['energy'] - np.sum([frml[s]*uxdict[s] for s in frml.keys()])
            ind.atoms.info['Eo'] = Eform
            ind.info['Eo'] = Eform
            ind.info['fitness']['Eform'] = -Eform

fit_dict = {
    'Enthalpy': EnthalpyFitness(),
    'Ehull': EhullFitness(),
    'Ercs': ErcsFitness(),
    }

def get_fitness_calculator(p_dict):
    fitness_calculator = []
    if 'Fitness' in p_dict:
        for fitness in p_dict['Fitness']:
            fitness_calculator.append(fit_dict[fitness])
    elif p_dict['formulaType'] == 'fix':
        fitness_calculator.append(fit_dict['Enthalpy'])
    elif p_dict['formulaType'] == 'var':
        fitness_calculator.append(fit_dict['Ehull'])
    return fitness_calculator
