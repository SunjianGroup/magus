from ..fitness import FitnessCalculator
import numpy as np
from .utils import RCSPhaseDiagram
import ase.io
import logging

log = logging.getLogger(__name__)


def symbols_and_formula(atoms):
    L = atoms.get_chemical_symbols()
    symbols = list(set(L))
    formula =  np.array([L.count(s) for s in symbols])
    return {s:i for s,i in zip(symbols, formula)}

class ErcsFitness(FitnessCalculator):
    def calc(self, pop):
        self.cal_refE(pop)

        if len(pop.Ind.slices) == 3:
            for nl in pop.Ind.symbol_numlist_pool:
                snp = pop.Ind.symbol_numlist_pool[nl]
                if len(snp) < np.sum([len(l) for l in snp]):
                    return self.Ehull(pop)

        return self.Eform(pop)


    def Ehull(self, pop):
        symbols = pop.Ind.symbol_list

        compound = pop.Ind.refE['compound']
        compoundE = pop.Ind.refE['compoundE']
        substrateE = pop.Ind.refE['substrateE']
        substrate = pop.Ind.refE['substrate']


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
            from .generator import formula_add
            frml = symbols_and_formula(ind)
            frml = formula_add({s: [frml[s]*scale] for s in frml.keys()}, {s: [-substrate[s]] for s in substrate.keys()})
            frml = {s: frml[s][0] for s in frml}
            delta_n.append(frml [symbols[0]] - frml[symbols[1]]*ref_num0) 

            Eo.append((ind.info['energy'] -frml[symbols[1]]*refE_perUnit)*scale - substrateE)

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
            pop[i].info['ehull'] = ehull
            pop[i].info['ehull'] = ehull
            pop[i].info['enthalpy'] = pop[i].info['enthalpy']
            pop[i].info['fitness']['ehull'] = -ehull
            pop[i].info['Eo'] = Eo[i]

    def Eform(self, pop):
    #define E_form = E_total - E_ideal - sum_x (nxux)
        uxdict = pop.Ind.refE['adEs']
        E_substrate = pop.Ind.refE['substrateE']
        substrate = pop.Ind.refE['substrate']

        for ind in pop:
            ind.info['enthalpy'] = ind.info['enthalpy']
            scale = ind.info['size'][0] * ind.info['size'][1]
            from .generator import formula_minus
            frml = formula_minus(symbols_and_formula(ind), {s: [substrate[s]*scale] for s in substrate})
            Eform = ind.info['energy'] - np.sum([frml[s]*uxdict[s] for s in frml.keys()]) - E_substrate*scale
            ind.info['Eo'] = Eform
            ind.info['Eo'] = Eform
            ind.info['fitness']['Eform'] = -Eform

    def cal_refE(self, pop):
        if pop.Ind.refE is None:
            from ..entrypoints.calculate import calculate
            from .tools import getslab
            slab = getslab(slabfile=None)
            res_pop = [ase.io.read("Ref/refslab.traj", index = 0)]
            res_pop.append(slab)
            ase.io.write("Ref/refslab.traj", res_pop)
            calculate(filename = "Ref/refslab.traj", output_file="Ref/refslab.traj")
            res_pop = ase.io.read("Ref/refslab.traj", index = ':')
            bulk, slab = res_pop[0], res_pop[1]

            substrateE, compoundE = slab.info['energy'], bulk.info['energy']
            substrate, compound = symbols_and_formula(slab), symbols_and_formula(bulk)
            pop.Ind.refE = {
                'substrateE': substrateE,
                'compoundE': compoundE,
                'substrate': substrate,
                'compound': compound,
                'adEs': {s: compoundE/len(bulk) for s in compound.keys() }
            }
        log.debug("default reference energy: {}".format(pop.Ind.refE))


import math, os

class AgeFitness(FitnessCalculator):
    def __init__(self, parameters) -> None: 
        self.age_scale = parameters['age_scale'] 
        #eg. [5,0.1], means fit = {enthalpy, age<5; enthalpy - 0.1*(age -5), age >=5}
 
    def calc(self, pop):
        for ind in pop:
            cur_n_gen = pop.gen
            born_n_gen = int((ind.info['identity'].split('-')[0]) [4:] )
            ind.info['fitness']['age'] = -ind.info['enthalpy'] - self.age_fit(cur_n_gen - born_n_gen)
            
    def age_fit(self, age):
        favor_age, scale_parm = self.age_scale
        if age < favor_age:
            return 0.0
        else:
            return scale_parm* (age - favor_age)

class AntiSeedFitness(FitnessCalculator):
    def __init__(self, parameters) -> None:
        self.anti_seeds = parameters['ANTISEED']
        self.anti_seeds['structs'] = []

    def refresh(self):
        try:
            self.anti_seeds['structs'] = ase.io.read(self.anti_seeds['file'], format='traj', index = ':')
        except:
            self.anti_seeds['structs'] = []
        
    def calc(self, pop):
        for ind in pop:
            #print('calc antiseed of enthalpy ', ind.info['enthalpy'])
            ind.info['fitness']['age'] = -ind.info['enthalpy'] - self.calc_anti_seed(ind)
            ind.info['fitness']['enthalpy'] = -ind.info['enthalpy'] 
    
    def calc_anti_seed(self, ind):

        summary = 0
        Wa = self.anti_seeds['W']
        SIGMAa2 = self.anti_seeds['SIGMA']**2
        #print('Wa', Wa, 'SIGMA2', SIGMAa2)

        for a in self.anti_seeds['structs']:
            Dia2 = np.average([x**2 for x in a.fingerprint - ind.fingerprint])
            summary += Wa * math.exp(- Dia2 /2 / SIGMAa2 )
            #if (Wa * math.exp(- Dia2 /2 / SIGMAa2 ) > 1e-3):
            print('dia2', Dia2, 'sum', Wa * math.exp(- Dia2 /2 / SIGMAa2 ), "\t", a.info['enthalpy'], ind.info["enthalpy"])
        print('summary', summary)
        return summary 
    
