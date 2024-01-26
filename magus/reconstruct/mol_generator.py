from magus.generators.random import MoleculeSPGGenerator, SPGGenerator
import numpy as np
import math, os, ase.io
from ..utils import check_parameters
import logging
from ase.geometry import cell_to_cellpar,cellpar_to_cell
import spglib
import itertools
from ase.neighborlist import neighbor_list
from collections import Counter
from ase import Atoms


log = logging.getLogger(__name__)


# ForMuLa Filter for calculation.
class FMLfilter:
    def __init__(self, symbols_formula_count):
        # eg. dict = atoms.symbols.formula.count
        if isinstance(symbols_formula_count, dict):
            self.formula = Counter(symbols_formula_count)
        elif isinstance(symbols_formula_count, FMLfilter):
            self.formula = symbols_formula_count.formula
            
    
    def __str__(self) -> str:
        return "{}".format(self.formula)

    def __mul__(self, integer):
        return FMLfilter({key: self.formula[key] * integer for key in self.formula})
    
    def __sub__(self, FML2):
        for key in FML2.formula:
            assert key in self.formula.keys(), 'cannot calculate {} - {} for {} not in {}'.format(self, FML2, key, self)
        return FMLfilter(self.formula - FML2.formula)
    
    def __add__(self, FML2) :
        return FMLfilter(self.formula + FML2.formula)
        
    def __truediv__(self, FML2):
        return sum(self.formula.values()) / sum(FML2.formula.values())

    def __ge__(self, FML2):
        for key in FML2.formula:
            if not (self.formula.get(key, 0) >= FML2.formula[key]):
                return False
        return True
    
    def __gt__(self, FML2):
        for key in FML2.formula:
            if not (self.formula.get(key, 0) > FML2.formula[key]):
                return False
        return True
    
    def __lt__(self, FML2):
        for key in FML2.formula:
            if not (self.formula.get(key, 0) < FML2.formula[key]):
                return False
        return True

    def __le__(self, FML2):
        for key in FML2.formula:
            if not (self.formula.get(key, 0) <= FML2.formula[key]):
                return False
        return True
    

def map_lyr_spg(spacegroup):
    mapper = {77: 183, 80:191}
    assert spacegroup in mapper, "map_lyr_spg err {}".format(spacegroup)
    return mapper[spacegroup]

from ase.spacegroup import Spacegroup
def recommand_multiplicity(spacegroup, dimention):
    if dimention == 2:
        spacegroup = map_lyr_spg(spacegroup)
    sg = Spacegroup(int(spacegroup))
    if spacegroup in list(range(123,143)) +  list( range(162,168)) + list(range(175,231)):
        return [0]
    
    multiplicity = len(sg.get_symop())

    rcd = [0, multiplicity]
    if multiplicity %3 ==0:
        rcd.append(multiplicity/3)
    if multiplicity %2==0:
        rcd.append(multiplicity/2)
    if multiplicity %4==0:
        rcd.append(multiplicity/4)
    if multiplicity %6==0:
        rcd.append(multiplicity/6)

    return  rcd


class SPGMinerGenerator(SPGGenerator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.mine_probabilities = parameters.get("mine_probabilities", 0.4)

    def update(self, *args, **kwargs):
        if self.mine_probabilities > 0:
            if not kwargs.get('miner_spgs', None) is None:
                self.adjust_spg_selection(kwargs['miner_spgs'])

    def adjust_spg_selection(self, miner_spgs):
        _sum = sum(miner_spgs.values())
        miner_spgs = {s:miner_spgs[s]/ _sum *self.mine_probabilities for s in miner_spgs}
        normal_spgs = {s:(1.0 - self.mine_probabilities)/len(self.spacegroup) for s in self.spacegroup}

        mixed_ratio = Counter(miner_spgs) + Counter(normal_spgs)
        
        # exclude spg not in self.spacegroup
        mixed_ratio = {s: mixed_ratio[s] for s in self.spacegroup}

        
        _sum = sum(mixed_ratio.values())
        if _sum < 1e-5:
            # change back to uniform 2-230
            mixed_ratio = {s:1.0 for s in self.spacegroup}
            _sum = sum(mixed_ratio.values())

        mixed_ratio = {s:mixed_ratio[s]/_sum for s in mixed_ratio}
        mixed_ratio_ = sorted(mixed_ratio.items(),key =lambda x:x[1],reverse=True)

        self.spacegroup = list(mixed_ratio.keys())
        self.spg_probabilities = list(mixed_ratio.values())
        if log.level <=10:
            info = "update probabilities when choosing spacegroups in random generator. \n-------------------\n"
            for ii,(s,p) in enumerate(mixed_ratio_):
                info = info + str(s).rjust(3) + ' : ' + ("{:.2%}".format(p)).rjust(6) + ' | '
                if ii%5 == 4:
                    info = info[:-3]
                    info += '\n'
            info += "\n-------------------"
            log.debug(info)



class OntheFlyFragSPGGenerator(SPGMinerGenerator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.update_fragments_pool()

        self.target_formula = FMLfilter(dict(zip(parameters['symbols'], parameters['formula'])))
        self.frag_ratio = parameters.get('frag_ratio', [0.3,0.8])
        self.mol_formula = parameters.get('mol_formula', None)
    
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        if np.sum(self.frag_ratio) > 0 and 'frags' in kwargs:
            self.update_fragments_pool(frags = kwargs['frags'])

    def update_fragments_pool(self, frags = None):
        if frags is None:
            fragments_file = 'fragments_pool.xyz'
            if os.path.exists(fragments_file):
                self.fragments = ase.io.read(fragments_file, ':')
            else:
                self.fragments = []
                for i in self.symbols:
                    isolate_atom = Atoms(symbols=i, positions=[[0,0,0]])
                    isolate_atom.info = {'ubc':-1,
                                        'dof':-1,
                                        'origin': "-1",
                                        'config_type': "isolate_atom"}
                    self.fragments.append(isolate_atom)
                    ase.io.write(fragments_file, self.fragments)
        else:
            self.fragments = frags
        
        self.input_mols = self.fragments

    def get_frag_formula(self, spg):
        
        formula_pool = []
        if self.mol_formula is None:
            max_n_frag = np.zeros(len(self.fragments), dtype=int)
            for i,f in enumerate(self.fragments):
                if len(f) == 1:
                    continue
                max_n_frag[i] = int(FMLfilter(self.target_formula) / FMLfilter(f.symbols.formula.count()))

                while not (FMLfilter(f.symbols.formula.count())*max_n_frag[i] <= self.target_formula):
                    max_n_frag[i] -= 1
            print("max_n_frag", max_n_frag)
            while len(formula_pool) < 10:
                print("recommand", recommand_multiplicity(spg, self.dimension))
                rand_formula = [np.random.choice([a for a in range(0, i+1) if a in recommand_multiplicity(spg, self.dimension)]+[0]) for i in max_n_frag]
                rand_formula = [i if np.random.uniform(0,1)< 2/len(rand_formula) else 0 for i in rand_formula]
                now_formula = FMLfilter({})
                for i,j in enumerate(rand_formula):
                    now_formula += FMLfilter(self.fragments[i].symbols.formula.count()) * j
                
                if now_formula <= self.target_formula: #and (self.frag_ratio[0] < now_formula/ self.target_formula <self.frag_ratio[1] or now_formula/ self.target_formula==0) :
                    fill = self.target_formula - now_formula
                    combine = list(rand_formula)
                    for i, f in enumerate(self.fragments):
                        if len(f)==1:
                            combine[i] = fill.formula[f[0].symbol]
                    
                    formula_pool.append(combine)    
            
            # GET ALL COMBINE IS SLOW...
            '''
            for combine in itertools.product(*[list(range(0,i+1)) for i in max_n_frag]):
                now_formula = FMLfilter({})
                for a,b in enumerate(combine):
                    now_formula += FMLfilter(self.fragments[a].symbols.formula.count()) * b
                

                if not np.all([c in recommand_multiplicity(spg, self.dimension) for c in combine]):
                    continue

                if now_formula <= self.target_formula and (self.frag_ratio[0] < now_formula/ self.target_formula <self.frag_ratio[1] or now_formula/ self.target_formula==0) :
                    fill = self.target_formula - now_formula
                    combine = list(combine)
                    for i, f in enumerate(self.fragments):
                        if len(f)==1:
                            combine[i] = fill.formula[f[0].symbol]
                
                    formula_pool.append(combine)
            '''
        else:
            formula_pool = [self.mol_formula]

        
        f = []
        for _f in formula_pool:
            if not _f in f:
                f.append(_f)

        log.debug("OntheFlyFragSPGGenerator: spg = {}\ninput fragments: {}\nfragment formula{}".format(spg, [str(FMLfilter(f.symbols.formula.count())) for f in self.input_mols], f))

        return f


    def generate_ind(self, spg, formula, n_split=1):
        print("self.input_mols", self.input_mols)
           
        self.mol_n_atoms, self.mol_radius = [], []
        radius_dict = dict(zip(self.symbols, self.radius))

        for i, mol in enumerate(self.fragments):
            assert isinstance(mol, Atoms), "input molucules must be instance of class Atoms, {}".format(mol)
            for s in mol.get_chemical_symbols():
                assert s in self.symbols, "{} of {} not in given symbols".format(s, mol.get_chemical_formula())
            assert not mol.pbc.any(), "Please provide a molecule ranther than a periodic system!"
            self.mol_n_atoms.append(len(mol))
            # get molecule radius
            center = np.mean(mol.positions, 0)
            radius = np.array([radius_dict[s] for s in mol.get_chemical_symbols()])
            distance = np.linalg.norm(mol.positions - center, axis=1)
            self.mol_radius.append(np.max(distance + radius))

            self.input_mols[i] = mol
            self.volume = np.array([sum([4 * np.pi * (radius_dict[s]) ** 3 / 3
                            for s in mol.get_chemical_symbols()])
                            for mol in self.input_mols])

            
            ff = self.get_frag_formula(spg)
            if len(ff):
                numlist = ff[np.random.choice(range(len(ff)))]
                self.symbols = []
                for m in self.input_mols:
                    self.symbols += list(m.symbols.species())
                if np.any(np.array(numlist)>0):
                    print(numlist)
                    return super().generate_ind(spg, numlist, np.random.choice(self.n_split))
            else:
                return False, None


    def get_generate_parm(self, spg, numlist):
        d = super().get_generate_parm(spg, numlist)
        d.update({
            'mol_mode': True,
            'input_mols': self.input_mols,
            'threshold_mol': getattr(self, 'threshold_mol') if hasattr(self, 'threshold_mol') else 0.1,
            })
        return d