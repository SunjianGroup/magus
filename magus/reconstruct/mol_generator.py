from magus.generators.random import SPGGenerator
import numpy as np
import  os, ase.io
import logging

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
        return "".join(["{}{}".format(s,self.formula[s]) for s in self.formula])

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
    

from .supergroupdb import spg_to_layer
from ase.spacegroup import Spacegroup
def recommand_multiplicity(spacegroup, dimention):
    if dimention == 2:
        sl = spg_to_layer()
        spacegroup = sl.layer_to_spg(spacegroup)
    sg = Spacegroup(int(spacegroup))
    if spacegroup in list(range(123,143)) +  list( range(162,168)) +list(range(187,231)): #list(range(175,231)):
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
        self.adjust_spg_selection({})

    def update(self, *args, **kwargs):
        if self.mine_probabilities > 0:
            if not kwargs.get('miner_spgs', None) is None:
                self.adjust_spg_selection(kwargs['miner_spgs'])

        
    def adjust_spg_selection(self, miner_spgs):
        if self.dimension == 2:
            s = spg_to_layer()
            miner_layers = {s.spg_to_layer(key): miner_spgs[key] for key in miner_spgs}
            if 1 in miner_layers:
                del miner_layers[1]
            miner_spgs = miner_layers

        _sum = sum(miner_spgs.values())
        miner_spgs = {s:miner_spgs[s]/ _sum *self.mine_probabilities for s in miner_spgs}
        normal_spgs = {s:(1.0 - self.mine_probabilities)/len(self.spacegroup) for s in self.spacegroup}
        mixed_ratio = Counter(miner_spgs) + Counter(normal_spgs) 

        if self.preset_spg_prob:
            _sum = np.sum(list(mixed_ratio.values()))
            mixed_ratio = {s: mixed_ratio[s] / _sum for s in mixed_ratio}
            _sum = np.sum(list(self.preset_spg_prob.values()))
            assert(_sum<=1.0), "the sum of set spg probabilities larger than 100%."
            mixed_ratio  = {s: mixed_ratio[s] * (1-_sum) for s in mixed_ratio}       
            mixed_ratio  = Counter(mixed_ratio) + Counter(self.preset_spg_prob)

        
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
            #print("max_n_frag", max_n_frag)
            while len(formula_pool) < 10:
                #print("recommand", recommand_multiplicity(spg, self.dimension))
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

        log.debug("SMFRGen: spg = {}; frags: {}; formula{}".format(spg, [str(FMLfilter(f.symbols.formula.count())) for f in self.input_mols], f))

        return f


    def generate_ind(self, spg, formula, n_split=1):
           
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
    
import copy, math
import numpy as np
# Spacegroup miner based on supergroup-subgroup relations.
# For example, spacegroup no.2, P-1 could mine into its supergroup no.10, P2/m.
from collections import Counter
import logging
from .supergroupdb import supergroup, _super_group_relation

def super_group_relation(spacegroup = -1, pointgroup = ""):
    if spacegroup > 0:
        for key in _super_group_relation:
            if spacegroup in list(range(key[0], key[1])):
                return {**_super_group_relation[key], 
                        "share_pointgroup": list(range(key[0], key[1])),
                        "name": key[2]}
    else:
        for key in _super_group_relation:
            if pointgroup == key[-1]:
                return {**_super_group_relation[key], 
                        "share_pointgroup": list(range(key[0], key[1])),
                        "name": key[2]}


class Miner:
    def __init__(self):
        pass
    
    @staticmethod
    def get_supergroup(supergroup_list, spacegroup = -1, pointgroup = ""):
        x = super_group_relation(supergroup_list, spacegroup=spacegroup, pointgroup=pointgroup)
        supergroup_list.extend([s for s in x['share_pointgroup'] if not s in supergroup_list])
        for name in x['super']:
            Miner.get_supergroup(supergroup_list, spacegroup=-1, pointgroup=name)
    
    def mine_spg(self, spg):
        x = super_group_relation(spacegroup = spg)
        log.debug("Mine into '{}' (pointgroup: {}; order: {})".format(spg, x['name'], x['order']))

        miner = {}
        isHighOrder = (x['order'] >7)
        isVeryHighOrder = (x['order'] >40)

        for s in supergroup[spg]['supergroups'].keys():
            if supergroup[spg]['supergroups'][s][0] <=(4 if isHighOrder else 8):
                miner[s] = 2
            else:
                miner[s] = 1

        if isHighOrder:
            miner[spg] = 50

        have_very_high_order = max([miner.get(s, 0) for s in range(221,231)])
        for s in range(221,231):
            # add all m-3m, things is weird when dealing with very high order
            miner[s] = have_very_high_order

        
        return miner

from collections import Counter
import yaml
import ase.io
import spglib

# miner_tracker: which spacegroups we mined into
# spg_trakcer:   spacegroups used to generate randoms
# analyzer:      some spacegroup just cannot generate useful structures.
#                the 'score' reflects this ability 

class MinerTracker:
    def __init__(self, trackfile = 'tracker.yaml', max_limit_per_spg = 10000, scale_num_spg=500):
        self.trackfile = trackfile
        self.analyzer = {}
        self.miner_tracker = Counter({})
        self.max_limit_per_spg = max_limit_per_spg
        self.scale_num_spg = scale_num_spg

    def read(self):
        with open(self.trackfile, 'r') as f:
            d = dict(yaml.load(f, Loader=yaml.FullLoader))
            self.analyzer = d['analyzer']
            self.miner_tracker = Counter(d['miner_tracker'])

    def write(self):
        with open(self.trackfile, 'w') as f:
            info = {'analyzer': dict(self.analyzer), 'miner_tracker':dict(self.miner_tracker)}
            yaml.dump(info, f)

    def add_to_analyzer(self, initspg, finspg, dominator):
        if initspg in self.analyzer:
            self.analyzer[initspg].append([finspg, dominator])
        else:
            self.analyzer[initspg] = [[finspg, dominator]]

    def add_generation_to_analyzer(self, gen):
        rawpop = ase.io.read('results/raw{}.traj'.format(gen), ':')
        rawpop = sorted([atom for atom in rawpop if atom.info['origin'] == 'random'], key = lambda x: x.info['enthalpy'])
        initpop = ase.io.read('results/init{}.traj'.format(gen), ':')
        initids = [x.info['identity'] for x in initpop]
        for i,atom in enumerate(rawpop):
            initatom = initpop[initids.index(atom.info['identity'])]
            initspg = spglib.get_symmetry_dataset(initatom, 0.1)['number']
            try:
                finspg = spglib.get_symmetry_dataset(atom, 0.1)['number']
            except:
                finspg = 1
            self.add_to_analyzer(initspg, finspg, i/len(rawpop))
    
    def add_miner_log_to_miner(self,miner):
        self.miner_tracker += miner

    def filter(self, miner):
        for initspg in miner:
            if not initspg in self.analyzer:
                # have no idea about initspg
                continue
            if len(self.analyzer[initspg]) < self.scale_num_spg:
                # have less idea about initspg
                continue
            elif len(self.analyzer[initspg]) > self.max_limit_per_spg:
                # used initspg too many times
                miner[initspg] = 0
            if not initspg in np.array(self.analyzer[initspg])[:,0]:
                # not stable
                miner[initspg] = 0
            if np.mean(np.array(self.analyzer[initspg])[:,1])>0.5:
                # unfavor
                miner[initspg] /= 2
        return miner



def pop_mine_good_spg(inst, good_ratio = 0.1, miner_tracker = Counter({})):
    inst.calc_dominators()
    spgs = [ind.info['spg'] for ind in sorted(inst.pop, key=lambda x: x.info['dominators']) if not ind.info['spg'] == 1]
    _miner_L = math.ceil(len(spgs) * good_ratio)
    
    miner = Counter({})
    

    for i,spg in enumerate(spgs):
        if i > _miner_L:
            break
        miner += Miner().mine_spg(spg)
    
    miner = miner_tracker.filter(miner)
    miner_tracker.add_miner_log_to_miner(miner)
    return miner

def pop_select(inst, n, remove_highE = 0., remove_p1 = 0.5):
    """
    good_pop selection: select first n-th (or less than n) population.
    Parameters:
        remove_highE: remove structures that have higher energy than 'remove_highE' * Min(energy).
        remove_p1: remove 'remove_p1' ratio of structures that have no symmetry.
    """
    
    inst.calc_dominators()
    inst.pop = sorted(inst.pop, key=lambda x: x.info['dominators'])
    
    if remove_highE > 0:
        enthalpys = [ind.info['enthalpy'] for ind in inst.pop]
        high = np.min(enthalpys) * remove_highE
        _oldLength = len(inst.pop)
        inst.pop = [ind for ind in inst.pop if ind.info['enthalpy'] <= high]
        logging.debug("select without enthalpy higher than {} eV/atom, pop length from {} to {}".format(high, _oldLength, len(inst.pop)))

    if remove_p1 > 0:
        _oldLength = len(inst.pop)
        inst.pop = [ind for ind in inst.pop if not (ind.info['spg']==1 and ind.info['dominators'] >= n * (1-remove_p1)) ]
        logging.debug("select without {:.2%} p1 symmetry structures, pop length from {} to {}".format(remove_p1, _oldLength, len(inst.pop)))

    if len(inst) > n:
        inst.pop = inst.pop[:n]