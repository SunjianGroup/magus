"""****************************************************************
THIS IS INTERFACE FILE TO MAGUS 2.0 of on the fly spacegroup miner and fragment reorganizer. 
****************************************************************""" 
import logging
from functools import partial
import prettytable as pt

log = logging.getLogger(__name__)

smfr_type_list = ['fly_SMFR']

def Magus_one_step(self):
    smfr_patch_to_Magus.update_volume_ratio(self)
    smfr_patch_to_Magus.update_spg(self)
    init_pop = self.get_init_pop()
    init_pop.save('init', self.curgen)
    #######  relax  #######
    self.set_current_pop(init_pop)
    self.cur_pop.save('gen', self.curgen)
    #######  analyze #######
    self.analysis()
    frags = smfr_patch_to_Magus.decompose_raw(self, self.cur_pop)
    smfr_patch_to_Magus.update_spg_tracker(self)
    smfr_patch_to_Magus.update_frag(self, frags)
    smfr_patch_to_Magus.get_pop_for_heredity(self)


def interface_smfr(magus_inst, restart = False):
    smfr_patch_to_Magus.init__(magus_inst, magus_inst.parameters, restart)

    atoms_generator = smfr_random_generator({**magus_inst.parameters, **magus_inst.spg_miner})
    if atoms_generator == 'ukn':
        magus_inst.spg_miner = {}
        magus_inst.frag_reorg = {}
    elif not atoms_generator == "don't change":
        setattr(magus_inst, "atoms_generator", atoms_generator)
     

    set_smfr_population(magus_inst.Population, magus_inst.parameters.get('Fitness', None))
    magus_inst.Population.atoms_generator = atoms_generator
   
    setattr(magus_inst, "one_step", partial(Magus_one_step, magus_inst))
    if restart:
        smfr_patch_to_Magus.get_pop_for_heredity(magus_inst)

"""**********************************************
#1. Change init random population generator.
**********************************************"""
from .mol_generator import OntheFlyFragSPGGenerator

def smfr_random_generator(p_dict): 
    if p_dict['structureType'] == 'bulk':
        return OntheFlyFragSPGGenerator(**p_dict)
    elif p_dict['structureType'] == 'surface':
        return "don't change"
    else:
        log.warning("fly_SMFR function doesnot support {}.".format(p_dict['structureType']))
        return 'ukn'


"""**********************************************
#2. Add funtion "remove P1/mine_good_spg" to class Population, and age fitness.
**********************************************"""
from .mol_generator import pop_select, pop_mine_good_spg
from .fitness import AgeFitness, FitnessCalculator

def set_smfr_population(inst, fitness_parm):
    if hasattr(inst, "pop_select"):
        pass
    else:
        setattr(inst, "pop_select", pop_select)
    if hasattr(inst, "pop_mine_good_spg"):
        pass
    else:
        setattr(inst, "pop_mine_good_spg", pop_mine_good_spg)
    if type(inst.fit_calcs[0]) is FitnessCalculator:
        assert 'Age' in fitness_parm, 'unknown fitness'
        inst.fit_calcs = [AgeFitness(fitness_parm.get('Age', {}))]

    return inst

"""**********************************************
#3. Add on-the-fly funtion to class Magus.
**********************************************"""

from magus.reconstruct.local_decompose import DECOMPOSE, CGIO_read, CG_ISOLATE_ATOM, CGIO_write, is_same_frag
from magus.reconstruct.mol_generator import MinerTracker
from ase import Atoms


class smfr_patch_to_Magus:
    @staticmethod
    def init__(inst, parameters, restart=False):
        inst.spg_miner = parameters.get("spg_miner", {})
        inst.frag_reorg = parameters.get("frag_gen", {})
        if inst.spg_miner:
            inst.miner_tracker = MinerTracker(max_limit_per_spg = inst.spg_miner.get("max_limit_per_spg", 1000), scale_num_spg = inst.spg_miner.get("scale_num_spg",1000))  
            if restart:
                inst.miner_tracker.read()
        if inst.frag_reorg and restart:
            try:
                inst.frags = CGIO_read('fragments_pool.xyz',':')
            except:
                pass
        if not hasattr(inst, 'frags'):
            inst.frags = CG_ISOLATE_ATOM(parameters['symbols'])    
        setattr(inst, "decompose_raw", smfr_patch_to_Magus.decompose_raw)

    @staticmethod
    def decompose_raw(inst, raw_pop):
        if hasattr(inst, "raw_frags"):
            all_frags = inst.raw_frags
            delattr(inst, "raw_frags")
            return all_frags
        for_decompose = list(map(lambda x:x.for_heredity(), raw_pop))
        if inst.frag_reorg:
            all_frags = DECOMPOSE(for_decompose, inst.frag_reorg.get("distance_dict", None), neighbor_dis = inst.frag_reorg.get("neighbor_dis", 5),
                                       path_length_cut = inst.frag_reorg.get("path_length_cut", 4), n_community = inst.frag_reorg.get("n_community",[3,12]))
        else:
            all_frags = []
        return all_frags
    
    @staticmethod
    def update_volume_ratio(inst):
        if inst.curgen > 1:
            log.debug(inst.cur_pop)
            new_volume_ratio = 0.7 * inst.good_pop[:5].volume_ratio + 0.3 * inst.atoms_generator.volume_ratio
            inst.atoms_generator.set_volume_ratio(new_volume_ratio)

    @staticmethod
    def get_pop_for_heredity(inst):
        inst.parent_pop = inst.cur_pop + inst.keep_pop
        inst.parent_pop.pop_select(len(inst.parent_pop), remove_p1 = inst.parameters.get('remove_p1', 0.0))
        inst.parent_pop.gen = inst.curgen
        inst.parent_pop.del_duplicate()
        inst.parent_pop.calc_dominators()


    @staticmethod
    def update_spg(inst):
        if inst.spg_miner:
            if inst.curgen > 1:
                spgs = inst.parent_pop.pop_mine_good_spg(inst.spg_miner.get('mine_ratio', 0.5), miner_tracker = inst.miner_tracker)
                inst.atoms_generator.update(miner_spgs = spgs)

    @staticmethod
    def update_spg_tracker(inst):
        if inst.spg_miner:
            inst.miner_tracker.add_generation_to_analyzer(inst.curgen)
            inst.miner_tracker.write()
            inst.miner_tracker.max_limit_per_spg = inst.curgen * inst.parameters['popSize']

    @staticmethod
    def update_frag(inst, frags):
        if inst.frag_reorg:
            CGIO_write('results/frag{}.xyz'.format(inst.curgen), frags)
            _frags = inst.frags.copy()
            _frags.extend(frags)   

            inst.frags = []
            n = 0
            for f in _frags:
                if f.info['config_type'] == "isolate_atom":
                    inst.frags.append(f)
                    n+=1
                else:
                    origin = f.info['origin']
                    origin = origin[:origin.find(":")]
                    if origin in [ind.info['identity'] for ind in inst.good_pop[:10]]:
                        for ff in inst.frags:
                            if is_same_frag(f, ff):
                                break
                        else:
                            inst.frags.append(f)
            inst.frags[n:] = sorted(inst.frags[n:], key = lambda x: (x.info['ubc'], x.info['dof'], 1/len(x)))
            inst.frags = inst.frags[:6]

            table = pt.PrettyTable()
            table.field_names = ['Natoms', 'ubc', 'dof', 'origin', 'config_type','dimension', 'density'] 
            for f in inst.frags:
                table.add_row([len(f)] + [f.info.get(key, None) for key in table.field_names[1:]])
            log.debug('resultant frags: \n'  + table.__str__())

            CGIO_write('fragments_pool.xyz', inst.frags)
            inst.atoms_generator.update(frags = list(map(lambda x:x.output_atoms(), inst.frags)))


