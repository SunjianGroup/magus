#Parallel Magus with parallel structure generation (random and GA) and relaxation
#(*only supports ASE-based local relaxation calculator and GULP calculator)

from magus.search.search import Magus
import multiprocessing as mp
import logging
import math
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import traceback
import os
import prettytable as pt
import ase.io
from magus.reconstruct.local_decompose import DECOMPOSE, CrystalGraph, CGIO_read, CG_ISOLATE_ATOM, CGIO_write
from magus.reconstruct.parentspg import MinerTracker
from ase import Atoms

_distance_dict =  {
   
    ('O', 'Mg'): 2.2769999999999997,
    ('Mg', 'O'): 2.2769999999999997,
    ('O', 'Si'): 1.9469999999999998,
    ('Si', 'O'): 1.9469999999999998,
    ('O', 'Al'): 2.057,
    ('Al', 'O'): 2.057,
    ('O', 'O'): 0.1,
    ('Mg', 'Mg'): 0.1,
    ('Mg', 'Si'): 0.1,
    ('Si', 'Mg'): 0.1,
    ('Mg', 'Al'): 0.1,
    ('Al', 'Mg'): 0.1,
    ('Si', 'Si'): 0.1,
    ('Si', 'Al'): 0.1,
    ('Al', 'Si'): 0.1,
    ('Al', 'Al'): 0.1,

   ('B', 'B'): 1.848,
   ('C', 'C'):1.6
   }


def is_same_frag(a,b):
    if isinstance(a, Atoms):
        acg = CrystalGraph()
        acg.input_atoms(a)
    else:
        acg = a
    if isinstance(b, Atoms):

        bcg = CrystalGraph()
        bcg.input_atoms(b)
    else:
        bcg = b

    if acg==bcg:
        return True
    else:
        return False

log = logging.getLogger(__name__)

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except mp.TimeoutError:
        log.warning("Process '{}' aborted due to timeout".format(os.getpid()))
        return None

class PaMagus(Magus):
    def __init__(self, parameters, restart=False):
        super().__init__(parameters, restart=restart)
        self.numParallelGen = self.parameters['num_para_generator'] if 'num_para_generator' in self.parameters else 10
        self.numParallelCalc = self.parameters['num_para_calc'] if 'num_para_calc' in self.parameters else 10
        self.kill_time = self.parameters['kill_time']  if 'kill_time' in self.parameters else 3600  # 1 hour

        log.info("\nMAGUS ver. parallel: \nResources for {} parallel <generator processes>".format(self.numParallelGen) + 
                " and {} parallel <calculator processes> are required.\n".format(self.numParallelCalc))
        #log.info("Parallelized functions only include <structure generation (random and GA), " + 
        #         "structure relaxation (ASE-based local relaxation calculator and GULP calculator)>. \n" + 
        #         "Attemptions in other systems may lead to unknown errors. ")
        self.on_the_fly_spg_miner = self.parameters.get("on_the_fly_spg_miner", False)
        self.on_the_fly_frager = self.parameters.get("on_the_fly_frager", False)
        if self.on_the_fly_spg_miner:
            self.miner_tracker = MinerTracker(max_limit_per_spg = self.parameters.get("max_limit_per_spg", 1000), scale_num_spg = self.parameters.get("scale_num_spg",1000))  
            if restart:
                self.miner_tracker.read()
        if self.on_the_fly_frager and restart:
            try:
                self.frags = CGIO_read('fragments_pool.xyz',':')
            except:
                pass
        if not hasattr(self, 'frags'):
            self.frags = CG_ISOLATE_ATOM( self.parameters['symbols'])
        
        
        if not restart:
            self.cur_pop = self.Population([])
    
        self.pop_generator.save_all_parm_to_yaml()


    def relax_serial(self, init_pop, thread_num = 0):
        np.random.seed(np.random.randint(100000) +thread_num)
        logfile = 'aserelax{}.log'.format(thread_num)
        trajname = 'calc{}.traj'.format(thread_num)

        relax_pop = self.main_calculator.relax(init_pop, logfile = logfile, trajname = trajname)
        # return raw data before checking
        raw_pop = relax_pop.copy()
        relax_pop.check()
        # find spg before delete duplicate
        log.debug("'{}'th process find spg...".format(thread_num))
        relax_pop.find_spg()
        relax_pop.del_duplicate()
        if self.on_the_fly_frager:
            decomposed_pop = DECOMPOSE(relax_pop, _distance_dict, neighbor_dis=5, path_length_cut = 4, minimal_n_community=3)
        else:
            decomposed_pop = []
        return raw_pop, relax_pop, decomposed_pop
        

    def relax(self, calcPop):
        numParallel = self.numParallelCalc
        PopList = [[] for _ in range(0, numParallel)]

        pop_num_per_thread = math.ceil(len(calcPop) / numParallel)

        for i in range(0,numParallel):
            for j in range(0, pop_num_per_thread):
                if pop_num_per_thread * i + j < len(calcPop): 
                    PopList[i].append(calcPop[pop_num_per_thread * i + j])
                
        pool = mp.Pool(len(PopList))
        
        runjob = partial(abortable_worker, self.relax_serial, timeout = self.kill_time)

        rawPop, resultPop, frag = None, None, None
        
        r1pool = [  pool.apply_async(runjob, args=(calcPop.__class__(PopList[i]), i)) for i in range(0, len(PopList))
        ]

        pool.close()
        pool.join()

        for i in range(0, len(PopList)):
            try:
                r2 = r1pool[i].get(timeout=0.1)
                if r2 is None:
                    log.warning("Exception timeout: '{}'th process terminated after {} seconds".format(i, self.kill_time))    
                else:
                    if rawPop is None:
                        rawPop = r2[0]
                    else: 
                        rawPop.extend(r2[0])

                    if resultPop is None:
                        resultPop = r2[1]
                    else: 
                        resultPop.extend(r2[1])
                    
                    if frag is None:
                        frag = r2[2]
                    else: 
                        frag.extend(r2[2])
                        
                    
            except Exception:
                log.warning("Process '{}'th:\n{}".format(i, traceback.format_exc()))
        
        return rawPop, resultPop, frag

    def get_init_pop_serial(self, initSize, popSize, thread_num, rand_ratio):
        np.random.seed(np.random.randint(100000) +thread_num)
        # mutate and crossover, empty for first generation
        if self.curgen == 1:
            random_frames = self.atoms_generator.generate_pop(math.ceil(initSize))
            init_pop = self.Population(random_frames, 'init', self.curgen)
            log_table = []
        else:
            self.pop_generator.gen = self.curgen
            log_table = []
            init_pop = self.pop_generator.get_next_pop(self.parent_pop, math.ceil(popSize*(1-rand_ratio)), 
                                                       log_table = log_table, thread_num = thread_num, need_change_op_ratio = False, dominators_calced = True)
            init_pop.gen = self.curgen
            init_pop.atoms_generator = self.atoms_generator
            init_pop.fill_up_with_random(targetLen = math.ceil(popSize))

        for i,atoms in enumerate(init_pop):
            atoms.info['gen'] = self.curgen
            atoms.info['identity'] = "{}{}-{}-{}".format(init_pop.name, self.curgen, thread_num, i)

        init_pop.check()

        return init_pop, log_table

    def get_init_pop(self):
        numParallel = self.numParallelGen
        pool = mp.Pool(numParallel)
        init_num_per_thread = self.parameters['initSize'] / numParallel
        pop_size_per_thread = math.ceil(self.parameters['popSize'] / numParallel)

        rand_ratio_ = self.parameters['rand_ratio']

        if self.curgen > 1:
            self.parent_pop = self.cur_pop + self.keep_pop
            self.parent_pop.select(len(self.parent_pop), remove_p1 = self.parameters['remove_p1'])
            self.parent_pop.gen = self.curgen
            self.parent_pop.del_duplicate()
            self.parent_pop.calc_dominators()

        # For AutoOPRatio GAGenerators, update its operation ratios before multiplying processes.
        if self.parameters['autoOpRatio'] and self.curgen > 1:
            self.pop_generator.gen = self.curgen
            self.pop_generator.change_op_ratio(self.parent_pop, output_log = True)
            rand_ratio_ = self.pop_generator.rand_ratio

        target_rand = np.round(self.parameters['popSize'] * rand_ratio_)
        rand_ratio_per_thread = []
        for i in range(0,numParallel):
            if len(rand_ratio_per_thread) * pop_size_per_thread < target_rand - pop_size_per_thread:
                rand_ratio_per_thread.append(1.0)
            elif len(rand_ratio_per_thread) * pop_size_per_thread < target_rand:
                rand_ratio_per_thread.append(target_rand / pop_size_per_thread - len(rand_ratio_per_thread))
            else:
                break
        
        # For Auto spg miner SPGGenerator, update its spg probabilities before multiplying processes.
        if self.on_the_fly_spg_miner:
            if self.curgen > 1:
                spgs = self.parent_pop.mine_good_spg(self.parameters['mine_ratio'], miner_tracker = self.miner_tracker)
                self.atoms_generator.update(miner_spgs = spgs)

        runjob = partial(abortable_worker, self.get_init_pop_serial, timeout = self.kill_time)

        init_pop, table = None, None

        r1pool = [ pool.apply_async(runjob, args=(init_num_per_thread, pop_size_per_thread, i, rand_ratio_per_thread[i] if i < len(rand_ratio_per_thread) else 0))  for i in range(0, numParallel)      ]
        
        pool.close()
        pool.join()
        for i in range(0, numParallel):
            try:
                r2 = r1pool[i].get(timeout=0.1)
                if r2 is None:
                    log.warning("Exception timeout: '{}'th process terminated after {} seconds".format(i, self.kill_time))   
                else:
                    if init_pop is None:
                        init_pop = r2[0]
                    else: 
                        init_pop.extend(r2[0])

                    if table is None:
                        table = np.array(r2[1])
                    else:
                        table = table + np.array(r2[1])

            except Exception:
                log.warning("Process '{}'th:\n{}".format(i, traceback.format_exc()))
        
        if self.curgen > 1 and log.level <= 20:
            
            sumtable = pt.PrettyTable()
            sumtable.field_names = ['Operator', 'Probability ', 'SelectedTimes', 'SuccessNum']
            for i in range(len(self.pop_generator.op_list)):
                sumtable.add_row([self.pop_generator.op_list[i].descriptor,
                            '{:.2%}'.format(self.pop_generator.op_prob[i]),
                            *table[i],
                            ])
            log.info("OP infomation: \n" + sumtable.__str__())
        
        
        # In cases self.parameters['popSize'] < numParallel * size_per_thread
        if self.curgen == 1 and len(init_pop) > self.parameters['initSize']:
            init_pop = init_pop[:self.parameters['initSize']]
        elif len(init_pop) > self.parameters['popSize']:
            init_pop = init_pop[:self.parameters['popSize']]

        # pass used times to self.parent_pop
        if self.curgen > 1:
            for ind in init_pop:
                if 'Mutation' in ind.info['origin'] or 'Pairing' in ind.info['origin']:
                    for parid in ind.info['parents']:
                        self.parent_pop[[ind.info['identity'] for ind in self.parent_pop].index(parid)].info['used'] +=1 
        
        ## read seeds
        log.info("Generate new initial population with {} individuals:".format(len(init_pop)))
        seed_pop = self.read_seeds()
        seed_pop.check()
        init_pop.extend(seed_pop)

        log.info("Extended new initial population to {} individuals with seeds".format(len(init_pop)))
        
        origins = [atoms.info['origin'] for atoms in init_pop]
        for origin in set(origins):
            log.info("  {}: {}".format(origin, origins.count(origin)))
        # del dulplicate?
        return init_pop
    """
    def update_anti_seeds(self):
        age_old = self.parameters['age_old']
        cur_n_gen = self.curgen
        
        for ind in self.good_pop:

            born_n_gen = int((ind.info['identity'].split('-')[0]) [4:] )
            age =  cur_n_gen - born_n_gen 

            if age >= age_old:
                for i in self.anti_seeds:
                    if i.info['identity'] == ind.info['identity']:
                        break
                else:
                    self.anti_seeds.append(ind)
                    log.debug("add anti_seeds: identity {} ".format(ind.info['identity']))
        
        '''
        if cur_n_gen > age_old:
            searched_space = ase.io.read('results/gen{}.traj'.format(cur_n_gen - age_old), index=':')
            for ind in searched_space:
                for i in self.anti_seeds:
                    if i.info['identity'] == ind.info['identity']:
                        break
                else:
                    self.anti_seeds.append(ind)

        n = len(self.anti_seeds)
        
        self.anti_seeds.del_duplicate()
        log.info("update antiseeds... from length {} to {}", n, len(self.anti_seeds))
        '''
        if len(self.anti_seeds):
            ase.io.write('antiseeds.traj', self.anti_seeds)
    """    

    def one_step(self):
        self.update_volume_ratio()
        init_pop = self.get_init_pop()
        init_pop.save('init', self.curgen)
        #######  relax  #######
        raw, relax_pop, frags = self.relax(init_pop)
        # __
        # \!/   sum relax_step not implied yet 
        """
        try:
            relax_step = sum([sum(atoms.info['relax_step']) for atoms in relax_pop])
            log.info('DFT relax {} structures with {} scf'.format(len(relax_pop), relax_step))
        except:
            pass
        """
        #save raw data
        raw.save('raw', self.curgen)
        if self.on_the_fly_spg_miner:
            self.miner_tracker.add_generation_to_analyzer(self.curgen)
            self.miner_tracker.write()
            self.miner_tracker.max_limit_per_spg = self.curgen * self.parameters['popSize']/15


        log.debug("delete duplicate structures...")
        relax_pop.del_duplicate()
        relax_pop.save('gen', self.curgen)
        self.cur_pop = relax_pop
        log.debug("set good population..")
        self.set_good_pop()
        self.good_pop.save('good', '')
        self.good_pop.save('good', self.curgen)

        CGIO_write('results/frag{}.xyz'.format(self.curgen), frags)
        if self.on_the_fly_frager:
            _frags = self.frags.copy()
            _frags.extend(frags)   

            self.frags = []
            n = 0
            for f in _frags:
                if f.info['config_type'] == "isolate_atom":
                    self.frags.append(f)
                    n+=1
                else:
                    origin = f.info['origin']
                    origin = origin[:origin.find(":")]
                    if origin in [ind.info['identity'] for ind in self.good_pop[:10]]:
                        for ff in self.frags:
                            if is_same_frag(f, ff):
                                break
                        else:
                            self.frags.append(f)
            self.frags[n:] = sorted(self.frags[n:], key = lambda x: (x.info['ubc'], x.info['dof'], 1/len(x)))
            self.frags = self.frags[:6]
            print('frags', self.frags)
            CGIO_write('fragments_pool.xyz', self.frags)
            self.atoms_generator.update(frags = list(map(lambda x:x.output_atoms(), self.frags)))

        log.debug("set keep population..")
        self.set_keep_pop()
        self.keep_pop.save('keep', self.curgen)
        self.update_best_pop()
        self.best_pop.save('best', '')
        #self.update_anti_seeds()
        #if len(self.anti_seeds):
        #    self.anti_seeds.save('antiseeds', self.curgen)
