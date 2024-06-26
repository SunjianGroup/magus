#Parallel Magus with parallel structure generation (random and GA) and relaxation
#(*For cluster-based ab-init calculator in mode parallel, the effect of parallelized genertion process 
# in saving time is limited THUS IT IS NOT SUPPORTED!)

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

        log.warning("\n\nMAGUS ver. parallel: \nResources for {} parallel <generator processes>".format(self.numParallelGen) + 
                " and {} parallel <calculator processes> are required.\n".format(self.numParallelCalc) + 
                "PLEASE NOTE THAT CLUSTER CALCULATOR IN PARALLEL MODE IS NOT SUPPORTED.\n")
        

        if not restart:
            self.cur_pop = self.Population([])
    
        #self.pop_generator.save_all_parm_to_yaml()


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
        relax_pop.gen = self.curgen
        relax_pop.del_duplicate()

        # decompose if needed

        if hasattr(self, "decompose_raw"):
            for_decompose = list(map(lambda x:x.for_heredity(), relax_pop))
            frags = self.decompose_raw(self, for_decompose)
        else:
            frags = []
        return raw_pop, relax_pop, frags
        

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
            init_pop = self.pop_generator.get_next_pop(self.parent_pop, n_next=math.ceil(popSize*(1-rand_ratio)) if len(self.parent_pop) else 0, 
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
    
    def set_current_pop(self, init_pop):
        raw, relax_pop, frags = self.relax(init_pop)
        relax_pop.gen = self.curgen 
        #save raw data
        raw.save('raw', self.curgen)
        
        log.debug("delete duplicate structures...")
        relax_pop.del_duplicate()
        self.cur_pop = relax_pop 
        
        if len(frags):
            self.raw_frags = frags

        return relax_pop

