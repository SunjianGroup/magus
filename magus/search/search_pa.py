#Parallel Magus with parallel structure generation (random and GA) and relaxation
#(*only supports ASE-based local relaxation calculator and GULP calculator)

from .search import Magus
import multiprocessing as mp
import logging
import math
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np

log = logging.getLogger(__name__)

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except mp.TimeoutError:
        log.warning("Aborting due to timeout")
        return None

class PaMagus(Magus):
    def __init__(self, parameters, restart=False):
        super().__init__(parameters, restart=restart)
        self.numParallel = self.parameters['num_parallel']
        log.info("Number of parallel: {}".format(self.numParallel))
        self.kill_time = self.parameters['kill_time']

    def relax_serial(self, init_pop, thread_num = 0):
        np.random.seed(np.random.randint(100000) +thread_num)
        logfile = 'aserelax{}.log'.format(thread_num)
        trajname = 'calc{}.traj'.format(thread_num)

        relax_pop = self.main_calculator.relax(init_pop, logfile = logfile, trajname = trajname)
        # save raw date before checking
        relax_pop.save('raw{}_'.format(thread_num), self.curgen)
        relax_pop.check()
        # find spg before delete duplicate
        log.debug("{}st thread find spg...".format(thread_num))
        relax_pop.find_spg()
        return relax_pop
        

    def relax(self, calcPop):
        PopList = [[] for _ in range(0, self.numParallel)]

        pop_num_per_thread = math.ceil(len(calcPop) / self.numParallel)

        for i in range(0,self.numParallel):
            for j in range(0, pop_num_per_thread):
                if pop_num_per_thread * i + j < len(calcPop): 
                    PopList[i].append(calcPop[pop_num_per_thread * i + j])
                
        pool = mp.Pool(len(PopList))
        
        runjob = partial(abortable_worker, self.relax_serial, timeout = self.kill_time)

        resultPop = None
        
        r1pool = [  pool.apply_async(runjob, args=(calcPop.__class__(PopList[i]), i)) for i in range(0, len(PopList))
        ]

        pool.close()
        pool.join()

        for i in range(0, len(PopList)):
            try:
                r2 = r1pool[i].get(timeout=0.1)
                if r2 is None:
                    log.warning("Exception timeout: subprocess {}th terminated after {} seconds".format(i, self.kill_time))    
                else:
                    if resultPop is None:
                        resultPop = r2
                    else: 
                        resultPop.extend(r2)
            except:
                pass
        
        return resultPop    

    def get_init_pop_serial(self, initSize, popSize, thread_num):
        np.random.seed(np.random.randint(100000) +thread_num)
        # mutate and crossover, empty for first generation
        if self.curgen == 1:
            random_frames = self.atoms_generator.generate_pop(initSize)
            init_pop = self.Population(random_frames, 'init', self.curgen)
        else:
            init_pop = self.pop_generator.get_next_pop(self.cur_pop + self.keep_pop, popSize)
            init_pop.gen = self.curgen
            init_pop.fill_up_with_random(targetLen = popSize)

        #log.info("refine coordination number ...")
        #extend_pop = init_pop.refine_coordination_number()
        #init_pop.extend(extend_pop)

        #log.info("refine atomic envirment ...")
        #extend_pop = init_pop.refine_env()
        #init_pop.extend(extend_pop)

        init_pop.check()

        return init_pop

    def get_init_pop(self):
        pool = mp.Pool(self.numParallel)
        init_num_per_thread = math.ceil(self.parameters['initSize'] / self.numParallel)
        pop_size_per_thread = math.ceil(self.parameters['popSize'] / self.numParallel)

        runjob = partial(abortable_worker, self.get_init_pop_serial, timeout = self.kill_time)

        init_pop = None

        r1pool = [ pool.apply_async(runjob, args=(init_num_per_thread, pop_size_per_thread, i))  for i in range(0, self.numParallel)      ]
        
        pool.close()
        pool.join()
        for i in range(0, self.numParallel):
            try:
                r2 = r1pool[i].get(timeout=0.1)
                if r2 is None:
                    log.warning("Exception timeout: subprocess {}th terminated after {} seconds".format(i, self.kill_time))    
                else:
                    if init_pop is None:
                        init_pop = r2
                    else: 
                        init_pop.extend(r2)
            except:
                pass
        
        ## read seeds
        seed_pop = self.read_seeds()
        init_pop.extend(seed_pop)
        # check and log
        # __
        # \!/   seeds checking not implied yet ??

        log.info("Generate new initial population with {} individuals:".format(len(init_pop)))
        for i,atoms in enumerate(init_pop):
            atoms.info['gen'] = self.curgen
            atoms.info['identity'] = "{}{}-{}".format(init_pop.name, self.curgen, i)
        
        origins = [atoms.info['origin'] for atoms in init_pop]
        for origin in set(origins):
            log.info("  {}: {}".format(origin, origins.count(origin)))
        # del dulplicate?
        return init_pop


    def one_step(self):
        self.update_volume_ratio()
        init_pop = self.get_init_pop()
        init_pop.save('init', self.curgen)
        #######  relax  #######
        relax_pop = self.relax(init_pop)
        # __
        # \!/   sum relax_step not implied yet 
        """
        try:
            relax_step = sum([sum(atoms.info['relax_step']) for atoms in relax_pop])
            log.info('DFT relax {} structures with {} scf'.format(len(relax_pop), relax_step))
        except:
            pass
        """

        log.debug("delete duplicate structures...")
        relax_pop.del_duplicate()
        relax_pop.save('gen', self.curgen)
        self.cur_pop = relax_pop
        log.debug("set good population..")
        self.set_good_pop()
        self.good_pop.save('good', '')
        self.good_pop.save('good', self.curgen)
        log.debug("set keep population..")
        self.set_keep_pop()
        self.keep_pop.save('keep', self.curgen)
        self.update_best_pop()
        self.best_pop.save('best', '')
