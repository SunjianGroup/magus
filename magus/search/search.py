import logging, os, shutil, subprocess
from magus.utils import read_seeds
from ase.io import read
# from ase.db import connect
import prettytable as pt

log = logging.getLogger(__name__)


class Magus:
    def __init__(self, parameters, restart=False):
        self.init_parms(parameters)
        self.seed_dir = '{}/Seeds'.format(self.parameters['workDir'])
        if restart:
            if not os.path.exists("results") or not os.path.exists("log.txt"):
                raise Exception("cannot restart without results or log.txt")
            content = 'grep "Generation" log.txt | tail -n 1'
            self.curgen = int(subprocess.check_output(content, shell=True).split()[-2])
            content = 'grep "volRatio" log.txt | tail -n 1'
            volume_ratio = float(subprocess.check_output(content, shell=True).split()[-1])
            self.atoms_generator.set_volume_ratio(volume_ratio)
            best_frames = read('results/best.traj', ':')
            good_frames = read('results/good.traj', ':')
            keep_frames = read('results/keep{}.traj'.format(self.curgen - 1), ':')
            cur_frames = read('results/gen{}.traj'.format(self.curgen - 1), ':')
            self.best_pop = self.Population(best_frames, 'best')
            self.good_pop = self.Population(good_frames, 'good')
            self.keep_pop = self.Population(keep_frames, 'keep', self.curgen - 1)
            self.cur_pop = self.Population(cur_frames, 'cur', self.curgen - 1)
            self.get_pop_for_heredity()
            log.warning("RESTART HERE!".center(40, "="))
        else:
            self.curgen = 1
            if os.path.exists("results"):
                i = 1
                while os.path.exists("results{}".format(i)):
                    i += 1
                shutil.move("results", "results{}".format(i))
            os.mkdir("results")
            self.best_pop = self.Population([], 'best')
            self.good_pop = self.Population([], 'good')
            self.keep_pop = self.Population([], 'keep')
            # self.db = connect("results/all_structures.db")


    def init_parms(self, parameters):
        self.parameters = parameters.p_dict
        self.atoms_generator = parameters.RandomGenerator
        self.pop_generator = parameters.NextPopGenerator
        self.main_calculator = parameters.MainCalculator
        self.Population = parameters.Population
        log.debug('Main Calculator information:\n{}'.format(self.main_calculator))
        log.debug('Random Generator information:\n{}'.format(self.atoms_generator))
        log.debug('Offspring Creator information:\n{}'.format(self.pop_generator))
        log.debug('Population information:\n{}'.format(self.Population([])))
        # For developers and testers only:
        # if you have the Global Minima structure[spg, enthalpy], write it in 'convergence_condition'
        # to stop the program and avoid useless further generations.
        if 'convergence_condition' in self.parameters:
            self.convergence_condition = self.parameters['convergence_condition']
            log.warning("WARNING: applied convergence_condition {}.".format(self.convergence_condition))
        else:
            self.convergence_condition = [-1, -1e+5]

    def read_seeds(self):
        log.info("Reading Seeds ...")
        seed_frames = read_seeds('{}/POSCARS_{}'.format(self.seed_dir, self.curgen))
        seed_frames.extend(read_seeds('{}/seeds_{}.traj'.format(self.seed_dir, self.curgen)))
        seed_pop = self.Population(seed_frames, 'seed', self.curgen)
        for i in range(len(seed_pop)):
            seed_pop[i].info['gen'] = self.curgen
        return seed_pop

    def get_pop_for_heredity(self):
        self.parent_pop = self.cur_pop + self.keep_pop

    def get_init_pop(self):
        # mutate and crossover, empty for first generation
        if self.curgen == 1:
            random_frames = self.atoms_generator.generate_pop(self.parameters['initSize'])
            init_pop = self.Population(random_frames, 'init', self.curgen)
        else:
            init_pop = self.pop_generator.get_next_pop(self.parent_pop, n_next=None if len(self.parent_pop) else 0)
            init_pop.gen = self.curgen
            init_pop.fill_up_with_random()
        ## read seeds
        seed_pop = self.read_seeds()
        init_pop.extend(seed_pop)
        # check and log
        init_pop.check()
        log.info("Generate new initial population with {} individuals:".format(len(init_pop)))
        for atoms in init_pop:
            atoms.info['gen'] = self.curgen
        origins = [atoms.info['origin'] for atoms in init_pop]
        for origin in set(origins):
            log.info("  {}: {}".format(origin, origins.count(origin)))
        # del dulplicate?
        return init_pop
    
    @staticmethod
    def show_pop_info(population, log_level = logging.DEBUG, show_pop_name = ''):
        if len(population) == 0:
            log.debug("no ind in {} pop.".format(population.name))
            return
        
        table = pt.PrettyTable()
        table.field_names = ['Dominator', 'Formula', 'Identity', 'Enthalpy'] + ['Fit-' + key for key in population[0].info.get('fitness', {}).keys()] 
        
        for ind in population:
            table.add_row([ind.info.get('dominators', None),
                           ind.get_chemical_formula(),
                           ind.info.get('identity', None), 
                           '{:.6f}'.format(ind.info.get('enthalpy', None)),
                           ] + ['{:.6f}'.format(ind.info.get('fitness', {})[key]) for key in ind.info.get('fitness', {})]    
                         )
        if hasattr(population, 'name'):
            name = population.name + " pop"
        else:
            name = 'Pop'
        
        name = show_pop_name or name
        log.log(log_level, name + " : \n" + table.__str__())


    def set_good_pop(self):
        log.info('construct goodPop')
        #target: good_pop = self.cur_pop + self.good_pop + self.keep_pop
        #Sometimes the mutated 'child' is relaxed back to 'parent'. 
        #The 'for...else' function purposely changes child.info['identity'] back to its parent.info['identity'] to get a correct history punish. 
        good_pop = self.good_pop + self.keep_pop
        for i, ind in enumerate(self.cur_pop):
            for ind1 in good_pop:
                if ind == ind1:
                    self.cur_pop[i] = ind1
                    break
            else:
                good_pop.append(ind)
        good_pop.gen = self.curgen
        good_pop.del_duplicate()
        good_pop.calc_dominators()
        good_pop.select(self.parameters['popSize'])
        self.good_pop = good_pop
        self.show_pop_info(good_pop)        

    def set_keep_pop(self):
        log.info('construct keepPop')
        _, keep_frames = self.good_pop.clustering(self.parameters['saveGood'])
        keep_pop = self.Population(keep_frames, 'keep', self.curgen)
        self.keep_pop = keep_pop
        self.show_pop_info(keep_pop)

    def update_best_pop(self):
        log.info("best ind:")
        bestind = self.good_pop.bestind()
        self.best_pop.extend(bestind)
        self.stop_signal = False

        if bestind[0].info['enthalpy'] <= self.convergence_condition[1]:
            if bestind[0].info['spg'] == self.convergence_condition[0] or self.convergence_condition[0] == -1:
                self.stop_signal = True

        self.show_pop_info(bestind, log_level=logging.INFO, show_pop_name = "BEST INDIVIDUALS")
        
    def run(self):
        while self.curgen <= self.parameters['numGen']:
            log.info(" Generation {} ".format(self.curgen).center(40, "="))
            self.one_step()
            self.curgen += 1
            if self.stop_signal:
                log.warning("Structure with spacegroup '{}', enthalpy lower than '{}' had appeared, which met the convergence_condition. GA loop break".format(*self.convergence_condition))
                break
        else:
            log.warning("Maximum number of generation reached")

    def update_volume_ratio(self):
        if self.curgen > 1:
            log.debug(self.cur_pop)
            new_volume_ratio = 0.7 * self.cur_pop.volume_ratio + 0.3 * self.atoms_generator.volume_ratio
            self.atoms_generator.set_volume_ratio(new_volume_ratio)

    def set_current_pop(self, init_pop):
        relax_pop = self.main_calculator.relax(init_pop)
        relax_pop.gen = self.curgen
        try:
            relax_step = sum([sum(atoms.info['relax_step']) for atoms in relax_pop])
            log.info('DFT relax {} structures with {} scf'.format(len(relax_pop), relax_step))
        except:
            pass
        # save raw data before checking
        relax_pop.save('raw', self.curgen)
        relax_pop.check()
        # find spg before delete duplicate
        log.debug("find spg...")
        relax_pop.find_spg()
        log.debug("delete duplicate structures...")
        relax_pop.del_duplicate()
        self.cur_pop = relax_pop 

    def analysis(self):
        log.debug("set good population..")
        self.set_good_pop()
        self.good_pop.save('good', '')
        self.good_pop.save('good', self.curgen)
        log.debug("set keep population..")
        self.set_keep_pop()
        self.keep_pop.save('keep', self.curgen)
        self.update_best_pop()
        self.best_pop.save('best', '')


    def one_step(self):
        self.update_volume_ratio()
        init_pop = self.get_init_pop()
        init_pop.save('init', self.curgen)
        #######  relax  #######
        self.set_current_pop(init_pop)
        self.cur_pop.save('gen', self.curgen)
        #######  analyze #######
        self.analysis()
        # prepare parent pop BEFORE next generation starts
        self.get_pop_for_heredity()
