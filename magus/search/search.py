import logging, os, shutil, subprocess
from magus.utils import read_seeds
from ase.io import read
# from ase.db import connect


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

    def read_seeds(self):
        log.info("Reading Seeds ...")
        seed_frames = read_seeds('{}/POSCARS_{}'.format(self.seed_dir, self.curgen))
        seed_frames.extend(read_seeds('{}/seeds_{}.traj'.format(self.seed_dir, self.curgen)))
        seed_pop = self.Population(seed_frames, 'seed', self.curgen)
        return seed_pop

    def get_init_pop(self):
        # mutate and crossover, empty for first generation
        if self.curgen == 1:
            random_frames = self.atoms_generator.generate_pop(self.parameters['initSize'])
            init_pop = self.Population(random_frames, 'init', self.curgen)
        else:
            init_pop = self.pop_generator.get_next_pop(self.cur_pop + self.keep_pop)
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

    def set_good_pop(self):
        log.info('construct goodPop')
        #good_pop = self.cur_pop + self.good_pop + self.keep_pop
        good_pop = self.good_pop + self.keep_pop
        for i, ind in enumerate(self.cur_pop):
            for ind1 in good_pop:
                if ind == ind1:
                    self.cur_pop[i] = ind1
            else:
                good_pop.append(ind)
        good_pop.del_duplicate()
        good_pop.calc_dominators()
        good_pop.select(self.parameters['popSize'])
        log.debug("good ind:")
        for ind in good_pop:
            log.debug("{strFrml} enthalpy: {enthalpy}, fit: {fitness}, dominators: {dominators}, id: {identity}"\
                .format(strFrml=ind.get_chemical_formula(), **ind.info))
        self.good_pop = good_pop

    def set_keep_pop(self):
        log.info('construct keepPop')
        _, keep_frames = self.good_pop.clustering(self.parameters['saveGood'])
        keep_pop = self.Population(keep_frames, 'keep', self.curgen)
        log.debug("keep ind:")
        for ind in keep_pop:
            log.debug("{strFrml} enthalpy: {enthalpy}, fit: {fitness}, dominators: {dominators}, id: {identity}"\
                .format(strFrml=ind.get_chemical_formula(), **ind.info))
        self.keep_pop = keep_pop

    def update_best_pop(self):
        log.info("best ind:")
        bestind = self.good_pop.bestind()
        self.best_pop.extend(bestind)
        for ind in bestind:
            log.info("{strFrml} enthalpy: {enthalpy}, fit: {fitness}"\
                .format(strFrml=ind.get_chemical_formula(), **ind.info))

    def run(self):
        while self.curgen <= self.parameters['numGen']:
            log.info(" Generation {} ".format(self.curgen).center(40, "="))
            self.one_step()
            self.curgen += 1

    def update_volume_ratio(self):
        if self.curgen > 1:
            log.debug(self.cur_pop)
            new_volume_ratio = 0.7 * self.cur_pop.volume_ratio + 0.3 * self.atoms_generator.volume_ratio
            self.atoms_generator.set_volume_ratio(new_volume_ratio)

    def one_step(self):
        self.update_volume_ratio()
        init_pop = self.get_init_pop()
        init_pop.save('init', self.curgen)
        #######  relax  #######
        relax_pop = self.main_calculator.relax(init_pop)
        try:
            relax_step = sum([sum(atoms.info['relax_step']) for atoms in relax_pop])
            log.info('DFT relax {} structures with {} scf'.format(len(relax_pop), relax_step))
        except:
            pass
        # save raw date before checking
        relax_pop.save('raw', self.curgen)
        relax_pop.check()
        # find spg before delete duplicate
        log.debug("find spg...")
        relax_pop.find_spg()
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
