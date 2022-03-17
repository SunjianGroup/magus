import logging, os, shutil, subprocess, time
from magus.utils import read_seeds
from magus.parallel import JobManager
from ase.io import read


log = logging.getLogger(__name__)


class CogusJobManager(JobManager):
    def check_gen(self):
        pass


# TODO
# restart
class Cogus:
    def __init__(self, parameters, restart=False):
        self.init_parms(parameters)
        self.seed_dir = '{}/Seeds'.format(self.parameters['workDir'])
        self.curgen = 1
        if os.path.exists("results"):
            i = 1
            while os.path.exists("results{}".format(i)):
                i += 1
            shutil.move("results", "results{}".format(i))
        os.mkdir("results")
        self.job_manager = parameters.CogusJobManager

    def read_seeds(self):
        log.info("Reading Seeds ...")
        seed_frames = read_seeds('{}/POSCARS_{}'.format(self.seed_dir, self.curgen))
        seed_pop = self.Population(seed_frames, 'seed', self.curgen)
        return seed_pop

    def run(self):
        while self.curgen <= self.parameters['numGen']:
            log.info(" Generation {} ".format(self.curgen).center(40, "="))
            
            self.share_good_individual()
            self.check_magus()
            time.sleep(180)

    def one_step(self):
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
        relax_pop.save('raw')
        relax_pop.check()
        # find spg before delete duplicate
        relax_pop.find_spg()
        relax_pop.del_duplicate()
        relax_pop.save('gen', self.curgen)
        self.cur_pop = relax_pop
        self.set_good_pop()
        self.good_pop.save('good', '')
        self.good_pop.save('good', self.curgen)
        self.set_keep_pop()
        self.keep_pop.save('keep', self.curgen)
        self.update_best_pop()
        self.best_pop.save('best', '')
