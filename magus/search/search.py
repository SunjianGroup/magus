import logging, os, shutil, subprocess
from magus.initstruct import read_seeds
from ase.io import read

"""
Pop:class, poplulation
pop:list, a list of atoms
Population: Population(pop) --> Pop
"""
#TODO 
# use same log among different functions
# change read parameters
# early converged


log = logging.getLogger(__name__)


class Magus:
    def __init__(self, parameters, atoms_generator, pop_generator,
                 main_calculator, Population, restart=False):
        self.parameters = parameters
        self.atoms_generator = atoms_generator
        self.pop_generator = pop_generator
        self.main_calculator = main_calculator
        log.debug('Main Calculator information:')
        log.debug(main_calculator.__str__())
        self.Population = Population
        self.seed_dir = '{}/Seeds'.format(self.parameters.workDir)
        if restart:
            if not os.path.exists("results") or not os.path.exists("log.txt"):
                raise Exception("cannot restart without results or log.txt")
            content = 'grep "Generation" log.txt | tail -n 1'
            self.curgen = int(subprocess.check_output(content, shell=True).split()[-2])
            try:
                content = 'grep "volRatio" log.txt | tail -n 1' 
                volume_ratio = float(subprocess.check_output(content, shell=True).split()[-1])
                self.atoms_generator.update_volume_ratio(volume_ratio)
            except:
                pass
            best_pop = read('results/best.traj', ':')
            good_pop = read('results/good.traj', ':')
            keep_pop = read('results/keep{}.traj'.format(self.curgen - 1), ':')
            cur_pop = read('results/gen{}.traj'.format(self.curgen - 1), ':')
            self.bestPop = self.Population(best_pop, 'bestPop')
            self.goodPop = self.Population(good_pop, 'goodPop')
            self.keepPop = self.Population(keep_pop, 'keepPop', self.curgen - 1)
            self.curPop = self.Population(cur_pop, 'curPop', self.curgen - 1)
            log.warning("RESTART HERE!".center(40, "="))
        else:
            self.curgen = 1
            if os.path.exists("results"):
                i = 1
                while os.path.exists("results{}".format(i)):
                    i += 1
                shutil.move("results", "results{}".format(i))
            os.mkdir("results")
            self.bestPop = self.Population([], 'bestPop')
            self.goodPop = self.Population([], 'goodPop')
            self.keepPop = self.Population([], 'keepPop')

    def read_seeds(self):
        log.info("Reading Seeds ...")
        #seedpop = read_seeds('{}/POSCARS_{}'.format(self.seed_dir, self.curgen))
        seedpop = read_seeds('{}/seed.traj'.format(self.seed_dir))
        seedPop = self.Population(seedpop, 'seedpop', self.curgen)
        if self.parameters.chkSeed:
            seedPop.check()
        return seedPop

    def get_initPop(self):

        def getseed():
            seedPop = self.read_seeds()
            seedPop.removebulk_relaxable_vacuum()
            for i, ind in enumerate(seedPop):
                ind.repair_atoms()
            seedPop.addbulk_relaxable_vacuum()
            return seedPop

        # mutate and crossover, empty for first generation
        if self.curgen == 1:
            initPop = self.Population([], 'initpop', self.curgen)
            n_random = self.parameters.initSize
            ## read seeds
            seedPop = self.read_seeds()
            #seedPop = getseed()
            initPop.extend(seedPop)
        else:
            initPop = self.pop_generator.next_Pop(self.curPop + self.keepPop)
            n_random = self.parameters.popSize - len(initPop)
        # random
        
        if n_random > 0:
            addpop = self.atoms_generator.Generate_pop(n_random, initpop=self.curgen==1)
            log.info("random generate population with {} strutures".format(len(addpop)))
            initPop.extend(addpop)
        """
        if n_random > 0:
            for _ in range(int(n_random/len(read_seeds('{}/seed.traj'.format(self.seed_dir))))):
                addpop = getseed()
                initPop.extend(addpop)
        """
        
        # check and log
        initPop.check()
        log.info("Generate new initial population with {} individuals:".format(len(initPop)))
        origins = [atoms.info['origin'] for atoms in initPop.frames]
        for origin in set(origins):
            log.info("  {}: {}".format(origin, origins.count(origin)))
        # del dulplicate?
        return initPop

    def set_goodPop(self):
        log.info('construct goodPop')
        goodPop = self.curPop + self.goodPop + self.keepPop
        goodPop.del_duplicate()
        goodPop.calc_dominators()
        goodPop.select(self.parameters.popSize, delete_highE = True, high = 0.8)
        log.debug("good ind:")
        for ind in goodPop.pop:
            log.debug("{strFrml} enthalpy: {enthalpy}, fit: {fitness}, dominators: {dominators}"\
                .format(strFrml=ind.atoms.get_chemical_formula(), **ind.info))
        self.goodPop = goodPop

    def set_keepPop(self):
        log.info('construct keepPop')
        _, keeppop = self.goodPop.clustering(self.parameters.saveGood)
        keepPop = self.Population(keeppop, 'keeppop', self.curgen)
        log.debug("keep ind:")
        for ind in keepPop.pop:
            log.debug("{strFrml} enthalpy: {enthalpy}, fit: {fitness}, dominators: {dominators}"\
                .format(strFrml=ind.atoms.get_chemical_formula(), **ind.info))
        self.keepPop = keepPop

    def update_bestPop(self):
        log.info("best ind:")
        bestind = self.goodPop.bestind()
        self.bestPop.extend(bestind)
        for ind in bestind:
            log.info("{strFrml} enthalpy: {enthalpy}, fit: {fitness}"\
                .format(strFrml=ind.atoms.get_chemical_formula(), **ind.info))

    def run(self):
        while self.curgen <= self.parameters.numGen:
            log.info(" Generation {} ".format(self.curgen).center(40, "="))
            self.one_step()
            self.curgen += 1

    def set_volume_ratio(self):
        if self.curgen > 1:
            volume_ratio = self.curPop.get_volRatio()
            new_volume_ratio = 0.7 * volume_ratio + 0.3 * self.atoms_generator.p.volRatio
            self.atoms_generator.update_volume_ratio(new_volume_ratio)

    def one_step(self):
        self.set_volume_ratio()
        initPop = self.get_initPop()
        initPop.save('init', self.curgen)
        #######  relax  #######
        relaxpop = self.main_calculator.relax(initPop.frames)
        #relax_step = sum([sum(atoms.info['relax_step']) for atoms in relaxpop])
        #log.info('DFT relax {} structures with {} scf'.format(len(relaxpop), relax_step))
        relaxPop = self.Population(relaxpop, 'relaxpop', self.curgen)
        # save raw date before checking
        relaxPop.save('raw')
        relaxPop.check(delP1 = True)
        # find spg before delete duplicate
        relaxPop.find_spg()
        relaxPop.del_duplicate()
        relaxPop.save('gen', self.curgen)
        self.curPop = relaxPop
        self.set_goodPop()
        self.goodPop.save('good', '')
        self.goodPop.save('good', self.curgen)
        self.set_keepPop()
        self.keepPop.save('keep', self.curgen)
        self.update_bestPop()
        self.bestPop.save('best', '')
