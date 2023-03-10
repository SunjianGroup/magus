import logging
from magus.utils import check_parameters
from magus.populations.populations import Population
from .search import Magus


log = logging.getLogger(__name__)


class MLMagus(Magus):
    def __init__(self, parameters, restart=False):
        super().__init__(parameters, restart=restart)
        self.get_initial_pot(epoch=self.init_times)

    def init_parms(self, parameters):
        super().init_parms(parameters)
        check_parameters(self, self.parameters, [], {'init_times': 2})
        self.ml_calculator = parameters.MLCalculator
        log.debug('ML Calculator information:\n{}'.format(self.ml_calculator))

    def get_initial_pot(self, epoch=1):
        if epoch == 0:
            log.warning("skip initial train, please make sure you have an trained "
                        "potential, otherwise you should make 'init_times' > 0")
            return
        log.info('try to get the initial potential, will repeat {} times'.format(epoch))
        for i in range(epoch):
            log.info('\tepoch {}'.format(i + 1))
            # get random populations
            random_frames = self.atoms_generator.generate_pop(self.parameters['poolSize'])
            random_pop = self.Population(random_frames, 'datapool')
            random_pop.check()
            log.info("\tRandom generate population with {} strutures\n"
                     "\tSelecting...".format(len(random_pop)))
            # select to add
            select_pop = self.ml_calculator.select(random_pop)
            log.info("\tDone! {} are selected\n\tscf...".format(len(select_pop)))
            scf_pop = self.main_calculator.scf(select_pop)
            scf_pop.check()
            self.ml_calculator.updatedataset(scf_pop)
            log.info("\tDone! {} structures in the dataset\n\ttraining...".format(len(self.ml_calculator.trainset)))
            self.ml_calculator.train()
        log.info('Done!')

    def select_to_relax(self, frames, init_num=3, min_num=20):
        try:
            ground_enthalpy = self.good_pop.bestind()[0].atoms.info['enthalpy']
        except:
            ground_enthalpy = min([atoms.info['enthalpy'] for atoms in frames])
        min_num = min(len(frames), min_num)
        trainset = self.ml_calculator.trainset
        energy_mse = self.ml_calculator.get_loss(trainset)[0]
        select_enthalpy = max(ground_enthalpy + init_num * energy_mse, 
                              sorted([atoms.info['enthalpy'] for atoms in frames])[min_num - 1])
        log.info('select good structures to relax\n'
                 '\tground enthalpy: {}\tenergy mse: {}\tselect enthaly: {}'
                 ''.format(ground_enthalpy, energy_mse, select_enthalpy))
        to_relax = [atoms for atoms in frames if atoms.info['enthalpy'] <= select_enthalpy]
        if isinstance(frames, Population):
            to_relax = frames.__class__(frames)
        return to_relax            

    def select_to_add(self, frames):
        trainset = self.ml_calculator.trainset
        energy_mae = self.ml_calculator.get_loss(trainset)[0]
        frames_ = self.ml_calculator.calc_efs(frames)
        to_add = []
        log.debug('compare begin...\ntarget\tpredict\n')
        for i, (atoms, atoms_) in enumerate(zip(frames, frames_)):
            target_energy = atoms.info['energy'] / len(atoms)
            predict_energy = atoms_.info['energy'] / len(atoms_)
            log.debug("{:.5f}\t{:.5f}\n".format(target_energy, predict_energy))
            error_per_atom = target_energy - predict_energy
            if abs(error_per_atom) > energy_mae:
                to_add.append(atoms)
        return to_add

    def one_step(self):
        self.update_volume_ratio()
        init_pop = self.get_init_pop()
        init_pop.save()
        #######  local relax by ML  #######
        relax_frames = self.ml_calculator.relax(init_pop)
        relax_pop = self.Population(relax_frames, 'relax', self.curgen)
        relax_pop.save("mlraw", self.curgen)
        relax_pop.check()
        # find spg before delete duplicate
        relax_pop.find_spg()
        relax_pop.del_duplicate()
        relax_pop.calc_dominators()
        relax_pop.save("mlgen", self.curgen)
        if self.parameters['DFTScf']:
            to_scf = self.select_to_relax(relax_pop, self.parameters['initNum'], self.parameters['minNum'])
            dft_scf_pop = self.main_calculator.scf(to_scf)
            log.info('{} structures need DFT scf'.format(len(dft_scf_pop)))
            dft_scf_pop.find_spg()
            dft_scf_pop.del_duplicate()
            self.cur_pop = dft_scf_pop
            to_add = self.select_to_add(dft_scf_pop)
            self.ml_calculator.updatedataset(to_add)
            self.ml_calculator.train()
        elif self.parameters['DFTRelax']:
            #######  select cfgs to do dft relax  #######
            to_relax = self.select_to_relax(relax_pop)
            #######  compare target and predict energy  #######   
            dft_relaxed_pop = self.main_calculator.relax(to_relax)
            try:
                relax_step = sum([atoms.info['relax_step'][-1] for atoms in dft_relaxed_pop])
                log.info('DFT relax {} structures with {} scf'.format(len(dft_relaxed_pop), relax_step))
            except:
                pass
            dft_relaxed_pop.find_spg()
            dft_relaxed_pop.del_duplicate()
            self.cur_pop = dft_relaxed_pop
            to_add = self.select_to_add(dft_relaxed_pop)
            self.ml_calculator.updatedataset(to_add)
            self.ml_calculator.train()
        else:
            self.cur_pop = relax_pop
        self.cur_pop.save('gen', self.curgen)
        self.set_good_pop()
        self.good_pop.save('good', '')
        self.good_pop.save('good', self.curgen)
        self.set_keep_pop()
        self.keep_pop.save('keep', self.curgen)
        self.update_best_pop()
        self.best_pop.save('best', '')
