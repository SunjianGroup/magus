import logging, os, shutil, subprocess, time, sys
from magus.utils import read_seeds
from magus.parallel import JobManager
from ase.io import read


log = logging.getLogger(__name__)


def get_job_gen(job):
    content = 'grep "Generation" {}/log.txt | tail -n 1'.format(job['workDir'])
    return int(subprocess.check_output(content, shell=True).split()[-2])


class SubMagus:
    pass

# TODO
# restart
class CoMagus:
    def __init__(self, parameters):
        self.seed_dir = '{}/Seeds'.format(self.parameters['workDir'])
        self.curgen = 1
        if os.path.exists("results"):
            i = 1
            while os.path.exists("results{}".format(i)):
                i += 1
            shutil.move("results", "results{}".format(i))
        os.mkdir("results")
        self.job_manager = parameters.CogusJobManager
        self.job_gen_dict = {}  # record generation of jobs
        self.all_gen = 0

    def read_seeds(self):
        log.info("Reading Seeds ...")
        seed_frames = read_seeds('{}/POSCARS_{}'.format(self.seed_dir, self.curgen))
        seed_pop = self.Population(seed_frames, 'seed', self.curgen)
        return seed_pop

    def run(self):
        while self.all_gen <= self.parameters['numGen']:
            log.info(" Generation {} ".format(self.curgen).center(40, "="))
            for job in self.jobs:
                job_gen = get_job_gen(job)
                # if job state has been updated
                if self.job_gen_dict[job['name']] != job_gen:
                    self.all_gen += job_gen - self.job_gen_dict[job['name']]
                    self.job_gen_dict[job['name']] = job_gen
                    good_frames = read('{}/results/good.traj'.format(job['workDir']), ':')
                    for atoms in good_frames:
                        atoms.info['origin'] = job['name']
                        atoms.info['gen'] = job_gen
                    self.good_pop.extend(good_frames)
                    self.good_pop.del_duplicate()

            self.good_pop.calc_dominators()
            remain_indices = []
            for i, ind in enumerate(self.good_pop):
                if ind.info['ehull'] <= 0.2:
                    remain_indices.append(i)
            self.good_pop = self.good_pop[remain_indices]

            self.send_to_all()        
            self.kill_bad_magus()     # 杀死10代未产生新的好结构的
            self.sub_new_magus()      # 根据组分评分产生新任务
            time.sleep(180)

    def kill_bad_magus(self):
        for job in self.jobs:
            job_gen = get_job_gen(job)
            for ind in self.good_pop:
                if ind.info['origin'] == job['name'] and ind.info['gen'] > job_gen - 10:
                    break
            else:
                self.job_manager.kill(job['id'])                
        
    def sub_new_magus(self):
        if len(self.job_manager.jobs) < self.parameters['maxNMagus']:
            for 
            