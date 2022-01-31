import logging, os, shutil, subprocess, time, itertools, yaml
import numpy as np
from magus.utils import get_units_numlist, read_seeds, get_unique_symbols, get_gcd_formula
from magus.parallel.queuemanage import JobManager
from magus.phasediagram import PhaseDiagram
from ase.io import read, write
from ase import Atoms


log = logging.getLogger(__name__)


def get_job_gen(job):
    content = 'grep "Generation" {}/log.txt | tail -n 1'.format(job['workDir'])
    return int(subprocess.check_output(content, shell=True).split()[-2])


def get_binary_on_convex(frames):
    candidate_pairs = []
    # only consider unitary and binary structures
    pd = PhaseDiagram([atoms for atoms in frames if len(set(atoms.symbols)) < 3])
    for simplex in pd.simplices:
        for i, j in itertools.combinations(simplex, 2):
            if len(get_unique_symbols([pd.frames[i], pd.frames[j]])) == 3:
                fi, fj = get_gcd_formula(pd.frames[i]), get_gcd_formula(pd.frames[j])
                if (fi, fj) not in candidate_pairs:
                    candidate_pairs.append((fi, fj))
    return candidate_pairs


def get_trinary_on_convex(frames):
    candidate_formula = []
    pd = PhaseDiagram(frames)
    for simplex in pd.simplices:
        for i in simplex:
            if len(get_unique_symbols(pd.frames[i])) == 3:
                f = get_gcd_formula(pd.frames[i])
                if f not in candidate_formula:
                    candidate_formula.append(f)
    return candidate_formula


class CogusJobManager(JobManager):
    pass
    

# TODO
# restart
# now only support trinary search, how to extend to 4 dimension
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
        self.job_manager = CogusJobManager(parameters['CogusJobManager'])
        self.job_gen_dict = {}  # record generation of jobs
        self.all_gen = 0
        self.sub_main_jobs()

    def run(self):
        while self.all_gen <= self.parameters['numGen']:
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
            time.sleep(60)

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
            self.sub_fix_trinary_jobs()
            self.sub_var_binary_jobs()

    def sub_var_binary_jobs(self):
        candidate_pairs = get_binary_on_convex(self.good_pop)
        # number of good structures on the convex line
        n_good = [0] * len(candidate_pairs)
        for i, pair in enumerate(candidate_pairs):
            if set(pair) not in self.running_jobs:
                for atoms in self.good_pop:
                    if len(get_unique_symbols(atoms)) == 3:
                        if get_units_numlist(atoms, [Atoms(pair[0]), Atoms(pair[1])]):
                            n_good[i] += 1
        if max(n_good) > 0:
            pair = candidate_pairs[np.argmax(n_good)]
            # update job list
            self.running_jobs.append(set(pair))
            job_name = "({})x({})y".format(pair[0], pair[1])
            var_binary_dir = "{}/{}".format(self.work_dir, job_name)
            os.mkdir(var_binary_dir)
            os.chdir(var_binary_dir)
            # prepare required files (inputFold, input.yaml, Seeds)
            # inputFold
            shutil.copytree("{}/inputFold".format(self.work_dir), ".")
            # input.yaml
            para = yaml.load("{}/input.yaml", Loader=yaml.FullLoader)
            para['formula'] = [[atoms.symbols.count(s) for s in para['symbols']] 
                                                       for atoms in [Atoms(pair[0]), Atoms(pair[1])]]
            para['checkSeed'] = True  # check Seeds so the undesirable Seeds are excluded
            with open("input.yaml", 'w') as f:
                f.write(yaml.dump(para))
            # Seeds
            os.mkdir("Seeds")
            write("Seeds/seeds_1.traj", self.good_pop)

            # sub job
            self.job_manager.sub('magus search', name=job_name)

    def sub_fix_trinary_jobs(self):
        candidate_formula = get_trinary_on_convex(self.good_pop)
        # number of good structures on the convex line
        for formula in candidate_formula:
            if formula not in self.running_jobs:
                # update job list
                self.running_jobs.append(formula)
                fix_trinary_dir = "{}/{}".format(self.work_dir, formula)
                os.mkdir(fix_trinary_dir)
                os.chdir(fix_trinary_dir)
                # prepare required files (inputFold, input.yaml, Seeds)
                # inputFold
                shutil.copytree("{}/inputFold".format(self.work_dir), ".")
                # input.yaml
                para = yaml.load("{}/input.yaml", Loader=yaml.FullLoader)
                para['formulaType'] = 'fix'
                para['formula'] = [Atoms(formula).symbols.count(s) for s in para['symbols']]
                para['checkSeed'] = True  # check Seeds so the undesirable Seeds are excluded
                with open("input.yaml", 'w') as f:
                    f.write(yaml.dump(para))
                # Seeds
                os.mkdir("Seeds")
                write("Seeds/seeds_1.traj", self.good_pop)

                # sub job
                self.job_manager.sub('magus search', name=formula)

    def sub_main_jobs(self):
        self.running_jobs.append('Main')
        main_dir = "{}/Main".format(self.work_dir)
        os.mkdir(main_dir)
        # prepare required files (inputFold, input.yaml, Seeds)
        shutil.copytree("{}/inputFold".format(self.work_dir), main_dir)
        shutil.copytree("{}/Seeds".format(self.work_dir), main_dir)
        shutil.copyfile("{}/input.yaml".format(self.work_dir), main_dir)
        self.job_manager.sub('magus search', name='Main')