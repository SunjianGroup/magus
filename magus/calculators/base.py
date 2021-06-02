import os, shutil, yaml
import numpy as np
import abc
import ase
import logging
from ase.atoms import Atoms
from magus.utils import checkParameters, EmptyClass
from magus.population import Individual
from magus.formatting.traj import write_traj
from magus.queuemanage import JobManager


log = logging.getLogger(__name__)


def split1(Njobs, Npara):
    Neach = int(np.ceil(Njobs / Npara))
    return [[i + j * Npara for j in range(Neach) if i + j * Npara < Njobs] for i in range(Npara)]

def split2(Njobs, Npara):
    Neach = int(np.ceil(Njobs / Npara))
    return [[i * Neach + j for j in range(Neach) if i * Neach + j < Njobs] for i in range(Npara)]

class Calculator(abc.ABC):
    def __init__(self, workDir, jobPrefix, pressure=0., *arg, **kwargs):
        self.work_dir = workDir
        self.pressure = pressure
        self.job_prefix = jobPrefix
        self.input_dir = '{}/inputFold/{}'.format(self.work_dir, self.job_prefix) 
        self.calc_dir = "{}/calcFold/{}".format(self.work_dir, self.job_prefix)
        if os.path.exists(self.calc_dir):
            shutil.rmtree(self.calc_dir)
        #os.makedirs(self.calc_dir)
        # make sure parameter files are copied, such as VASP's vdw kernel file and XTB's parameters
        shutil.copytree(self.input_dir, self.calc_dir)
        self.main_info = ['job_prefix', 'pressure', 'input_dir', 'calc_dir']

    def __str__(self):
        d = {info: getattr(self, info) if hasattr(self, info) else None for info in self.main_info}
        out  = self.__class__.__name__ + ':\n'
        out += yaml.dump(d)
        return out
        
    def cp_input_to(self, path='.'):
        for filename in os.listdir(self.input_dir):
            shutil.copy(os.path.join(self.input_dir, filename), 
                        os.path.join(path, filename))

    def pre_processing(self, calcPop):
        if isinstance(calcPop[0], Individual):
            self.atomstype = 'Individual'
            self.Pop = calcPop
            return calcPop.frames
        elif isinstance(calcPop[0], Atoms):
            self.atomstype = 'Atoms'
            return calcPop

    def post_processing(self, pop):
        if self.atomstype == 'Atoms':
            return pop
        elif self.atomstype == 'Individual':
            return self.Pop(pop)

    @abc.abstractmethod
    def relax(self,calcPop):
        pass
    
    @abc.abstractmethod
    def scf(self,calcPop):
        pass

class ClusterCalculator(Calculator, abc.ABC):
    def __init__(self, workDir, queueName, numCore, numParallel, jobPrefix,
                 pressure=0., Preprocessing='', waitTime=200, verbose=False, 
                 killtime=100000, mode='parallel'):
        super().__init__(workDir=workDir, pressure=pressure, jobPrefix=jobPrefix)
        self.num_parallel = numParallel
        self.wait_time = waitTime
        assert mode in ['serial', 'parallel'], "only support 'serial' and 'parallel'"
        self.mode = mode
        self.main_info.append('mode')
        if self.mode == 'parallel':
            self.J = JobManager(
                queue_name=queueName,
                num_core=numCore, 
                pre_processing=Preprocessing,
                verbose=verbose,
                kill_time=killtime,
                control_file="{}/job_controller".format(self.calc_dir))

    def paralleljob(self, calcPop, runjob):
        job_queues = split1(len(calcPop), self.num_parallel)
        os.chdir(self.calc_dir)
        self.prepare_for_calc()
        for i, job_queue in enumerate(job_queues):
            if len(job_queue) == 0:
                continue
            currdir = str(i).zfill(2)
            if os.path.exists(currdir):
                shutil.rmtree(currdir)
            os.mkdir(currdir)
            os.chdir(currdir)
            write_traj('initPop.traj', [calcPop[j] for j in job_queue])
            runjob(index=i)
            os.chdir(self.calc_dir)
        self.J.wait_jobs_done(self.wait_time)
        os.chdir(self.work_dir)

    def scf(self, calcPop):
        if self.mode == 'parallel':
            self.paralleljob(calcPop, self.scf_job)
            scfPop = self.read_parallel_results()
            self.J.clear()
        else:
            scfPop = self.scf_serial(calcPop)
        return scfPop

    def relax(self, calcPop):
        if self.mode == 'parallel':
            self.paralleljob(calcPop, self.relax_job)
            relaxPop = self.read_parallel_results()
            self.J.clear()
        else:
            relaxPop = self.relax_serial(calcPop)
        return relaxPop

    def read_parallel_results(self):
        pop = []
        for job in self.J.jobs:
            try:
                a = ase.io.read("{}/optPop.traj".format(job['workDir']), 
                                format='traj', index=':')
                pop.extend(a)
            except:
                log.warning("ERROR in read results {}".format(job['workDir']))
        return pop

    def scf_job(self, index):
        raise NotImplementedError

    def relax_job(self, index):
        raise NotImplementedError

    def scf_serial(self, index):
        raise NotImplementedError

    def relax_serial(self, index):
        raise NotImplementedError

    def prepare_for_calc(self):
        pass


class AdjointCalculator(Calculator):
    def __init__(self, calclist):
        self.calclist = calclist
    
    def __str__(self):
        out  = self.__class__.__name__ + ':\n'
        for i, calc in enumerate(self.calclist):
            out += 'Calculator {}: {}'.format(i, calc.__str__())
        return out

    def relax(self, calcPop):
        for calc in self.calclist:
            calcPop = calc.relax(calcPop)
        return calcPop

    def scf(self, calcPop):
        calc = self.calclist[-1]
        calcPop = calc.scf(calcPop)
        return calcPop
