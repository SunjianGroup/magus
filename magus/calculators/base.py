import os, shutil, yaml
import numpy as np
import abc
import ase
import logging
from ase.atoms import Atoms
from magus.populations.populations import Population
from magus.formatting.traj import write_traj
from magus.queuemanage import JobManager
from magus.utils import CALCULATOR_CONNECT_PLUGIN
from ase.constraints import ExpCellFilter
from ase.units import GPa, eV, Ang
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from ase.io import read, write


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
        self.main_info = ['job_prefix', 'pressure', 'input_dir', 'calc_dir']  # main information to print

    def __repr__(self):
        ret = self.__class__.__name__
        ret += "\n-------------------"
        for info in self.main_info:
            if hasattr(self, info):
                value = getattr(self, info)
                if isinstance(value, dict):
                    value = yaml.dump(value).rstrip('\n').replace('\n', '\n'.ljust(18))
                ret += "\n{}: {}".format(info.ljust(15, ' '), value)
        ret += "\n-------------------\n"
        return ret

    def cp_input_to(self, path='.'):
        for filename in os.listdir(self.input_dir):
            shutil.copy(os.path.join(self.input_dir, filename), 
                        os.path.join(path, filename))

    def pre_processing(self, calcPop):
        pass

    def post_processing(self, calcPop, pop):
        if isinstance(calcPop, Population):
            pop = calcPop.__class__(pop)
        return pop

    def relax(self, calcPop):
        self.pre_processing(calcPop)
        pop = self.relax_(calcPop)
        return self.post_processing(pop)

    def scf(self, calcPop):
        self.pre_processing(calcPop)
        pop = self.scf_(calcPop)
        return self.post_processing(calcPop, pop)

    @abc.abstractmethod
    def relax_(self, calcPop):
        pass

    @abc.abstractmethod
    def scf_(self, calcPop):
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
        self.main_info.append(mode)
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

    def scf_(self, calcPop):
        if self.mode == 'parallel':
            self.paralleljob(calcPop, self.scf_job)
            scfPop = self.read_parallel_results()
            self.J.clear()
        else:
            scfPop = self.scf_serial(calcPop)
        return scfPop

    def relax_(self, calcPop):
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


class ASECalculator(Calculator):
    optimizer_dict = {
        'bfgs': BFGS, 
        'lbfgs': LBFGS,
        'fire': FIRE,
    }
    def __init__(self, workDir, jobPrefix='ASE', pressure=0., eps=0.05, maxStep=100,
                 optimizer='bfgs', maxMove=0.1, relaxLattice=True, *arg, **kwargs):
        super().__init__(workDir=workDir, pressure=pressure, jobPrefix=jobPrefix)
        self.eps = eps
        self.max_step = maxStep
        self.max_move = maxMove
        self.relax_lattice = relaxLattice
        self.optimizer = self.optimizer_dict[optimizer]
        self.set_calc()

    def set_calc(self):
        raise NotImplementedError

    def relax_(self, calcPop, logfile='aserelax.log', trajname='calc.traj'):
        os.chdir(self.calc_dir)
        new_frames = []
        error_frames = []
        for i, atoms in enumerate(calcPop):
            atoms.set_calculator(self.relax_calc)
            if self.relax_lattice:
                ucf = ExpCellFilter(atoms, scalar_pressure=self.pressure * GPa)
            else:
                ucf = atoms
            gopt = self.optimizer(ucf, maxstep=self.max_move, logfile=logfile, trajectory=trajname)
            try:
                label = gopt.run(fmax=self.eps, steps=self.max_step)
                traj = read(trajname, ':')
            except Converged:
                pass
            except TimeoutError:
                error_frames.append(atoms)
                log.warning("Calculator:{} relax Timeout".format(self.__class__.__name__))
                continue
            except:
                error_frames.append(atoms)
                log.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
                log.warning("Calculator:{} relax fail".format(self.__class__.__name__))
                continue
            atoms.info['energy'] = atoms.get_potential_energy()
            atoms.info['forces'] = atoms.get_forces()
            try:
                atoms.info['stress'] = atoms.get_stress()
            except:
                pass
            enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa)/ len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 3)
            atoms.wrap()
            atoms.set_calculator(None)
            new_frames.append(atoms)
        write('errorTraj.traj', error_frames)
        os.chdir(self.work_dir)
        return new_frames

    def scf_(self, calcPop):
        for atoms in calcPop:
            atoms.set_calculator(self.scf_calc)
            try:
                atoms.info['energy'] = atoms.get_potential_energy()
                atoms.info['forces'] = atoms.get_forces()
                try:
                    atoms.info['stress'] = atoms.get_stress()
                except:
                    pass
                enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa) / len(atoms)
                atoms.info['enthalpy'] = round(enthalpy, 3)
                atoms.set_calculator(None)
            except:
                log.debug('{} scf Error'.format(self.__class__.__name__))
        return calcPop


@CALCULATOR_CONNECT_PLUGIN.register('naive')
class AdjointCalculator(Calculator):
    def __init__(self, calclist):
        self.calclist = calclist
    
    def __repr__(self):
        out  = self.__class__.__name__ + ':\n'
        for i, calc in enumerate(self.calclist):
            out += 'Calculator {}: {}'.format(i + 1, calc.__repr__())
        return out

    def relax_(self, calcPop):
        for calc in self.calclist:
            calcPop = calc.relax(calcPop)
        return calcPop

    def scf_(self, calcPop):
        calc = self.calclist[-1]
        calcPop = calc.scf(calcPop)
        return calcPop
