import os, shutil, yaml, traceback
import numpy as np
import abc
import ase
import logging
from magus.populations.populations import Population
from magus.formatting.traj import write_traj
from magus.parallel.queuemanage import JobManager
from magus.utils import CALCULATOR_CONNECT_PLUGIN, check_parameters
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
    def __init__(self, **parameters):
        self.all_parameters = parameters
        Requirement = ['work_dir', 'job_prefix']
        Default={'pressure': 0.}
        check_parameters(self, parameters, Requirement, Default)
        self.input_dir = '{}/inputFold/{}'.format(self.work_dir, self.job_prefix)
        self.calc_dir = "{}/calcFold/{}".format(self.work_dir, self.job_prefix)
        os.makedirs(self.calc_dir, exist_ok=True)
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
            source = os.path.join(self.input_dir, filename)
            target = os.path.join(path, filename)
            if not os.path.exists(target):
                if os.path.isdir(source):
                    shutil.copytree(source, target)
                else:
                    shutil.copy(source, target)

    def calc_pre_processing(self, calcPop):
        to_calc = []
        for ind in calcPop:
            convert_op = getattr(ind, 'for_calculate', None)
            if callable(convert_op):
                to_calc.append(ind.for_calculate())
            else:
                to_calc.append(ind)
        return to_calc

    def calc_post_processing(self, calcPop, pop):
        if isinstance(calcPop, Population):
            pop = calcPop.__class__(pop)
        return pop

    def relax(self, calcPop):
        to_relax = self.calc_pre_processing(calcPop)
        pop = self.relax_(to_relax)
        return self.calc_post_processing(calcPop, pop)

    def scf(self, calcPop):
        to_scf = self.calc_pre_processing(calcPop)
        pop = self.scf_(to_scf)
        return self.calc_post_processing(calcPop, pop)

    @abc.abstractmethod
    def relax_(self, calcPop):
        pass

    @abc.abstractmethod
    def scf_(self, calcPop):
        pass


class ClusterCalculator(Calculator, abc.ABC):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        check_parameters(self, parameters, [], {'mode': 'parallel'})
        assert self.mode in ['serial', 'parallel'], "only support 'serial' and 'parallel'"
        self.main_info.append('mode')
        if self.mode == 'parallel':
            Requirement = ['queue_name', 'num_core']
            Default={
                'pre_processing': '',
                'wait_time': 200,
                'verbose': False,
                'kill_time': 100000,
                'num_parallel': 1,
                'memory': '1000M',
                }
            check_parameters(self, parameters, Requirement, Default)

            self.J = JobManager(
                queue_name=self.queue_name,
                num_core=self.num_core,
                pre_processing=self.pre_processing,
                verbose=self.verbose,
                kill_time=self.kill_time,
                memory=self.memory,
                control_file="{}/job_controller".format(self.calc_dir))

    def paralleljob(self, calcPop, runjob):
        job_queues = split1(len(calcPop), self.num_parallel)
        os.chdir(self.calc_dir)
        self.prepare_for_calc()
        for i, job_queue in enumerate(job_queues):
            if len(job_queue) == 0:
                continue
            currdir = str(i).zfill(2)
            if not os.path.exists(currdir):
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
            os.chdir(self.calc_dir)
            scfPop = self.scf_serial(calcPop)
            os.chdir(self.work_dir)
        return scfPop

    def relax_(self, calcPop):
        if self.mode == 'parallel':
            self.paralleljob(calcPop, self.relax_job)
            relaxPop = self.read_parallel_results()
            self.J.clear()
        else:
            os.chdir(self.calc_dir)
            relaxPop = self.relax_serial(calcPop)
            os.chdir(self.work_dir)
        return relaxPop

    def read_parallel_results(self):
        pop = []
        for job in self.J.jobs:
            try:
                a = read("{}/optPop.traj".format(job['workDir']), format='traj', index=':')
                pop.extend(a)
            except:
                log.warning("ERROR in read results {}".format(job['workDir']))
        write("{}/optPop.traj".format(self.calc_dir), pop)
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
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = []
        Default={
            'eps': 0.05,
            'max_step': 100,
            'optimizer': 'bfgs',
            'max_move': 0.1,
            'relax_lattice': True,
            }
        check_parameters(self, parameters, Requirement, Default)
        self.optimizer = self.optimizer_dict[self.optimizer]
        self.main_info.extend(['eps', 'max_step', 'optimizer', 'max_move', 'relax_lattice'])

    def relax_(self, calcPop, logfile='aserelax.log', trajname='calc.traj'):
        log.debug('Using Calculator:\n{}log_path:{}\ntraj_path:{}\n'.format(self, logfile, trajname))
        os.chdir(self.calc_dir)
        new_frames = []
        error_frames = []
        for i, atoms in enumerate(calcPop):
            if isinstance(self.relax_calc, dict):
                # For dftb+ calculator, a 'kpts' dict should be set together with 'atoms'
                atoms.set_calculator(self.ase_calc_type(atoms=atoms,**self.relax_calc))
            else:
                atoms.set_calculator(self.relax_calc)
            if self.relax_lattice:
                ucf = ExpCellFilter(atoms, scalar_pressure=self.pressure * GPa)
            else:
                ucf = atoms
            gopt = self.optimizer(ucf, maxstep=self.max_move, logfile=logfile, trajectory=trajname)
            try:
                label = gopt.run(fmax=self.eps, steps=self.max_step)
                traj = read(trajname, ':')
                log.debug('{} relax steps: {}'.format(self.__class__.__name__, len(traj)))
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
            # atoms.info['enthalpy'] = round(enthalpy, 6)
            atoms.info['enthalpy'] = enthalpy
            atoms.info['trajs'] = traj
            atoms.wrap()
            atoms.set_calculator(None)
            new_frames.append(atoms)
        write('errorTraj.traj', error_frames)
        os.chdir(self.work_dir)
        return new_frames

    def scf_(self, calcPop):
        for atoms in calcPop:
            if isinstance(self.scf_calc, dict):
                # For dftb+ calculator, a 'kpts' dict should be set together with 'atoms'
                atoms.set_calculator(self.ase_calc_type(atoms=atoms,**self.scf_calc))
            else:
                atoms.set_calculator(self.scf_calc)
            try:
                atoms.info['energy'] = atoms.get_potential_energy()
                atoms.info['forces'] = atoms.get_forces()
                try:
                    atoms.info['stress'] = atoms.get_stress()
                except:
                    pass
                enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa) / len(atoms)
                # atoms.info['enthalpy'] = round(enthalpy, 6)
                atoms.info['enthalpy'] = enthalpy
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

# TODO
# AdjointCalculator(ClusterCalculator)?
