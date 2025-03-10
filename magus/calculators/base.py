import os, shutil, yaml, traceback, sys
import numpy as np
import abc
import ase
import logging
from magus.populations.populations import Population
from magus.formatting.traj import write_traj
from magus.parallel.queuemanage import JobManager
from magus.utils import CALCULATOR_CONNECT_PLUGIN, check_parameters
from ase.units import GPa, eV, Ang
from ase.optimize import BFGS, LBFGS, FIRE, GPMin, BFGSLineSearch 
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from ase.io import read, write
try:
    from ase.filters import ExpCellFilter, FrechetCellFilter
except:
    from ase.constraints import ExpCellFilter


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

    def relax(self, calcPop, **kwargs):
        to_relax = self.calc_pre_processing(calcPop)
        pop = self.relax_(to_relax, **kwargs)
        return self.calc_post_processing(calcPop, pop)

    def scf(self, calcPop):
        to_scf = self.calc_pre_processing(calcPop)
        pop = self.scf_(to_scf)
        return self.calc_post_processing(calcPop, pop)

    @abc.abstractmethod
    def relax_(self, calcPop, *args, **kwargs):
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
                'memory': None,
                'mem_per_cpu': '1G',
                # 'memory': '1000M',
                'wait_params': '--mem=10M',
                }
            check_parameters(self, parameters, Requirement, Default)

            self.J = JobManager(
                queue_name=self.queue_name,
                num_core=self.num_core,
                pre_processing=self.pre_processing,
                verbose=self.verbose,
                kill_time=self.kill_time,
                memory=self.memory,
                mem_per_cpu=self.mem_per_cpu,
                wait_params=self.wait_params,
                control_file="{}/job_controller".format(self.calc_dir))
        elif self.mode == 'serial':
            check_parameters(self, parameters, Requirement=[], Default={'num_core': 1})

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

    def relax_(self, calcPop, *args, **kwargs):
        if self.mode == 'parallel':
            self.paralleljob(calcPop, self.relax_job)
            relaxPop = self.read_parallel_results()
            self.J.clear()
        else:
            # make multiple paths if 'serial' mode is used with pa_magus.
            # WARNING: PLEASE DO NOT CHANGE LOGFILE NAME IN "search/search_pa.py"!
            try:
                _dir = "/" + str(int(kwargs['logfile'][8:-4]))
            except:
                _dir = ""
            
            os.makedirs(self.calc_dir + _dir, exist_ok=True)
            os.chdir(self.calc_dir + _dir)
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
try:
    from ase.spacegroup.symmetrize import FixSymmetry
except:
    from ase.constraints import FixSymmetry

class ASECalculator(Calculator):
    optimizer_dict = {
        'bfgs': BFGS,
        'lbfgs': LBFGS,
        'fire': FIRE,
        'gpmin': GPMin,
        'bfgsline': BFGSLineSearch, 
        'scipyfminbfgs': SciPyFminBFGS, 
        'scipyfmincg': SciPyFminCG,
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
            'fix_symmetry': False,
            'max_force': None,
            'fix_volume': False
            }
        check_parameters(self, parameters, Requirement, Default)

        class MagusOptimizer(self.optimizer_dict[self.optimizer]):
            def converged(self, forces=None):
                # # Note: here self.atoms is a Filter not Atoms, so self.atoms.atoms is used.
                # cutoffs = natural_cutoffs(self.atoms.atoms, mult=min_mace_dratio)
                # nlInds = neighbor_list('i', self.atoms.atoms, cutoffs)
                # if len(nlInds) > 0:
                #     #write('dist.vasp', self.optimizable.atom)
                #     raise Exception('Too small distance during relaxation')
                if forces is None:
                    forces = self.optimizable.get_forces()
                if self.max_force != None:
                    if np.abs(forces).max() > self.max_force:
                        raise Exception('Too large forces during relaxation')
                return self.optimizable.converged(forces, self.fmax)
        self.optimizer = MagusOptimizer

        # self.optimizer = self.optimizer_dict[self.optimizer]
        self.main_info.extend(list(Default.keys()))
    
    # set parameters like 'eps', 'max_step' etc. for specific calclators
    def update_parameters(self, parameters):
        for key, val in parameters.items():
            if hasattr(self, key):
                setattr(self, key, val)
        

    def relax_(self, calcPop, logfile='aserelax.log', trajname='calc.traj'):
        #Main calculator information is avail in Magus.init_parms(), no need to print again
        #log.debug('Using Calculator:\n{}log_path:{}\ntraj_path:{}\n'.format(self, logfile, trajname))
        os.chdir(self.calc_dir)
        new_frames = []
        error_frames = []
        for i, atoms in enumerate(calcPop):
            if isinstance(self.relax_calc, dict):
                # For dftb+ calculator, a 'kpts' dict should be set together with 'atoms'
                atoms.set_calculator(self.ase_calc_type(atoms=atoms,**self.relax_calc))
            else:
                atoms.set_calculator(self.relax_calc)
                
            if self.fix_symmetry:
                atoms.constraints += [FixSymmetry(atoms,symprec=0.1)]
            if self.relax_lattice:
                try:
                    # Try to use newest FrechetCellFilter
                    ucf = FrechetCellFilter(atoms, scalar_pressure=self.pressure * GPa, constant_volume=self.fix_volume)
                except:
                    ucf = ExpCellFilter(atoms, scalar_pressure=self.pressure * GPa, constant_volume=self.fix_volume)
            else:
                ucf = atoms
            
            kwargs = {'logfile':logfile, 'trajectory': trajname}
            if not self.optimizer.__name__ == 'SciPyFminCG':
                kwargs['maxstep'] = self.max_move

            gopt = self.optimizer(ucf, **kwargs)
            # SciPyFminCG raises error if maxstep parameter is used

            # set max force for optimizer
            # Naive implmentation. It should be imporved later.
            setattr(gopt, 'max_force', self.max_force)

            try:
                label = gopt.run(fmax=self.eps, steps=self.max_step)
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
            try:
                traj = read(trajname, ':')
                log.debug('{} relax steps: {}'.format(self.__class__.__name__, len(traj)))
            except:
                traj = []
                log.warning("ERROR in reading relaxation traj, traceback.format_exc():\n{}".format(traceback.format_exc()))
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

# ASE calculator supporting parallel mode
class ASEClusterCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.ASECalc = ASECalculator(**parameters)
    
    def scf_(self, calcPop):
        if self.mode == 'parallel':
            self.paralleljob(calcPop, self.scf_job)
            scfPop = self.read_parallel_results()
            self.J.clear()
        else:
            # serial model: use ASECalculator
            os.chdir(self.calc_dir)
            scfPop = self.ASECalc.scf_(calcPop)
            os.chdir(self.work_dir)
        return scfPop

    def relax_(self, calcPop):
        if self.mode == 'parallel':
            self.paralleljob(calcPop, self.relax_job)
            relaxPop = self.read_parallel_results()
            self.J.clear()
        else:
            os.chdir(self.calc_dir)
            relaxPop = self.ASECalc.relax_(calcPop)
            os.chdir(self.work_dir)
        return relaxPop
    



@CALCULATOR_CONNECT_PLUGIN.register('naive')
class AdjointCalculator(Calculator):
    def __init__(self, calclist):
        self.calclist = calclist

    def __repr__(self):
        out  = self.__class__.__name__ + ':\n'
        for i, calc in enumerate(self.calclist):
            out += 'Calculator {}: {}'.format(i + 1, calc.__repr__())
        return out

    def relax_(self, calcPop, *args, **kwargs):
        for calc in self.calclist:
            calcPop = calc.relax(calcPop, *args, **kwargs)
        return calcPop

    def scf_(self, calcPop):
        calc = self.calclist[-1]
        calcPop = calc.scf(calcPop)
        return calcPop

# TODO
# AdjointCalculator(ClusterCalculator)?
