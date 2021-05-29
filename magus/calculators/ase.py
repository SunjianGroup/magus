from magus.calculators.base import Calculator, split1
from ase.constraints import ExpCellFilter
import logging, traceback, os, yaml
import numpy as np
from ase.units import GPa, eV, Ang
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from ase.io import read, write
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
import multiprocessing as mp
from ase.calculators.calculator import FileIOCalculator

try:
    from xtb.ase.calculator import XTB
except:
    pass
try:
    from quippy.potential import Potential as QUIP
except:
    pass


log = logging.getLogger(__name__)


class ASECalculator(Calculator):
    optimizer_dict = {
        'bfgs': BFGS, 
        'lbfgs': LBFGS,
        'fire': FIRE,
    }
    def __init__(self, workDir, jobPrefix='ASE', pressure=0., eps=0.05, maxStep=100,
                 optimizer='bfgs', maxMove=0.1, relaxLattice=True, mode = 'serial', kill_time =1000000, *arg, **kwargs):
        super().__init__(workDir=workDir, pressure=pressure, jobPrefix=jobPrefix)
        self.eps = eps
        self.max_step = maxStep
        self.max_move = maxMove
        self.relax_lattice = relaxLattice
        self.optimizer = self.optimizer_dict[optimizer]
        self.mode = mode
        self.kill_time = kill_time
        #self.set_calc()

    def set_relax_calc(self):
        raise NotImplementedError
    def set_scf_calc(self):
        raise NotImplementedError

    def calculate(self, calcPop, runjob):
        calPop = None
        if self.mode == 'parallel':
            calPop = self.paralleljob(calcPop, runjob)
        else:
            calPop = runjob(calcPop, workdir=self.calc_dir)
        os.chdir(self.work_dir)
        return calPop
        
    def paralleljob(self, calcPop, runjob):
        numParallel = mp.cpu_count()
        popLen = len(calcPop)
        eachLen = popLen//numParallel
        remainder = popLen%numParallel

        runArray = []
        for i in range(numParallel):
            tmpList = [ i + numParallel*j for j in range(eachLen)]
            if i < remainder:
                tmpList.append(numParallel*eachLen + i)
            if len(tmpList) > 0:
                runArray.append(tmpList)

        _numparallel = np.min((numParallel, len(runArray)))

        pool = mp.Pool(_numparallel)
        results = []
        c = self.relax_calc if hasattr(self, 'relax_calc') else self.scf_calc
        if isinstance(c, FileIOCalculator):
        #for FileIOCalculator in ASE, calculate jobs use command "command in.in > out.out".
        #So if we don't separate dictionary, all parallel processes just write into the same in.in file, which will leads to an error.
            dirname = self.calc_dir + self.__class__.__name__[:-10]
            for i in range(_numparallel):
                if not os.path.exists(dirname+str(i)):
                    os.makedir(dirname+str(i))

            results = [pool.apply_async(runjob, args=([calcPop[j] for j in runArray[i]], 'ase{}.log'.format(i), 'calc{}.traj'.format(i), "{}{}".format(dirname,i))) \
                for i in range(_numparallel)]

        else:
            results = [pool.apply_async(runjob, args=([calcPop[j] for j in runArray[i]], 'ase{}.log'.format(i), 'calc{}.traj'.format(i), "{}".format(self.calc_dir))) \
                for i in range(_numparallel)]
        
        resultPop = []
        try:
            resultPop = [ind for p in results for ind in p.get(timeout=self.kill_time)]
        except mp.TimeoutError:
            log.warning("Exception timeout: calculator {} {} terminated after {} seconds".format(self.__class__.__name__, runjob.__name__, self.kill_time))
        return resultPop
    
    def relax(self, calcPop):
        self.set_relax_calc()
        return self.calculate(calcPop, self.relax_serial)
    def scf(self, calcPop):
        self.set_scf_calc()
        return self.calculate(calcPop, self.scf_serial)

    def relax_serial(self, calcPop, logfile='aserelax.log', trajname='calc.traj', workdir = '.'):
        os.chdir(workdir)
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
        write('errorTraj.traj', error_frames, append = True)

        return new_frames

    #TODO: cannot add *args here, or calcPop will be divided into list of args. How to slove it?
    #def scf_serial(self, calcPop, *args):
    def scf_serial(self, calcPop, logfile=None, trajname=None, workdir = '.'):
        os.chdir(workdir)
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


class EMTCalculator(ASECalculator):
    def set_relax_calc(self):
        self.relax_calc = EMT()
    def set_scf_calc(self):
        self.scf_calc = EMT()


class LJCalculator(ASECalculator):
    def set_relax_calc(self):
        with open("{}/lj_relax.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.relax_calc = LennardJones(**params)

    def set_scf_calc(self):   
        with open("{}/lj_scf.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.scf_calc = LennardJones(**params)


## TODO zhe ge zen me she zhi de lai zhe? xu yao fen kai lai ma?
class QUIPCalculator(ASECalculator):
    def set_relax_calc(self):
        with open("{}/quip_relax.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.relax_calc = QUIP(**params)
    def set_scf_calc(self):
        with open("{}/quip_scf.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.scf_calc = QUIP(**params)

class XTBCalculator(ASECalculator):
    def set_relax_calc(self):
        with open("{}/xtb_relax.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.relax_calc = XTB(**params)
    def set_scf_calc(self):
        with open("{}/xtb_scf.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.scf_calc = XTB(**params)
