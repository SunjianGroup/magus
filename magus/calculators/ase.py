from magus.calculators.base import Calculator
from ase.constraints import ExpCellFilter
import logging, traceback, os, yaml
import numpy as np
from ase.units import GPa, eV, Ang
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from ase.io import read, write
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
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

    def relax(self, calcPop, logfile='aserelax.log', trajname='calc.traj'):
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

    def scf(self, calcPop):
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
    def set_calc(self):
        self.relax_calc = EMT()
        self.scf_calc = EMT()


class LJCalculator(ASECalculator):
    def set_calc(self):
        self.relax_calc = LennardJones()
        self.scf_calc = LennardJones()


## TODO zhe ge zen me she zhi de lai zhe? xu yao fen kai lai ma?
class QUIPCalculator(ASECalculator):
    def set_calc(self):
        with open("{}/quip_relax.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.relax_calc = QUIP(**params)
        with open("{}/quip_scf.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.scf_scf = QUIP(**params)

class XTBCalculator(ASECalculator):
    def set_calc(self):
        with open("{}/xtb.yaml".format(self.input_dir)) as f:
            params = yaml.load(f)
            self.relax_calc = XTB(**params)
            self.scf_calc = XTB(**params)
#        with open("{}/xtb_relax.yaml".format(self.input_dir)) as f:
#            params = yaml.load(f)
#            self.relax_calc = XTB(**params)
#        with open("{}/xtb_scf.yaml".format(self.input_dir)) as f:
#            params = yaml.load(f)
#            self.scf_calc = XTB(**params)
