from __future__ import print_function, division
from ase import Atoms
import ase.io
from .readvasp import *
import sys, math, os, shutil, subprocess, logging, copy, yaml, traceback

from ase.calculators.lj import LennardJones
from ase.calculators.vasp import Vasp
from ase.spacegroup import crystal
# from parameters import parameters
from .writeresults import write_yaml, read_yaml, write_traj
from .utils import *
from ase.units import GPa
try:
    from xtb import GFN0_PBC
    from ase.constraints import ExpCellFilter
except:
    pass

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ase.calculators.lj import LennardJones
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from .queue import JobManager
from .runvasp import calc_vasp
from .rungulp import calc_gulp

class Calculator:
    def __init__(self):
        pass
    
    def relax(self,calcPop):
        pass
    
    def scf(self,calcPop):
        pass

class ASECalculator(Calculator):
    def __init__(self,parameters,calc):
        self.parameters=parameters
        self.calc=calc

    def relax(self, calcPop):
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            os.mkdir('calcFold')
        os.chdir('calcFold')

        relaxPop = []
        errorPop = []
        for i, ind in enumerate(calcPop):
            ind.set_calculator(self.calc)
            try:
                ucf = ExpCellFilter(ind, scalar_pressure=self.parameters.pressure*GPa)
            except:
                ucf = UnitCellFilter(ind, scalar_pressure=self.parameters.pressure*GPa)
            if self.parameters.mainoptimizer == 'cg':
                gopt = SciPyFminCG(ucf, logfile='aseOpt.log',)
            elif self.parameters.mainoptimizer == 'BFGS':
                gopt = BFGS(ucf, logfile='aseOpt.log', maxstep=self.parameters.maxRelaxStep)
            elif self.parameters.mainoptimizer == 'fire':
                gopt = FIRE(ucf, logfile='aseOpt.log', maxmove=self.parameters.maxRelaxStep)

            try:
                label=gopt.run(fmax=self.parameters.epsArr, steps=self.parameters.stepArr)
            except Converged:
                pass
            except TimeoutError:
                errorPop.append(ind)
                logging.info("Timeout")
                continue
            except:
                errorPop.append(ind)
                logging.debug("traceback.format_exc():\n{}".format(traceback.format_exc()))
                logging.info("ASE relax fail")
                continue

            if label:
                # save energy, forces, stress for trainning potential
                ind.info['energy'] = ind.get_potential_energy()
                ind.info['forces'] = ind.get_forces()
                ind.info['stress'] = ind.get_stress()
                enthalpy = (ind.info['energy'] + self.parameters.pressure * ind.get_volume() * GPa)/len(ind)
                ind.info['enthalpy'] = round(enthalpy, 3)

                ind.set_calculator(None)
                relaxPop.append(ind)
        os.chdir(self.parameters.workDir)
        return relaxPop

    def scf(self, calcPop):
        scfPop = []
        for ind in calcPop:
            atoms=copy.deepcopy(ind)
            atoms.set_calculator(self.calc)
            try:
                atoms.info['energy'] = atoms.get_potential_energy()
                atoms.info['forces'] = atoms.get_forces()
                atoms.info['stress'] = atoms.get_stress()
                enthalpy = (atoms.info['energy'] + self.parameters.pressure * atoms.get_volume() * GPa)/len(atoms)
                atoms.info['enthalpy'] = round(enthalpy, 3)
                atoms.set_calculator(None)
                scfPop.append(atoms)
            except:
                pass
        return scfPop

class LJCalculator(ASECalculator):
    def __init__(self,parameters):
        calc = LennardJones()
        return super(LJCalculator, self).__init__(parameters,calc)
    
    def relax(self, calcPop):
        return super(LJCalculator, self).relax(calcPop)

    def scf(self, calcPop):
        return super(LJCalculator, self).scf(calcPop)

class xtbCalculator:
    def __init__(self,parameters):
        self.parameters=parameters

    def calc_xtb(self,calcs,structs):
        newStructs = []
        for i, ind in enumerate(structs):
            for j, calc in enumerate(calcs):
                ind.set_calculator(calc)
                logging.info("Structure {} Step {}".format(i, j))
                try:
                    ucf = ExpCellFilter(ind, scalar_pressure=self.parameters.pressure*GPa)
                except:
                    ucf = UnitCellFilter(ind, scalar_pressure=self.parameters.pressure*GPa)
                if self.parameters.xtboptimizer == 'cg':
                    gopt = SciPyFminCG(ucf, logfile='aseOpt.log',)
                elif self.parameters.xtboptimizer == 'BFGS':
                    gopt = BFGS(ucf, logfile='aseOpt.log', maxstep=self.parameters.maxRelaxStep)
                elif self.parameters.xtboptimizer == 'fire':
                    gopt = FIRE(ucf, logfile='aseOpt.log', maxmove=self.parameters.maxRelaxStep)

                try:
                    label=gopt.run(fmax=self.parameters.epsArr[j], steps=self.parameters.stepArr[j])
                except Converged:
                    pass
                except TimeoutError:
                    logging.info("Timeout")
                    break
                except:
                    logging.debug("traceback.format_exc():\n{}".format(traceback.format_exc()))
                    logging.info("XTB fail")
                    break

            else:
                if label:
                    # save energy, forces, stress for trainning potential
                    ind.info['energy'] = ind.get_potential_energy()
                    ind.info['forces'] = ind.get_forces()
                    ind.info['stress'] = ind.get_stress()
                    enthalpy = (ind.info['energy'] + self.parameters.pressure * ind.get_volume() * GPa)/len(ind)
                    ind.info['enthalpy'] = round(enthalpy, 3)

                    logging.info("XTB finish")
                    ind.set_calculator(None)
                    newStructs.append(ind)

        return newStructs

    def relax(self,calcPop):
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            os.mkdir('calcFold')
        os.chdir('calcFold')
        calcs = []
        for i in range(1, self.parameters.calcNum + 1):
            params = yaml.load(open("{}/inputFold/xtb_{}.yaml".format(self.parameters.workDir, i)))
            calc = GFN0_PBC(**params)
            calcs.append(calc)
        relaxPop = self.calc_xtb(calcs, calcPop)
        os.chdir(self.parameters.workDir)
        return relaxPop

    def scf(self,calcPop):   
        params = yaml.load(open("{}/inputFold/xtb_scf.yaml".format(self.parameters.workDir)))
        calc = GFN0_PBC(**params)
        scfPop=[]
        for ind in calcPop:
            atoms=copy.deepcopy(ind)
            atoms.set_calculator(calc)
            try:
                atoms.info['energy'] = atoms.get_potential_energy()
                atoms.info['forces'] = atoms.get_forces()
                atoms.info['stress'] = atoms.get_stress()
                enthalpy = (atoms.info['energy'] + self.parameters.pressure * atoms.get_volume() * GPa)/len(atoms)
                atoms.info['enthalpy'] = round(enthalpy, 3)
                atoms.set_calculator(None)
                scfPop.append(atoms)
            except:
                pass
        return scfPop

class ABinitCalculator(Calculator):
    def __init__(self,parameters,prefix):
        self.parameters=parameters
        if self.parameters.mode == 'serial':
            self.scf = self.scf_serial
            self.relax = self.relax_serial
        elif self.parameters.mode == 'parallel':
            self.J=JobManager()
            self.scf = self.scf_parallel
            self.relax = self.relax_parallel
            self.prefix=prefix
    
    def cdcalcFold(self):
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            os.mkdir('calcFold')
        os.chdir('calcFold')

    def scf_serial(self,calcPop):
        pass
           
    def relax_serial(self,calcPop):
        pass

    def paralleljob(self,calcPop,runjob):
        numParallel = self.parameters.numParallel
        popLen = len(calcPop)
        eachLen = popLen//numParallel
        remainder = popLen%numParallel

        runArray = []
        for i in range(numParallel):
            tmpList = [ i + numParallel*j for j in range(eachLen)]
            if i < remainder:
                tmpList.append(numParallel*eachLen + i)
            runArray.append(tmpList)

        for i in range(numParallel):
            if not os.path.exists("{}{}".format(self.prefix, i)):
                os.mkdir("{}{}".format(self.prefix, i))
            os.chdir("{}{}".format(self.prefix, i))

            tmpPop = [calcPop[j] for j in runArray[i]]
            write_traj('initPop.traj', tmpPop)

            runjob()
            
            os.chdir("%s/calcFold" %(self.parameters.workDir))

        self.J.WaitJobsDone(self.parameters.waitTime)
        os.chdir(self.parameters.workDir)

    def scf_parallel(self,calcPop):
        self.cdcalcFold()
        self.paralleljob(calcPop,self.scfjob)
        scfPop = self.read_parallel_results()
        self.J.clear()
        return scfPop
    
    def scfjob(self):
        pass

    def relax_parallel(self,calcPop):
        self.cdcalcFold()
        self.paralleljob(calcPop,self.relaxjob)
        relaxPop = self.read_parallel_results()
        self.J.clear()
        return relaxPop

    def relaxjob(self):
        pass

    def read_parallel_results(self):
        pop = []
        for job in self.J.jobs:
            try:
                pop.extend(ase.io.read("{}/optPop.traj".format(job['workDir']), format='traj', index=':'))
            except:
                logging.info("ERROR in read results {}".format(job['workDir']))
        return pop

class VaspCalculator(ABinitCalculator):
    def __init__(self,parameters,prefix='calcVasp'):
        super().__init__(parameters,prefix)

    def scf_serial(self,calcPop):
        self.cdcalcFold()
        calc = Vasp()
        calc.read_incar('INCAR_scf')
        calc.set(xc=self.parameters.xc,setups=dict(zip(self.parameters.symbols, self.parameters.ppLabel)),pstress=self.parameters.pressure*10)
        scfPop = calc_vasp([calc], calcPop)
        return scfPop

    def relax_serial(self,calcPop):
        self.cdcalcFold()
        incars = ['INCAR_{}'.format(i) for i in range(1, self.parameters.calcNum+1)]
        calcs = []
        for incar in incars:
            calc = Vasp()
            calc.read_incar(incar)
            calc.set(xc=self.parameters.xc,setups=dict(zip(self.parameters.symbols, self.parameters.ppLabel)),pstress=self.parameters.pressure*10)
            calcs.append(calc)
        relaxPop = calc_vasp(calcs, calcPop)
        return relaxPop

    def scfjob(self):
        shutil.copy("{}/inputFold/INCAR_scf".format(self.parameters.workDir),'INCAR_scf')
        vaspSetup = dict(zip(self.parameters.symbols, self.parameters.ppLabel))
        with open('vaspSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(vaspSetup))

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -W %s\n"
                "#BSUB -J Vasp_%s\n"% (self.parameters.queueName, self.parameters.numCore, self.parameters.maxRelaxTime*len(tmpPop), i))
        f.write("{}\n".format(self.parameters.jobPrefix))
        f.write("python -m newcsp.runvasp 0 {} vaspSetup.yaml {} initPop.traj optPop.traj\n".format(self.parameters.xc, self.parameters.pressure))
        f.close()
        self.J.bsub('bsub < parallel.sh')

    def relaxjob(self):
        vaspSetup = dict(zip(self.parameters.symbols, self.parameters.ppLabel))
        with open('vaspSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(vaspSetup))

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -W %s\n"
                "#BSUB -J Vasp_%s\n"% (self.parameters.queueName, self.parameters.numCore, self.parameters.maxRelaxTime*len(tmpPop), i))
        f.write("{}\n".format(self.parameters.jobPrefix))
        f.write("python -m newcsp.runvasp {} {} vaspSetup.yaml {} initPop.traj optPop.traj\n".format(self.parameters.calcNum, self.parameters.xc, self.parameters.pressure))
        f.close()
        self.J.bsub('bsub < parallel.sh')

class gulpCalculator(ABinitCalculator):
    def __init__(self, parameters,prefix='calcGulp'):
        super().__init__(parameters,prefix)

    def scf_serial(self,calcPop):
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            os.mkdir('calcFold')
        os.chdir('calcFold')

        calcNum = 0
        exeCmd = self.parameters.exeCmd
        pressure = self.parameters.pressure
        inputDir = "{}/inputFold".format(self.parameters.workDir)

        scfPop = calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir)
        write_traj('optPop.traj', scfPop)
        return scfPop

    def relax_serial(self,calcPop):
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            os.mkdir('calcFold')
        os.chdir('calcFold')

        calcNum = self.parameters.calcNum
        exeCmd = self.parameters.exeCmd
        pressure = self.parameters.pressure
        inputDir = "{}/inputFold".format(self.parameters.workDir)

        relaxPop = calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir)
        write_traj('optPop.traj', relaxPop)
        return relaxPop

    def scfjob(self):
        calcDic = {
            'calcNum': 0,
            'pressure': self.parameters.pressure,
            'exeCmd': self.parameters.exeCmd,
            'inputDir': "{}/inputFold".format(self.parameters.workDir),
        }
        with open('gulpSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(calcDic))

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -W %s\n"
                "#BSUB -J Gulp_%s\n"% (self.parameters.queueName, self.parameters.numCore, self.parameters.maxRelaxTime*len(tmpPop), i))
        f.write("{}\n".format(self.parameters.jobPrefix))
        f.write("python -m csp.rungulp gulpSetup.yaml")
        f.close()

        self.J.bsub('bsub < parallel.sh')

    def relaxjob(self):
        tmpPop = [calcPop[j] for j in runArray[i]]
        write_traj('initPop.traj', tmpPop)

        with open('gulpSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(calcDic))

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -W %s\n"
                "#BSUB -J Gulp_%s\n"% (self.parameters.queueName, self.parameters.numCore, self.parameters.maxRelaxTime*len(tmpPop), i))
        f.write("{}\n".format(self.parameters.jobPrefix))
        f.write("python -m csp.rungulp gulpSetup.yaml")
        f.close()

        self.J.bsub('bsub < parallel.sh')

    