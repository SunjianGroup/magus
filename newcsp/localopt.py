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


class VaspCalculator:
    def __init__(self,parameters,prefix='calcVasp'):
        self.parameters=parameters
        self.J=JobManager()
        self.prefix=prefix

    def scf(self,calcPop):
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            os.mkdir('calcFold')
        os.chdir('calcFold')
        numParallel = self.parameters.numParallel
        vaspSetup = dict(zip(self.parameters.symbols, self.parameters.ppLabel))
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
            shutil.copy("{}/inputFold/INCAR_scf".format(self.parameters.workDir),'INCAR_scf')

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
            os.chdir("%s/calcFold" %(self.parameters.workDir))

        self.J.WaitJobsDone(self.parameters.waitTime)
        os.chdir(self.parameters.workDir)
        scfPop=self.read_parallel_results()
        self.J.clear()
        return scfPop

    def relax(self,calcPop):
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            os.mkdir('calcFold')
        os.chdir('calcFold')
        numParallel = self.parameters.numParallel
        vaspSetup = dict(zip(self.parameters.symbols, self.parameters.ppLabel))
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
            for j in range(1, self.parameters.calcNum + 1):
                shutil.copy("{}/inputFold/INCAR_{}".format(self.parameters.workDir, j), 'INCAR_{}'.format(j))

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
            os.chdir("%s/calcFold" %(self.parameters.workDir))

        self.J.WaitJobsDone(self.parameters.waitTime)
        os.chdir(self.parameters.workDir)
        relaxPop=self.read_parallel_results()
        self.J.clear()
        return relaxPop
    
    def read_parallel_results(self):
        optPop = []
        for job in self.J.jobs:
            try:
                optPop.extend(ase.io.read("{}/optPop.traj".format(job['workDir']), format='traj', index=':'))
            except:
                logging.info("ERROR in read results {}".format(job['workDir']))
        #bjobs.clear()
        return optPop
    