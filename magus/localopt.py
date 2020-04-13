from __future__ import print_function, division
from ase import Atoms
import ase.io
from .readvasp import *
from .tolammps import Atomic
from .readlmps import read_lammps_dump
import sys, math, os, shutil, subprocess, logging, copy, yaml, traceback

from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp
from ase.calculators.gulp import GULP, Conditions
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lj import LennardJones
from ase.spacegroup import crystal
# from parameters import parameters
from .writeresults import write_traj
from .utils import *
from ase.units import GPa, eV, Ang
try:
    from xtb import GFN0, GFN1
    from ase.constraints import ExpCellFilter
    from quippy.potential import Potential as QUIP
except:
    pass

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from .queue import JobManager
# from .runvasp import calc_vasp
# from .rungulp import calc_gulp

__all__ = ['VaspCalculator','XTBCalculator','LJCalculator',
    'EMTCalculator','GULPCalculator','LammpsCalculator','QUIPCalculator','ASECalculator']
class RelaxVasp(Vasp):
    """
    Slightly modify ASE's Vasp Calculator so that it will never check relaxation convergence.
    """
    def read_relaxed(self):
        return True

class Calculator:
    def __init__(self,parameters):
        if not hasattr(self,'p'):
            self.p = EmptyClass()
        Requirement = ['workDir']
        Default = {'pressure':0}
        checkParameters(self.p,parameters,Requirement,Default)

    def relax(self,calcPop):
        pass

    def scf(self,calcPop):
        pass

class ASECalculator(Calculator):
    def __init__(self,parameters):
        super().__init__(parameters)
        Requirement = ['epsArr','stepArr','calcNum']
        Default = {'optimizer':'bfgs','maxRelaxStep':0.1}
        checkParameters(self.p,parameters,Requirement,Default)
        assert len(self.p.epsArr) == self.p.calcNum
        assert len(self.p.stepArr) == self.p.calcNum

    def relax(self, calcPop ,calcs):
        os.chdir(self.p.workDir)
        if not os.path.exists('calcFold'):
            # os.mkdir('calcFold')
            shutil.copytree("{}/inputFold".format(self.p.workDir), "calcFold")
        os.chdir('calcFold')
        logfile = 'aserelax.log'
        relaxPop = []
        errorPop = []
        for i, ind in enumerate(calcPop):
            for j, calc in enumerate(calcs):
                ind.set_calculator(calc)
                logging.debug("Structure {} Step {}".format(i, j))
                ucf = ExpCellFilter(ind, scalar_pressure=self.p.pressure*GPa)
                if self.p.optimizer == 'cg':
                    gopt = SciPyFminCG(ucf, logfile=logfile,trajectory='calc.traj')
                elif self.p.optimizer == 'bfgs':
                    gopt = BFGS(ucf, logfile=logfile, maxstep=self.p.maxRelaxStep,trajectory='calc.traj')
                elif self.p.optimizer == 'lbfgs':
                    gopt = LBFGS(ucf, logfile=logfile, maxstep=self.p.maxRelaxStep,trajectory='calc.traj')
                elif self.p.optimizer == 'fire':
                    gopt = FIRE(ucf, logfile=logfile, maxmove=self.p.maxRelaxStep,trajectory='calc.traj')
                try:
                    label = gopt.run(fmax=self.p.epsArr[j], steps=self.p.stepArr[j])
                    traj = ase.io.read('calc.traj',':')
                    # save relax steps
                    logging.debug('{} relax steps: {}'.format(self.__class__.__name__,len(traj)))
                except Converged:
                    pass
                except TimeoutError:
                    errorPop.append(ind)
                    logging.warning("Calculator:{} relax Timeout".format(self.__class__.__name__))
                    continue
                except:
                    errorPop.append(ind)
                    logging.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
                    logging.warning("Calculator:{} relax fail".format(self.__class__.__name__))
                    continue

            else:
                #if label:
                # save energy, forces, stress for trainning potential
                ind.info['energy'] = ind.get_potential_energy()
                ind.info['forces'] = ind.get_forces()
                ind.info['stress'] = ind.get_stress()
                enthalpy = (ind.info['energy'] + self.p.pressure * ind.get_volume() * GPa)/len(ind)
                ind.info['enthalpy'] = round(enthalpy, 3)

                ind.set_calculator(None)
                relaxPop.append(ind)
        os.chdir(self.p.workDir)
        return relaxPop

    def scf(self, calcPop, calcs):
        os.chdir(self.p.workDir)
        if not os.path.exists('calcFold'):
            # os.mkdir('calcFold')
            shutil.copytree("{}/inputFold".format(self.p.workDir), "calcFold")
        os.chdir('calcFold')

        scfPop = []
        for ind in calcPop:
            atoms=copy.deepcopy(ind)
            atoms.set_calculator(calcs[0])
            try:
                atoms.info['energy'] = atoms.get_potential_energy()
                atoms.info['forces'] = atoms.get_forces()
                atoms.info['stress'] = atoms.get_stress()
                enthalpy = (atoms.info['energy'] + self.p.pressure * atoms.get_volume() * GPa)/len(atoms)
                atoms.info['enthalpy'] = round(enthalpy, 3)
                atoms.set_calculator(None)
                scfPop.append(atoms)
                logging.debug('{} scf steps: 0'.format(self.__class__.__name__))
            except:
                pass
        os.chdir(self.p.workDir)
        return scfPop

class LJCalculator(ASECalculator):
    def __init__(self,parameters):
        self.calcs = [LennardJones() for _ in range(parameters.calcNum)]
        return super(LJCalculator, self).__init__(parameters)

    def relax(self, calcPop):
        return super(LJCalculator, self).relax(calcPop,self.calcs)

    def scf(self, calcPop):
        return super(LJCalculator, self).scf(calcPop,self.calcs)

class EMTCalculator(ASECalculator):
    def __init__(self,parameters):
        self.calcs = [EMT() for _ in range(parameters.calcNum)]
        return super(EMTCalculator, self).__init__(parameters)

    def relax(self, calcPop):
        return super(EMTCalculator, self).relax(calcPop,self.calcs)

    def scf(self, calcPop):
        return super(EMTCalculator, self).scf(calcPop,self.calcs)

class QUIPCalculator(ASECalculator):
    def __init__(self,parameters):
        self.calcs = []
        for i in range(1, parameters.calcNum + 1):
            params = yaml.load(open("{}/inputFold/quip_{}.yaml".format(parameters.workDir, i)))
            calc = QUIP(**params)
            self.calcs.append(calc)
        return super(QUIPCalculator, self).__init__(parameters)
    def relax(self, calcPop):
        return super(QUIPCalculator, self).relax(calcPop,self.calcs)

    def scf(self, calcPop):
        return super(QUIPCalculator, self).scf(calcPop,self.calcs)

class XTBCalculator(ASECalculator):
    def __init__(self,parameters):
        self.calcs = []
        for i in range(1, parameters.calcNum + 1):
            xtbParams = yaml.load(open("{}/inputFold/xtb_{}.yaml".format(parameters.workDir, i)))
            if xtbParams['type'] == 0:
                calc = GFN0(**xtbParams)
            elif xtbParams['type'] == 1:
                calc = GFN1(**xtbParams)
            self.calcs.append(calc)
        return super(XTBCalculator, self).__init__(parameters)

    def relax(self, calcPop):
        return super(XTBCalculator, self).relax(calcPop,self.calcs)

    def scf(self, calcPop):
        return super(XTBCalculator, self).scf(calcPop,self.calcs)

class ASEGULPCalculator(ASECalculator):
    """
    GULP Calculator based on ASE's GULP Calculator
    Still have bugs.
    """
    def __init__(self,parameters):
        self.calcs = []
        for i in range(1, parameters.calcNum + 1):
            with open("{}/inputFold/goptions_{}".format(parameters.workDir, i), 'r') as f:
                keywords = f.readline()
            with open("{}/inputFold/ginput_{}".format(parameters.workDir, i), 'r') as f:
                options = f.readlines()
            calc = GULP(keywords=keywords, options=options, library='')
            self.calcs.append(calc)
        return super(ASEGULPCalculator, self).__init__(parameters)

    def relax(self, calcPop):
        return super(ASEGULPCalculator, self).relax(calcPop,self.calcs)

    def scf(self, calcPop):
        return super(ASEGULPCalculator, self).scf(calcPop,self.calcs)

class ABinitCalculator(Calculator):
    def __init__(self,parameters,prefix):
        super().__init__(parameters)
        Requirement = ['mode','calcNum']
        Default = {}
        checkParameters(self.p,parameters,Requirement,Default)
        if self.p.mode == 'serial':
            self.scf = self.scf_serial
            self.relax = self.relax_serial
        elif self.p.mode == 'parallel':
            Requirement = ['queueName','numCore','numParallel']
            Default = {'Preprocessing':'','waitTime':200,'verbose':False}
            checkParameters(self.p,parameters,Requirement,Default)
            self.J=JobManager(self.p.verbose)
            self.scf = self.scf_parallel
            self.relax = self.relax_parallel
            self.prefix=prefix
        else:
            raise Exception("'{}' shi ge sha mo shi".format(parameters.mode))

    def cdcalcFold(self):
        os.chdir(self.p.workDir)
        if not os.path.exists('calcFold'):
            # os.mkdir('calcFold')
            # logging.debug('make calcFold')
            shutil.copytree('inputFold', 'calcFold')
        os.chdir('calcFold')

    def scf_serial(self,calcPop):
        pass

    def relax_serial(self,calcPop):
        pass

    def paralleljob(self,calcPop,runjob):
        numParallel = self.p.numParallel
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
                # os.mkdir("{}{}".format(self.prefix, i))
                shutil.copytree("{}/inputFold".format(self.p.workDir), "{}{}".format(self.prefix, i))
            os.chdir("{}{}".format(self.prefix, i))

            tmpPop = [calcPop[j] for j in runArray[i]]
            write_traj('initPop.traj', tmpPop)

            runjob(index=i)

            os.chdir("%s/calcFold" %(self.p.workDir))

        self.J.WaitJobsDone(self.p.waitTime)
        os.chdir(self.p.workDir)

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

    def relaxjob(self, index):
        pass

    def read_parallel_results(self):
        pop = []
        for job in self.J.jobs:
            try:
                pop.extend(ase.io.read("{}/optPop.traj".format(job['workDir']), format='traj', index=':'))
            except:
                logging.warning("ERROR in read results {}".format(job['workDir']))
        return pop

class VaspCalculator(ABinitCalculator):
    def __init__(self,parameters,prefix='calcVasp'):
        super().__init__(parameters,prefix)
        Requirement = ['symbols']
        Default = {'xc':'PBE','jobPrefix':'Vasp'}
        checkParameters(self.p,parameters,Requirement,Default)
        self.p.ppLabel = parameters.ppLabel if hasattr(parameters,'ppLabel') \
            else['' for _ in parameters.symbols]
        self.p.setup = dict(zip(self.p.symbols, self.p.ppLabel))

    def scf_serial(self,calcPop):
        self.cdcalcFold()
        calc = RelaxVasp()
        calc.read_incar('INCAR_0')
        calc.set(xc=self.p.xc,setups=self.p.setup,pstress=self.p.pressure*10)
        scfPop = calc_vasp([calc], calcPop)
        os.chdir(self.p.workDir)
        return scfPop

    def relax_serial(self,calcPop):
        self.cdcalcFold()
        incars = ['INCAR_{}'.format(i) for i in range(1, self.p.calcNum+1)]
        calcs = []
        for incar in incars:
            calc = RelaxVasp()
            calc.read_incar(incar)
            calc.set(xc=self.p.xc,setups=self.p.setup,pstress=self.p.pressure*10)
            calcs.append(calc)
        relaxPop = calc_vasp(calcs, calcPop)
        os.chdir(self.p.workDir)
        return relaxPop

    def scfjob(self,index):
        shutil.copy("{}/inputFold/INCAR_0".format(self.p.workDir),'INCAR_0')
        with open('vaspSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(self.p.setup))
        #jobName = self.p.jobPrefix + '_scf_' + str(index)
        jobName = self.p.jobPrefix + '_s_' + str(index)
        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J %s\n"% (self.p.queueName, self.p.numCore,jobName))
        f.write("{}\n".format(self.p.Preprocessing))
        f.write("python -m magus.runvasp 0 {} vaspSetup.yaml {} initPop.traj optPop.traj\n".format(self.p.xc, self.p.pressure))
        f.close()
        self.J.bsub('bsub < parallel.sh',jobName)

    def relaxjob(self,index):
        with open('vaspSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(self.p.setup))
        #jobName = self.p.jobPrefix + '_relax_' + str(index)
        jobName = self.p.jobPrefix + '_' + str(index)
        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J %s\n"% (self.p.queueName, self.p.numCore, jobName))
        f.write("{}\n".format(self.p.Preprocessing))
        f.write("python -m magus.runvasp {} {} vaspSetup.yaml {} initPop.traj optPop.traj\n".format(self.p.calcNum, self.p.xc, self.p.pressure))
        f.close()
        self.J.bsub('bsub < parallel.sh',jobName)

class GULPCalculator(ABinitCalculator):
    def __init__(self, parameters,prefix='calcGulp'):
        super().__init__(parameters,prefix)
        Requirement = ['symbols']
        Default = {'exeCmd':'','jobPrefix':'Gulp'}
        checkParameters(self.p,parameters,Requirement,Default)

    def scf_serial(self,calcPop):
        self.cdcalcFold()

        calcNum = 0
        exeCmd = self.p.exeCmd
        pressure = self.p.pressure
        inputDir = "{}/inputFold".format(self.p.workDir)

        scfPop = calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir)
        write_traj('optPop.traj', scfPop)
        os.chdir(self.p.workDir)
        return scfPop

    def relax_serial(self,calcPop):
        self.cdcalcFold()

        calcNum = self.p.calcNum
        exeCmd = self.p.exeCmd
        pressure = self.p.pressure
        inputDir = "{}/inputFold".format(self.p.workDir)

        relaxPop = calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir)
        write_traj('optPop.traj', relaxPop)
        os.chdir(self.p.workDir)
        return relaxPop

    def scfjob(self,index):
        calcDic = {
            'calcNum': 0,
            'pressure': self.p.pressure,
            'exeCmd': self.p.exeCmd,
            'inputDir': "{}/inputFold".format(self.p.workDir),
        }
        with open('gulpSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(calcDic))
        #jobName = self.p.jobPrefix + '_scf_' + str(index)
        jobName = self.p.jobPrefix + '_s_' + str(index)
        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J %s\n"% (self.p.queueName, self.p.numCore, jobName))
        f.write("{}\n".format(self.p.Preprocessing))
        f.write("python -m magus.rungulp gulpSetup.yaml")
        f.close()

        self.J.bsub('bsub < parallel.sh',jobName)

    def relaxjob(self,index):
        calcDic = {
            'calcNum': self.p.calcNum,
            'pressure': self.p.pressure,
            'exeCmd': self.p.exeCmd,
            'inputDir': "{}/inputFold".format(self.p.workDir),
        }
        with open('gulpSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(calcDic))
        #jobName = self.p.jobPrefix + '_relax_' + str(index)
        jobName = self.p.jobPrefix + '_' + str(index)
        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J %s\n"% (self.p.queueName, self.p.numCore,jobName))
        f.write("{}\n".format(self.p.Preprocessing))
        f.write("python -m magus.rungulp gulpSetup.yaml")
        f.close()

        self.J.bsub('bsub < parallel.sh',jobName)

class LammpsCalculator(ABinitCalculator):
    def __init__(self, parameters,prefix='calcLammps'):
        super().__init__(parameters,prefix)
        Requirement = ['symbols']
        Default = {'exeCmd':'','jobPrefix':'Lammps'}
        checkParameters(self.p,parameters,Requirement,Default)

    def scf_serial(self,calcPop):
        self.cdcalcFold()

        calcNum = 0
        exeCmd = self.p.exeCmd
        pressure = self.p.pressure
        inputDir = "{}/inputFold".format(self.p.workDir)

        scfPop = calc_lammps(calcNum, calcPop, pressure, exeCmd, inputDir)
        write_traj('optPop.traj', scfPop)
        os.chdir(self.p.workDir)
        return scfPop

    def relax_serial(self,calcPop):
        self.cdcalcFold()

        calcNum = self.p.calcNum
        exeCmd = self.p.exeCmd
        pressure = self.p.pressure
        inputDir = "{}/inputFold".format(self.p.workDir)

        relaxPop = calc_lammps(calcNum, calcPop, pressure, exeCmd, inputDir)
        write_traj('optPop.traj', relaxPop)
        os.chdir(self.p.workDir)
        return relaxPop

    def scfjob(self,index):
        calcDic = {
            'calcNum': 0,
            'pressure': self.p.pressure,
            'exeCmd': self.p.exeCmd,
            'inputDir': "{}/inputFold".format(self.p.workDir),
        }
        with open('gulpSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(calcDic))
        jobName = self.p.jobPrefix + '_scf_' + str(index)
        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J %s\n"% (self.p.queueName, self.p.numCore, jobName))
        f.write("{}\n".format(self.p.Preprocessing))
        f.write("python -m magus.rungulp gulpSetup.yaml")
        f.close()

        self.J.bsub('bsub < parallel.sh',jobName)

    def relaxjob(self,index):
        calcDic = {
            'calcNum': self.p.calcNum,
            'pressure': self.p.pressure,
            'exeCmd': self.p.exeCmd,
            'inputDir': "{}/inputFold".format(self.p.workDir),
        }
        with open('gulpSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(calcDic))
        jobName = self.p.jobPrefix + '_relax_' + str(index)
        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J %s\n"% (self.p.queueName, self.p.numCore,jobName))
        f.write("{}\n".format(self.p.Preprocessing))
        f.write("python -m magus.rungulp gulpSetup.yaml")
        f.close()

        self.J.bsub('bsub < parallel.sh',jobName)

def calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir):
    optPop = []
    for n, ind in enumerate(calcPop):
        if calcNum == 0:
            ind = calc_gulp_once(0, ind, pressure, exeCmd, inputDir)
            logging.debug("Structure %s scf" %(n))
            if ind:
                optPop.append(ind)
            else:
                logging.warning("fail in gulp scf")
        else:
            for i in range(1, calcNum + 1):
                logging.debug("Structure %s Step %s" %(n, i))
                ind = calc_gulp_once(i, ind, pressure, exeCmd, inputDir)
            if ind:
                optPop.append(ind)
                shutil.copy('output', "gulp_out-{}-{}".format(n, i))
            else:
                logging.warning("fail in gulp relax")
    return optPop

def calc_gulp_once(calcStep, calcInd, pressure, exeCmd, inputDir):
    """
    exeCmd should be "gulp < input > output"
    """
    if os.path.exists('output'):
        os.remove('output')
    try:
        # for f in os.listdir(inputDir):
        #     filepath = "{}/{}".format(inputDir, f)
        #     if os.path.isfile(filepath):
        #         shutil.copy(filepath, f)
        shutil.copy("goptions_{}".format(calcStep), "input")

        with open('input', 'a') as gulpIn:
            gulpIn.write('cell\n')
            a, b, c, alpha, beta, gamma = calcInd.get_cell_lengths_and_angles()
            gulpIn.write("%g %g %g %g %g %g\n" %(a, b, c, alpha, beta, gamma))
            gulpIn.write('fractional\n')
            for atom in calcInd:
                gulpIn.write("%s %.6f %.6f %.6f\n" %(atom.symbol, atom.a, atom.b, atom.c))
            gulpIn.write('\n')

            with open("ginput_{}".format(calcStep), 'r') as gin:
                gulpIn.write(gin.read())

            gulpIn.write('pressure\n{}\n'.format(pressure))
            gulpIn.write("dump every optimized.structure")

        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('Gulp exited with exit code: %d.  ' % exitcode)

        fout = open('optimized.structure', 'r')
        output = fout.readlines()
        fout.close()

        for i, line in enumerate(output):
            if 'cell' in line:
                cellIndex = i + 1
            if 'fractional' in line:
                posIndex = i + 1

        cellpar = output[cellIndex].split()
        cellpar = [float(par) for par in cellpar]

        pos = []
        for line in output[posIndex:posIndex + len(calcInd)]:
            pos.append([eval(i) for i in line.split()[2:5]])

        optInd = crystal(symbols=calcInd, cellpar=cellpar)
        optInd.set_scaled_positions(pos)
        optInd.info = calcInd.info.copy()

        relaxsteps = os.popen("grep Cycle output | tail -1 | awk '{print $2}'").readlines()[0]
        logging.debug('gulp relax steps:{}'.format(relaxsteps))
        enthalpy = os.popen("grep Energy output | tail -1 | awk '{print $4}'").readlines()[0]
        enthalpy = float(enthalpy)
        volume = optInd.get_volume()
        energy = enthalpy - pressure * GPa * volume
        optInd.info['energy'] = energy
        optInd.info['enthalpy'] = round(enthalpy/len(optInd), 3)

        #TODO The following code are adapted from ASE, need modification
        with open('output') as f:
            lines = f.readlines()
        for i,line in enumerate(lines):
            if 'Final internal derivatives' in line:
                s = i + 5
                break
        forces = []
        while(True):
            s = s + 1
            if "------------" in lines[s]:
                break
            g = lines[s].split()[3:6]                    
            G = [-float(x) * eV / Ang for x in g]
            forces.append(G)
        forces = np.array(forces)
        optInd.info['forces'] = forces
        return optInd

    except:
        logging.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
        logging.warning("GULP fail")
        return None


def calc_vasp_once(
    calc,    # ASE calculator
    struct,
    index,
    ):
    struct.set_calculator(calc)

    try:
        energy = struct.get_potential_energy()
        forces = struct.get_forces()
        stress = struct.get_stress()
        gap = read_eigen()
    except:
        s = sys.exc_info()
        logging.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
        logging.warning("VASP fail")
        return None

    if calc.float_params['pstress']:
        pstress = calc.float_params['pstress']
    else:
        pstress = 0

    struct.info['pstress'] = pstress

    volume = struct.get_volume()
    # the unit of pstress is kBar = GPa/10
    enthalpy = energy + pstress * GPa * volume / 10
    enthalpy = enthalpy/len(struct)

    struct.info['gap'] = round(gap, 3)
    struct.info['enthalpy'] = round(enthalpy, 3)

    # save energy, forces, stress for trainning potential
    struct.info['energy'] = energy
    struct.info['forces'] = forces
    struct.info['stress'] = stress

    # save relax trajectory
    traj = ase.io.read('OUTCAR', index=':', format='vasp-out')

    # save relax steps
    logging.debug('vasp relax steps: {}'.format(len(traj)))
    trajDict = [extract_atoms(ats) for ats in traj]
    if index == 0:
        struct.info['trajs'] = []
        struct.info['relaxStep'] = []
    struct.info['trajs'].append(trajDict)
    struct.info['relaxStep'].append(len(trajDict))

    logging.debug("VASP finish")
    return struct[:]

def calc_vasp(
    calcs,    #a list of ASE calculator
    structs,    #a list of structures
    ):

    newStructs = []
    for i, ind in enumerate(structs):
        initInd = ind.copy()
        initInd.info = {}
        for j, calc in enumerate(calcs):
            logging.debug("Structure {} Step {}".format(i, j))
            ind = calc_vasp_once(copy.deepcopy(calc), ind, j)
            press = calc.float_params['pstress']/10
            shutil.copy("OUTCAR", "OUTCAR-{}-{}-{}".format(i, j, press))
            if ind is None:
                break
        else:
            # ind.info['initStruct'] = extract_atoms(initInd)
            newStructs.append(ind)
    return newStructs

def calc_lammps(calcNum, calcPop, pressure, exeCmd, inputDir):
    optPop = []
    for n, ind in enumerate(calcPop):
        if calcNum == 0:
            ind = calc_lammps_once(0, ind, pressure, exeCmd, inputDir)
            logging.debug("Structure %s scf" %(n))
            if ind:
                optPop.append(ind)
            else:
                logging.warning("fail in lammps scf")
        else:
            for i in range(1, calcNum + 1):
                logging.debug("Structure %s Step %s" %(n, i))
                ind = calc_lammps_once(i, ind, pressure, exeCmd, inputDir)
            if ind:
                optPop.append(ind)
            else:
                logging.warning("fail in lammps relax")
    return optPop

def calc_lammps_once(calcStep, calcInd, pressure, exeCmd, inputDir):
    """
    exeCmd should be "lmp -in in.lammps"
    """
    if os.path.exists('output'):
        os.remove('output')
    try:
        shutil.copy("in.lammps_{}".format(calcStep), "in.lammps")
        #TODO more systems
        Atomic(calcInd).dump('data')
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('Lammps exited with exit code: %d.  ' % exitcode)
        numlist = [0]
        numlist.extend(calcInd.get_chemical_symbols())
        with open('out.dump') as f:
            struct = read_lammps_dump(f,numlist=numlist)[-1]
        volume = struct.get_volume()
        energy = float(os.popen("grep energy energy.out | tail -1 | awk '{print $3}'").readlines()[0])
        # the unit of pstress is kBar = GPa/10
        enthalpy = energy + pressure * GPa * volume / 10
        enthalpy = enthalpy/len(struct)

        struct.info['enthalpy'] = round(enthalpy, 3)

        # save energy, forces, stress for trainning potential
        struct.info['energy'] = energy
        return struct
    except:
        logging.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
        logging.warning("Lammps fail")
        return None
