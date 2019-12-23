from __future__ import print_function, division
from ase import Atoms
import ase.io
from .readvasp import *
import sys, math, os, shutil, subprocess, logging, copy, yaml, traceback

from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp
from ase.calculators.gulp import GULP, Conditions
from ase.spacegroup import crystal
# from parameters import parameters
from .writeresults import write_traj
from .utils import *
from ase.units import GPa, eV, Ang
try:
    from xtb import GFN0, GFN1
    from ase.constraints import ExpCellFilter
except:
    pass

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ase.calculators.lj import LennardJones
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from .queue import JobManager
# from .runvasp import calc_vasp
# from .rungulp import calc_gulp

class Calculator:
    def __init__(self):
        pass

    def relax(self,calcPop):
        pass

    def scf(self,calcPop):
        pass

class ASECalculator(Calculator):
    def __init__(self,parameters,calcs):
        self.parameters=parameters
        self.calcs=calcs

    def relax(self, calcPop):
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            # os.mkdir('calcFold')
            shutil.copytree("{}/inputFold".format(self.parameters.workDir), "calcFold")
        os.chdir('calcFold')

        relaxPop = []
        errorPop = []
        for i, ind in enumerate(calcPop):
            for j, calc in enumerate(self.calcs):
                ind.set_calculator(calc)
                logging.info("Structure {} Step {}".format(i, j))
                ucf = ExpCellFilter(ind, scalar_pressure=self.parameters.pressure*GPa)
                # ucf = UnitCellFilter(ind, scalar_pressure=self.parameters.pressure*GPa)
                if self.parameters.mainoptimizer == 'cg':
                    gopt = SciPyFminCG(ucf, logfile='aseOpt.log',)
                elif self.parameters.mainoptimizer == 'BFGS':
                    gopt = BFGS(ucf, logfile='aseOpt.log', maxstep=self.parameters.maxRelaxStep)
                elif self.parameters.mainoptimizer == 'fire':
                    gopt = FIRE(ucf, logfile='aseOpt.log', maxmove=self.parameters.maxRelaxStep)

                try:
                    label=gopt.run(fmax=self.parameters.epsArr[j], steps=self.parameters.stepArr[j])
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

            else:
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
        os.chdir(self.parameters.workDir)
        if not os.path.exists('calcFold'):
            # os.mkdir('calcFold')
            shutil.copytree("{}/inputFold".format(self.parameters.workDir), "calcFold")
        os.chdir('calcFold')

        scfPop = []
        for ind in calcPop:
            atoms=copy.deepcopy(ind)
            atoms.set_calculator(self.calcs[0])
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
        os.chdir(self.parameters.workDir)
        return scfPop

class LJCalculator(ASECalculator):
    def __init__(self,parameters):
        calcs = [LennardJones() for _ in range(parameters.calcNum)]
        return super(LJCalculator, self).__init__(parameters,calcs)

    def relax(self, calcPop):
        return super(LJCalculator, self).relax(calcPop)

    def scf(self, calcPop):
        return super(LJCalculator, self).scf(calcPop)

class EMTCalculator(ASECalculator):
    def __init__(self,parameters):
        calcs = [EMT() for _ in range(parameters.calcNum)]
        return super(EMTCalculator, self).__init__(parameters,calcs)

    def relax(self, calcPop):
        return super(EMTCalculator, self).relax(calcPop)

    def scf(self, calcPop):
        return super(EMTCalculator, self).scf(calcPop)

class XTBCalculator(ASECalculator):
    def __init__(self,parameters):
        calcs = []
        for i in range(1, parameters.calcNum + 1):
            xtbParams = yaml.load(open("{}/inputFold/xtb_{}.yaml".format(parameters.workDir, i)))
            if xtbParams['type'] == 0:
                calc = GFN0(**xtbParams)
            elif xtbParams['type'] == 1:
                calc = GFN1(**xtbParams)
            calcs.append(calc)
        return super(XTBCalculator, self).__init__(parameters,calcs)

    def relax(self, calcPop):
        return super(XTBCalculator, self).relax(calcPop)

    def scf(self, calcPop):
        return super(XTBCalculator, self).scf(calcPop)

class ASEGULPCalculator(ASECalculator):
    """
    Still have bugs
    """
    def __init__(self,parameters):
        calcs = []
        for i in range(1, parameters.calcNum + 1):
            with open("{}/inputFold/goptions_{}".format(parameters.workDir, i), 'r') as f:
                keywords = f.readline()
            with open("{}/inputFold/ginput_{}".format(parameters.workDir, i), 'r') as f:
                options = f.readlines()
            calc = GULP(keywords=keywords, options=options, library='')
            calcs.append(calc)
        return super(ASEGULPCalculator, self).__init__(parameters,calcs)

    def relax(self, calcPop):
        return super(ASEGULPCalculator, self).relax(calcPop)

    def scf(self, calcPop):
        return super(ASEGULPCalculator, self).scf(calcPop)

class OldXTBCalculator:
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
            calc = GFN0(**params)
            calcs.append(calc)
        relaxPop = self.calc_xtb(calcs, calcPop)
        os.chdir(self.parameters.workDir)
        return relaxPop

    def scf(self,calcPop):
        params = yaml.load(open("{}/inputFold/xtb_scf.yaml".format(self.parameters.workDir)))
        calc = GFN0(**params)
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
        os.chdir(self.parameters.workDir)
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

            runjob(index=i)

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

    def relaxjob(self, index):
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
        os.chdir(self.parameters.workDir)
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
        os.chdir(self.parameters.workDir)
        return relaxPop

    def scfjob(self,index):
        shutil.copy("{}/inputFold/INCAR_scf".format(self.parameters.workDir),'INCAR_scf')
        vaspSetup = dict(zip(self.parameters.symbols, self.parameters.ppLabel))
        with open('vaspSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(vaspSetup))

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J Vasp_%s\n"% (self.parameters.queueName, self.parameters.numCore, index))
        f.write("{}\n".format(self.parameters.jobPrefix))
        f.write("python -m magus.runvasp 0 {} vaspSetup.yaml {} initPop.traj optPop.traj\n".format(self.parameters.xc, self.parameters.pressure))
        f.close()
        self.J.bsub('bsub < parallel.sh')

    def relaxjob(self,index):
        vaspSetup = dict(zip(self.parameters.symbols, self.parameters.ppLabel))
        with open('vaspSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(vaspSetup))

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J Vasp_%s\n"% (self.parameters.queueName, self.parameters.numCore, index))
        f.write("{}\n".format(self.parameters.jobPrefix))
        f.write("python -m magus.runvasp {} {} vaspSetup.yaml {} initPop.traj optPop.traj\n".format(self.parameters.calcNum, self.parameters.xc, self.parameters.pressure))
        f.close()
        self.J.bsub('bsub < parallel.sh')

class GULPCalculator(ABinitCalculator):
    def __init__(self, parameters,prefix='calcGulp'):
        super().__init__(parameters,prefix)

    def scf_serial(self,calcPop):
        self.cdcalcFold()

        calcNum = 0
        exeCmd = self.parameters.exeCmd
        pressure = self.parameters.pressure
        inputDir = "{}/inputFold".format(self.parameters.workDir)

        scfPop = calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir)
        write_traj('optPop.traj', scfPop)
        os.chdir(self.parameters.workDir)
        return scfPop

    def relax_serial(self,calcPop):
        self.cdcalcFold()

        calcNum = self.parameters.calcNum
        exeCmd = self.parameters.exeCmd
        pressure = self.parameters.pressure
        inputDir = "{}/inputFold".format(self.parameters.workDir)

        relaxPop = calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir)
        write_traj('optPop.traj', relaxPop)
        os.chdir(self.parameters.workDir)
        return relaxPop

    def scfjob(self,index):
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
                "#BSUB -J Gulp_%s\n"% (self.parameters.queueName, self.parameters.numCore, index))
        f.write("{}\n".format(self.parameters.jobPrefix))
        f.write("python -m magus.rungulp gulpSetup.yaml")
        f.close()

        self.J.bsub('bsub < parallel.sh')

    def relaxjob(self,index):
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
                "#BSUB -J Gulp_%s\n"% (self.parameters.queueName, self.parameters.numCore, index))
        f.write("{}\n".format(self.parameters.jobPrefix))
        f.write("python -m magus.rungulp gulpSetup.yaml")
        f.close()

        self.J.bsub('bsub < parallel.sh')



def calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir):
    optPop = []
    for n, ind in enumerate(calcPop):
        if calcNum == 0:
            ind = calc_gulp_once(i, ind, pressure, exeCmd, inputDir)
            logging.info("Structure %s scf" %(n))
            if ind:
                optPop.append(ind)
            else:
                logging.info("fail in scf")
        else:
            for i in range(1, calcNum + 1):
                logging.info("Structure %s Step %s" %(n, i))
                ind = calc_gulp_once(i, ind, pressure, exeCmd, inputDir)
                shutil.copy('output', "gulp_out-{}-{}".format(n, i))
            if ind:
                optPop.append(ind)
            else:
                logging.info("fail in localopt")
    logging.info('\n')
    return optPop

def calc_gulp_once(calcStep, calcInd, pressure, exeCmd, inputDir):
    """
    exeCmd should be "gulp < input > output"
    """
    if os.path.exists('output'):
        os.remove('output')
    try:
        for f in os.listdir(inputDir):
            filepath = "{}/{}".format(inputDir, f)
            if os.path.isfile(filepath):
                shutil.copy(filepath, f)
        if calcStep == 0:
            shutil.copy("goptions_scf", "input")
        else:
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

        enthalpy = os.popen("grep Energy output | tail -1 | awk '{print $4}'").readlines()[0]
        enthalpy = float(enthalpy)
        volume = optInd.get_volume()
        energy = enthalpy + pressure * GPa * volume
        optInd.info['energy'] = energy
        optInd.info['enthalpy'] = round(enthalpy/len(optInd), 3)

        return optInd

    except:
        logging.debug("traceback.format_exc():\n{}".format(traceback.format_exc()))
        logging.info("GULP fail")
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
        logging.info("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
        logging.info("VASP fail")
        print("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))

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
    trajDict = [extract_atoms(ats) for ats in traj]
    if index == 0:
        struct.info['trajs'] = []
    struct.info['trajs'].append(trajDict)

    logging.info("VASP finish")
    return struct[:]

def calc_vasp(
    calcs,    #a list of ASE calculator
    structs,    #a list of structures
    ):

    newStructs = []
    logging.info('1')
    for i, ind in enumerate(structs):
        initInd = ind.copy()
        logging.info('1')
        initInd.info = {}
        for j, calc in enumerate(calcs):
            # logging.info('Structure ' + str(structs.index(ind)) + ' Step '+ str(calcs.index(calc)))
            logging.info("Structure {} Step {}".format(i, j))
            # print("Structure {} Step {}".format(i, j))
            ind = calc_vasp_once(copy.deepcopy(calc), ind, j)
            press = calc.float_params['pstress']/10
            shutil.copy("OUTCAR", "OUTCAR-{}-{}-{}".format(i, j, press))
            # shutil.copy("INCAR", "INCAR-{}-{}".format(i, j))

            if ind is None:
                break

        else:
            # ind.info['initStruct'] = extract_atoms(initInd)
            newStructs.append(ind)

    return newStructs

def read_gulp_results(filename):

    with open(filename) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.match(r'\s*Total lattice energy\s*=\s*(\S+)\s*eV', line)
        if m:
            energy = float(m.group(1))

        elif line.find('Final Cartesian derivatives') != -1:
            s = i + 5
            forces = []
            while(True):
                s = s + 1
                if lines[s].find("------------") != -1:
                    break
                if lines[s].find(" s ") != -1:
                    continue
                g = lines[s].split()[3:6]
                G = [-float(x) * eV / Ang for x in g]
                forces.append(G)
            forces = np.array(forces)

        elif line.find('Final internal derivatives') != -1:
            s = i + 5
            forces = []
            while(True):
                s = s + 1
                if lines[s].find("------------") != -1:
                    break
                g = lines[s].split()[3:6]

                    # Uncomment the section below to separate the numbers when there is no space between them, in the case of long numbers. This prevents the code to break if numbers are too big.

                '''for t in range(3-len(g)):
                    g.append(' ')
                for j in range(2):
                    min_index=[i+1 for i,e in enumerate(g[j][1:]) if e == '-']
                    if j==0 and len(min_index) != 0:
                        if len(min_index)==1:
                            g[2]=g[1]
                            g[1]=g[0][min_index[0]:]
                            g[0]=g[0][:min_index[0]]
                        else:
                            g[2]=g[0][min_index[1]:]
                            g[1]=g[0][min_index[0]:min_index[1]]
                            g[0]=g[0][:min_index[0]]
                            break
                    if j==1 and len(min_index) != 0:
                        g[2]=g[1][min_index[0]:]
                        g[1]=g[1][:min_index[0]]'''

                G = [-float(x) * eV / Ang for x in g]
                forces.append(G)
            forces = np.array(forces)

        elif line.find('Final stress tensor components') != -1:
            res=[0.,0.,0.,0.,0.,0.]
            for j in range(3):
                var=lines[i+j+3].split()[1]
                res[j]=float(var)
                var=lines[i+j+3].split()[3]
                res[j+3]=float(var)
            stress=np.array(res)

    return energy, forces, stress