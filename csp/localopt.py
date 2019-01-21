from __future__ import print_function, division
from ase import Atoms
import ase.io
from .readvasp import *
import sys
import math
import os
import shutil
import subprocess
import logging
import copy
import yaml
from ase.calculators.lj import LennardJones
from ase.calculators.vasp import Vasp
from ase.spacegroup import crystal
# from parameters import parameters
from .writeresults import write_yaml, read_yaml, write_traj
from .utils import *

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
    enthalpy = energy + pstress * volume / 1602.262
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
    for i, ind in enumerate(structs):
        initInd = ind.copy()
        initInd.info = {}
        for j, calc in enumerate(calcs):
            # logging.info('Structure ' + str(structs.index(ind)) + ' Step '+ str(calcs.index(calc)))
            logging.info("Structure {} Step {}".format(i, j))
            # print("Structure {} Step {}".format(i, j))
            ind = calc_vasp_once(copy.deepcopy(calc), ind, j)
            shutil.copy("OUTCAR", "OUTCAR-{}-{}".format(i, j))
            # shutil.copy("INCAR", "INCAR-{}-{}".format(i, j))

            if ind is None:
                break

        else:
            # ind.info['initStruct'] = extract_atoms(initInd)
            newStructs.append(ind)

    return newStructs

def generate_calcs(calcNum, parameters):
    workDir = parameters['workDir']
    xc = parameters['xc']
    pressure = parameters['pressure']
    symbols = parameters['symbols']
    ppLabel = parameters['ppLabel']

    vaspSetup = dict(zip(symbols, ppLabel))

    incars = ["{}/inputFold/INCAR_{}".format(workDir, i) for i in range(1, calcNum + 1)]

    calcs = []
    for incar in incars:
        calc = Vasp()
        calc.read_incar(incar)
        calc.set(xc=xc)
        calc.set(setups=vaspSetup)
        calc.set(pstress=pressure*10)
        calcs.append(calc)

    return calcs


# def calc_vasp_parallel_old(calcNum, calcPop, parameters, prefix='calcVasp'):
#     workDir = parameters['workDir']
#     queueName = parameters['queueName']
#     numParallel = parameters['numParallel']
#     numCore = parameters['numCore']
#     xc = parameters['xc']
#     pressure = parameters['pressure']
#     symbols = parameters['symbols']
#     ppLabel = parameters['ppLabel']

#     vaspSetup = dict(zip(symbols, ppLabel))

#     incars = ["{}/inputFold/INCAR_{}".format(workDir, i) for i in range(1, calcNum + 1)]

#     popLen = len(calcPop)
#     eachLen = popLen//numParallel
#     remainder = popLen%numParallel

#     runArray = []
#     for i in range(numParallel):
#         tmpList = [ i + numParallel*j for j in range(eachLen)]
#         if i < remainder:
#             tmpList.append(numParallel*eachLen + i)
#         runArray.append(tmpList)

#     runVasp = open('run_vasp.py', 'w')
#     runVasp.write("from __future__ import print_function\n")
#     runVasp.write("from ase.calculators.vasp import Vasp\n")
#     runVasp.write("import ase.io\n")
#     # runVasp.write("import sys\nsys.path.append('%s')\n" %(workDir))
#     runVasp.write("from csp.localopt import calc_vasp\n")
#     runVasp.write("from csp.writeresults import write_traj, write_yaml, read_yaml\n")
#     # runVasp.write("from csp.setincar import calcs\n")
#     runVasp.write("calcs = []\n")
#     runVasp.write("incars = {}\n".format(incars))
#     runVasp.write("for incar in incars:\n")
#     runVasp.write("    calc = Vasp()\n")
#     runVasp.write("    calc.read_incar(incar)\n")
#     runVasp.write("    calc.set(xc={})\n".format(repr(xc)))
#     runVasp.write("    calc.set(setups={})\n".format(repr(vaspSetup)))
#     runVasp.write("    calc.set(pstress={})\n".format(repr(pressure * 10)))
#     runVasp.write("    calcs.append(calc)\n")
#     runVasp.write("initPop = ase.io.read('initPop.traj', format='traj', index=':',)\n")
#     runVasp.write("optPop = calc_vasp(calcs, initPop, )\n")
#     runVasp.write("write_traj('optPop.traj', optPop)")
#     runVasp.close()

#     runJobs = []
#     for i in range(numParallel):
#         if not os.path.exists("{}{}".format(prefix, i)):
#             os.mkdir("{}{}".format(prefix, i))
#         os.chdir("{}{}".format(prefix, i))

#         tmpPop = [calcPop[j] for j in runArray[i]]
#         write_traj('initPop.traj', tmpPop)
#         shutil.copy("../run_vasp.py", "run_vasp.py")

#         f = open('parallel.sh', 'w')
#         f.write("#BSUB -q %s\n"
#                 "#BSUB -n %s\n"
#                 "#BSUB -o out\n"
#                 "#BSUB -e err\n"
#                 "#BSUB -J Vasp_%s\n"% (queueName, numCore, i))
#         f.write("python run_vasp.py > vasplog")
#         f.close()

#         jobID = subprocess.check_output("bsub < parallel.sh", shell=True).split()[1]
#         jobID = jobID[1: -1]
#         runJobs.append(jobID)

#         os.chdir("%s/calcFold" %(workDir))

#     return runJobs

def calc_vasp_parallel(calcNum, calcPop, parameters, prefix='calcVasp'):
    workDir = parameters['workDir']
    queueName = parameters['queueName']
    numParallel = parameters['numParallel']
    numCore = parameters['numCore']
    xc = parameters['xc']
    pressure = parameters['pressure']
    symbols = parameters['symbols']
    ppLabel = parameters['ppLabel']
    maxRelaxTime = parameters['maxRelaxTime']

    vaspSetup = dict(zip(symbols, ppLabel))

    # incars = ["{}/inputFold/INCAR_{}".format(workDir, i) for i in range(1, calcNum + 1)]

    popLen = len(calcPop)
    eachLen = popLen//numParallel
    remainder = popLen%numParallel

    runArray = []
    for i in range(numParallel):
        tmpList = [ i + numParallel*j for j in range(eachLen)]
        if i < remainder:
            tmpList.append(numParallel*eachLen + i)
        runArray.append(tmpList)



    runJobs = []
    for i in range(numParallel):
        if not os.path.exists("{}{}".format(prefix, i)):
            os.mkdir("{}{}".format(prefix, i))
        os.chdir("{}{}".format(prefix, i))

        tmpPop = [calcPop[j] for j in runArray[i]]
        write_traj('initPop.traj', tmpPop)
        for j in range(1, calcNum + 1):
            shutil.copy("{}/inputFold/INCAR_{}".format(workDir, j), 'INCAR_{}'.format(j))
        # shutil.copy("../run_vasp.py", "run_vasp.py")
        with open('vaspSetup.yaml', 'w') as setupF:
            setupF.write(yaml.dump(vaspSetup))

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -W %s\n"
                "#BSUB -J Vasp_%s\n"% (queueName, numCore, maxRelaxTime*len(tmpPop), i))
        f.write("python -m csp.runvasp {} {} vaspSetup.yaml {} initPop.traj optPop.traj".format(calcNum, xc, pressure))
        f.close()

        jobID = subprocess.check_output("bsub < parallel.sh", shell=True).split()[1]
        jobID = jobID[1: -1]
        runJobs.append(jobID)

        os.chdir("%s/calcFold" %(workDir))

    return runJobs

def read_parallel_results(jobStat, parameters, prefix):

    optPop = []
    for i, stat in enumerate(jobStat):
        if stat == 'DONE':
            # logging.info(os.getcwd())
            try:
                # optPop.extend(read_yaml("{}/calcFold/{}{}/optPop.yaml".format(parameters['workDir'], prefix, i)))
                optPop.extend(ase.io.read("{}/calcFold/{}{}/optPop.traj".format(parameters['workDir'], prefix, i), format='traj', index=':'))
            except:
                logging.info("ERROR in read results")

    return optPop




def calc_lj(structs):
    ljCalc = LennardJones()
    newStructs = []
    for ind in structs:
        ind.set_calculator(ljCalc)
        energy = ind.get_potential_energy()
        energy = energy/len(ind)
        ind.info['enthalpy'] = round(energy, 3)
        ind.info['gap'] = 0
        if not math.isnan(energy):
            newStructs.append(ind)

    return newStructs

def calc_gulp(calcNum, calcPop, pressure, exeCmd, inputDir):
    optPop = []
    for n, ind in enumerate(calcPop):
        for i in range(1, calcNum + 1):
            ind = calc_gulp_once(i, ind, pressure, exeCmd, inputDir)
            logging.info("Structure %s Step %s" %(n, i))

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
        subprocess.call("cp {}/* .".format(inputDir), shell=True)
        subprocess.call("cat goptions_{} > input".format(calcStep), shell=True)


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
        enthalpy = float(enthalpy)/len(optInd)
        optInd.info['enthalpy'] = round(enthalpy, 3)

        return optInd

    except:
        s = sys.exc_info()
        print("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
        logging.info("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
        logging.info("GULP fail")
        return None

def calc_gulp_parallel(calcNum, calcPop, parameters,):

    workDir = parameters['workDir']
    queueName = parameters['queueName']
    numParallel = parameters['numParallel']
    numCore = parameters['numCore']

    pressure = parameters['pressure']
    exeCmd = parameters['exeCmd']
    inputDir = "{}/inputFold".format(parameters['workDir'])

    popLen = len(calcPop)
    eachLen = popLen//numParallel
    remainder = popLen%numParallel

    runArray = []
    for i in range(numParallel):
        tmpList = [ i + numParallel*j for j in range(eachLen)]
        if i < remainder:
            tmpList.append(numParallel*eachLen + i)
        runArray.append(tmpList)

    runGulp = open('run_gulp.py', 'w')
    runGulp.write("from __future__ import print_function\n")
    runGulp.write("import sys\nsys.path.append('%s')\n" %(workDir))
    runGulp.write("import ase.io" %(workDir))
    runGulp.write("from csp.localopt import calc_gulp\n")
    runGulp.write("from csp.writeresults import write_traj, write_yaml, read_yaml\n")
    runGulp.write("initPop = ase.io.read('initPop.traj', format='traj', index=':',)\n")
    runGulp.write("optPop = calc_gulp({}, initPop, {}, {!r}, {!r})\n".format(calcNum, pressure, exeCmd, inputDir))
    runGulp.write("write_traj('optPop.traj', optPop)")
    runGulp.close()


    runJobs = []
    for i in range(numParallel):
        if not os.path.exists('calcGulp%s' %(i)):
            os.mkdir('calcGulp%s' %(i))
        os.chdir('calcGulp%s' %(i))

        tmpPop = [calcPop[j] for j in runArray[i]]
#        logging.info("runArray[i]: %s"% (runArray[i]))
        write_traj('initPop.traj', tmpPop)
        shutil.copy("../run_gulp.py", "run_gulp.py")

        f = open('parallel.sh', 'w')
        f.write("#BSUB -q %s\n"
                "#BSUB -n %s\n"
                "#BSUB -o out\n"
                "#BSUB -e err\n"
                "#BSUB -J Gulp_%s\n"% (queueName, numCore ,i))
        f.write("python run_gulp.py > gulplog")
        f.close()

        jobID = subprocess.check_output("bsub < parallel.sh", shell=True).split()[1]
        jobID = jobID[1: -1]
        runJobs.append(jobID)

        os.chdir("%s/calcFold" %(workDir))

    return runJobs

def jobs_stat(runJobs):
    """
    Check if the job has been done.
    """
    jobStat = []
    for jobID in runJobs:
        try:
            stat = subprocess.check_output("bjobs %s | grep %s | awk '{print $3}'"% (jobID, jobID), shell=True)
            stat = stat.decode()[:-1]
        except:
            s = sys.exc_info()
            logging.info("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
            stat = 'DONE'
        # logging.debug(jobID, stat)
        if stat == 'DONE':
            jobStat.append('DONE')
        elif stat == 'PEND' or stat == 'RUN':
            jobStat.append('RUN')
        else:
            jobStat.append('ERROR')

    return jobStat

# def read_gulp_results(jobStat):

#     optPop = []
#     for i, stat in enumerate(jobStat):
#         if stat == 'DONE':
#             # logging.info(os.getcwd())
#             try:
#                 optPop.extend(read_yaml("%s/calcFold/calcGulp%s/optPop.yaml"% (parameters['workDir'], i)))
#             except:
#                 logging.info("GULP ERROR!")

#     return optPop
