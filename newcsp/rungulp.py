import sys, os,shutil,subprocess,logging,traceback
import yaml
from ase.calculators.vasp import Vasp
import ase.io
from csp.writeresults import write_traj
from ase.spacegroup import crystal

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
        enthalpy = float(enthalpy)/len(optInd)
        optInd.info['enthalpy'] = round(enthalpy, 3)

        return optInd

    except:
        logging.debug("traceback.format_exc():\n{}".format(traceback.format_exc()))
        logging.info("GULP fail")
        return None

if  __name__ == "__main__":
    calcDic = yaml.load(open(sys.argv[1]))
    calcNum = calcDic['calcNum']
    exeCmd = calcDic['exeCmd']
    pressure = calcDic['pressure']
    inputDir = calcDic['inputDir']

    initPop = ase.io.read('initPop.traj', index=':', format='traj')

    optPop = calc_gulp(calcNum, initPop, pressure, exeCmd, inputDir)
    write_traj('optPop.traj', optPop)