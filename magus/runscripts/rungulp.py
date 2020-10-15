import sys, os,shutil,subprocess,logging,traceback
import yaml
from ase.calculators.vasp import Vasp
import ase.io
from ase.spacegroup import crystal
from magus.writeresults import write_traj
from magus.localopt import calc_gulp

if  __name__ == "__main__":
    calcDic = yaml.load(open(sys.argv[1]))
    calcNum = calcDic['calcNum']
    exeCmd = calcDic['exeCmd']
    pressure = calcDic['pressure']
    inputDir = calcDic['inputDir']
    fixcell = calcDic['fixcell']
    initPop = ase.io.read('initPop.traj', index=':', format='traj')

    optPop = calc_gulp(calcNum, initPop, pressure, exeCmd, inputDir, fixcell)
    write_traj('optPop.traj', optPop)
