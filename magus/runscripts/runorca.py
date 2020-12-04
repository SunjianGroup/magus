import sys, os,shutil,subprocess,logging,traceback
import yaml
import ase.io
from magus.writeresults import write_traj
from magus.localopt import calc_orca

if  __name__ == "__main__":
    calcDic = yaml.load(open(sys.argv[1]))
    calcNum = calcDic['calcNum']
    exeCmd = calcDic['exeCmd']
    pressure = calcDic['pressure']
    initPop = ase.io.read('initPop.traj', index=':', format='traj')

    optPop = calc_orca(calcNum, initPop, pressure, exeCmd)
    write_traj('optPop.traj', optPop)
