from __future__ import print_function
import sys, os
import yaml
from ase.calculators.vasp import Vasp
import ase.io
from csp.localopt import calc_gulp
from csp.writeresults import write_traj

if  __name__ == "__main__":
    calcDic = yaml.load(open(sys.argv[1]))
    calcNum = calcDic['calcNum']
    exeCmd = calcDic['exeCmd']
    pressure = calcDic['pressure']
    inputDir = calcDic['inputDir']

    initPop = ase.io.read('initPop.traj', index=':', format='traj')

    # remove previous label
    if os.path.exists('DONE'):
        os.remove('DONE')

    optPop = calc_gulp(calcNum, initPop, pressure, exeCmd, inputDir)
    write_traj('optPop.traj', optPop)
    with open('DONE', 'w') as f:
        f.write('DONE')