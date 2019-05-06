from __future__ import print_function
import sys, os
import yaml
from ase.calculators.vasp import Vasp
import ase.io
from csp.localopt import calc_vasp
from csp.writeresults import write_traj

if  __name__ == "__main__":
    vaspDict = yaml.load(sys.argv[1])
    calcNum = vaspDict['calcNum']
    xc = vaspDict['xc']
    vaspSetup = vaspDict['vaspSetup']
    pressArr = vaspDict['pressArr']
    inputTraj = vaspDict['inputTraj']
    outTraj = vaspDict['outTraj']
    # calcNum, xc, vaspStpFile, pressure, inputTraj, outTraj = sys.argv[1:]
    # calcNum = int(calcNum)
    # pressure = float(pressure)
    # vaspSetup = yaml.load(open(vaspStpFile))

    # remove previous label
    if os.path.exists('DONE'):
        os.remove('DONE')

    calcs = []
    incars = ['INCAR_{}'.format(i) for i in range(1, calcNum+1)]
    for pressure in pressArr:
        for incar in incars:
            calc = Vasp()
            calc.read_incar(incar)
            calc.set(xc=xc)
            calc.set(setups=vaspSetup)
            calc.set(pstress=pressure*10)
            calcs.append(calc)
        initPop = ase.io.read(inputTraj, format='traj', index=':',)
        optPop = calc_vasp(calcs, initPop, )
        write_traj("{}_{}".format(outTraj, pressure), optPop)
    with open('DONE', 'w') as f:
        f.write('DONE')