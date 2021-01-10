import sys, os, shutil, yaml, copy
from ase.calculators.vasp import Vasp
import ase.io
import logging

from magus.writeresults import write_traj
from magus.readvasp import *
from magus.utils import *
from magus.localopt import calc_vasp, RelaxVasp



if  __name__ == "__main__":
    calcNum, xc, vaspStpFile, pressure, ignore_constraints, inputTraj, outTraj = sys.argv[1:]
    calcNum = int(calcNum)
    pressure = float(pressure)
    ignore_constraints = bool(int(ignore_constraints))
    vaspSetup = yaml.load(open(vaspStpFile))

    calcs = []
    if calcNum == 0:
        incars = ['INCAR_0']
    else:
        incars = ['INCAR_{}'.format(i) for i in range(1, calcNum+1)]
    for incar in incars:
        calc = RelaxVasp()
        calc.read_incar(incar)
        calc.set(xc=xc)
        calc.set(setups=vaspSetup)
        calc.set(pstress=pressure*10)
        calc.set(ignore_constraints=ignore_constraints)
        calcs.append(calc)
    initPop = ase.io.read(inputTraj, format='traj', index=':',)
    optPop = calc_vasp(calcs, initPop, )
    write_traj(outTraj, optPop)
