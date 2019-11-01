from __future__ import print_function
import sys
import yaml
from csp.mopac import MOPAC
import ase.io
from csp.localopt import calc_vasp
from csp.writeresults import write_traj

if  __name__ == "__main__":
    calcNum, xc, vaspStpFile, pressure, inputTraj, outTraj = sys.argv[1:]
    calcNum = int(calcNum)
    pressure = float(pressure)
    vaspSetup = yaml.load(open(vaspStpFile))

    # calcs = []
    # incars = ['INCAR_{}'.format(i) for i in range(1, calcNum+1)]
    # for incar in incars:
    #     calc = Vasp()
    #     calc.read_incar(incar)
    #     calc.set(xc=xc)
    #     calc.set(setups=vaspSetup)
    #     calc.set(pstress=pressure*10)
    #     calcs.append(calc)
    # initPop = ase.io.read(inputTraj, format='traj', index=':',)
    # optPop = calc_vasp(calcs, initPop, )
    # write_traj(outTraj, optPop)