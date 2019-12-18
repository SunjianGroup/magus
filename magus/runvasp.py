import sys, os, shutil, yaml, copy
from ase.calculators.vasp import Vasp
import ase.io
from magus.writeresults import write_traj
import logging

from .readvasp import *
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

if  __name__ == "__main__":
    calcNum, xc, vaspStpFile, pressure, inputTraj, outTraj = sys.argv[1:]
    calcNum = int(calcNum)
    pressure = float(pressure)
    vaspSetup = yaml.load(open(vaspStpFile))

    calcs = []
    if calcNum == 0:
        incars = ['INCAR_scf']
    else:
        incars = ['INCAR_{}'.format(i) for i in range(1, calcNum+1)]
    for incar in incars:
        calc = Vasp()
        calc.read_incar(incar)
        calc.set(xc=xc)
        calc.set(setups=vaspSetup)
        calc.set(pstress=pressure*10)
        calcs.append(calc)
    initPop = ase.io.read(inputTraj, format='traj', index=':',)
    optPop = calc_vasp(calcs, initPop, )
    write_traj(outTraj, optPop)
