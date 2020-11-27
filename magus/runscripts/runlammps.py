import logging, yaml, os, sys, traceback
from ase.io import read
from magus.writeresults import write_traj
from magus.formatting.lammps import dump_lmps, load_lmps


def calc_lammps(calcNum, calcPop, pressure, exeCmd, inputDir, numCore, symbol_to_type, type_to_symbol):
    optPop = []
    for n, ind in enumerate(calcPop):
        if calcNum == 0:
            ind = calc_lammps_once(0, ind, pressure, exeCmd, inputDir, numCore, symbol_to_type, type_to_symbol)
            logging.debug("Structure %s scf" %(n))
            if ind:
                optPop.append(ind)
            else:
                logging.warning("fail in lammps scf")
        else:
            for i in range(1, calcNum + 1):
                logging.debug("Structure {} Step {}".format(n, i))
                ind = calc_lammps_once(i, ind, pressure, exeCmd, inputDir, numCore,symbol_to_type, type_to_symbol)
            if ind:
                optPop.append(ind)
            else:
                logging.warning("fail in lammps relax")
    return optPop


def calc_lammps_once(calcStep, calcInd, pressure, exeCmd, inputDir, numCore, symbol_to_type, type_to_symbol):
    if os.path.exists('output'):
        os.remove('output')
    try:
        shutil.copy("in.lammps_{}".format(calcStep), "in.lammps")
        #TODO more systems
        dump_lmps(calcInd, 'data', symbol_to_type)
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('Lammps exited with exit code: %d.  ' % exitcode)
        struct = load_lmps('out.dump', type_to_symbol)[-1]
        volume = struct.get_volume()
        energy = float(os.popen("grep energy energy.out | tail -1 | awk '{print $3}'").readlines()[0])
        # the unit of pstress is kBar = GPa/10
        enthalpy = energy + pressure * GPa * volume / 10
        enthalpy = enthalpy / len(struct)
        struct.info['enthalpy'] = round(enthalpy, 3)
        # save energy, forces, stress for trainning potential
        struct.info['energy'] = energy
        return struct
    except:
        logging.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
        logging.warning("Lammps fail")
        return None


if  __name__ == "__main__":
    calcDic = yaml.load(open(sys.argv[1]))
    initPop = read('initPop.traj', index=':', format='traj')
    optPop = calc_lammps(**calcDic)
    write_traj('optPop.traj', optPop)
