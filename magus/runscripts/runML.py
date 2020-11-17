import sys, os,shutil,subprocess,logging,traceback
import yaml
import ase.io
from ase.spacegroup import crystal
from magus.writeresults import write_traj
from magus.parameters import magusParameters
from ase.constraints import ExpCellFilter
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from magus.utils import *
from ase.units import GPa, eV, Ang
import copy


if  __name__ == "__main__":
    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format="%(asctime)s  %(message)s",datefmt='%H:%M:%S')

    calcDic = yaml.load(open(sys.argv[1]))
    calcNum = calcDic['calcNum']
    pressure = calcDic['pressure']
    workDir = calcDic['workDir']
    maxRelaxStep = calcDic['maxRelaxStep']
    logfile = calcDic['logfile']
    trajname = calcDic['trajname']
    epsArr = calcDic['epsArr'] 
    stepArr = calcDic['stepArr'] 
    relaxLattice = calcDic['relaxLattice'] 
    optimizer = calcDic['optimizer']
    calcPop = ase.io.read('initPop.traj', index=':', format='traj')
    logging.info('read initPop')

    p = magusParameters('{}/input.yaml'.format(workDir))
    mlc = p.get_MLCalculator()
    mlc.p.mlDir = '{}/mlFold'.format(workDir)
    mlc.load_model('para')
    logging.info('load model')
    calc = mlc.get_calculator()
    if calcNum == 0:
        for n, ind in enumerate(calcPop):
            scfPop = []
            for ind in calcPop:
                atoms=copy.deepcopy(ind)
                atoms.set_calculator(calc)
                try:
                    atoms.info['energy'] = atoms.get_potential_energy()
                    atoms.info['forces'] = atoms.get_forces()
                    try:
                        atoms.info['stress'] = atoms.get_stress()
                    except:
                        pass
                    enthalpy = (atoms.info['energy'] + pressure * atoms.get_volume() * GPa)/len(atoms)
                    atoms.info['enthalpy'] = round(enthalpy, 3)
                    atoms.set_calculator(None)
                    scfPop.append(atoms)
                    logging.debug('{} scf steps: 0'.format(self.__class__.__name__))
                except:
                    pass
        write_traj('optPop.traj', scfPop)
    else:
        relaxPop = []
        errorPop = []
        for i, ind in enumerate(calcPop):
            for j in range(calcNum):
                ind.set_calculator(calc)
                logging.debug("Structure {} Step {}".format(i, j))
                if relaxLattice:
                    ucf = ExpCellFilter(ind, scalar_pressure=pressure*GPa)
                else:
                    ucf = ind
                if optimizer == 'cg':
                    gopt = SciPyFminCG(ucf, logfile=logfile,trajectory=trajname)
                elif optimizer == 'bfgs':
                    gopt = BFGS(ucf, logfile=logfile, maxstep=maxRelaxStep,trajectory=trajname)
                elif optimizer == 'lbfgs':
                    gopt = LBFGS(ucf, logfile=logfile, maxstep=maxRelaxStep,trajectory=trajname)
                elif optimizer == 'fire':
                    gopt = FIRE(ucf, logfile=logfile, maxmove=maxRelaxStep,trajectory=trajname)
                try:
                    label = gopt.run(fmax=epsArr[j], steps=stepArr[j])
                    traj = ase.io.read(trajname,':')
                    # save relax steps
                    logging.debug('{} relax steps: {}'.format(mlc.__class__.__name__,len(traj)))
                except Converged:
                    pass
                except TimeoutError:
                    errorPop.append(ind)
                    logging.warning("Calculator:{} relax Timeout".format(mlc.__class__.__name__))
                    continue
                except:
                    errorPop.append(ind)
                    logging.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
                    logging.warning("Calculator:{} relax fail".format(mlc.__class__.__name__))
                    continue

            else:
                ind.info['energy'] = ind.get_potential_energy()
                ind.info['forces'] = ind.get_forces()
                try:
                    ind.info['stress'] = ind.get_stress()
                except:
                    pass
                enthalpy = (ind.info['energy'] + pressure * ind.get_volume() * GPa)/len(ind)
                ind.info['enthalpy'] = round(enthalpy, 3)
                ind.wrap()
                ind.set_calculator(None)
                relaxPop.append(ind)
        write_traj('optPop.traj', relaxPop)
