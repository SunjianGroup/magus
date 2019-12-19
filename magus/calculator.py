from __future__ import print_function, division
from ase import Atoms
import ase.io
from .readvasp import *
import sys
import math
import os
import shutil
import subprocess
import logging
import copy
import yaml
from ase.calculators.lj import LennardJones
from ase.calculators.vasp import Vasp
from ase.spacegroup import crystal
# from parameters import parameters
from .writeresults import write_yaml, read_yaml, write_traj
from .utils import *
from ase.units import GPa


class calculator:
    def __init__(self,parameters):
        Requirement=['symbols','formula','numFrml']
        Default={'threshold':1.0,'maxAttempts':50,'method':2,'volRatio':1.5,'spgs':np.arange(1,231),'maxtryNum':100}

        for key in Requirement:
            if not hasattr(self.p, key):
                raise Exception("Mei you '{}' wo suan ni ma?".format(key))
            setattr(self,key,getattr(self.p,key))

        for key in Default.keys():
            if not hasattr(self.p,key):
                setattr(self,key,Default[key])
            else:
                setattr(self,key,getattr(self.p,key))
        self.workDir = parameters['workDir']
        queueName = parameters['queueName']
        numParallel = parameters['numParallel']
        numCore = parameters['numCore']
        xc = parameters['xc']
        pressure = parameters['pressure']
        symbols = parameters['symbols']
        ppLabel = parameters['ppLabel']
        maxRelaxTime = parameters['maxRelaxTime']
        jobPrefix = parameters['jobPrefix']


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
        for i, ind in enumerate(structs):
            initInd = ind.copy()
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

    def calc_vasp_parallel(self, calcPop, J, prefix='calcVasp'):

        vaspSetup = dict(zip(symbols, ppLabel))

        popLen = len(calcPop)
        eachLen = popLen//numParallel
        remainder = popLen%numParallel

        runArray = []
        for i in range(numParallel):
            tmpList = [ i + numParallel*j for j in range(eachLen)]
            if i < remainder:
                tmpList.append(numParallel*eachLen + i)
            runArray.append(tmpList)

        runJobs = []
        for i in range(numParallel):
            if not os.path.exists("{}{}".format(prefix, i)):
                os.mkdir("{}{}".format(prefix, i))
            os.chdir("{}{}".format(prefix, i))

            tmpPop = [calcPop[j] for j in runArray[i]]
            write_traj('initPop.traj', tmpPop)
            for j in range(1, calcNum + 1):
                shutil.copy("{}/inputFold/INCAR_{}".format(workDir, j), 'INCAR_{}'.format(j))

            with open('vaspSetup.yaml', 'w') as setupF:
                setupF.write(yaml.dump(vaspSetup))

            f = open('parallel.sh', 'w')
            f.write("#BSUB -q %s\n"
                    "#BSUB -n %s\n"
                    "#BSUB -o out\n"
                    "#BSUB -e err\n"
                    "#BSUB -W %s\n"
                    "#BSUB -J Vasp_%s\n"% (queueName, numCore, maxRelaxTime*len(tmpPop), i))
            f.write("{}\n".format(jobPrefix))
            f.write("python -m magus.runvasp {} {} vaspSetup.yaml {} initPop.traj optPop.traj\n".format(calcNum, xc, pressure))
            f.close()

            J.bsub('bsub < parallel.sh')
            os.chdir("%s/calcFold" %(workDir))

