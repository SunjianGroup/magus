import os, subprocess, shutil, logging, copy, sys, yaml, traceback
import numpy as np
from ase.io import read, write
from ase.units import GPa, eV, Ang
from magus.calculators.base import ClusterCalculator
from magus.formatting.orca import OrcaIo, RelaxOrca

log = logging.getLogger(__name__)

class OrcaCalculator(ClusterCalculator):
    def __init__(self, symbols, workDir, queueName, numCore, numParallel, jobPrefix='Orca',
                 pressure=0., Preprocessing='', waitTime=200, verbose=False, killtime=100000,
                 exeCmd='orca orca.inp > orca.out', *arg, **kwargs):
        super().__init__(workDir=workDir, queueName=queueName, numCore=numCore, 
                         numParallel=numParallel, jobPrefix=jobPrefix, pressure=pressure, 
                         Preprocessing=Preprocessing, waitTime=waitTime, 
                         verbose=verbose, killtime=killtime)
        self.orca_setup = {
            'pressure': pressure,
            'exe_cmd': exeCmd,
        }
        self.main_info.append('orca_setup')

    def prep(self):
        shutil.copy("{}/orcainput".format(self.input_dir), 'orcainput')
        shutil.copy("{}/orcablock".format(self.input_dir), 'orcablock')
        with open('orcaSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.orca_setup))
    
    def job(self, index, name):
        self.prep()
        job_name = self.job_prefix + '_{}_'.format(name[0]) + str(index)
        content = "python -m magus.calculators.orca orcaSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='{}.sh'.format(name),
                   out='{}-out'.format(name), err='{}-err'.format(name))

    def scf_job(self, index):
        self.job(index, 'scf')
    """
    It is still not clear to me how to calculate scf in orca and I will fix it in coming updates, maybe. ;) -yh
    """
    def relax_job(self, index):
        self.job(index, 'relax')

    def scf_serial(self, calcPop):
        self.prep()
        return calc_orca(self.orca_setup, calcPop)

    def relax_serial(self, calcPop):
        self.prep()
        return calc_orca(self.orca_setup, calcPop)

def calc_orca(orca_setup, frames):
    exe_cmd = orca_setup['exe_cmd']
    pressure = orca_setup['pressure']
    new_frames = []
    for i, atoms in enumerate(frames):
        try:
            with open("orcainput_{}".format(calcStep), 'r') as f:
                orcasimpleinput = f.readline()
            with open("orcablock_{}".format(calcStep), 'r') as f:
                orcablocks = f.readlines()
            label = exeCmd.split()
            assert label[1] [:-4] == label[3] [:-4] 
            orcaio = OrcaIo(label = label[1][:-4])
            orcaio.write_input(frames, orcasimpleinput = orcasimpleinput, orcablocks = orcablocks)

            exitcode = subprocess.call(exeCmd, shell=True)
            if exitcode != 0:
                raise RuntimeError('orca exited with exit code: %d.  ' % exitcode)
            numlist , positions = orcaio.read_positions()
            energy = orcaio.read_energy()
            forces = orcaio.read_forces()
            relaxsteps = orcaio.read_relaxsteps()
            log.debug('orca relax steps:{}'.format(relaxsteps))
            struct = Atoms(positions=positions, numbers=numlist, pbc = (0,0,0))
            struct.info = calcInd.info.copy()

            #Todo: how to add pressure here?
            #volume = struct.get_volume()
            enthalpy = energy #+ pressure * GPa * volume / 10
            enthalpy = enthalpy/len(struct)

            struct.info['enthalpy'] = round(enthalpy, 6)

            # save energy, forces, stress for trainning potential
            struct.info['energy'] = energy
            struct.info['forces'] = forces
            new_frames.append(struct)
        except:
            log.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
            log.warning("orca fail")
        
        shutil.copy('orca.out', "orca_out-{}".format(i))
        shutil.copy('orca.engrad', "orca_engrad-{}".format(i))
    return new_frames

if  __name__ == "__main__":
    orca_setup_file, input_traj, output_traj = sys.argv[1:]
    orca_setup = yaml.load(open(orca_setup_file), Loader=yaml.FullLoader)

    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_orca(orca_setup, init_pop)
    write(output_traj, opt_pop)
