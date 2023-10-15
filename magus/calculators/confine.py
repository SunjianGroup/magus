from fileinput import filename
import shutil, logging, copy, sys, yaml, linecache, os, traceback, subprocess
import numpy as np
from ase.io import read, write
from ase.units import GPa, eV, Ang
from magus.calculators.base import ClusterCalculator
from magus.utils import check_parameters
from ase.calculators.vasp import Vasp
from magus.utils import CALCULATOR_PLUGIN
from magus.formatting.vaspc import dump_vaspc,load_incar

log = logging.getLogger(__name__)


def read_eigen(filename='EIGENVAL'):
    linecache.clearcache()
    l6 = linecache.getline(filename, 6).split()
    eN = int(l6[0])
    filled_band = int(eN/2)
    n_kpoint = int(l6[1])
    n_band = int(l6[2])
    start, end = np.zeros((2, n_kpoint))
    for i in range(n_kpoint):
        line = i * (n_band + 2) + filled_band + 8
        start[i] = float(linecache.getline(filename, line).split()[1])
        end[i] = float(linecache.getline(filename, line + 1).split()[1])
    indirect_gap = max(np.min(end) - np.max(start), 0)
    direct_gap = max(np.min(end - start), 0)
    return direct_gap, indirect_gap


class RelaxVasp(Vasp):
    """
    Slightly modify ASE's Vasp Calculator so that it will never check relaxation convergence.
    """
    def read_relaxed(self):
        return True


@CALCULATOR_PLUGIN.register('confine')
class VaspCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols']
        Default={
            'xc': 'PBE',
            'pp_label': None,
            'ppDir': '~/POT/',
            'LJ_D': None,
            'LJ_A': None,
            'LJ_W': None,
            'job_prefix': 'Vasp',
            'structureType': 'confined_2d',
            'exe_cmd': 'mpirun vasp_std',
            }
        check_parameters(self, parameters, Requirement, Default)

        pp_label = self.pp_label or [''] * len(self.symbols)
        self.vasp_setup = {
            'pp_setup': dict(zip(self.symbols, pp_label)),
            'LJD': dict(zip(self.symbols, self.LJ_D)),
            'LJA': dict(zip(self.symbols, self.LJ_A)),
            'LJW': dict(zip(self.symbols, self.LJ_W)),
            'pp_dir':self.ppDir,
            'xc': self.xc,
            'pressure': self.pressure,
            'structure_type': self.structureType,
            'exe_cmd':self.exe_cmd,
            'restart':False}
        self.main_info.append('vasp_setup')

    def scf_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_s_' + str(index)
        with open('vaspSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.vasp_setup))
            f.write('scf: True')
        content = "python -m magus.calculators.confine vaspSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh',
                   out='scf-out', err='scf-err')

    def relax_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_r_' + str(index)
        with open('vaspSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.vasp_setup))
        content = "python -m magus.calculators.confine vaspSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='relax.sh',
                   out='relax-out', err='relax-err')

    def scf_serial(self, calcPop):
        self.cp_input_to()
        calc = get_calc(self.vasp_setup)
        calc.set(nsw=0)
        opt_pop = calc_vasp(calc, calcPop)
        return opt_pop     

    def relax_serial(self, calcPop):
        self.cp_input_to()
        calc = get_calc(self.vasp_setup)
        opt_pop = calc_vasp(calc, calcPop)
        return opt_pop    

def calc_vasp(vasp_setup, frames):
    exe_cmd = vasp_setup['exe_cmd']
    pressure = vasp_setup['pressure']
    incar = load_incar()
    new_frames = []
    for i, atoms in enumerate(frames):
        atoms.pbc = True
        dump_vaspc(atoms, vasp_setup, incar)
        try:
            exitcode = subprocess.call(exe_cmd, shell=True)
            if exitcode != 0:
                raise RuntimeError('vasp exited with exit code: %d.  ' % exitcode)
            atoms_tmp = read('OUTCAR', format='vasp-out')
            new_atoms = atoms_tmp.copy()
            energy = atoms_tmp.get_potential_energy()
            forces = atoms_tmp.get_forces()
            stress = atoms_tmp.get_stress()
            direct_gap, indirect_gap = read_eigen()
            new_atoms.info['direct_gap'] = direct_gap
            new_atoms.info['indirect_gap'] = indirect_gap
            new_atoms.info['energy'] = energy
            new_atoms.info['forces'] = forces
            new_atoms.info['stress'] = stress
            enthalpy = (new_atoms.info['energy'] + pressure * GPa * new_atoms.get_volume()) / len(new_atoms)
            new_atoms.info['enthalpy'] = round(enthalpy, 6)
            traj = read('OUTCAR', index=':', format='vasp-out')
            # save relax steps
            log.debug('vasp relax steps: {}'.format(len(traj)))
            if 'relax_step' not in new_atoms.info:
                new_atoms.info['relax_step'] = []
            else:
                new_atoms.info['relax_step'] = list(new_atoms.info['relax_step'])
            new_atoms.info['relax_step'].append(len(traj))
            log.debug("VASP finish")
            shutil.copy("OUTCAR", "OUTCAR-{}".format(i))
            new_frames.append(new_atoms)
        except:
            log.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
            log.warning("vasp fail")
    return new_frames

if  __name__ == "__main__":
    vasp_setup_file, input_traj, output_traj = sys.argv[1:]
    vasp_setup = yaml.load(open(vasp_setup_file), Loader=yaml.FullLoader)
    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_vasp(vasp_setup, init_pop)
    write(output_traj, opt_pop)
