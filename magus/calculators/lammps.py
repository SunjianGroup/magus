import os, subprocess, shutil, logging, copy, sys, yaml
import numpy as np
from ase.io import read, write
from ase.units import GPa, eV, Ang
from magus.calculators.base import ClusterCalculator
from magus.utils import CALCULATOR_PLUGIN, check_parameters
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
#TODO: return None

# units must be metal!!
@CALCULATOR_PLUGIN.register('lammps')
class LammpsCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols']
        Default={
            'exe_cmd': '', 
            'save_traj': False, 
            'atomStyle': 'atomic',
            'job_prefix': 'Lammps',
            }
        check_parameters(self, parameters, Requirement, Default)
        self.lammps_setup = {
            'pressure': self.pressure,
            'symbols': self.symbols,
            'atom_style': self.atom_style,
            'exe_cmd': self.exe_cmd,
            'save_traj': self.save_traj,
        }
        self.main_info.append('lammps_setup')

    def scf_job(self, index):
        self.cp_input_to()
        shutil.copy('in.scf', 'in.lammps')
        job_name = self.job_prefix + '_s_' + str(index)
        with open('lammpsSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.lammps_setup))
        content = "python -m magus.calculators.lammps lammpsSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh', out='scf-out', err='scf-err')

    def relax_job(self, index):
        self.cp_input_to()
        shutil.copy('in.relax', 'in.lammps')
        job_name = self.job_prefix + '_r_' + str(index)
        with open('lammpsSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.lammps_setup))
        content = "python -m magus.calculators.lammps lammpsSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='relax.sh', out='relax-out', err='relax-err')

    def scf_serial(self, calcPop):
        self.cp_input_to()
        shutil.copy('in.scf', 'in.lammps')
        opt_pop = calc_lammps(self.lammps_setup, calcPop)
        return opt_pop     

    def relax_serial(self, calcPop):
        self.cp_input_to()
        shutil.copy('in.relax', 'in.lammps')
        opt_pop = calc_lammps(self.lammps_setup, calcPop)
        return opt_pop


def calc_lammps_once(lammps_setup, atoms):
    specorder = lammps_setup['symbols']
    atom_style = lammps_setup['atom_style']
    exe_cmd = lammps_setup['exe_cmd']
    pressure = lammps_setup['pressure']
    save_traj = lammps_setup['save_traj']
    write_lammps_data('data', atoms, specorder=specorder, atom_style=atom_style)
    exitcode = subprocess.call(exe_cmd, shell=True)
    if exitcode != 0 and exitcode != 8:
        raise RuntimeError('Lammps exited with exit code: %d.  ' % exitcode)
    # break because of MTP
    if exitcode == 8:
        return None
    with open('out.dump') as f:
        new_atoms = read_lammps_dump_text(f, specorder=specorder)
    thermo_content = []
    if not os.path.exists('log.lammps'):
        raise RuntimeError('Lammps failed, no log.lammps!')
    with open('log.lammps') as f:
        line = 'chongchongchong!'
        while line:
            line = f.readline()
            if 'Error' in line:
                raise RuntimeError('Lammps failed, please check log.lammps!')
            if 'Step Temp Press' in line:
                thermo_args = line.split()
                line = f.readline()
                while 'Loop time of' not in line:
                    thermo_content.append(
                        {arg: float(value) for arg, value in zip(thermo_args, line.split())}
                    )
                    line = f.readline()
                break
    energy = thermo_content[-1]['PotEng']
    enthalpy = (energy + pressure * GPa * new_atoms.get_volume()) / len(new_atoms)
    new_atoms.info['enthalpy'] = round(enthalpy, 6)
    new_atoms.info['energy'] = energy
    new_atoms.info['forces'] = new_atoms.get_forces()
    new_atoms.info['stress'] = np.array(
        [-thermo_content[-1][arg] for arg in ("Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy")]) * 1e-4
    if save_traj:
        with open('out.dump') as f:
            traj = read_lammps_dump_text(f, index=slice(None, None, None), specorder=specorder)
        for i, atoms in enumerate(traj):
            energy = thermo_content[i]['PotEng']
            enthalpy = (energy + pressure * GPa * atoms.get_volume()) / len(atoms)
            new_atoms.info['enthalpy'] = round(enthalpy, 6)
            new_atoms.info['energy'] = energy
            atoms.info['forces'] = atoms.get_forces()
            atoms.info['stress'] = np.array(
                [-thermo_content[i][arg] for arg in ("Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy")]) * 1e-4
        new_atoms.info['traj'] = traj
    return new_atoms


def calc_lammps(lammps_setup, frames):
    new_frames = []
    for i, atoms in enumerate(frames):
        new_atoms = calc_lammps_once(lammps_setup, atoms)
        if new_atoms is not None:
            new_frames.append(new_atoms)
    return new_frames


if  __name__ == "__main__":
    lammps_setup_file, input_traj, output_traj = sys.argv[1:]
    lammps_setup = yaml.load(open(lammps_setup_file), Loader=yaml.FullLoader)

    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_lammps(lammps_setup, init_pop)
    write(output_traj, opt_pop)
