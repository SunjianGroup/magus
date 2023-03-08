import os, subprocess, shutil, logging, copy, sys, yaml, traceback
import numpy as np
from ase.io import read, write
from ase.units import GPa, eV, Ang
from magus.calculators.base import ClusterCalculator
from magus.formatting.gulp import load_gulp, dump_gulp
from magus.utils import CALCULATOR_PLUGIN, check_parameters


log = logging.getLogger(__name__)


# units must be real!!
@CALCULATOR_PLUGIN.register('gulp')
class GulpCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = []
        Default={
            'exe_cmd': 'gulp < input > output',
            'job_prefix': 'Gulp',
            'shell': None,
            }
        check_parameters(self, parameters, Requirement, Default)

        self.gulp_setup = {
            'pressure': self.pressure,
            'exe_cmd': self.exe_cmd,
            'shell': self.shell,
        }
        self.main_info.append('gulp_setup')
        if not os.path.exists("{}/goption.scf".format(self.input_dir)):
            with open("{}/goption.scf".format(self.input_dir), 'w') as f:
                f.write('nosymmetry conp gradients\n')
        if not os.path.exists("{}/goption.relax".format(self.input_dir)):
            with open("{}/goption.relax".format(self.input_dir), 'w') as f:
                f.write('opti conjugate nosymmetry conp\n')

    def scf_job(self, index):
        self.cp_input_to('.')
        job_name = self.job_prefix + '_s_' + str(index)
        shutil.copy('goption.scf', 'goption')
        with open('gulpSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.gulp_setup))
        content = "python -m magus.calculators.gulp gulpSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh', out='scf-out', err='scf-err')

    def relax_job(self, index):
        self.cp_input_to('.')
        job_name = self.job_prefix + '_r_' + str(index)
        shutil.copy('goption.relax', 'goption')
        with open('gulpSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.gulp_setup))
        content = "python -m magus.calculators.gulp gulpSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='relax.sh', out='relax-out', err='relax-err')

    def scf_serial(self, calcPop):
        self.cp_input_to('.')
        shutil.copy('goption.scf', 'goption')
        opt_pop = calc_gulp(self.gulp_setup, calcPop)
        return opt_pop     

    def relax_serial(self, calcPop):
        self.cp_input_to('.')
        shutil.copy('goption.relax', 'goption')
        opt_pop = calc_gulp(self.gulp_setup, calcPop)
        return opt_pop


def calc_gulp(gulp_setup, frames):
    exe_cmd = gulp_setup['exe_cmd']
    pressure = gulp_setup['pressure']
    shell = gulp_setup['shell']
    new_frames = []
    for i, atoms in enumerate(frames):
        if os.path.exists('output'):
            os.remove('output')
        try:
            dump_gulp(atoms, 'structure', shell=shell)
            subprocess.call('cat goption structure gpot > input', shell=True)
            with open('input', 'a') as f:
                f.write('\npressure\n{}\n'.format(pressure))
            exitcode = subprocess.call(exe_cmd, shell=True)
            if exitcode != 0:
                raise RuntimeError('Gulp exited with exit code: %d.  ' % exitcode)
            new_atoms = load_gulp('output')
            atoms.info.update(new_atoms.info)
            new_atoms.info = atoms.info
            enthalpy = (new_atoms.info['energy'] + pressure * GPa * new_atoms.get_volume()) / len(new_atoms)
            new_atoms.info['enthalpy'] = round(enthalpy, 6)
            new_frames.append(new_atoms)
        except:
            log.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
            log.warning("GULP fail")
    return new_frames


if  __name__ == "__main__":
    gulp_setup_file, input_traj, output_traj = sys.argv[1:]
    gulp_setup = yaml.load(open(gulp_setup_file), Loader=yaml.FullLoader)

    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_gulp(gulp_setup, init_pop)
    write(output_traj, opt_pop)
