import os, subprocess, shutil, logging, copy, sys, yaml, traceback
import numpy as np
from ase.io import read, write
from ase.units import GPa, eV, Ang
from magus.calculators.base import ClusterCalculator
from magus.formatting.espresso import load_espresso, dump_espresso
from magus.utils import CALCULATOR_PLUGIN, check_parameters


log = logging.getLogger(__name__)


# units must be real!!
@CALCULATOR_PLUGIN.register('espresso')
class EspressoCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols']
        Default={
            'exe_cmd': 'mpirun pw.x < input > output',
            'job_prefix': 'espresso',
            'kmesh': 0.03,
            'pp_label': None
            }
        check_parameters(self, parameters, Requirement, Default)
        pp_label = self.pp_label or [''] * len(self.symbols)  
        self.espresso_setup = {
            'pp_setup': dict(zip(self.symbols, pp_label)),
            'pressure': self.pressure,
            'exe_cmd': self.exe_cmd,
            'kmesh': self.kmesh,
        }
        self.main_info.append('espresso_setup')
        
    def scf_job(self, index):
        self.cp_input_to('.')
        job_name = self.job_prefix + '_s_' + str(index)
        shutil.copy('pw.scf', 'pwi')
        with open('espressoSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.espresso_setup))
        content = "python -m magus.calculators.espresso espressoSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh', out='scf-out', err='scf-err')

    def relax_job(self, index):
        self.cp_input_to('.')
        job_name = self.job_prefix + '_r_' + str(index)
        shutil.copy('pw.relax', 'pwi')
        with open('espressoSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.espresso_setup))
        content = "python -m magus.calculators.espresso espressoSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='relax.sh', out='relax-out', err='relax-err')

    def scf_serial(self, calcPop):
        self.cp_input_to('.')
        shutil.copy('pw.scf', 'pwi')
        opt_pop = calc_espresso(self.espresso_setup, calcPop)
        return opt_pop     

    def relax_serial(self, calcPop):
        self.cp_input_to('.')
        shutil.copy('pw.relax', 'pwi')
        opt_pop = calc_espresso(self.espresso_setup, calcPop)
        return opt_pop


def calc_espresso(espresso_setup, frames):
    exe_cmd = espresso_setup['exe_cmd']
    pressure = espresso_setup['pressure']
    new_frames = []
    for i, atoms in enumerate(frames):
        if os.path.exists('output'):
            os.remove('output')
        try:
            dump_espresso(atoms, espresso_setup, 'pwi2')
            subprocess.call('cat pwi pwi2 > input', shell=True)
            exitcode = subprocess.call(exe_cmd, shell=True)
            if exitcode != 0:
                raise RuntimeError('espresso exited with exit code: %d.  ' % exitcode)
            new_atoms = load_espresso('output')
            atoms.info.update(new_atoms.info)
            new_atoms.info = atoms.info
            enthalpy = (new_atoms.info['energy'] + pressure * GPa * new_atoms.get_volume()) / len(new_atoms)
            new_atoms.info['enthalpy'] = round(enthalpy, 6)
            new_frames.append(new_atoms)
        except:
            log.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
            log.warning("espresso fail")
    return new_frames


if  __name__ == "__main__":
    espresso_setup_file, input_traj, output_traj = sys.argv[1:]
    espresso_setup = yaml.load(open(espresso_setup_file), Loader=yaml.FullLoader)

    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_espresso(espresso_setup, init_pop)
    write(output_traj, opt_pop)
