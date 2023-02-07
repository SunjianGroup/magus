import os
import logging
import copy
import sys
import yaml

from ase.io import read, write
from ase.calculators.castep import Castep

from magus.calculators.base import ClusterCalculator
from magus.utils import check_parameters
from magus.utils import CALCULATOR_PLUGIN


log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('castep')
class CastepCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols', ]
        Default = {
            'xc_functional': 'PBE',
            'pspot': '00PBE',
            'suffix': 'usp',
            'job_prefix': 'Castep',
            'kpts': "{'density': 10, 'gamma': True, 'even': False}",
            'castep_command': 'castep',
            'castep_pp_path': None,
        }
        check_parameters(self, parameters, Requirement, Default)

        self.castep_setup = {
            'pressure': self.pressure,
        }
        self.castep_setup.update(parameters)

    def scf_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_s_' + str(index)
        with open('CastepSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.castep_setup))
            f.write('scf: True')
        content = "python -m magus.calculators.castep CastepSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh',
                   out='scf-out', err='scf-err')

    def relax_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_r_' + str(index)
        with open('CastepSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.castep_setup))
        content = "python -m magus.calculators.castep CastepSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='relax.sh',
                   out='relax-out', err='relax-err')


def calc_castep(castep_setup, frames):
    new_frames = []
    for i, atoms in enumerate(frames):
        calc = Castep(
            castep_command=castep_setup['castep_command'], castep_pp_path=castep_setup['castep_pp_path'])
        atoms.pbc = True
        atoms.calc = calc
        calc._label = "magus_castep_job"
        # read param_file
        param_file = None
        for filename in os.listdir("."):
            if filename.endswith(".param"):
                param_file = filename
                break
        if param_file == None:
            raise Exception("Cannot find .param file!")
        calc.find_pspots(
            pspot=castep_setup['pspot'], suffix=castep_setup['suffix'])
        calc.set_kpts(castep_setup['kpts'])
        calc.param.xc_functional = castep_setup['xc_functional']
        calc.merge_param(param_file)
        # write pressure (hydrostatic pressure, pxx = pyy = pzz)
        p = str(castep_setup['pressure'])
        calc.cell.external_pressure = f"{p} 0 0\n {p} 0 \n {p}"
        if 'scf' in castep_setup.keys():
            calc.param.task = "SinglePoint"
        else:
            calc.param.task = "GeometryOptimization"
        pbc = atoms.get_pbc()

        try:
            atoms.pbc = True
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
        except:
            s = sys.exc_info()
            log.warning("Error '%s' happened on line %d" %
                        (s[1], s[2].tb_lineno))
            log.warning("Castep fail")
            continue
        # unit already converted in ase
        pressure = stress[:3].mean()
        volume = atoms.get_volume()
        enthalpy = (energy + pressure * volume / 10) / len(atoms)
        atoms.set_pbc(pbc)
        atoms.info['enthalpy'] = enthalpy
        # save energy, forces, stress for trainning potential
        atoms.info['energy'] = energy
        atoms.info['forces'] = forces
        atoms.info['stress'] = stress
        # save relax trajectory
        traj = read(f'{calc._label}.geom', index=':', format='castep-geom')
        # save relax steps
        log.debug('castep relax steps: {}'.format(len(traj)))
        if 'relax_step' not in atoms.info:
            atoms.info['relax_step'] = []
        else:
            atoms.info['relax_step'] = list(atoms.info['relax_step'])
        atoms.info['relax_step'].append(len(traj))
        # remove calculator becuase some strange error when save .traj
        atoms.set_calculator(None)
        log.debug("Castep finish.")
        new_frames.append(atoms)
    return new_frames


if __name__ == "__main__":
    castep_setup_file, input_traj, output_traj = sys.argv[1:]
    castep_setup = yaml.load(open(castep_setup_file), Loader=yaml.FullLoader)
    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_castep(castep_setup, init_pop)
    write(output_traj, opt_pop)
