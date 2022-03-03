import os, subprocess, shutil, logging, copy, sys, yaml
import numpy as np
from ase.io import read, write
from ase.units import GPa, eV, Ang
from magus.calculators.base import ClusterCalculator
from magus.utils import check_parameters
from ase.calculators.vasp import Vasp
from magus.utils import CALCULATOR_PLUGIN


log = logging.getLogger(__name__)


class RelaxVasp(Vasp):
    """
    Slightly modify ASE's Vasp Calculator so that it will never check relaxation convergence.
    """
    def read_relaxed(self):
        return True


@CALCULATOR_PLUGIN.register('vasp')
class VaspCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols']
        Default={
            'xc': 'PBE', 
            'pp_label': None, 
            'job_prefix': 'Vasp',
            }
        check_parameters(self, parameters, Requirement, Default)

        pp_label = self.pp_label or [''] * len(self.symbols)
        self.vasp_setup = {
            'pp_setup': dict(zip(self.symbols, pp_label)),
            'xc': self.xc,
            'pressure': self.pressure}
        self.main_info.append('vasp_setup')

    def scf_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_s_' + str(index)
        with open('vaspSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.vasp_setup))
            f.write('scf: True')
        content = "python -m magus.calculators.vasp vaspSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh',
                   out='scf-out', err='scf-err')

    def relax_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_r_' + str(index)
        with open('vaspSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.vasp_setup))
        content = "python -m magus.calculators.vasp vaspSetup.yaml initPop.traj optPop.traj"
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


def calc_vasp(calc, frames):
    new_frames = []
    for i, atoms in enumerate(frames):
        atoms.set_calculator(copy.deepcopy(calc))
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            # get the energy without PV becaruse new ase version gives enthalpy, should be removed if ase fix the bug
            atoms_tmp = read('OUTCAR', format='vasp-out')
            energy = atoms_tmp.get_potential_energy()
        except:
            s = sys.exc_info()
            log.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
            log.warning("VASP fail")
            break
        pstress = calc.float_params['pstress']
        volume = atoms.get_volume()
        # the unit of pstress is kBar = GPa / 10
        enthalpy = (energy + pstress * GPa * volume / 10) / len(atoms)
        # struct.info['gap'] = round(gap, 3)
        atoms.info['enthalpy'] = round(enthalpy, 3)
        # save energy, forces, stress for trainning potential
        atoms.info['energy'] = energy
        atoms.info['forces'] = forces
        atoms.info['stress'] = stress
        # save relax trajectory
        traj = read('OUTCAR', index=':', format='vasp-out')
        # save relax steps
        log.debug('vasp relax steps: {}'.format(len(traj)))
        if 'relax_step' not in atoms.info:
            atoms.info['relax_step'] = []
        else:
            atoms.info['relax_step'] = list(atoms.info['relax_step'])
        atoms.info['relax_step'].append(len(traj))
        # remove calculator becuase some strange error when save .traj
        atoms.set_calculator(None)
        log.debug("VASP finish")
        shutil.copy("OUTCAR", "OUTCAR-{}".format(i))
        new_frames.append(atoms)
    return new_frames


def get_calc(vasp_setup):
    calc = RelaxVasp()
    calc.read_incar('INCAR')
    calc.set(xc=vasp_setup['xc'])
    calc.set(setups=vasp_setup['pp_setup'])
    calc.set(pstress=vasp_setup['pressure'] * 10)
    calc.set(lwave=False)
    calc.set(lcharg=False)
    if 'scf' in vasp_setup.keys():
        calc.set(nsw=0)
    return calc


if  __name__ == "__main__":
    vasp_setup_file, input_traj, output_traj = sys.argv[1:]
    vasp_setup = yaml.load(open(vasp_setup_file), Loader=yaml.FullLoader)
    calc = get_calc(vasp_setup)
    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_vasp(calc, init_pop)
    write(output_traj, opt_pop)
