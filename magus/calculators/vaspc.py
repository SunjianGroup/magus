from fileinput import filename
import shutil, logging, copy, sys, yaml, linecache, os
import numpy as np
from ase.io import read, write
from ase.units import GPa, eV, Ang
from magus.calculators.base import ClusterCalculator
from magus.utils import check_parameters
from ase.calculators.vasp import Vasp
from magus.utils import CALCULATOR_PLUGIN
from magus.formatting.vaspc import dump_vaspc

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


@CALCULATOR_PLUGIN.register('vaspc')
class VaspCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols']
        Default={
            'xc': 'PBE', 
            'pp_label': None, 
            'job_prefix': 'Vasp',
            'structureType': 'confined_2d'
            }
        check_parameters(self, parameters, Requirement, Default)

        pp_label = self.pp_label or [''] * len(self.symbols)
        self.vasp_setup = {
            'pp_setup': dict(zip(self.symbols, pp_label)),
            'xc': self.xc,
            'pressure': self.pressure,
            'restart':False,
            'structure_type':self.structureType}
        self.main_info.append('vasp_setup')

    def scf_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_s_' + str(index)
        with open('vaspSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.vasp_setup))
            f.write('scf: True')
        content = "python -m magus.calculators.vaspc vaspSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh',
                   out='scf-out', err='scf-err')

    def relax_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_r_' + str(index)
        with open('vaspSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.vasp_setup))
        content = "python -m magus.calculators.vaspc vaspSetup.yaml initPop.traj optPop.traj"
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
        pbc = atoms.get_pbc()
        atoms.pbc = True
        dump_vaspc(atoms, vasp_setup)
        calc2 = copy.deepcopy(calc)
        calc2.read_kpoints('KPOINTS')
        atoms.set_calculator(calc2)
        try:
            atoms.pbc=True
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            # get the energy without PV becaruse new ase version gives enthalpy, should be removed if ase fix the bug
            atoms_tmp = read('OUTCAR', format='vasp-out')
            energy = atoms_tmp.get_potential_energy()
            direct_gap, indirect_gap = read_eigen()
        except:
            s = sys.exc_info()
            log.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
            log.warning("VASP fail")
            continue
        pstress = calc.float_params['pstress']
#        pstress = 0
        volume = atoms.get_volume()
        # the unit of pstress is kBar = GPa / 10
        enthalpy = (energy + pstress * GPa * volume / 10) / len(atoms)
        atoms.set_pbc(pbc)
        atoms.info['direct_gap'] = direct_gap
        atoms.info['indirect_gap'] = indirect_gap
        atoms.info['enthalpy'] = enthalpy
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
    calc = RelaxVasp(restart = vasp_setup['restart'])
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
