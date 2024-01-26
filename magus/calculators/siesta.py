import shutil, logging, copy, sys, yaml, os
from ase.io import read, write
from ase.units import GPa, eV, Ang, Ry
from magus.calculators.base import ClusterCalculator
from magus.utils import check_parameters
from ase.calculators.siesta import Siesta
from magus.utils import CALCULATOR_PLUGIN
import os

log = logging.getLogger(__name__)

#ASE doesnot do local relaxation...? Treated as append seting in 'preset.fdf'
class Relaxsiesta(Siesta):
    def set(self, **kwargs):
        if 'append_fdf' in kwargs:
            self.append_fdf = kwargs['append_fdf']
            del kwargs['append_fdf']
        Siesta.set(self, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        append_fdf = getattr(self, "append_fdf", None)
        
        Siesta.write_input(self, atoms, properties=properties, system_changes=system_changes)
        if not append_fdf is None:
            print("used preset settings in", append_fdf)
            filename = self.getpath(ext='fdf')
            os.system("mv {} {}1".format(filename, filename))
            os.system("cat {}1 {} > {}".format(filename, append_fdf, filename))
            os.system("rm {}1".format(filename))
        
@CALCULATOR_PLUGIN.register('siesta')
class siestaCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols']
        Default={
            'xc': 'LDA', 
            'job_prefix': 'siesta',
            }
        check_parameters(self, parameters, Requirement, Default)

        self.siesta_setup = {
            'xc': self.xc,
            'restart':None,
            'append_fdf': parameters.get('append_fdf', None),
            'atomic_coord_format': parameters.get('atomic_coord_format','xyz'), #'zmatrix'),    #support constraints by default?
            #it didn't work...?
            'pressure': 0}
        self.main_info.append('siesta_setup')

    def scf_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_s_' + str(index)
        with open('siestaSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.siesta_setup))
            f.write('scf: True')
        content = "python -m magus.calculators.siesta siestaSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh',
                   out='scf-out', err='scf-err')

    def relax_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_r_' + str(index)
        with open('siestaSetup.yaml', 'w') as f:
            f.write(yaml.dump(self.siesta_setup))
        content = "python -m magus.calculators.siesta siestaSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='relax.sh',
                   out='relax-out', err='relax-err')

    
    def scf_serial(self, calcPop):
        self.cp_input_to()
        calc = get_calc(self.siesta_setup)
        calc.set(nsw=0)
        opt_pop = calc_siesta(calc, calcPop)
        return opt_pop     

    def relax_serial(self, calcPop):
        self.cp_input_to()
        calc = get_calc(self.siesta_setup)
        opt_pop = calc_siesta(calc, calcPop)
        return opt_pop    


def calc_siesta(calc, frames, savetraj=False):
    new_frames = []
    for i, atoms in enumerate(frames):
        pbc = atoms.get_pbc()
        atoms.pbc = True
        atoms.set_calculator(copy.deepcopy(calc))
        #try:
        atoms.pbc=True
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        atoms_tmp = read('struct.XV')
        """
            '''
            if savetraj:
                # save relax trajectory
                traj = read('OUTCAR', index=':', format='siesta-out')
                # save relax steps
                log.debug('siesta relax steps: {}'.format(len(traj)))
                if 'relax_step' not in atoms.info:
                    atoms.info['relax_step'] = []
                else:
                    atoms.info['relax_step'] = list(atoms.info['relax_step'])
                atoms.info['relax_step'].append(len(traj))
            '''

        except:
            s = sys.exc_info()
            log.warning("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
            log.warning("siesta fail")
            continue
        """
        #I don't mind stress for now, someone fix it pls
        pstress = 0 #calc.float_params['pstress']
        
        volume = atoms.get_volume()
        # the unit of pstress is kBar = GPa / 10
        enthalpy = (energy + pstress * GPa * volume / 10) / len(atoms)
        atoms.set_pbc(pbc)
        atoms.info['enthalpy'] = enthalpy
        # save energy, forces, stress for trainning potential
        atoms.info['energy'] = energy
        atoms.info['forces'] = forces
        atoms.info['stress'] = stress

        # change to relaxed positions and cell
        atoms.set_positions(atoms_tmp.get_positions())
        atoms.set_cell(atoms_tmp.get_cell())

        # remove calculator becuase some strange error when save .traj
        atoms.set_calculator(None)
        log.debug("SIESTA finish")
        shutil.copy("struct.out", "struct.out-{}".format(i))
        shutil.copy("struct.XV", "struct.XV-{}".format(i))
        new_frames.append(atoms)
    return new_frames


def get_calc(siesta_setup):
    calc = Relaxsiesta(restart = siesta_setup.get('restart', None), label = 'struct')
    calc.set(append_fdf = siesta_setup.get('append_fdf', None), \
            xc=siesta_setup['xc'], \
            atomic_coord_format = siesta_setup['atomic_coord_format'])
            #pstress=siesta_setup['pressure'] * 10)
    return calc


if  __name__ == "__main__":
    siesta_setup_file, input_traj, output_traj = sys.argv[1:]
    siesta_setup = yaml.load(open(siesta_setup_file), Loader=yaml.FullLoader)
    calc = get_calc(siesta_setup)
    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_siesta(calc, init_pop)
    write(output_traj, opt_pop)
