import os, logging, shutil, copy
from pathlib import Path
import numpy as np
from magus.calculators.base import ClusterCalculator, ASECalculator
from ase.io import read, write
from ase.calculators.calculator import Calculator
from ase.units import GPa, eV, Ang
from ase.atoms import Atoms
from magus.utils import CALCULATOR_PLUGIN, check_parameters
from magus.populations.populations import Population
from deepmd import DeepPotential
from dpdata import MultiSystems, LabeledSystem


log = logging.getLogger(__name__)


class ASEDPData(MultiSystems):
    def from_traj(self, frames):
        if isinstance(frames, Atoms):
            frames = [frames]
        for atoms in frames:
            symbols = atoms.get_chemical_symbols()
            atom_names = list(set(symbols))
            atom_numbs = [symbols.count(symbol) for symbol in atom_names]
            atom_types = np.array([atom_names.index(symbol) for symbol in symbols]).astype(int)

            cells = atoms.cell[:]
            coords = atoms.get_positions()
            energies = atoms.info['energy']
            forces = atoms.info['forces']
            info_dict = {
                'atom_names': atom_names,
                'atom_numbs': atom_numbs,
                'atom_types': atom_types,
                'cells': np.array([cells]).astype('float32'),
                'coords': np.array([coords]).astype('float32'),
                'energies': np.array([energies]).astype('float32'),
                'forces': np.array([forces]).astype('float32'),
                'orig': [0,0,0],
            }
            if 'stress' in atoms.info:
                stress = atoms.info['stress']
                if stress.size == 6:
                    xx, yy, zz, yz, xz, xy = stress
                    stress = np.array([(xx, xy, xz),
                                       (xy, yy, yz),
                                       (xz, yz, zz)])
                virials = np.array([-atoms.get_volume() * stress]).astype('float32')
                info_dict['virials'] = virials
            system = LabeledSystem(data=info_dict)
            system.sort_atom_names()
            self.append(system)


class DPEnsemble(Calculator):
    name = "DPEnsemble"
    implemented_properties = ["energy", "forces", "stress", "max_force_std"]
    def __init__(self, model, type_dict=None, n_ensemble=1, keep_prob=1.):
        Calculator.__init__(self)
        self.keep_prob = keep_prob
        self.n_ensemble = n_ensemble
        self.dp = DeepPotential(str(Path(model).resolve()))
        if type_dict:
            self.type_dict = type_dict
        else:
            self.type_dict = dict(
                zip(self.dp.get_type_map(), range(self.dp.get_ntypes()))
            )

    def calculate(self, atoms, *arg, **kwargs):
        if atoms is not None:
            self.atoms = atoms.copy()
        coord = self.atoms.get_positions().reshape([1, -1])
        if sum(self.atoms.get_pbc()) > 0:
            cell = self.atoms.get_cell().reshape([1, -1])
        else:
            cell = None
        symbols = self.atoms.get_chemical_symbols()
        atype = [self.type_dict[k] for k in symbols]
        e, f, v = self.dp.eval(coords=coord, cells=cell, atom_types=atype, keep_prob=1.)
        self.results["energy"] = e[0][0]
        self.results["forces"] = f[0]
        if sum(atoms.get_pbc()) > 0:
            stress = -v[0].reshape(3, 3) / self.atoms.get_volume()
            stress = 0.5 * (stress.copy() + stress.copy().T)
            self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]

        f = np.zeros((self.n_ensemble, 1, len(self.atoms), 3))
        for i in range(self.n_ensemble):
            f[i] = self.dp.eval(coords=coord, cells=cell, atom_types=atype, keep_prob=self.keep_prob)[1]
        sf = np.std(f, axis=0)[0]
        self.results["max_force_std"] = sf.max()
        atoms.info['max_force_std'] = self.results["max_force_std"]


@CALCULATOR_PLUGIN.register('dp')
class DPCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols']
        Default = {
            'model_dir': self.input_dir,
            'n_ensemble': 1,
            'keep_prob': 1.0,
        }
        check_parameters(self, parameters, Requirement, Default)
        type_dict = {j: i for i, j in enumerate(self.symbols)}
        model_path="{}/graph.pb".format(self.model_dir)
        self.relax_calc = self.scf_calc = DPEnsemble(model_path, type_dict, self.n_ensemble, self.keep_prob)


@CALCULATOR_PLUGIN.register('dp-otf')
class OTFDPCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['query_calculator', 'symbols']
        Default={
            'break_threshold': 2.0,
            'record_threshold': 0.5,
            'job_prefix': 'DP',
            'select_ratio': 0.5,
            'n_epoch': 200, 
            'n_ensemble': 5,
            'keep_prob': 0.9,
            }
        check_parameters(self, parameters, Requirement, Default)
        self.ml_dir = "{}/mlFold/{}".format(self.work_dir, self.job_prefix)
        os.makedirs(self.ml_dir, exist_ok=True)

        if not os.path.exists('{}/input.json'.format(self.ml_dir)):
            shutil.copy('{}/input.json'.format(self.input_dir), 
                        '{}/input.json'.format(self.ml_dir))
        if not os.path.exists('{}/graph.pb'.format(self.ml_dir)):
            if os.path.exists('{}/graph.pb'.format(self.input_dir)):
                shutil.copy('{}/graph.pb'.format(self.input_dir), 
                            '{}/graph.pb'.format(self.ml_dir))
        if not os.path.exists('{}/data.traj'.format(self.ml_dir)):
            if os.path.exists('{}/data.traj'.format(self.input_dir)):
                shutil.copy('{}/data.traj'.format(self.input_dir), 
                            '{}/data.traj'.format(self.ml_dir))
                self.trainset = read('{}/data.traj'.format(self.ml_dir), ':')
            else:
                self.trainset = []
        else:
            self.trainset = read('{}/data.traj'.format(self.ml_dir), ':')
        self.dpdata = ASEDPData(type_map=self.symbols)
        self.dpdata.from_traj(self.trainset)

    def train(self, epoch=None):
        epoch = epoch or self.n_epoch
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        content = "dp train input.json"
        if os.path.exists('graph.pb'):
            content += " -f graph.pb"
        if self.mode == 'parallel':
            self.J.sub(content, name='train', file='train.sh', out='train-out', err='train-err')
            self.J.wait_jobs_done(self.wait_time)
            self.J.clear()
        else:
            os.system(content)
        os.system("dp freeze -o graph.pb")
        os.chdir(nowpath)

    def updatedataset(self, frames):
        self.trainset.extend(frames)
        write('{}/data.traj'.format(self.ml_dir), self.trainset)
        self.dpdata.from_traj(frames)
        self.dpdata.to_deepmd_npy('{}/data'.format(self.ml_dir))

    def calc_efs(self, frames):
        frames = copy.deepcopy(frames)
        if isinstance(frames, Atoms):
            frames = [frames]
        type_dict = {j: i for i, j in enumerate(self.symbols)}
        calc = DPEnsemble('{}/graph.pb'.format(self.ml_dir), type_dict)
        for atoms in frames:
            calc.calculate(atoms)
            atoms.info['energy'] = calc.results['energy']
            atoms.info['forces'] = calc.results['forces']
            atoms.info['stress'] = calc.results['stress']            
        return frames

    def get_loss(self, frames):
        predict_frames = self.calc_efs(frames)
        e_error, f_error, s_error = [], [], []
        for atoms1, atoms2 in zip(frames, predict_frames):
            e_error.append(atoms1.info['energy'] - atoms2.info['energy'])
            f_error.append((atoms1.info['forces'] - atoms2.info['forces']).reshape(-1))
            s_error.append((atoms1.info['stress'] - atoms2.info['stress']).reshape(-1))
        mae_energies = np.mean(np.abs(e_error))
        mae_forces = np.mean(np.abs(f_error))
        mae_stress = np.mean(np.abs(s_error))
        return mae_energies, 0., mae_forces, 0., mae_stress, 0.

    def relax_with_dp(self, calcPop):
        calc = DPCalculator(**self.all_parameters, model_dir=self.ml_dir, 
                            n_ensemble=self.n_ensemble, keep_prob=self.keep_prob)
        relax_pop = calc.relax(calcPop)
        return relax_pop

    def preselect(self, pop):
        preselect_frames = []
        for atoms_ in pop:
            for atoms in atoms_.info['trajs']:
                if atoms.info['max_force_std'] < self.break_threshold:
                    if atoms.info['max_force_std'] > self.record_threshold:
                        preselect_frames.append(atoms)
                else:
                    break
        return preselect_frames

    # TODO now random, should use other method in future
    def select(self, pop):
        size = int(len(pop) * self.select_ratio)
        select_frames = [pop[i] for i in np.random.choice(len(pop), size)]
        if isinstance(pop, Population):
            return pop.__class__(select_frames)
        return select_frames

    def get_train_set(self, select_frames):
        currdir = os.getcwd()
        self.scf_num += len(select_frames)
        scf_frames = self.query_calculator.scf(select_frames)
        os.chdir(currdir)
        return scf_frames

    def relax_(self, calcPop, max_epoch=20):
        self.scf_num = 0
        # remain info
        for i, atoms in enumerate(calcPop):
            atoms.info['identification'] = i
        nowpath = os.getcwd()
        calc_dir = self.calc_dir
        for epoch in range(1, max_epoch):
            log.info('{} active relax epoch {}'.format(self.job_prefix, epoch))
            # prepare new calcDir
            currdir = '{}/epoch{:02d}'.format(calc_dir, epoch)
            os.makedirs(currdir, exist_ok=True)
            os.chdir(currdir)
            shutil.copy("{}/graph.pb".format(self.ml_dir), "graph.pb")
            # 01: do relax with dpmd
            log.info('\tstep 01: do relax with dp')
            relax_frames = self.relax_with_dp(calcPop)
            bad_frames = self.preselect(relax_frames)
            # 02: select bad cfg
            log.info('\tstep 02: select bad frames')
            select_frames = self.select(bad_frames)
            if len(select_frames) == 0:
                log.info('\thao ye, no bad frames')
                break
            # 03: DFT
            log.info('\tstep 03: {} DFT scf need to be calculated'.format(len(select_frames)))
            scf_frames = self.get_train_set(select_frames)
            # 04: update dataset
            log.info('\tstep 04: update dataset')
            self.updatedataset(scf_frames)
            # 04: retrain and freeze
            log.info('\tstep 05: retrain')
            self.train(epoch=self.n_epoch)
        else:
            log.info('\tbu hao ye, some relax failed')
        log.info('{} DFT scf calculated'.format(self.scf_num))

        for atoms in relax_frames:
            enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa) / len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 6)
            origin_atoms = calcPop[atoms.info['identification']]
            origin_atoms.info.update(atoms.info)
            atoms.info = origin_atoms.info
            atoms.info.pop('identification')
            atoms.info.pop('trajs')    # to save memory
        os.chdir(nowpath)
        return relax_frames

    def scf_(self, calcPop):
        calc = DPCalculator(**self.all_parameters, model_dir=self.ml_dir)
        return calc.scf(calcPop)
