import os, subprocess, shutil, time
import numpy as np
from magus.calculators.base import Calculator, ClusterCalculator
from magus.formatting.mtp import load_cfg, dump_cfg
from ase.units import GPa, eV, Ang
from ase.atoms import Atoms
import logging
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from magus.calculators.lammps import calc_lammps_once
from magus.utils import CALCULATOR_PLUGIN, CALCULATOR_CONNECT_PLUGIN, check_parameters
from magus.populations.populations import Population
#if len(os.popen('which mlp').readlines()) == 0:
#    raise ImportError("No 'mlp' detected")


log = logging.getLogger(__name__)


@CALCULATOR_PLUGIN.register('mtp-noselect')
class MTPNoSelectCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['symbols']
        Default={
            'force_tolerance': 0.05,
            'stress_tolerance': 1.,
            'min_dist': 0.5,
            'n_epoch': 200,
            'job_prefix': 'MTP',
            'mtp_exe': 'mlp',
            'mtp_runner': 'mpirun',
            }
        check_parameters(self, parameters, Requirement, Default)
        self.symbol_to_type = {j: i for i, j in enumerate(self.symbols)}
        self.type_to_symbol = {i: j for i, j in enumerate(self.symbols)}

        self.main_info.extend(['force_tolerance', 'stress_tolerance'])

    def relax_with_mtp(self):
        #content = "mpirun -np {0} mlp relax mlip.ini "\
        #          "--pressure={1} --cfg-filename=to_relax.cfg "\
        #          "--force-tolerance={2} --stress-tolerance={3} "\
        #          "--min-dist={4} --log=mtp_relax.log "\
        #          "--save-relaxed=relaxed.cfg\n"\
        #          "cat relaxed.cfg* > relaxed.cfg\n"\
        #          "".format(self.num_core, self.pressure, self.force_tolerance,
        #                    self.stress_tolerance, self.min_dist)
        content = f"{self.mtp_runner} -n {self.num_core} {self.mtp_exe} relax mlip.ini "\
                  f"--pressure={self.pressure} --cfg-filename=to_relax.cfg "\
                  f"--force-tolerance={self.force_tolerance} --stress-tolerance={self.stress_tolerance} "\
                  f"--min-dist={self.min_dist} --log=mtp_relax.log "\
                  f"--save-relaxed=relaxed.cfg\n"\
                  f"cat relaxed.cfg?* > relaxed.cfg\n"
        self.J.sub(content, name='relax', file='relax.sh', out='relax-out', err='relax-err')
        self.J.wait_jobs_done(self.wait_time)
        self.J.clear()
        time.sleep(10)

    def relax_(self, calcPop, max_epoch=20):
        self.scf_num = 0
        # remain info
        for i, atoms in enumerate(calcPop):
            atoms.info['identification'] = i
        nowpath = os.getcwd()
        calc_dir = self.calc_dir
        basedir = '{}/epoch{:02d}'.format(calc_dir, 0)
        os.makedirs(basedir, exist_ok=True)
        shutil.copy("{}/mlip.ini".format(self.input_dir), "{}/mlip.ini".format(basedir))
        shutil.copy("{}/pot.mtp".format(self.input_dir), "{}/pot.mtp".format(basedir))
        os.chdir(basedir)
        dump_cfg(calcPop, "{}/to_relax.cfg".format(basedir), self.symbol_to_type)
        self.relax_with_mtp()
        relaxpop = load_cfg("relaxed.cfg", self.type_to_symbol)
        for atoms in relaxpop:
            enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa) / len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 6)
            origin_atoms = calcPop[atoms.info['identification']]
            origin_atoms.info.update(atoms.info)
            atoms.info = origin_atoms.info
            atoms.info.pop('identification')
        os.chdir(nowpath)
        return relaxpop

    def scf_(self, calcPop):
        calc_dir = self.calc_dir
        basedir = '{}/epoch{:02d}'.format(calc_dir, 0)
        os.makedirs(basedir, exist_ok=True)
        shutil.copy("{}/mlip.ini".format(self.input_dir), "{}/pot.mtp".format(basedir))
        shutil.copy("{}/pot.mtp".format(self.ml_dir), "{}/pot.mtp".format(basedir))
        dump_cfg(calcPop, "{}/to_scf.cfg".format(basedir), self.symbol_to_type)
        exeCmd = f"{self.mtp_runner} -n {self.num_core} {self.mtp_exe} calc-efs {basedir}/pot.mtp {basedir}/to_scf.cfg {basedir}/scf_out.cfg"
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('MTP exited with exit code: %d.  ' % exitcode)
        scfpop = load_cfg("{}/scf_out.cfg".format(basedir), self.type_to_symbol)
        for atoms in scfpop:
            enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa) / len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 6)
        return scfpop


@CALCULATOR_PLUGIN.register('mtp')
class MTPSelectCalculator(ClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['query_calculator', 'symbols']
        Default={
            'xc': 'PBE',
            'weights': [1., 0.01, 0.001],
            'scaled_by_force': 0.,
            'force_tolerance': 0.05,
            'stress_tolerance': 1.,
            'min_dist': 0.5,
            'n_epoch': 200,
            'ignore_weights': True,
            'job_prefix': 'MTP',
            'n_fail': 0,
            'mtp_exe': 'mlp',
            'mtp_runner': 'mpirun',
            }
        check_parameters(self, parameters, Requirement, Default)
        self.symbol_to_type = {j: i for i, j in enumerate(self.symbols)}
        self.type_to_symbol = {i: j for i, j in enumerate(self.symbols)}
        self.ml_dir = "{}/mlFold/{}".format(self.work_dir, self.job_prefix)
        if not os.path.exists('{}/train.cfg'.format(self.input_dir)):
            with open('{}/train.cfg'.format(self.input_dir), 'w') as f:
                pass
        if not os.path.exists(self.ml_dir):
            os.makedirs(self.ml_dir)
        if not os.path.exists('{}/pot.mtp'.format(self.ml_dir)):
            shutil.copy('{}/pot.mtp'.format(self.input_dir),
                        '{}/pot.mtp'.format(self.ml_dir))
        if not os.path.exists('{}/train.cfg'.format(self.ml_dir)):
            shutil.copy('{}/train.cfg'.format(self.input_dir),
                        '{}/train.cfg'.format(self.ml_dir))
        if not os.path.exists('{}/datapool.cfg'.format(self.ml_dir)):
            with open('{}/datapool.cfg'.format(self.ml_dir), 'w') as f:
                pass
        self.scf_num = 0
        self.static_need_update = True
        self.main_info.extend(
            ['weights', 'scaled_by_force', 'min_dist', 'force_tolerance',
             'stress_tolerance', 'n_epoch', 'ignore_weights',
             'ml_dir', 'n_fail'])

    @property
    def E_min(self):
        if self.static_need_update:
            self.update_static()
        return self.E_min_

    @property
    def E_mean(self):
        if self.static_need_update:
            self.update_static()
        return self.E_mean_

    def update_static(self):
        enthalpy = [(atoms.info['energy'] + self.pressure *
                     atoms.get_volume() * GPa) / len(atoms) for atoms in self.trainset]
        self.E_min_ = np.min(enthalpy)
        self.E_mean_ = np.mean(enthalpy)
        self.static_need_update = False

    def get_weight(self, atoms, w0=10):
        if self.static_need_update:
            self.update_static()
        enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa) / len(atoms)
        return w0 * np.exp(np.log(w0) * (enthalpy - self.E_min) / (self.E_min - self.E_mean))

    def reweighting(self):
        new_set = []
        if self.static_need_update:
            self.update_static()
        for atoms in self.trainset:
            atoms.info['energy_weight'] = self.get_weight(atoms)
            new_set.append(atoms)
        dump_cfg(new_set, 'train.cfg', self.symbol_to_type)

    def train(self, epoch=None):
        epoch = epoch or self.n_epoch
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        if not self.ignore_weights:
            self.reweighting()
        #content = "mpirun -np {0} mlp train "\
        #          "pot.mtp train.cfg --trained-pot-name=pot.mtp --max-iter={1} "\
        #          "--energy-weight={2} --force-weight={3} --stress-weight={4} "\
        #          "--scale-by-force={5} "\
        #          "--weighting=structures "\
        #          "--update-mindist "\
        #          "--ignore-weights={6}"\
        #          "".format(self.num_core, epoch, *self.weights, self.scaled_by_force, self.ignore_weights)
        content = f"{self.mtp_runner} -n {self.num_core} {self.mtp_exe} train "\
                  f"pot.mtp train.cfg --trained-pot-name=pot.mtp --max-iter={epoch} "\
                  f"--energy-weight={self.weights[0]} --force-weight={self.weights[1]} --stress-weight={self.weights[2]} "\
                  f"--scale-by-force={self.scaled_by_force} "\
                  f"--weighting=structures "\
                  f"--update-mindist "\
                  f"--ignore-weights={self.ignore_weights}\n"


        self.J.sub(content, name='train', file='train.sh', out='train-out', err='train-err')
        self.J.wait_jobs_done(self.wait_time)
        self.J.clear()
        os.chdir(nowpath)

    @property
    def trainset(self):
        return load_cfg('{}/train.cfg'.format(self.ml_dir), self.type_to_symbol)

    def calc_efs(self, frames):
        if isinstance(frames, Atoms):
            frames = [frames]
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        dump_cfg(frames, 'tmp.cfg', self.symbol_to_type)
        exeCmd = f"{self.mtp_runner} -n {self.num_core} {self.mtp_exe} calc-efs pot.mtp tmp.cfg out.cfg"
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('MTP calc-efs exited with exit code: {}.'.format(exitcode))
        result = load_cfg('out.cfg', self.type_to_symbol)
        os.remove('tmp.cfg')
        os.remove('out.cfg')
        os.chdir(nowpath)
        return result

    def updatedataset(self, frames):
        dump_cfg(frames, '{}/train.cfg'.format(self.ml_dir), self.symbol_to_type, mode='a')
        self.static_need_update = True

    def get_loss(self, frames):
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        dump_cfg(frames, 'tmp.cfg', self.symbol_to_type)
        exeCmd = f"{self.mtp_runner} -n {self.num_core} {self.mtp_exe} calc-errors pot.mtp tmp.cfg | grep 'Average absolute difference' | awk {{'print $5'}}"
        loss = os.popen(exeCmd).readlines()
        mae_energies, r2_energies = float(loss[1]), 0.
        mae_forces, r2_forces = float(loss[2]), 0.
        mae_stress, r2_stress = float(loss[3]), 0.
        os.remove('tmp.cfg')
        os.chdir(nowpath)
        return mae_energies, r2_energies, mae_forces, r2_forces, mae_stress, r2_stress

    def calc_grade(self):
        # must have: pot.mtp, train.cfg
        log.info('\tstep 01: calculate grade')
        exeCmd = f"{self.mtp_runner} -n 1 {self.mtp_exe} calc-grade pot.mtp train.cfg train.cfg "\
                 "temp.cfg --als-filename=A-state.als"
        #exeCmd = f"mlp calc-grade pot.mtp train.cfg train.cfg "\
        #         "temp.cfg --als-filename=A-state.als"
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('MTP exited with exit code: %d.  ' % exitcode)

    def relax_with_mtp(self):
        # must have: mlip.ini, to_relax.cfg, pot.mtp, A-state.als
        log.info('\tstep 02: do relax with mtp')
        #content = "mpirun -np {0} mlp relax mlip.ini "\
        #          "--pressure={1} --cfg-filename=to_relax.cfg "\
        #          "--force-tolerance={2} --stress-tolerance={3} "\
        #          "--min-dist={4} --log=mtp_relax.log "\
        #          "--save-relaxed=relaxed.cfg\n"\
        #          "cat B-preselected.cfg* > B-preselected.cfg\n"\
        #          "cat relaxed.cfg* > relaxed.cfg\n"\
        #          "".format(self.num_core, self.pressure, self.force_tolerance,
        #                    self.stress_tolerance, self.min_dist)
        content = f"{self.mtp_runner} -n {self.num_core} {self.mtp_exe} relax mlip.ini "\
                  f"--pressure={self.pressure} --cfg-filename=to_relax.cfg "\
                  f"--force-tolerance={self.force_tolerance} --stress-tolerance={self.stress_tolerance} "\
                  f"--min-dist={self.min_dist} --log=mtp_relax.log "\
                  f"--save-relaxed=relaxed.cfg\n"\
                  f"sleep 10\n"\
                  f"cat B-preselected.cfg?* > B-preselected.cfg\n"\
                  f"cat relaxed.cfg?* > relaxed.cfg\n"
        self.J.sub(content, name='relax', file='relax.sh', out='relax-out', err='relax-err')
        self.J.wait_jobs_done(self.wait_time)
        self.J.clear()

    def select(self, pop):
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        dump_cfg(pop, "new.cfg", self.symbol_to_type)
        #content = "mpirun -np {0} mlp select-add "\
        #          "pot.mtp train.cfg new.cfg diff.cfg "\
        #          "--weighting=structures"\
        #          "".format(self.num_core)
        content = f"{self.mtp_runner} -n {self.num_core} {self.mtp_exe} select-add "\
                  f"pot.mtp train.cfg new.cfg diff.cfg "\
                  f"--weighting=structures\n"
        self.J.sub(content, name='select', file='select.sh',
                   out='select-out', err='select-err')
        self.J.wait_jobs_done(self.wait_time)
        self.J.clear()
        time.sleep(10)
        diff_frames = load_cfg("diff.cfg", self.type_to_symbol)
        os.chdir(nowpath)
        if isinstance(pop, Population):
            return pop.__class__(diff_frames)
        return diff_frames

    def select_bad_frames(self):
        # must have: train.cfg, pot.mtp, A-state.als, B-preselected.cfg
        log.info('\tstep 03: select bad frames')
        to_select = load_cfg("B-preselected.cfg", self.type_to_symbol)
        selected = self.select(to_select)
        dump_cfg(selected, "C-selected.cfg", self.symbol_to_type)
        return selected

    def get_train_set(self):
        currdir = os.getcwd()
        to_scf = load_cfg("C-selected.cfg", self.type_to_symbol)
        log.info('\tstep 04: {} DFT scf need to be calculated'.format(len(to_scf)))
        self.scf_num += len(to_scf)
        scfpop = self.query_calculator.scf(to_scf)
        os.chdir(currdir)
        dump_cfg(scfpop, "D-computed.cfg", self.symbol_to_type)

    def retrain(self):
        log.info('\tstep 05: retrain mtp')
        exeCmd = "cat train.cfg D-computed.cfg > E-train.cfg\n"\
                 "cp E-train.cfg {0}/train.cfg".format(self.ml_dir)
        subprocess.call(exeCmd, shell=True)
        self.train(epoch=self.n_epoch)
        shutil.copy("{}/train-out".format(self.ml_dir), "train-out")

    def relax_(self, calcPop, max_epoch=20):
        self.scf_num = 0
        # remain info
        for i, atoms in enumerate(calcPop):
            atoms.info['identification'] = i
        nowpath = os.getcwd()
        calc_dir = self.calc_dir
        basedir = '{}/epoch{:02d}'.format(calc_dir, 0)
        os.makedirs(basedir, exist_ok=True)
        shutil.copy("{}/mlip.ini".format(self.input_dir), "{}/mlip.ini".format(basedir))
        shutil.copy("{}/pot.mtp".format(self.ml_dir), "{}/pot.mtp".format(basedir))
        shutil.copy("{}/train.cfg".format(self.ml_dir), "{}/train.cfg".format(basedir))
        dump_cfg(calcPop, "{}/to_relax.cfg".format(basedir), self.symbol_to_type)
        for epoch in range(1, max_epoch):
            log.info('{} active relax epoch {}'.format(self.job_prefix, epoch))
            prevdir = '{}/epoch{:02d}'.format(calc_dir, epoch - 1)
            currdir = '{}/epoch{:02d}'.format(calc_dir, epoch)
            os.makedirs(currdir, exist_ok=True)
            os.chdir(currdir)
            shutil.copy("{}/mlip.ini".format(prevdir), "mlip.ini")
            shutil.copy("{}/pot.mtp".format(self.ml_dir), "pot.mtp")
            shutil.copy("{}/to_relax.cfg".format(prevdir), "to_relax.cfg")
            shutil.copy("{}/train.cfg".format(self.ml_dir), "train.cfg")
            # 01: calculate grade
            self.calc_grade()
            # 02: do relax with mtp
            self.relax_with_mtp()
            if os.path.getsize("B-preselected.cfg") == 0:
                log.info('\thao ye, no bad frames')
                break
            # 03: select bad cfg
            selected = self.select_bad_frames()
            if len(selected) <= self.n_fail:
                log.info('\tselected frames less than threshold {}'.format(self.n_fail))
                break
            # 04: DFT
            self.get_train_set()
            # 05: train
            self.retrain()
        else:
            log.info('\tbu hao ye, some relax failed')
        log.info('{} DFT scf calculated'.format(self.scf_num))
        shutil.copy("pot.mtp", "{}/pot.mtp".format(self.ml_dir))
        shutil.copy("train.cfg", "{}/train.cfg".format(self.ml_dir))
        relaxpop = load_cfg("relaxed.cfg", self.type_to_symbol)
        for atoms in relaxpop:
            enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa) / len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 6)
            origin_atoms = calcPop[atoms.info['identification']]
            origin_atoms.info.update(atoms.info)
            atoms.info = origin_atoms.info
            atoms.info.pop('identification')
        os.chdir(nowpath)
        return relaxpop

    def scf_(self, calcPop):
        calc_dir = self.calc_dir
        basedir = '{}/epoch{:02d}'.format(calc_dir, 0)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        shutil.copy("{}/mlip.ini".format(self.input_dir), "{}/pot.mtp".format(basedir))
        shutil.copy("{}/pot.mtp".format(self.ml_dir), "{}/pot.mtp".format(basedir))
        shutil.copy("{}/train.cfg".format(self.ml_dir), "{}/train.cfg".format(basedir))
        dump_cfg(calcPop, "{}/to_scf.cfg".format(basedir), self.symbol_to_type)

        exeCmd = f"{self.mtp_runner} -n {self.num_core} {self.mtp_exe} calc-efs {basedir}/pot.mtp {basedir}/to_scf.cfg {basedir}/scf_out.cfg"
        #exeCmd = f"mlp calc-efs {0}/pot.mtp {0}/to_scf.cfg {0}/scf_out.cfg".format(basedir)
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('MTP exited with exit code: %d.  ' % exitcode)
        scfpop = load_cfg("{}/scf_out.cfg".format(basedir), self.type_to_symbol)
        for atoms in scfpop:
            enthalpy = (atoms.info['energy'] + self.pressure * atoms.get_volume() * GPa) / len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 6)
        return scfpop


@CALCULATOR_CONNECT_PLUGIN.register('share-trainset')
class TwoShareMTPCalculator(Calculator):
    def __init__(self, mtps):
        assert isinstance(mtps, list), "TwoShareMTP input should be list"
        assert len(mtps) == 2, "length of mtps must be 2"
        self.mtp1 = mtps[0]
        self.mtp2 = mtps[1]
        self.mtp1.ignore_weights, self.mtp2.ignore_weights = True, False

        self.symbol_to_type = self.mtp1.symbol_to_type
        self.type_to_symbol = self.mtp1.type_to_symbol

        self.max_enthalpy = 0.
        self.mtp2_train_len = len(self.mtp2.trainset)

    def __repr__(self):
        out  = self.__class__.__name__ + ':\n'
        out += 'Robust MTP:' + self.mtp1.__repr__()
        out += 'Accurate MTP:' + self.mtp2.__repr__()
        return out

    def update_threshold(self, enthalpy):
        self.max_enthalpy = enthalpy

    def relax_(self, calcPop):
        relaxpop = self.mtp1.relax(calcPop)
        shutil.copy('{}/train.cfg'.format(self.mtp1.ml_dir),
                    '{}/train.cfg'.format(self.mtp2.ml_dir))
        if self.mtp2_train_len != len(self.mtp2.trainset):
            log.info('The share train set update, updating the accurate potential...')
            self.mtp2_train_len = len(self.mtp2.trainset)
            self.mtp2.train()
            self.mtp2.static_need_update = True
        selectpop = [atoms for atoms in relaxpop if atoms.info['enthalpy'] < self.mtp2.E_min + 1.5]
        relaxpop = self.mtp2.relax(selectpop)
        shutil.copy('{}/train.cfg'.format(self.mtp2.ml_dir),
                    '{}/train.cfg'.format(self.mtp1.ml_dir))
        return relaxpop

    def scf_(self, calcPop, level='accurate'):
        if level == 'robust':
            scfpop = self.mtp1.scf(calcPop)
        elif level == 'accurate':
            scfpop = self.mtp2.scf(calcPop)
        return scfpop

    def updatedataset(self, frames):
        self.mtp1.updatedataset(frames)
        self.mtp2.updatedataset(frames)

    def get_loss(self, frames, level='accurate'):
        if level == 'robust':
            return self.mtp1.get_loss(frames)
        elif level == 'accurate':
            return self.mtp2.get_loss(frames)

    def select(self, new_frames):
        diff_frames = self.mtp1.select(new_frames)
        return diff_frames

    def train(self):
        log.debug('train robust...')
        self.mtp1.train()
        shutil.copy('{}/train.cfg'.format(self.mtp1.ml_dir),
                    '{}/train.cfg'.format(self.mtp2.ml_dir))
        if self.mtp2_train_len != len(self.mtp2.trainset):
            self.mtp2_train_len = len(self.mtp2.trainset)
            self.mtp2.static_need_update = True
        log.debug('train accurate...')
        self.mtp2.train()

    @property
    def trainset(self):
        return self.mtp2.trainset

    def calc_efs(self, frames, level='accurate'):
        if level == 'robust':
            return self.mtp1.calc_efs(frames)
        elif level == 'accurate':
            return self.mtp2.calc_efs(frames)


@CALCULATOR_PLUGIN.register('mtp-lammps')
class MTPLammpsCalculator(MTPSelectCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.lammps_setup = {
            'symbols': self.symbols,
            'atom_style': 'atomic',
            'pressure': self.pressure,
            'save_traj': True,
            'exe_cmd': 'lmp_mtp -in in.lammps',
        }

    def relax_with_lammps(self, atoms):
        new_atoms = calc_lammps_once(self.lammps_setup, atoms)
        if new_atoms is None:
            return None
        return new_atoms.info['traj']

    def relax_(self, calcPop, max_epoch=20):
        assert len(calcPop) == 1 , 'MTP active lammps only support one atoms now'
        self.scf_num = 0
        nowpath = os.getcwd()
        calc_dir = self.calc_dir
        basedir = '{}/epoch{:02d}'.format(calc_dir, 0)
        os.makedirs(basedir, exist_ok=True)
        shutil.copy("{}/mlip.ini".format(self.input_dir), "{}/mlip.ini".format(basedir))
        shutil.copy("{}/in.lammps".format(self.input_dir), "{}/in.lammps".format(basedir))
        shutil.copy("{}/pot.mtp".format(self.ml_dir), "{}/pot.mtp".format(basedir))
        shutil.copy("{}/train.cfg".format(self.ml_dir), "{}/train.cfg".format(basedir))
        for epoch in range(1, max_epoch):
            log.info('{} active relax epoch {}'.format(self.job_prefix, epoch))
            prevdir = '{}/epoch{:02d}'.format(calc_dir, epoch - 1)
            currdir = '{}/epoch{:02d}'.format(calc_dir, epoch)
            os.makedirs(currdir, exist_ok=True)
            os.chdir(currdir)
            shutil.copy("{}/mlip.ini".format(prevdir), "mlip.ini")
            shutil.copy("{}/in.lammps".format(prevdir), "in.lammps")
            shutil.copy("{}/pot.mtp".format(self.ml_dir), "pot.mtp")
            shutil.copy("{}/train.cfg".format(self.ml_dir), "train.cfg")
            # 01: calculate grade
            self.calc_grade()
            # 02: do relax with lammps
            traj = self.relax_with_lammps(calcPop[0])
            if not os.path.exists("B-preselected.cfg"):
                log.info('\thao ye, no bad frames')
                break
            # 03: select bad cfg
            self.select_bad_frames()
            if os.path.getsize("C-selected.cfg") == 0:
                log.info('\thao ye, no bad frames')
                break
            # 04: DFT
            self.get_train_set()
            # 05: train
            self.retrain()
        else:
            log.info('\tbu hao ye, some relax failed')
        log.info('{} DFT scf calculated'.format(self.scf_num))
        shutil.copy("pot.mtp", "{}/pot.mtp".format(self.ml_dir))
        shutil.copy("train.cfg", "{}/train.cfg".format(self.ml_dir))
        os.chdir(nowpath)
        return traj
