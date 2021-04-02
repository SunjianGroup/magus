import os, subprocess, shutil
import numpy as np
from magus.calculators.base import Calculator
from magus.utils import *
from magus.queuemanage import JobManager
from magus.formatting.mtp import load_cfg, dump_cfg
from ase.units import GPa, eV, Ang
from ase.atoms import Atoms


class MTPCalculator(Calculator):
    def __init__(self, query_calculator, level, parameters):
        super().__init__(parameters)
        self.level = level
        Requirement = ['symbols', 'queueName', 'numCore']
        Default = {
            'jobPrefix': 'MTP',
            'Preprocessing': '# export I_MPI_DEBUG=100',
            'waitTime': 50,
            'verbose': True,
            'w_energy': 1.0, 
            'w_forces': 0.01, 
            'w_stress': 0.001,
            'scale_by_force': 0,
            'min_dist': 0.5,
            'ft': 0.05,
            'st': 1.,
            'init_epoch': 200,
            'n_epoch': 200,
            'ignore_weights': True,
            }
        checkParameters(self.p, parameters, Requirement, Default)
        self.symbol_to_type = {j: i for i, j in enumerate(self.p.symbols)}
        self.type_to_symbol = {i: j for i, j in enumerate(self.p.symbols)}
        self.query_calculator = query_calculator
        self.J = JobManager(self.p.verbose)
        self.calcDir = "{}/calcFold/MTP/{}".format(self.p.workDir, level)
        self.mlDir = "{}/mlFold/MTP/{}".format(self.p.workDir, level)
        if not os.path.exists(self.mlDir):
            os.makedirs(self.mlDir)
        if not os.path.exists('{}/pot.mtp'.format(self.mlDir)):
            shutil.copy('{}/inputFold/pot.mtp'.format(self.p.workDir), 
                        '{}/pot.mtp'.format(self.mlDir))
        if not os.path.exists('{}/train.cfg'.format(self.mlDir)):
            with open('{}/train.cfg'.format(self.mlDir), 'w') as f:
                pass
        if not os.path.exists('{}/datapool.cfg'.format(self.mlDir)):
            with open('{}/datapool.cfg'.format(self.mlDir), 'w') as f:
                pass
        self.scf_num = 0

    def reweighting(self):
        def get_weight(e, e_min, e_mean, w0=10):
            return w0 * np.exp(np.log(w0) * (e - e_min) / (e_min - e_mean))

        if not hasattr(self, 'Emin'):
            self.Emin = 999
        # reweighting must in the folder with train.cfg
        Emean = 0.
        raw_data = load_cfg('train.cfg', self.type_to_symbol)
        for atoms in raw_data:
            enthalpy = (atoms.info['energy'] + self.p.pressure * atoms.get_volume() * GPa) / len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 3)
            Emean += enthalpy
            if enthalpy < self.Emin:
                self.Emin = enthalpy
        Emean /= len(raw_data)
        for atoms in raw_data:
            atoms.info['energy_weight'] = get_weight(atoms.info['enthalpy'], self.Emin, Emean)
        dump_cfg(raw_data, 'train.cfg', self.symbol_to_type)

    def train(self):
        nowpath = os.getcwd()
        os.chdir(self.mlDir)
        we = self.p.w_energy
        wf = self.p.w_forces
        ws = self.p.w_stress
        sbf = self.p.scale_by_force
        if not self.p.ignore_weights:
            self.reweighting()
        with open('train.sh', 'w') as f:
            f.write(
                "#BSUB -q {0}\n"
                "#BSUB -n {1}\n"
                "#BSUB -o train-out\n"
                "#BSUB -e train-err\n"
                "#BSUB -J mtp-train\n"
                #"#BSUB -x\n"
                "{2}\n"
                "mpirun -np {1} mlp train "
                "pot.mtp train.cfg --trained-pot-name=pot.mtp --max-iter={6} "
                "--energy-weight={3} --force-weight={4} --stress-weight={5} "
                "--scale-by-force={7} "
                "--weighting=structures "
                "--update-mindist "
                "--ignore-weights={8}"
                "".format(self.p.queueName, self.p.numCore, self.p.Preprocessing, 
                    we, wf, ws, self.p.init_epoch, sbf, self.p.ignore_weights))
        self.J.bsub('bsub < train.sh', 'train')
        self.J.WaitJobsDone(self.p.waitTime)
        self.J.clear()
        os.chdir(nowpath)
    @property
    def trainset(self):
        return load_cfg('{}/train.cfg'.format(self.mlDir), self.type_to_symbol)

    def predict_energies(self, frames):
        if isinstance(frames, Atoms):
            frames = [frames]
        nowpath = os.getcwd()
        os.chdir(self.mlDir)
        dump_cfg(frames, 'tmp.cfg', self.symbol_to_type)
        exeCmd = "mlp calc-efs pot.mtp tmp.cfg out.cfg"
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('MTP predict_energies exited with exit code: {}.'.format(exitcode))
        result = load_cfg('out.cfg', self.type_to_symbol)
        enengies = [atoms.info['energy'] for atoms in result]
        os.remove('tmp.cfg')
        os.remove('out.cfg')
        os.chdir(nowpath)
        return enengies        
        
    def updatedataset(self, frames):
        dump_cfg(frames, '{}/train.cfg'.format(self.mlDir), self.symbol_to_type, mode='a')

    def get_loss(self, frames):
        nowpath = os.getcwd()
        os.chdir(self.mlDir)
        dump_cfg(frames, 'tmp.cfg', self.symbol_to_type)
        exeCmd = "mlp calc-errors pot.mtp tmp.cfg | grep 'Average absolute difference' | awk {'print $5'}"
        loss = os.popen(exeCmd).readlines()
        mae_energies, r2_energies = float(loss[1]), 0.
        mae_forces, r2_forces = float(loss[2]), 0.
        mae_stress, r2_stress = float(loss[3]), 0.
        os.remove('tmp.cfg')
        os.chdir(nowpath)
        return mae_energies, r2_energies, mae_forces, r2_forces, mae_stress, r2_stress

    def calc_grade(self):
        # must have: pot.mtp, train.cfg
        logging.info('\tstep 01: calculate grade')
        exeCmd = "mlp calc-grade pot.mtp train.cfg train.cfg "\
                 "temp.cfg --als-filename=A-state.als"
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('MTP exited with exit code: %d.  ' % exitcode)
            
    def relax_with_mtp(self):
        # must have: mlip.ini, to_relax.cfg, pot.mtp, A-state.als
        logging.info('\tstep 02: do relax with mtp')
        with open('relax.sh', 'w') as f:
            f.write(
                "#BSUB -q {0}\n"
                "#BSUB -n {1}\n"
                "#BSUB -o relax-out\n"
                "#BSUB -e relax-err\n"
                "#BSUB -J mtp-relax\n"
                "{2}\n"
                "mpirun -np {1} mlp relax mlip.ini "
                "--pressure={3} --cfg-filename=to_relax.cfg "
                "--force-tolerance={4} --stress-tolerance={5} "
                "--min-dist={6} --log=mtp_relax.log "
                "--save-relaxed=relaxed.cfg\n"
                "cat B-preselected.cfg* > B-preselected.cfg\n"
                "cat relaxed.cfg* > relaxed.cfg\n"
                "".format(self.p.queueName, self.p.numCore, self.p.Preprocessing, self.p.pressure,
                          self.p.ft, self.p.st, self.p.min_dist))
        self.J.bsub('bsub < relax.sh', 'relax')
        self.J.WaitJobsDone(self.p.waitTime)
        self.J.clear()

    def select_bad_frames(self):
        # must have: train.cfg, pot.mtp, A-state.als, B-preselected.cfg
        logging.info('\tstep 03: select bad frames')
        """
        exeCmd = "mlp select-add pot.mtp train.cfg B-preselected.cfg C-selected.cfg --weighting=structures"
        subprocess.call(exeCmd, shell=True)
        """
        with open('select.sh', 'w') as f:
            f.write(
                "#BSUB -q {0}\n"
                "#BSUB -n {1}\n"
                "#BSUB -o select-out\n"
                "#BSUB -e select-err\n"
                "#BSUB -J mtp-select\n"
                "{2}\n"
                "mpirun -np {1} mlp select-add "
                "pot.mtp train.cfg B-preselected.cfg C-selected.cfg "
                "--weighting=structures"
                "".format(self.p.queueName, self.p.numCore, self.p.Preprocessing))
        self.J.bsub('bsub < select.sh', 'select')
        self.J.WaitJobsDone(self.p.waitTime)
        self.J.clear()
        
    def get_train_set(self):
        currdir = os.getcwd()
        to_scf = load_cfg("C-selected.cfg", self.type_to_symbol)
        logging.info('\tstep 04: {} DFT scf need to be calculated'.format(len(to_scf)))
        self.scf_num += len(to_scf)
        scfpop = self.query_calculator.scf(to_scf)
        os.chdir(currdir)
        dump_cfg(scfpop, "D-computed.cfg", self.symbol_to_type)

    def retrain(self):
        logging.info('\tstep 05: retrain mtp')
        exeCmd = "cat train.cfg D-computed.cfg >> E-train.cfg\n"\
                 "cp E-train.cfg {0}/train.cfg".format(self.mlDir)
        subprocess.call(exeCmd, shell=True)
        self.train()
        shutil.copy("{}/train-out".format(self.mlDir), "train-out")

    def relax(self, calcPop, max_epoch=20):
        self.scf_num = 0
        # remain info
        for i, atoms in enumerate(calcPop):
            atoms.info['identification'] = i
        nowpath = os.getcwd()
        self.cdcalcFold()
        pressure = self.p.pressure
        calcDir = self.calcDir
        basedir = '{}/epoch{:02d}'.format(calcDir, 0)
        if os.path.exists(basedir):
            shutil.rmtree(basedir)
        os.makedirs(basedir)
        shutil.copy("{}/inputFold/mlip{}.ini".format(self.p.workDir, self.level), "{}/mlip.ini".format(basedir))
        shutil.copy("{}/pot.mtp".format(self.mlDir), "{}/pot.mtp".format(basedir))
        shutil.copy("{}/train.cfg".format(self.mlDir), "{}/train.cfg".format(basedir))
        dump_cfg(calcPop, "{}/to_relax.cfg".format(basedir), self.symbol_to_type)
        for epoch in range(1, max_epoch):
            logging.info('MTP{} active relax epoch {}'.format(self.level, epoch))
            prevdir = '{}/epoch{:02d}'.format(calcDir, epoch - 1)
            currdir = '{}/epoch{:02d}'.format(calcDir, epoch)
            if os.path.exists(currdir):
                shutil.rmtree(currdir)
            os.mkdir(currdir)
            os.chdir(currdir)
            shutil.copy("{}/mlip.ini".format(prevdir), "mlip.ini")
            shutil.copy("{}/pot.mtp".format(self.mlDir), "pot.mtp")
            shutil.copy("{}/to_relax.cfg".format(prevdir), "to_relax.cfg")
            shutil.copy("{}/train.cfg".format(self.mlDir), "train.cfg")
            # 01: calculate grade
            self.calc_grade()
            # 02: do relax with mtp
            self.relax_with_mtp()
            # 03: select bad cfg
            self.select_bad_frames()
            if os.path.getsize("C-selected.cfg") == 0:
                logging.info('\thao ye, no bad frames')
                break
            # 04: DFT
            self.get_train_set()
            # 05: train
            self.retrain()
        else:
            logging.info('\tbu hao ye, some relax failed')
        logging.info('{} DFT scf calculated'.format(self.scf_num))
        shutil.copy("pot.mtp", "{}/pot.mtp".format(self.mlDir))
        shutil.copy("train.cfg", "{}/train.cfg".format(self.mlDir))
        relaxpop = load_cfg("relaxed.cfg", self.type_to_symbol)
        for atoms in relaxpop:
            enthalpy = (atoms.info['energy'] + self.p.pressure * atoms.get_volume() * GPa) / len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 3)
            origin_atoms = calcPop[atoms.info['identification']]
            origin_atoms.info.update(atoms.info)
            atoms.info = origin_atoms.info
            atoms.info.pop('identification')
        os.chdir(nowpath)
        return relaxpop

    def scf(self, calcPop):
        self.cdcalcFold()
        pressure = self.p.pressure
        calcDir = self.calcDir
        basedir = '{}/epoch{:02d}'.format(calcDir, 0)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        shutil.copy("{}/inputFold/mlip.ini".format(self.p.workDir), "{}/pot.mtp".format(basedir))
        shutil.copy("{}/pot.mtp".format(self.mlDir), "{}/pot.mtp".format(basedir))
        shutil.copy("{}/train.cfg".format(self.mlDir), "{}/train.cfg".format(basedir))
        dump_cfg(calcPop, "{}/to_scf.cfg".format(basedir), self.symbol_to_type)

        exeCmd = "mlp calc-efs {0}/pot.mtp {0}/to_scf.cfg {0}/scf_out.cfg".format(basedir)
        exitcode = subprocess.call(exeCmd, shell=True)
        if exitcode != 0:
            raise RuntimeError('MTP exited with exit code: %d.  ' % exitcode)
        scfpop = load_cfg("{}/scf_out.cfg".format(basedir), self.type_to_symbol)
        for atoms in scfpop:
            enthalpy = (atoms.info['energy'] + pressure * atoms.get_volume() * GPa) / len(atoms)
            atoms.info['enthalpy'] = round(enthalpy, 3)
        return scfpop


class TwoShareMTPCalculator(Calculator):
    def __init__(self, query_calculator, parameters):
        self.mtp1 = MTPCalculator(query_calculator, 'robust', parameters)
        self.mtp2 = MTPCalculator(query_calculator, 'accurate', parameters)
        self.mtp1.p.scale_by_force = parameters.sbf_r
        self.mtp2.p.scale_by_force = parameters.sbf_a
        self.mtp2.p.ignore_weights = False
        if not hasattr(self, 'p'):
            self.p = EmptyClass()
        self.mlDir = self.mtp1.mlDir
        self.symbol_to_type = self.mtp1.symbol_to_type
        self.type_to_symbol = self.mtp1.type_to_symbol
        self.p.robust_mtp = self.mtp1.p
        self.p.accurate_mtp = self.mtp2.p
        self.max_enthalpy = 0.

    def update_threshold(self, enthalpy):
        self.max_enthalpy = enthalpy

    def relax(self, calcPop):
        relaxpop = self.mtp1.relax(calcPop)
        shutil.copy('{}/train.cfg'.format(self.mtp1.mlDir), '{}/train.cfg'.format(self.mtp2.mlDir))
        self.mtp2.train()
        # selectpop = relaxpop
        selectpop = [atoms for atoms in relaxpop if atoms.info['enthalpy'] < self.mtp2.Emin + 1.5]
        relaxpop = self.mtp2.relax(selectpop)
        shutil.copy('{}/train.cfg'.format(self.mtp2.mlDir), '{}/train.cfg'.format(self.mtp1.mlDir))
        return relaxpop

    def scf(self, calcPop, level='accurate'):
        if level == 'robust':
            scfpop = self.mtp1.scf(calcPop)
        elif level == 'accurate':
            scfpop = self.mtp2.scf(calcPop)
        return scfpop
    
    def updatedataset(self, frames):                                                                                        
        self.mtp1.updatedataset(frames)

    def get_loss(self, frames):
        return self.mtp2.get_loss(frames)

    def train(self):
        self.mtp1.train()
        shutil.copy('{}/train.cfg'.format(self.mtp1.mlDir), '{}/train.cfg'.format(self.mtp2.mlDir))
        shutil.copy('{}/pot.mtp'.format(self.mtp1.mlDir), '{}/pot.mtp'.format(self.mtp2.mlDir))
        self.mtp2.train()

    @property
    def trainset(self):
        return self.mtp2.trainset
    
    def predict_energies(self, frames):
        return self.mtp2.predict_energies(frames)
