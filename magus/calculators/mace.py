import logging, os, shutil, sys, random, glob
import yaml
import numpy as np
from ase.io import read, write
from mace.calculators import mace_mp, mace_off, mace_anicc
from mace.calculators import MACECalculator as MACEAseCalculator
import torch

from magus.calculators.base import ASECalculator, ClusterCalculator, ASEClusterCalculator
from magus.utils import CALCULATOR_PLUGIN, check_parameters, apply_peturb
from magus.populations.populations import Population

from ase.filters import ExpCellFilter
from ase.units import GPa, eV, Ang
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from ase import Atoms
from ase.utils.abc import Optimizable
from ase.neighborlist import neighbor_list, natural_cutoffs
log = logging.getLogger(__name__)

from scipy.spatial.distance import cdist


class FarthestPointSample:
    """Farthest point sampler

    Example:

    1. Select points from random 2d points:

    >>> data = np.random.randn(100000, 2)
    >>> selector = FarthestPointSample(min_distance=0.05)
    >>> indices = selector.select(data)

    2. Select atoms with structure descriptors

    Suppose we already have frames to be selected and a NEP calculator.
    In fact, you can get descriptors by any other method, such as SOAP.

    >>> des = np.array([np.mean(calc.get_property('descriptor', atoms), axis=0) for atoms in frames])
    # Use average of each atomic descriptors to get structure descriptors, shape: (Nframes, Ndescriptor)
    >>> sampler = FarthestPointSample()
    >>> indices = sampler.select(des, [])
    >>> selected_structures = [frames[i] for  i in indices]

    3. Select atoms with atomic latent descriptors

    >>> lat = np.concatenate([calc.get_property('latent', atoms) for atoms in frames])
    # shape: (Natoms, Nlatent)
    >>> comesfrom = np.concatenate([[i] * len(atoms) for i, atoms in enumerate(frames)])
    # shape: (Natoms, )  the ith data in lat belong to the atoms: frames[comesfrom[i]]
    >>> sampler = FarthestPointSample()
    >>> indices = [comesfrom[i] for i in sampler.select(lat, [])]
    >>> indices = set(indices)  # remove repeated indices because two points may come from the same structure
    >>> selected_structures = [frames[i] for  i in indices]

    """
    def __init__(self, min_distance=0.1, metric='euclidean', metric_para={}):
        """Initial the sampler

        Args:
            min_distance (float, optional): minimum distance between selected data. Defaults to 0.1.
            metric (str, optional): metric of distance between data.
                Defaults to 'euclidean'. Any metric can be used by 'scipy.spatial.distance.cdist'
                such as 'cosine', 'minkowski' can also be used.
            metric_para (dict, optional): Extra arguments to metric.
                Defaults to {}.

            More information about metric can be found: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        """
        self.min_distance = min_distance
        self.metric = metric
        self.metric_para = {}

    def select(self, new_data, now_data=[], min_distance=None, min_select=1, max_select=None):
        """Select those data fartheset from given data

        Args:
            new_data (2d list or array): A series of points to be selected
            now_data (2d list or array): Points already in the dataset.
                Defaults to []. (No existed data)
            min_distance (float, optional):
                If distance between two points exceeded the minimum distance, stop the selection.
                Defaults to None (use the self.min_distance)
            min_select (int, optional): Minimal numbers of points to be selected. This may cause
                some distance between points less than given min_distance.
                Defaults to 1.
            max_select (int, optional): Maximum numbers of points to be selected.
                Defaults to None. (No limitation)

        Returns:
            A list of int: index of selected points
        """
        min_distance = min_distance or self.min_distance
        max_select = max_select or len(new_data)
        to_add = []
        if len(new_data) == 0:
            return to_add
        if len(now_data) == 0:
            to_add.append(0)
            now_data.append(new_data[0])
        distances = np.min(cdist(new_data, now_data, metric=self.metric, **self.metric_para), axis=1)

        while np.max(distances) > min_distance or len(to_add) < min_select:
            i = np.argmax(distances)
            to_add.append(i)
            if len(to_add) >= max_select:
                break
            distances = np.minimum(distances, cdist([new_data[i]], new_data, metric=self.metric)[0])
        return to_add

def split_dataset(dataset, ratios):
    # Training:Validation:Test
    assert len(ratios) == 3

    ratios = np.array(ratios)
    ratios = ratios/ratios.sum()

    allNum = len(dataset)
    numTr = int(np.round(ratios[0]*allNum))
    numVa = int(np.round(ratios[1]*allNum))
    numTe = allNum - numTr - numVa

    # print(f"Training: {numTr}, Validation: {numVa}, Test: {numTe}")
    random.shuffle(dataset)

    trainSet = dataset[:numTr]
    validSet = dataset[numTr:numTr+numVa]
    testSet = dataset[numTr+numVa:]

    return trainSet, validSet, testSet


@CALCULATOR_PLUGIN.register('mace-noselect')
class MACENoSelectCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        with open("{}/mace_setting.yaml".format(self.input_dir)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        if params['modelType'] == 'mp':
            self.relax_calc = mace_mp(**params)
            self.scf_calc = mace_mp(**params)
        elif params['modelType'] == 'off':
            self.relax_calc = mace_off(**params)
            self.scf_calc = mace_off(**params)
        elif params['modelType'] == 'anicc':
            self.relax_calc = mace_anicc(**params)
            self.scf_calc = mace_anicc(**params)
        else:
            self.relax_calc = MACEAseCalculator(**params)
            self.scf_calc = MACEAseCalculator(**params)


@CALCULATOR_PLUGIN.register('mace')
class MACECalculator(ASEClusterCalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        Requirement = ['query_calculator', 'symbols']
        Default = {
            'xc': 'PBE',
            'job_prefix': 'MACE',
            # 'generation': 100,
            'init_times': 1,
            # 'device': 'cuda',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mace_runner': '',
            'mode': 'parallel',
            'max_step': 100,
            'optimizer': 'bfgs',
            'max_move': 0.1,
            'eps': 0.05,
            'split_ratios': [8, 1, 1],
            'n_sample': None,
            'n_perturb': 0,
            'perturb_keep_init': True,
            'std_atom_move': 0.05,
            'std_lat_move': 0.05,
            'selection': 'fps',
            'filter_force': True,
            'train_mode': 'all', # all: use all the data; new: only new data; mix: mix previous and current data
            'mix_ratio': 1, # No. previous data / No. current data
            'max_mace_force': 1000, # Max force for MACE. If forces are larger than this value, the relaxation will be stopped
            'min_mace_dratio': 0.5, # Min distance ratio for MACE. If distance ratio is less than this value, the relaxation will stop
        }
        check_parameters(self, parameters, Requirement, Default)

        self.ml_dir = "{}/mlFold/{}".format(self.work_dir, self.job_prefix)
        self.mace_setup = {
            'pressure': self.pressure,
            'model_paths': f'{self.ml_dir}/mace_pot.model',
            'max_step': self.max_step,
            'optimizer': self.optimizer,
            'max_move': self.max_move,
            'eps': self.eps,
            'filter_force': self.filter_force,
            'max_mace_force': self.max_mace_force,
            'min_mace_dratio': self.min_mace_dratio,
        }
        self.main_info.append('mace_setup')

        # copy files
        if not os.path.exists(self.ml_dir):
            os.makedirs(self.ml_dir)
        # It is messy because *.model and *compiled.model in MACR are different.
        # Original models can be trained files but not compatible to ASE calculator.
        # Compiled models are compatible to ASE calculator but can't be initial potential for training.
        # But mace_mp foundation models work for both cases.
        if not os.path.exists(f'{self.ml_dir}/mace_pot.model'):
            if os.path.exists(f'{self.input_dir}/mace_pot.model'):
                shutil.copy(f'{self.input_dir}/mace_pot.model',
                            f'{self.ml_dir}/mace_pot.model')
                self.relax_calc = MACEAseCalculator(model_paths=f"{self.ml_dir}/mace_pot.model", device=self.device)
                self.scf_calc = MACEAseCalculator(model_paths=f"{self.ml_dir}/mace_pot.model", device=self.device)
            else:
                log.warning("No initial mace potential.")
        if not os.path.exists(f'{self.ml_dir}/init_mace_pot.model'):
            if os.path.exists(f'{self.input_dir}/init_mace_pot.model'):
                shutil.copy(f'{self.input_dir}/init_mace_pot.model',
                            f'{self.ml_dir}/init_mace_pot.model')
            elif os.path.exists(f'{self.input_dir}/mace_pot.model'):
                shutil.copy(f'{self.input_dir}/mace_pot.model',
                            f'{self.ml_dir}/init_mace_pot.model')
            else:
                log.warning("No initial mace potential for training.")
        if not os.path.exists(f'{self.ml_dir}/mace_desc.model'):
            if os.path.exists(f'{self.input_dir}/mace_desc.model'):
                shutil.copy(f'{self.input_dir}/mace_desc.model',
                            f'{self.ml_dir}/mace_desc.model')
            elif os.path.exists(f'{self.input_dir}/mace_pot.model'):
                shutil.copy(f'{self.input_dir}/mace_pot.model',
                            f'{self.ml_dir}/mace_desc.model')
            else:
                log.warning("No initial mace potential for descriptor.")
        if not os.path.exists(f'{self.ml_dir}/mace_train.yaml'):
            if os.path.exists(f'{self.input_dir}/mace_train.yaml'):
                shutil.copy(f'{self.input_dir}/mace_train.yaml',
                            f'{self.ml_dir}/mace_train.yaml')
            else:
                raise Exception("No mace training setting file.")
        fnames = ['train.xyz', 'valid.xyz', 'test.xyz']
        for fn in fnames:
            if not os.path.exists(f'{self.ml_dir}/{fn}'):
                if os.path.exists(f'{self.input_dir}/{fn}'):
                     shutil.copy(f'{self.input_dir}/{fn}',f'{self.ml_dir}/{fn}')
                     shutil.copy(f'{self.input_dir}/{fn}',f'{self.ml_dir}/new_{fn}')
                     shutil.copy(f'{self.input_dir}/{fn}',f'{self.ml_dir}/all_{fn}')
                else:
                    log.info(f"No inital {fn}")
                    os.system(f"touch {self.ml_dir}/{fn}")

    def train(self):
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        if os.path.exists(f'{self.ml_dir}/mace_fit.model'):
            # Use the fitted model in previous training
            shutil.copy(f'{self.ml_dir}/mace_fit.model', f'{self.ml_dir}/init_mace_pot.model')
        #shutil.copy(f'{self.ml_dir}/mace_pot.model', f'{self.ml_dir}/init_mace_pot.model')
        train_command = f"{self.mace_runner} mace_run_train --config mace_train.yaml"
        self.J.sub(train_command, name='train', file='train.sh',
                   out='train-out-%j', err='train-err-%j')
        self.J.wait_jobs_done(self.wait_time)
        self.J.clear()
        # read trained pot in
        shutil.copy(f'{self.ml_dir}/mace_fit_compiled.model', f'{self.ml_dir}/mace_pot.model')
        # save model for descriptor
        desc_fn = glob.glob(f'{self.ml_dir}/checkpoints/mace_fit*model')[0]
        shutil.copy(desc_fn, f'{self.ml_dir}/mace_desc.model')
        self.relax_calc = MACEAseCalculator(model_paths=f"{self.ml_dir}/mace_pot.model", device=self.device)
        self.scf_calc = MACEAseCalculator(model_paths=f"{self.ml_dir}/mace_pot.model", device=self.device)
        os.chdir(nowpath)

    def relax_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_r_' + str(index)
        mace_setup = self.mace_setup.copy()
        mace_setup['task'] = 'relax'
        with open('maceSetup.yaml', 'w') as f:
            f.write(yaml.dump(mace_setup))
        content = "python -m magus.calculators.mace maceSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='relax.sh',
                   out='relax-out', err='relax-err')

    def scf_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_s_' + str(index)
        mace_setup = self.mace_setup.copy()
        mace_setup['task'] = 'scf'
        with open('maceSetup.yaml', 'w') as f:
            f.write(yaml.dump(mace_setup))
        content = "python -m magus.calculators.mace maceSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='scf.sh',
                   out='scf-out', err='relax-err')

    def desc_job(self, index):
        self.cp_input_to()
        job_name = self.job_prefix + '_d_' + str(index)
        mace_setup = self.mace_setup.copy()
        mace_setup['task'] = 'desc'
        mace_setup['model_paths'] = f'{self.ml_dir}/mace_desc.model'
        with open('maceSetup.yaml', 'w') as f:
            f.write(yaml.dump(mace_setup))
        content = "python -m magus.calculators.mace maceSetup.yaml initPop.traj optPop.traj"
        self.J.sub(content, name=job_name, file='desc.sh',
                   out='desc-out', err='relax-err')

    def desc_(self, calcPop):
        assert self.mode == 'parallel', 'Currently only parallel calculation is supported for descriptor calculation.'
        if self.mode == 'parallel':
            self.paralleljob(calcPop, self.desc_job)
            descPop = self.read_parallel_results()
            self.J.clear()

        return descPop

    def select(self, pop):
        assert self.selection in ['fps', 'random'], "selectiom must be fps or random!"
        log.debug(f"Number of selection : {self.n_sample}")
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        if not os.path.exists("mace_pot.model"):
            log.warning("No mace_pot.model, so we have to select all the structures.")
            return pop
        if len(self.trainset) == 0:
            log.warning("No initial training set, so we have to select all the structures.")
            return pop
        # pot = MACEAseCalculator(model_paths="mace_pot.model", device=self.device)
        # des_current = np.array(
        #     [np.mean(pot.get_descriptors(atoms), axis=0) for atoms in self.trainset])
        # des_new = np.array(
        #     [np.mean(pot.get_descriptors(atoms), axis=0) for atoms in pop])
        if self.selection == 'fps':
            log.debug("Select structures by FPS.")
            log.debug("Compute descriptor of new structures.")
            new_pop = self.desc_(pop)
            log.debug("Compute descriptor of training structures.")
            train_pop = self.desc_(self.trainset)
            des_new = np.array([atoms.info['descriptor'] for atoms in new_pop])
            des_train = np.array([atoms.info['descriptor'] for atoms in train_pop])
            sampler = FarthestPointSample(min_distance=0)
            indices = sampler.select(des_new, des_train, max_select=self.n_sample)
            # log.debug(f"FPS indices: {indices}")
            ret = [new_pop[i] for i in indices]
        elif self.selection == 'random':
            log.debug("Select structures randomly.")
            if self.n_sample == None:
                log.debug("No n_sample. Sample all the structures")
                return pop
            else:
                indices = random.sample(range(len(pop)), self.n_sample)
                ret = [pop[i] for i in indices]


        os.chdir(nowpath)
        if isinstance(pop, Population):
            return pop.__class__(ret)
        return ret

    def expand_perturb(self, pop):
        # expand dataset by perturbation
        if self.n_perturb > 0:
            log.debug("Expand dataset by perturbation")
            ret = apply_peturb(pop, self.n_perturb,  self.std_atom_move, self.std_lat_move, self.perturb_keep_init, rndType='normal')
            if isinstance(pop, Population):
                return pop.__class__(ret)
            return ret
        else:
            return pop

    def get_loss(self, frames):
        mace_result = self.calc_efs(frames)
        mace_e = np.array([atoms.info['energy'] / len(atoms)
                         for atoms in mace_result])
        dft_e = np.array([atoms.get_potential_energy() / len(atoms)
                         for atoms in frames])
        mae = np.abs(mace_e-dft_e).mean()
        return [mae]

    def calc_efs(self, frames):
        if isinstance(frames, Atoms):
            frames = [frames]
        # need to copy, prevent losing dft info
        return self.scf_([atoms.copy() for atoms in frames])

    @property
    def trainset(self):
        try:
            # return read(f'{self.ml_dir}/train.xyz', ':')
            return read(f'{self.ml_dir}/all_train.xyz', ':')
        except:
            log.warning("No training set now.")
            return []

    def updatedataset(self, frames):
        nowpath = os.getcwd()
        os.chdir(self.ml_dir)
        cleanArr = []
        for atoms in frames:
            clAts = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
            clAts.arrays['forces'] = atoms.info['forces']
            clAts.info['energy'] = atoms.info['energy']
            clAts.info['stress'] = atoms.info['stress']
            cleanArr.append(clAts)
        trainSet, validSet, testSet = split_dataset(cleanArr, self.split_ratios)
        setDic = {
                'train': trainSet,
                'valid': validSet,
                'test': testSet,
                }
        for key, data in setDic.items():
            write(f'new_{key}.xyz', data)
            os.system(f"cat new_{key}.xyz >> all_{key}.xyz")
            if self.train_mode == 'all':
                os.system(f"cat all_{key}.xyz > {key}.xyz")
            elif self.train_mode == 'new':
                os.system(f"cat new_{key}.xyz > {key}.xyz")
            elif self.train_mode == 'mix':
                preLen = round(len(data)*self.mix_ratio)
                # read previous dataset before appending
                # randomly sample previous dataset
                if os.path.exists(f"all_{key}.xyz"):
                    preData = read(f"all_{key}.xyz", ':')
                else:
                    # No initial dataset in this case
                    preData = []
                if preLen >= len(preData):
                    write(f'sample_prev_{key}.xyz', preData)
                else:
                    write(f'sample_prev_{key}.xyz', random.sample(preData, preLen))
                # merge datasets
                if key == 'test':
                    os.system(f"cat all_{key}.xyz > {key}.xyz")
                else:
                    os.system(f"cat sample_prev_{key}.xyz new_{key}.xyz > {key}.xyz")
            else:
                raise Exception("train_mode must be all, new or mix")

        #write('new_train.xyz', trainSet)
        #write('new_valid.xyz', validSet)
        #write('new_test.xyz', testSet)
        #os.system("cat new_train.xyz >> all_train.xyz")
        #os.system("cat new_valid.xyz >> all_valid.xyz")
        #os.system("cat new_test.xyz >> all_test.xyz")


        os.chdir(nowpath)

def calc_mace(mace_setup, frames):
    optimizer_dict = {
        'bfgs': BFGS,
        'lbfgs': LBFGS,
        'fire': FIRE,
    }
    task = mace_setup['task']
    assert task in ['relax', 'scf', 'desc']
    model_paths = mace_setup['model_paths']
    pressure = mace_setup['pressure']
    max_move = mace_setup['max_move']
    max_step = mace_setup['max_step']
    max_mace_force = mace_setup['max_mace_force']
    min_mace_dratio = mace_setup['min_mace_dratio']
    eps = mace_setup['eps']
    filter_force = mace_setup['filter_force']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    logfile='aserelax.log'
    trajname='calc.traj'
    new_frames = []

    # define optimizer which ends when forces are too large
    class optimizer(optimizer_dict[mace_setup['optimizer']]):
        def converged(self, forces=None):
            # Note: here self.atoms is a Filter not Atoms, so self.atoms.atoms is used.
            cutoffs = natural_cutoffs(self.atoms.atoms, mult=min_mace_dratio)
            nlInds = neighbor_list('i', self.atoms.atoms, cutoffs)
            if len(nlInds) > 0:
                #write('dist.vasp', self.optimizable.atom)
                raise Exception('Too small distance during relaxation')
            if forces is None:
                forces = self.optimizable.get_forces()
            if np.abs(forces).max() > max_mace_force:
                raise Exception('Too large forces during relaxation')
            return self.optimizable.converged(forces, self.fmax)

    calc = MACEAseCalculator(model_paths=model_paths, device=device)
    for i, atoms in enumerate(frames):
        atoms.calc = calc
        if task == 'desc':
            desc = np.mean(calc.get_descriptors(atoms), axis=0)
            atoms.info['descriptor'] = desc
        else:
            if task == 'relax':
                ucf = ExpCellFilter(atoms, scalar_pressure=pressure * GPa)
                gopt = optimizer(ucf, maxstep=max_move, logfile=logfile, trajectory=trajname)
                try:
                    label = gopt.run(fmax=eps, steps=max_step)
                    traj = read(trajname, ':')
                    log.debug(f'relax steps: {len(traj)}')
                # except Converged:
                #     continue
                except Exception:
                    continue
                if filter_force:
                    finalF = atoms.get_forces()
                    fmax = np.sqrt((finalF ** 2).sum(axis=1).max())
                    if fmax > eps:
                        continue

            atoms.info['energy'] = atoms.get_potential_energy()
            atoms.info['forces'] = atoms.get_forces()
            try:
                atoms.info['stress'] = atoms.get_stress()
            except:
                pass


            enthalpy = (atoms.info['energy'] + pressure * atoms.get_volume() * GPa)/ len(atoms)
            # atoms.info['enthalpy'] = round(enthalpy, 6)
            atoms.info['enthalpy'] = enthalpy
            # atoms.info['trajs'] = traj

        atoms.wrap()
        atoms.calc = None
        new_frames.append(atoms)

    return new_frames

if  __name__ == "__main__":
    mace_setup_file, input_traj, output_traj = sys.argv[1:]
    mace_setup = yaml.load(open(mace_setup_file), Loader=yaml.FullLoader)
    # calc = get_calc(mace_setup)
    init_pop = read(input_traj, format='traj', index=':',)
    opt_pop = calc_mace(mace_setup, init_pop)
    write(output_traj, opt_pop)
