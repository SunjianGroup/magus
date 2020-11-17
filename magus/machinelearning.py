#ML module
import os, logging,traceback,copy
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from ase import Atoms, Atom, units
from ase.optimize import BFGS,FIRE
from ase.units import GPa
from ase.io import read,write
from ase.constraints import UnitCellFilter,ExpCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from ase.data import atomic_numbers
from .utils import *
import copy
import yaml
try:
    from ani.environment import ASEEnvironment
    from ani.kernel import RBF
    from ani.cutoff import CosineCutoff
    from ani.model import ANI, GPR
    from ani.dataloader import AtomsData, convert_frames
    from ani.symmetry_functions import BehlerG1, Zernike, CombinationRepresentation
    from torch.utils.data import DataLoader
    import torch
    from ani.prior import *
except:
    pass
class MachineLearning:
    def __init__(self):
        pass

    def train(self):
        pass

    def updatedataset(self,images):
        pass

    def getloss(self,images):
        pass


from .descriptor import ZernikeFp
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
from sklearn.metrics import mean_absolute_error
from ase.calculators.calculator import Calculator, all_changes
from .localopt import ASECalculator, MLCalculator_tmp
class LRCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    nolabel = True

    def __init__(self, cf, reg=None ,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.reg = reg
        self.cf = cf

    def load_calculator(self,filename):
        with open(filename) as f:
            data = yaml.load(f)
        #TODO use bayesian linear regression to calculate uncertainty
        #self.reg = LinearRegression()
        #self.reg = Lasso()
        self.reg = BayesianRidge()
        self.reg.coef_=data['coef_']
        self.reg.intercept_ = data['intercept_']
        cutoff = data['cutoff']
        nmax = data['nmax']
        lmax = data['lmax']
        ncut = data['ncut']
        diag = data['diag']
        elems = data['elems']
        self.cf = ZernikeFp(cutoff, nmax, lmax, ncut, elems,diag=diag)

    def save_calculator(self,filename):
        data={'coef_':self.reg.coef_,'intercept_':self.reg.intercept_,
            'cutoff':self.cf.cutoff,'nmax':self.cf.nmax,'lmax':self.cf.lmax,
            'ncut':self.cf.ncut,'elems':self.cf.elems,'diag':self.cf.diag}
        with open(filename,'w') as f:
            yaml.dump(data,f)   

    def get_potential_energies(self, atoms=None):
        if atoms is not None:
            self.atoms = atoms.copy()
        eFps, _, _ = self.cf.get_all_fingerprints(self.atoms)
        X=np.concatenate((np.ones((len(self.atoms),1)),eFps),axis=1)
        y=self.reg.predict(X)
        return y

    def calculate(self, atoms=None,properties=['energy'],system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()
        X,n=[],[]

        eFps, fFps, sFps = self.cf.get_all_fingerprints(self.atoms)
        fFps = np.sum(fFps,axis=0)
        sFps = np.sum(sFps,axis=0)
        X.append(np.mean(eFps,axis=0))
        X.extend(fFps.reshape(-1,eFps.shape[1]))
        X.extend(sFps.reshape(-1,eFps.shape[1]))
        n.extend([1.0]+[0.0]*len(self.atoms)*3+[0.0]*6)
        X=np.array(X)
        n=np.array(n)
        X=np.concatenate((n.reshape(-1,1),X),axis=1)

        y=self.reg.predict(X)
        self.results['energy'] = y[0]*len(self.atoms)
        self.results['free_energy'] = y[0]*len(self.atoms)
        self.results['forces'] = y[1:-6].reshape((len(self.atoms),3))
        self.results['stress'] = y[-6:]/self.atoms.get_volume()/2


class gpr_calculator(Calculator):
    implemented_properties = ['energy', 'forces']
    default_parameters = {}

    def __init__(self, gpr, kappa=None, **kwargs):
        self.gpr = gpr
        self.kappa = kappa
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            if self.kappa is None:
                E = self.gpr.predict_energy(atoms, eval_std=False)
            else:
                E, Estd = self.gpr.predict_energy(atoms, eval_std=True)
                E = E - self.kappa*Estd
            self.results['energy'] = E

        if 'forces' in properties:
            if self.kappa is None:
                F = self.gpr.predict_forces(atoms)
            else:
                F, Fstd = self.gpr.predict_forces(atoms, eval_with_energy_std=True)
                F = F - self.kappa*Fstd
            self.results['forces'] = F

class torch_gpr_calculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {}

    def __init__(self, gpr, kappa=None, **kwargs):
        self.gpr = gpr
        self.kappa = kappa
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            if self.kappa is None:
                E = self.gpr.predict_energy(atoms, eval_std=False)
            else:
                E, Estd = self.gpr.predict_energy(atoms, eval_std=True)
                E = E - self.kappa*Estd
            self.results['energy'] = E

        if 'forces' in properties:
            F = self.gpr.predict_forces(atoms)
            self.results['forces'] = F

        if 'stress' in properties:
            S = self.gpr.predict_stress(atoms)
            self.results['stress'] = S

class bayeslr_calculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {}

    def __init__(self, bayeslr, kappa=None, **kwargs):
        self.bayeslr = bayeslr
        self.kappa = kappa
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            if self.kappa is None:
                E = self.bayeslr.predict_energy(atoms, eval_std=False)
            else:
                E, Estd = self.bayeslr.predict_energy(atoms, eval_std=True)
                E = E - self.kappa*Estd
            self.results['energy'] = E

        if 'forces' in properties:
            if self.kappa is None:
                F = self.bayeslr.predict_forces(atoms)
            else:
                F, Fstd = self.bayeslr.predict_forces(atoms, eval_std=True)
                F = F - self.kappa*Fstd
            self.results['forces'] = F

        if 'stress' in properties:
            S = self.bayeslr.predict_stress(atoms)
            self.results['stress'] = S

optimizers={'BFGS':BFGS,'FIRE':FIRE}
class LRmodel(MachineLearning,ASECalculator):
    def __init__(self,parameters):
        self.p = EmptyClass()
        
        Requirement = ['mlDir', 'maxDataset']
        Default = {'w_energy':30.0,'w_force':1.0,'w_stress':-1.0}
        checkParameters(self.p,parameters,Requirement,Default)

        train_property = []
        if self.p.w_energy > 0:
            train_property.append('energy')
        if self.p.w_force > 0:
            train_property.append('forces')
        if self.p.w_stress > 0:
            train_property.append('stress')
        self.p.train_property = train_property

        p = copy.deepcopy(parameters)
        for key, val in parameters.MLCalculator.items():
            setattr(p, key, val)
        p.workDir = parameters.workDir
        ASECalculator.__init__(self,p)

        self.X = None
        self.cf = ZernikeFp(parameters)
        self.dataset = []

        if not os.path.exists(self.p.mlDir):
            os.mkdir(self.p.mlDir)

        self.calc = LRCalculator(self.cf)

    def train(self):
        logging.info('{} in dataset,training begin!'.format(len(self.dataset)))
        #self.reg = LinearRegression().fit(self.X, self.y, self.w)
        #self.reg = Lasso().fit(self.X, self.y)
        self.reg = BayesianRidge().fit(self.X, self.y)
        self.calc.reg = self.reg
        self.calc.save_calculator('{}/mlparameters'.format(self.p.mlDir))
        logging.info('training end')

    def get_data(self,images,implemented_properties = ['energy', 'forces']):
        X,y,w,n=[],[],[],[]
        for atoms in images:
            eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
            totNd = eFps.shape[1]
            if 'energy' in implemented_properties:
                X.append(np.mean(eFps,axis=0))
                w.append(self.p.w_energy)
                n.append(1.0)
                # y.append(atoms.info['energy']/len(atoms))
                try:
                    y.append(atoms.info['energy']/len(atoms))
                except:
                    y.append(0.0)
            if 'forces' in implemented_properties:
                fFps = np.sum(fFps, axis=0)
                X.extend(fFps.reshape(-1,totNd))
                w.extend([self.p.w_force]*len(atoms)*3)
                n.extend([0.0]*len(atoms)*3)
                y.extend(atoms.info['forces'].reshape(-1))
            if 'stress' in implemented_properties:
                sFps = np.sum(sFps, axis=0)
                X.extend(sFps.reshape(-1,totNd))
                w.extend([self.p.w_stress]*6)
                n.extend([0.0]*6)
                y.extend(atoms.info['stress'].reshape(-1))
        X=np.array(X)
        w=np.array(w)
        y=np.array(y)
        n=np.array(n)
        X=np.concatenate((n.reshape(-1,1),X),axis=1)
        return X,y,w

    def dataNum(self):
        return len(self.dataset)

    def cleardataset(self):
        self.dataset = []

    def updatedataset(self,images):
        alldata=copy.deepcopy(self.dataset)
        alldata.extend(images)
        alldata=del_duplicate(alldata)
        if len(alldata) > self.p.maxDataset:
            self.cleardataset()
            alldata = alldata[-self.p.maxDataset:]
        newdata=[]
        for data in alldata:
            if data not in self.dataset:
                newdata.append(data)
        if newdata:
            self.dataset.extend(newdata)
            X,y,w = self.get_data(newdata,self.p.train_property)
            if self.X is None:
                self.X,self.y,self.w=X,y,w
            else:
                self.X=np.concatenate((self.X,X),axis=0)
                self.y=np.concatenate((self.y,y),axis=0)
                self.w=np.concatenate((self.w,w),axis=0)

    def get_loss(self,images):
        # Evaluate energy
        X,y,w = self.get_data(images,['energy'])
        yp = self.reg.predict(X)
        mae_energies = mean_absolute_error(y, yp)
        r2_energies = self.reg.score(X, y, w)

        # Evaluate force
        X,y,w = self.get_data(images,['forces'])
        yp = self.reg.predict(X)
        mae_forces = mean_absolute_error(y, yp)
        r2_forces = self.reg.score(X, y, w)
        """
        # Evaluate stress
        X,y,w = self.get_data(images,['stress'])
        yp = self.reg.predict(X)
        mae_stresses = mean_absolute_error(y, yp)
        r2_stresses = self.reg.score(X, y, w)
        #np.savez('stress',y=y,yp=yp)
        """
        return mae_energies, r2_energies, mae_forces, r2_forces #,mae_stresses ,r2_stresses

    def relax(self,calcPop):
        calcs = [self.calc]
        return super().relax(calcPop,calcs)
    
    def scf(self,calcPop):
        calcs = [self.calc]
        return super().scf(calcPop,calcs)

class Prior:
    def energy(self,atoms):
        return 0
    def forces(self,atoms):
        return np.zeros_like(atoms.positions).reshape(-1)

class GPRmodel(MachineLearning,ASECalculator):
    def __init__(self,parameters,cf=None):
        self.p = EmptyClass()
        
        Requirement = ['mlDir']
        Default = {'w_energy':30.0,'w_force':1.0,'w_stress':-1.0,'norm':False,'cf':'zernike'}
        checkParameters(self.p,parameters,Requirement,Default)

        p = copy.deepcopy(parameters)
        for key, val in parameters.MLCalculator.items():
            setattr(p, key, val)
        p.workDir = parameters.workDir
        ASECalculator.__init__(self,p)

        self.X = None
        if self.p.cf == 'gofee':
            raise Exception('Shi da bian le, da ren')
        elif self.p.cf == 'zernike':
            self.cf = ZernikeFp(parameters)
        self.dataset = []

        from .kernel import GaussKernel
        self.kernel = GaussKernel()

        self.prior = Prior()
        if not os.path.exists(self.p.mlDir):
            os.mkdir(self.p.mlDir)

    def update_bias(self):
        self.bias = np.mean(self.E - self.prior_values)

    def update_mean_std(self):
        if self.p.norm:
            self.mean = np.mean(self.X,axis=0)
            self.std = np.std(self.X,axis=0)
            self.std[np.where(self.std == 0.0)] = 1
        else:
            self.mean,self.std = 0.0,1.0

    def train(self):
        logging.info('{} in dataset,training begin!'.format(len(self.dataset)))
        X_ = (self.X-self.mean)/self.std
        K = self.kernel(X_)
        L = cholesky(K, lower=True)
        
        self.alpha = cho_solve((L, True), self.Y)
        self.K_inv = cho_solve((L, True), np.eye(K.shape[0]))
        self.K0 = self.kernel.kernel_value(X_[0], X_[0])
        logging.info('training end')

    def get_loss(self,images):
        # Evaluate energy
        Ypredict = np.array([self.predict_energy(atoms) for atoms in images])
        Y = np.array([atoms.info['energy'] for atoms in images])
        mae_energies = np.mean(np.abs(Ypredict-Y))
        r2_energies = 1 - np.sum((Y - Ypredict)**2)/np.sum((Y- np.mean(Y))**2)

        # Evaluate force
        Ypredict = np.array([self.predict_forces(atoms) for atoms in images])
        Y = np.array([atoms.info['forces'] for atoms in images])
        mae_forces = np.mean(np.abs(Ypredict-Y))
        r2_forces = 1 - np.sum((Y - Ypredict)**2)/np.sum((Y- np.mean(Y))**2)

        return mae_energies, r2_energies, mae_forces, r2_forces

    def get_data(self,images):
        X,E,p=[],[],[]
        for atoms in images:
            eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
            X.append(eFps)
            E.append(atoms.info['energy'])
            p.append(self.prior.energy(atoms))

        X=np.array(X)
        E=np.array(E)
        p=np.array(p)
        return X,E,p

    def updatedataset(self,images):
        alldata=copy.deepcopy(self.dataset)
        alldata.extend(images)
        alldata=del_duplicate(alldata)
        newdata=[]
        for data in alldata:
            if data not in self.dataset:
                newdata.append(data)
        if newdata:
            self.dataset.extend(newdata)
            X,E,prior_values = self.get_data(newdata)
            if self.X is None:
                self.X,self.E,self.prior_values=X,E,prior_values
            else:
                self.X=np.concatenate((self.X,X),axis=0)
                self.E=np.concatenate((self.E,E),axis=0)
                self.prior_values=np.concatenate((self.prior_values,prior_values),axis=0)
        self.update_bias()
        self.update_mean_std()
        self.Y = self.E - self.prior_values - self.bias

    def relax(self,calcPop):
        calcs = [self.get_calculator()]
        return super().relax(calcPop,calcs)

    def scf(self,calcPop):
        calcs = [self.get_calculator()]
        return super().scf(calcPop,calcs)

    def predict_energy(self, atoms, eval_std=False):
        eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
        x = (eFps-self.mean)/self.std
        X_ = (self.X-self.mean)/self.std
        k = self.kernel.kernel_vector(x, X_)
        E = np.dot(k,self.alpha) + self.bias + self.prior.energy(atoms)
        if eval_std:
            vk = np.dot(self.K_inv, k)
            E_std = np.sqrt(self.K0 - np.dot(k, vk))
            return E, E_std
        else:
            return E

    def predict_forces(self, atoms, eval_with_energy_std=False):
        # Calculate descriptor and its gradient
        eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
        x_ = (eFps - self.mean)/self.std
        X_ = (self.X-self.mean)/self.std
        x_ddr = (fFps/self.std).T

        # Calculate kernel and its derivative
        k_ddx = self.kernel.kernel_jacobian(x_, X_)
        k_ddr = np.dot(k_ddx, x_ddr)

        F = -np.dot(k_ddr.T, self.alpha) + self.prior.forces(atoms)

        if eval_with_energy_std:
            k = self.kernel.kernel_vector(x_, X_)
            vk = np.dot(self.K_inv, k)
            g = self.K0 - np.dot(k.T, vk)
            assert g >= 0
            F_std = 1/np.sqrt(g) * np.dot(k_ddr.T, vk)
            return F.reshape((-1,3)), F_std.reshape(-1,3)
        else:
            return F.reshape(-1,3)

    def get_calculator(self, kappa=0):
        return gpr_calculator(self, kappa)

class BayesLRmodel(MachineLearning,ASECalculator):
    def __init__(self,parameters):
        self.p = EmptyClass()
        self.reg = BayesianRidge()
        
        Requirement = ['mlDir']
        Default = {'w_energy':30.0,'w_force':-1.0,'w_stress':-1.0,'norm':False,'cf':'zernike'}
        checkParameters(self.p,parameters,Requirement,Default)

        train_property = []
        if self.p.w_energy > 0:
            train_property.append('energy')
        if self.p.w_force > 0:
            train_property.append('forces')
            if self.p.norm:
                logging.info('norm cannot be True when forces need train')
                self.p.norm = False
        if self.p.w_stress > 0:
            train_property.append('stress')
        self.p.train_property = train_property

        p = copy.deepcopy(parameters)
        for key, val in parameters.MLCalculator.items():
            setattr(p, key, val)
        p.workDir = parameters.workDir
        ASECalculator.__init__(self,p)

        self.X = None
        if self.p.cf == 'gofee':
            raise Exception('Shi da bian le, da ren')
        elif self.p.cf == 'zernike':
            self.cf = ZernikeFp(parameters)
        self.dataset = []

        if not os.path.exists(self.p.mlDir):
            os.mkdir(self.p.mlDir)

    def get_loss(self,images):
        # Evaluate energy
        Ypredict = np.array([self.predict_energy(atoms) for atoms in images])
        Y = np.array([atoms.info['energy'] for atoms in images])
        mae_energies = np.mean(np.abs(Ypredict-Y))
        r2_energies = 1 - np.sum((Y - Ypredict)**2)/np.sum((Y- np.mean(Y))**2)

        # Evaluate force
        Ypredict, Y = [], []
        for atoms in images:
            Ypredict.extend(list(self.predict_forces(atoms).reshape(-1)))
            Y.extend(list(atoms.info['forces'].reshape(-1)))
        Ypredict = np.array(Ypredict)
        Y = np.array(Y)
        mae_forces = np.mean(np.abs(Ypredict-Y))
        r2_forces = 1 - np.sum((Y - Ypredict)**2)/np.sum((Y- np.mean(Y))**2)

        # Evaluate stress
        Ypredict = np.array([self.predict_stress(atoms) for atoms in images])
        Y = np.array([atoms.info['stress'] for atoms in images])
        mae_stress = np.mean(np.abs(Ypredict-Y))
        r2_stress = 1 - np.sum((Y - Ypredict)**2)/np.sum((Y- np.mean(Y))**2)

        return mae_energies, r2_energies, mae_forces, r2_forces, mae_stress, r2_stress

    def train(self):
        logging.info('{} in dataset,training begin!'.format(len(self.dataset)))
        self.reg.fit(self.X, self.y,self.w)
        self.K0 = self.reg.predict(self.X, True)[1].mean()**2

    def get_data(self,images,implemented_properties = ['energy', 'forces']):
        X,y,w,n=[],[],[],[]
        for atoms in images:
            eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
            totNd = len(eFps)
            if 'energy' in implemented_properties:
                X.append(eFps)
                w.append(self.p.w_energy)
                n.append(1.0)
                y.append(atoms.info['energy'])
            if 'forces' in implemented_properties:
                X.extend(-fFps.reshape(-1,totNd))
                w.extend([self.p.w_force]*len(atoms)*3)
                n.extend([0.0]*len(atoms)*3)
                y.extend(atoms.info['forces'].reshape(-1))
            if 'stress' in implemented_properties:
                sFps = np.sum(sFps, axis=0)
                X.extend(sFps.reshape(-1,totNd))
                w.extend([self.p.w_stress]*6)
                n.extend([0.0]*6)
                y.extend(atoms.info['stress'].reshape(-1))
        X=np.array(X)
        w=np.array(w)
        y=np.array(y)
        n=np.array(n)
        X=np.concatenate((n.reshape(-1,1),X),axis=1)
        return X,y,w

    def dataNum(self):
        return len(self.dataset)

    def cleardataset(self):
        self.dataset = []

    def updatedataset(self,images):
        write('{}/dataset.traj'.format(self.p.mlDir),self.dataset)
        alldata=copy.deepcopy(self.dataset)
        alldata.extend(images)
        alldata=del_duplicate(alldata)
        newdata=[]
        for data in alldata:
            if data not in self.dataset:
                newdata.append(data)
        if newdata:
            self.dataset.extend(newdata)
            X,y,w = self.get_data(newdata,self.p.train_property)
            if self.X is None:
                self.X,self.y,self.w=X,y,w
            else:
                self.X=np.concatenate((self.X,X),axis=0)
                self.y=np.concatenate((self.y,y),axis=0)
                self.w=np.concatenate((self.w,w),axis=0)

    def relax(self,calcPop):
        calcs = [self.get_calculator()]
        return super().relax(calcPop,calcs)

    def scf(self,calcPop):
        calcs = [self.get_calculator()]
        return super().scf(calcPop,calcs)

    def predict_energy(self, atoms, eval_std=False):
        eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
        X = np.ones(len(eFps)+1)
        X[1:] = eFps
        result = self.reg.predict(X.reshape(1,-1),eval_std)
        if eval_std:
            return result[0][0],result[1][0]
        else:
            return result[0]

    def predict_forces(self, atoms, eval_std=False):
        # Calculate descriptor and its gradient
        eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
        X = np.zeros([3*len(atoms),len(eFps)+1])
        X[:,1:] = -fFps
        result = self.reg.predict(X,eval_std)
        if eval_std:
            return result[0].reshape((len(atoms),3)),result[1].reshape((len(atoms),3))
        else:
            return result.reshape((len(atoms),3))

    def predict_stress(self, atoms, eval_std=False):
        # Calculate descriptor and its gradient
        eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
        X = np.zeros([6,len(eFps)+1])
        X[:,1:] = np.sum(sFps, axis=0)

        result = self.reg.predict(X,eval_std)
        if eval_std:
            return result[0].reshape(6),result[1].reshape(6)
        else:
            return result.reshape(6)
            
    def get_calculator(self, kappa=0):
        return bayeslr_calculator(self, kappa)


class pytorchGPRmodel(MachineLearning, MLCalculator_tmp):
    def __init__(self,parameters):
        self.p = EmptyClass()
        Requirement = ['mlDir']
        Default = {'w_energy':30.0,'w_force':1.0,'w_stress':-1.0,'norm':False,
        'cutoff': 5.0,'n_radius':30,'n_angular':0,'n_max':8,'l_max':8,'diag':False,
        'n_cut':2,'prior':None}
        checkParameters(self.p,parameters,Requirement,Default)

        p = copy.deepcopy(parameters)
        for key, val in parameters.MLCalculator.items():
            setattr(p, key, val)
        p.workDir = parameters.workDir
        MLCalculator_tmp.__init__(self,p)

        if not os.path.exists(self.p.mlDir):
            os.mkdir(self.p.mlDir)

        cutoff = self.p.cutoff
        n_radius = self.p.n_radius
        n_angular = self.p.n_angular
        n_max = self.p.n_max
        l_max = self.p.l_max
        diag = self.p.diag
        n_cut = self.p.n_cut
        self.environment_provider = ASEEnvironment(cutoff)
        cut_fn = CosineCutoff(cutoff)

        elements = tuple(set([atomic_numbers[element] for element in parameters.symbols]))
        # rdf = BehlerG1(n_radius, cut_fn)
        zer = Zernike(elements,n_max,l_max,diag,cutoff,n_cut)
        representation = CombinationRepresentation(zer)
        # self.ani_model = ANI(representation, self.environment_provider)
        kern = RBF(representation.dimension)
        if self.p.prior is None:
            prior = None
        elif self.p.prior == 'repulsive':    
            prior = RepulsivePrior(r_max=cutoff)
        self.model = GPR(representation, kern, self.environment_provider, prior)

    def train(self, epoch1=1000, epoch2=30000):
        # self.ani_model.train(epoch1)
        # self.model.recompute_X_array()
        self.model.train(epoch2)
        tmp = self.model.kern.variance.get().detach().numpy()
        self.K0 = tmp

    def get_loss(self,images):
        batch_data = convert_frames(images, self.environment_provider)
        predict_energy = self.model.get_energies(batch_data).detach().numpy()
        predict_forces = self.model.get_forces(batch_data).detach().numpy()
        predict_stress = self.model.get_stresses(batch_data).detach().numpy()
        target_energy = batch_data['energy'].numpy()
        target_forces = batch_data['forces'].numpy()
        target_stress = batch_data['stress'].numpy()
        mae_energies = np.mean(np.abs(predict_energy - target_energy))
        r2_energies = 1 - np.sum((predict_energy - target_energy)**2) / \
            np.sum((target_energy - np.mean(target_energy))**2)
        mae_forces = np.mean(np.abs(predict_forces - target_forces))
        r2_forces = 1 - np.sum((predict_forces - target_forces)**2) / \
            np.sum((target_forces - np.mean(target_forces))**2)
        mae_stress = np.mean(np.abs(predict_stress - target_stress))
        r2_stress = 1 - np.sum((predict_stress - target_stress)**2) / \
            np.sum((target_stress - np.mean(target_stress))**2)
        return mae_energies, r2_energies, mae_forces, r2_forces, mae_stress, r2_stress

    def updatedataset(self,images):
        self.model.update_dataset(images)
        # self.ani_model.update_dataset(images)

    def relax(self,calcPop):
        calcs = [self.get_calculator()]
        return super().relax(calcPop,calcs)

    def scf(self,calcPop):
        calcs = [self.get_calculator()]
        return super().scf(calcPop,calcs)

    def predict_energy(self, atoms, eval_std=False):
        batch_data = convert_frames([atoms], self.environment_provider)
        E, E_std = self.model.get_energies(batch_data, True)
        E, E_std = E.detach().item(), E_std.detach().item()
        if eval_std:
            return E, E_std
        else:
            return E

    def predict_forces(self, atoms):
        batch_data = convert_frames([atoms], self.environment_provider)
        F = self.model.get_forces(batch_data)
        return F.squeeze().detach().numpy()

    def predict_stress(self, atoms, eval_with_energy_std=False):
        batch_data = convert_frames([atoms], self.environment_provider)
        S = self.model.get_stresses(batch_data)
        return S.squeeze().detach().numpy()

    def get_calculator(self, kappa=0):
        return torch_gpr_calculator(self, kappa)

    def save_model(self, filename):
        torch.save(self.model.state_dict(), '{}/{}.pt'.format(self.p.mlDir, filename))
        L, V = self.model.L.numpy(), self.model.V.numpy()
        mean, std = self.model.mean.numpy(), self.model.std.numpy()
        X_array = self.model.X_array.numpy()
        np.savez('{}/{}.npz'.format(self.p.mlDir, filename),L=L,V=V,mean=mean,std=std,X_array=X_array)
    
    def load_model(self, filename):
        state_dict = torch.load('{}/{}.pt'.format(self.p.mlDir, filename))
        d = np.load('{}/{}.npz'.format(self.p.mlDir, filename))
        self.model.load_state_dict(state_dict)
        self.model.L = torch.tensor(d['L'])
        self.model.V = torch.tensor(d['V'])
        self.model.mean = torch.tensor(d['mean'])
        self.model.std = torch.tensor(d['std'])
        self.model.X_array = torch.tensor(d['X_array'])
