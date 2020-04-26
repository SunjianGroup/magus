#ML module
import os, logging,traceback,copy
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from ase import Atoms, Atom, units
from ase.optimize import BFGS,FIRE
from ase.units import GPa
from ase.constraints import UnitCellFilter,ExpCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from ase.data import atomic_numbers
from .utils import *
import copy
import yaml
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from ase.calculators.calculator import Calculator, all_changes
from .localopt import ASECalculator
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
        self.reg = LinearRegression()
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





optimizers={'BFGS':BFGS,'FIRE':FIRE}
class LRmodel(MachineLearning,ASECalculator):
    def __init__(self,parameters):
        self.p = EmptyClass()
        
        Requirement = ['mlDir']
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
        self.reg = LinearRegression().fit(self.X, self.y, self.w)
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
    def energy(self):
        return 0
class GPRmodel(MachineLearning,ASECalculator):
    def __init__(self,parameters):
        self.p = EmptyClass()
        
        Requirement = ['mlDir']
        Default = {'w_energy':30.0,'w_force':1.0,'w_stress':-1.0}
        checkParameters(self.p,parameters,Requirement,Default)

        p = copy.deepcopy(parameters)
        for key, val in parameters.MLCalculator.items():
            setattr(p, key, val)
        p.workDir = parameters.workDir
        ASECalculator.__init__(self,p)

        self.X = None
        self.cf = ZernikeFp(parameters)
        self.dataset = []

        from .kernel import GaussKernel
        self.kernel = GaussKernel()

        self.prior = Prior()
        if not os.path.exists(self.p.mlDir):
            os.mkdir(self.p.mlDir)

    def update_bias(self):
        self.bias = np.mean(self.E - self.prior_values)

    def train(self):
        logging.info('{} in dataset,training begin!'.format(len(self.dataset)))
        K = self.kernel(self.X)
        L = cholesky(K, lower=True)
        
        self.alpha = cho_solve((L, True), self.Y)
        self.K_inv = cho_solve((L, True), np.eye(K.shape[0]))
        self.K0 = self.kernel.kernel_value(self.X[0], self.X[0])
        logging.info('training end')

    def get_data(self,images):
        X,E,p=[],[],[]
        for atoms in images:
            eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
            X.append(np.mean(eFps,axis=0))
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
                self.p=np.concatenate((self.prior_values,prior_values),axis=0)
        self.update_bias()
        self.Y = self.E - self.prior_values - self.bias

    def relax(self,calcPop):
        calcs = [self.calc]
        return super().relax(calcPop,calcs)

    def scf(self,calcPop):
        calcs = [self.calc]
        return super().scf(calcPop,calcs)

    def predict_energy(self, atoms, eval_std=False):
        eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
        x = np.sum(eFps,axis=0)
        k = self.kernel.kernel_vector(x, self.X)
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
        x = x = np.sum(eFps,axis=0)
        x_ddr = fFps.T

        # Calculate kernel and its derivative
        k_ddx = self.kernel.kernel_jacobian(x, self.X)
        k_ddr = np.dot(k_ddx, x_ddr)

        F = -np.dot(k_ddr.T, self.alpha) + self.prior.forces(atoms)

        if eval_with_energy_std:
            k = self.kernel.kernel_vector(x, self.X)
            vk = np.dot(self.K_inv, k)
            g = self.K0 - np.dot(k.T, vk)
            assert g >= 0
            F_std = 1/np.sqrt(g) * np.dot(k_ddr.T, vk)
            return F.reshape((-1,3)), F_std.reshape(-1,3)
        else:
            return F.reshape(-1,3)