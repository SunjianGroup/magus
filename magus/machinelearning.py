#ML module
import os, logging,traceback,copy
import numpy as np
from ase import Atoms, Atom, units
from ase.optimize import BFGS,FIRE
from ase.units import GPa
from ase.constraints import UnitCellFilter,ExpCellFilter
from ase.optimize import BFGS, LBFGS, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, Converged
from ase.data import atomic_numbers
from .utils import del_duplicate
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

    def __init__(self, reg, cf ,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.reg = reg
        self.cf = cf

    def load_calculator(self,filename):
        with open(filename) as f:
            data = yaml.load(f)
        self.reg = LinearRegression()
        self.reg.coef_=data['coef_']
        self.reg.intercept_ = data['intercept_']
        cutoff = data['cutoff']
        nmax = data['nmax']
        lmax = data['lmax']
        ncut = data['ncut']
        diag = data['diag']
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

optimizers={'BFGS':BFGS,'FIRE':FIRE}
class LRmodel(MachineLearning,ASECalculator):
    def __init__(self,parameters):
        self.parameters=parameters
        self.cf = ZernikeFp(parameters)
        self.w_energy = 30.0
        self.w_force = 1.0
        self.w_stress = 1.0
        self.X = None
        #self.optimizer=optimizers[parameters.mloptimizer]
        self.dataset = []
        self.mlDir = '{}/MLFold'.format(self.parameters.workDir)
        if not os.path.exists(self.mlDir):
            os.mkdir(self.mlDir)
        ASECalculator.__init__(self,parameters)

    def train(self):
        logging.info('{} in dataset,training begin!'.format(len(self.dataset)))
        self.reg = LinearRegression().fit(self.X, self.y, self.w)
        calc = LRCalculator(self.reg,self.cf)
        calc.save_calculator('{}/mlparameters'.format(self.mlDir))
        logging.info('training end')

    def get_data(self,images,implemented_properties = ['energy', 'forces']):
        X,y,w,n=[],[],[],[]
        for atoms in images:
            eFps, fFps, sFps = self.cf.get_all_fingerprints(atoms)
            totNd = eFps.shape[1]
            if 'energy' in implemented_properties:
                X.append(np.mean(eFps,axis=0))
                w.append(self.w_energy)
                n.append(1.0)
                # y.append(atoms.info['energy']/len(atoms))
                try:
                    y.append(atoms.info['energy']/len(atoms))
                except:
                    y.append(0.0)
            if 'forces' in implemented_properties:
                fFps = np.sum(fFps, axis=0)
                X.extend(fFps.reshape(-1,totNd))
                w.extend([self.w_force]*len(atoms)*3)
                n.extend([0.0]*len(atoms)*3)
                y.extend(atoms.info['forces'].reshape(-1))
            if 'stress' in implemented_properties:
                sFps = np.sum(sFps, axis=0)
                X.extend(sFps.reshape(-1,totNd))
                w.extend([self.w_stress]*6)
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
            X,y,w = self.get_data(newdata)
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

        # Evaluate stress
        X,y,w = self.get_data(images,['stress'])
        yp = self.reg.predict(X)
        mae_stresses = mean_absolute_error(y, yp)
        r2_stresses = self.reg.score(X, y, w)
        #np.savez('stress',y=y,yp=yp)
        return mae_energies, r2_energies, mae_forces, r2_forces ,mae_stresses ,r2_stresses

    def get_fp(self,pop):
        for ind in pop:
            properties = ['energy']
            for s in ['forces', 'stress']:
                if s in ind.info:
                    properties.append(s)
            X,_,_ = self.get_data([ind], implemented_properties=properties)
            ind.info['image_fp']=X[0,1:]

    def relax(self,calcPop):
        calcs = [LRCalculator(self.reg,self.cf)]
        return super().relax(calcPop,calcs)
        

    def scf(self,calcPop):
        calcs = [LRCalculator(self.reg,self.cf)]
        return super().scf(calcPop,calcs)