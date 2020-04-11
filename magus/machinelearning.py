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
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
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
        #self.reg = LinearRegression()
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
        #self.reg = LinearRegression().fit(self.X, self.y, self.w)
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

    def get_fp(self,pop):
        for ind in pop:
            properties = ['energy']
            for s in ['forces', 'stress']:
                if s in ind.info:
                    properties.append(s)
            X,_,_ = self.get_data([ind], implemented_properties=properties)
            ind.info['image_fp']=X[0,1:]

    def relax(self,calcPop):
        calcs = [self.calc]
        return super().relax(calcPop,calcs)
        

    def scf(self,calcPop):
        calcs = [self.calc]
        return super().scf(calcPop,calcs)

class GPRmodel(MachineLearning,ASECalculator):
    def __init__(self, descriptor=None, kernel='double', prior=None, n_restarts_optimizer=1, template_structure=None):
        if descriptor is None:
            self.descriptor = Fingerprint()
        else:
            self.descriptor = descriptor
        Nsplit_eta = None
        if template_structure is not None:
            self.descriptor.initialize_from_atoms(template_structure)
            if hasattr(self.descriptor, 'use_angular'):
                if self.descriptor.use_angular:
                    Nsplit_eta = self.descriptor.Nelements_2body

        if kernel is 'single':
            self.kernel = GaussKernel(Nsplit_eta=Nsplit_eta)
        elif kernel is 'double':
            self.kernel = DoubleGaussKernel(Nsplit_eta=Nsplit_eta)
        else:
            self.kernel = kernel

        if prior is None:
            self.prior = RepulsivePrior()
        else:
            self.prior = prior

        self.n_restarts_optimizer = n_restarts_optimizer

        self.memory = gpr_memory(self.descriptor, self.prior)

    def predict_energy(self, a, eval_std=False):
        """Evaluate the energy predicted by the GPR-model.

        parameters:

        a: Atoms object
            The structure to evaluate.

        eval_std: bool
            In addition to the force, predict also force contribution
            arrising from including the standard deviation of the
            predicted energy.
        """
        x = self.descriptor.get_feature(a)
        k = self.kernel.kernel_vector(x, self.X)

        E = np.dot(k,self.alpha) + self.bias + self.prior.energy(a)

        if eval_std:
            # Lines 5 and 6 in GPML
            vk = np.dot(self.K_inv, k)
            E_std = np.sqrt(self.K0 - np.dot(k, vk))
            return E, E_std
        else:
            return E

    def predict_forces(self, a, eval_with_energy_std=False):
        """Evaluate the force predicted by the GPR-model.

        parameters:

        a: Atoms object
            The structure to evaluate.

        eval_with_energy_std: bool
            In addition to the force, predict also force contribution
            arrising from including the standard deviation of the
            predicted energy.
        """

        # Calculate descriptor and its gradient
        x = self.descriptor.get_feature(a)
        x_ddr = self.descriptor.get_featureGradient(a).T

        # Calculate kernel and its derivative
        k_ddx = self.kernel.kernel_jacobian(x, self.X)
        k_ddr = np.dot(k_ddx, x_ddr)

        F = -np.dot(k_ddr.T, self.alpha) + self.prior.forces(a)

        if eval_with_energy_std:
            k = self.kernel.kernel_vector(x, self.X)
            vk = np.dot(self.K_inv, k)
            g = self.K0 - np.dot(k.T, vk)
            assert g >= 0
            F_std = 1/np.sqrt(g) * np.dot(k_ddr.T, vk)
            return F.reshape((-1,3)), F_std.reshape(-1,3)
        else:
            return F.reshape(-1,3)

    def update_bias(self):
        self.bias = np.mean(self.memory.energies - self.memory.prior_values)

    def train(self, atoms_list=None, add_data=True):
        if atoms_list is not None:
            assert isinstance(atoms_list, list)
            if not len(atoms_list) == 0:
                self.memory.save_data(atoms_list, add_data)

        self.update_bias()
        self.E, self.X, self.prior_values = self.memory.get_data()
        self.Y = self.E - self.prior_values - self.bias
        
        K = self.kernel(self.X)
        L = cholesky(K, lower=True)
        
        self.alpha = cho_solve((L, True), self.Y)
        self.K_inv = cho_solve((L, True), np.eye(K.shape[0]))
        self.K0 = self.kernel.kernel_value(self.X[0], self.X[0])
    
    def optimize_hyperparameters(self, atoms_list=None, add_data=True, comm=None):
        if self.n_restarts_optimizer == 0:
            self.train(atoms_list)
            return

        if atoms_list is not None:
            assert isinstance(atoms_list, list)
            if not len(atoms_list) == 0:
                self.memory.save_data(atoms_list, add_data)

        self.update_bias()
        self.E, self.X, self.prior_values = self.memory.get_data()
        self.Y = self.E - self.prior_values - self.bias

        results = []
        for i in range(self.n_restarts_optimizer):
            theta_initial = np.random.uniform(self.kernel.theta_bounds[:, 0],
                                              self.kernel.theta_bounds[:, 1])
            if i == 0:
                # Make sure that the previously currently choosen
                # hyperparameters are always tried as initial values.
                if comm is not None:
                    # But only on a single communicator, if multiple are present.
                    if comm.rank == 0:
                        theta_initial = self.kernel.theta
                else:
                    theta_initial = self.kernel.theta
                        
            res = self.constrained_optimization(theta_initial)
            results.append(res)
        index_min = np.argmin(np.array([r[1] for r in results]))
        result_min = results[index_min]
        
        if comm is not None:
        # Find best hyperparameters among all communicators and broadcast.
            results_all = comm.gather(result_min, root=0)
            if comm.rank == 0:
                index_all_min = np.argmin(np.array([r[1] for r in results_all]))
                result_min = results_all[index_all_min]
            else:
                result_min = None
            result_min = comm.bcast(result_min, root=0)
                
        self.kernel.theta = result_min[0]
        self.lml = -result_min[1]

        self.train()
    
    def neg_log_marginal_likelihood(self, theta=None, eval_gradient=True):
        if theta is not None:
            self.kernel.theta = theta

        if eval_gradient:
            K, K_gradient = self.kernel(self.X, eval_gradient)
        else:
            K = self.kernel(self.X)

        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), self.Y)

        lml = -0.5 * np.dot(self.Y, alpha)
        lml -= np.sum(np.log(np.diag(L)))
        lml -= K.shape[0]/2 * np.log(2*np.pi)
        
        if eval_gradient:
            # Equation (5.9) in GPML
            K_inv = cho_solve((L, True), np.eye(K.shape[0]))
            tmp = np.einsum("i,j->ij", alpha, alpha) - K_inv

            lml_gradient = 0.5*np.einsum("ij,kij->k", tmp, K_gradient)
            return -lml, -lml_gradient
        else:
            return -lml

    def constrained_optimization(self, theta_initial):
        theta_opt, func_min, convergence_dict = \
            fmin_l_bfgs_b(self.neg_log_marginal_likelihood,
                          theta_initial,
                          bounds=self.kernel.theta_bounds)
        return theta_opt, func_min

    def numerical_neg_lml(self, dx=1e-4):
        N_data = self.X.shape[0]
        theta = np.copy(self.kernel.theta)
        N_hyper = len(theta)
        lml_ddTheta = np.zeros((N_hyper))
        for i in range(N_hyper):
            theta_up = np.copy(theta)
            theta_down = np.copy(theta)
            theta_up[i] += 0.5*dx
            theta_down[i] -= 0.5*dx

            lml_up = self.neg_log_marginal_likelihood(theta_up, eval_gradient=False)
            lml_down = self.neg_log_marginal_likelihood(theta_down, eval_gradient=False)
            lml_ddTheta[i] = (lml_up - lml_down)/dx
        return lml_ddTheta

    def numerical_forces(self, a, dx=1e-4, eval_std=False):
        Na, Nd = a.positions.shape
        if not eval_std:
            F = np.zeros((Na,Nd))
            for ia in range(Na):
                for idim in range(Nd):
                    a_up = a.copy()
                    a_down = a.copy()
                    a_up.positions[ia,idim] += 0.5*dx
                    a_down.positions[ia,idim] -= 0.5*dx
                    
                    E_up = self.predict_energy(a_up)
                    E_down = self.predict_energy(a_down)
                    F[ia,idim] = -(E_up - E_down)/dx
            return F
        else:
            F = np.zeros((Na,Nd))
            Fstd = np.zeros((Na,Nd))
            for ia in range(Na):
                for idim in range(Nd):
                    a_up = a.copy()
                    a_down = a.copy()
                    a_up.positions[ia,idim] += 0.5*dx
                    a_down.positions[ia,idim] -= 0.5*dx
                    
                    E_up, Estd_up = self.predict_energy(a_up, eval_std=True)
                    E_down, Estd_down = self.predict_energy(a_down, eval_std=True)
                    F[ia,idim] = -(E_up - E_down)/dx
                    Fstd[ia,idim] = -(Estd_up - Estd_down)/dx
            return F, Fstd

    def get_calculator(self, kappa):
        return gpr_calculator(self, kappa)
