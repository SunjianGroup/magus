#ML module
import os, logging,traceback,copy
import numpy as np
from ase import Atoms, Atom, units
from ase.optimize import BFGS,FIRE
from ase.units import GPa

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

class LRCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    nolabel = True

    def __init__(self, reg, cf ,**kwargs):
        Calculator.__init__(self, **kwargs)
        self.reg = reg
        self.cf = cf

    def calculate(self, atoms=None,properties=['energy'],system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        X,n=[],[]

        eFps, fFps ,sFps= self.cf.get_all_fingerprints(self.atoms)
        fFps = np.sum(fFps,axis=0)

        X.append(np.mean(eFps,axis=0))
        X.extend(fFps.reshape(-1,eFps.shape[1]))
        n.extend([1.0]+[0.0]*len(atoms)*3)
        X=np.array(X)
        n=np.array(n)
        X=np.concatenate((n.reshape(-1,1),X),axis=1)
        
        y=self.reg.predict(X)
        self.results['energy'] = y[0]*len(self.atoms)
        self.results['free_energy'] = y[0]*len(self.atoms)
        self.results['forces'] = y[1:].reshape((len(self.atoms),3))

        X=np.concatenate((np.zeros((9,1)),sFps),axis=1)
        y=self.reg.predict(X)
        self.results['stress'] = y.reshape((3,3))


optimizers={'BFGS':BFGS,'FIRE':FIRE}
class LRmodel(MachineLearning):
    def __init__(self,parameters):
        self.parameters=parameters
        cutoff = 5
        elems = [13]
        nmax = 8
        ncut = 4
        self.cf = ZernikeFp(cutoff, nmax, None, ncut, elems,diag=False)
        self.w_energy = 10.0
        self.w_force = 1.0
        self.X = None
        self.optimizer=optimizers[parameters.mloptimizer]
        
    def train(self):
        logging.info('OvO!')
        self.reg = LinearRegression().fit(self.X, self.y, self.w)

    def get_data(self,images):
        X,y,w,n=[],[],[],[]
        for atoms in images:
            eFps, fFps ,_= self.cf.get_all_fingerprints(atoms)
            fFps = np.sum(fFps,axis=0)

            X.append(np.mean(eFps,axis=0))
            X.extend(fFps.reshape(-1,eFps.shape[1]))
            w.extend([self.w_energy]+[self.w_force]*len(atoms)*3)
            
            y.append(atoms.info['energy']/len(atoms))
            y.extend(atoms.info['forces'].reshape(-1))
            n.extend([1.0]+[0.0]*len(atoms)*3)
        X=np.array(X)
        w=np.array(w)
        y=np.array(y)
        n=np.array(n)
        X=np.concatenate((n.reshape(-1,1),X),axis=1)
        return X,y,w

    def updatedataset(self,images):
        X,y,w = self.get_data(images)
        if self.X is None:
            self.X,self.y,self.w=X,y,w
        else:
            self.X=np.concatenate((self.X,X),axis=0)  
            self.y=np.concatenate((self.y,y),axis=0)  
            self.w=np.concatenate((self.w,w),axis=0)  
        
    def get_loss(self,images):
        X,y,w = self.get_data(images)
        X_forces, X_energies = [], []
        y_forces, y_energies = [], []
        w_forces, w_energies = [], []

        for i, x in enumerate(X):
            if x[0] == 0.:
                X_forces.append(X[i])
                y_forces.append(y[i])
                w_forces.append(w[i]+1.0)
            else:
                X_energies.append(X[i])
                y_energies.append(y[i])
                w_energies.append(w[i])
                
        # Evaluate energy
        yp_energies = self.reg.predict(X_energies)
        mae_energies = mean_absolute_error(y_energies, yp_energies)
        r2_energies = self.reg.score(X_energies, y_energies, w_energies)

        # Evaluate force
        yp_forces = self.reg.predict(X_forces)
        mae_forces = mean_absolute_error(y_forces, yp_forces)
        r2_forces = self.reg.score(X_forces, y_forces, w_forces)
        return mae_energies, r2_energies, mae_forces, r2_forces 

    def relax(self,structs):
        calc = LRCalculator(self.reg,self.cf)
        newStructs = []
        structs_=copy.deepcopy(structs)
        for ind in structs_:
            ind.set_calculator(calc)
            dyn = self.optimizer(ind,logfile="{}/MLrelax.log".format(self.parameters.MLpath))
            for j in range(self.parameters.mlrelaxNum):
                try:
                    label=dyn.run(fmax=self.parameters.epsArr[j], steps=self.parameters.stepArr[j])
                except Converged:
                    pass
                except TimeoutError:
                    logging.info("Timeout")
                    break
                except:
                    logging.debug("traceback.format_exc():\n{}".format(traceback.format_exc()))
                    logging.info("ML relax fail")
                    break
            else:
                if label:
                    ind.info['energy'] = ind.get_potential_energy()
                    ind.info['forces'] = ind.get_forces()
                    #ind.info['stress'] = ind.get_stress()
                    enthalpy = (ind.info['energy'] + self.parameters.pressure * ind.get_volume() * GPa)/len(ind)
                    ind.info['enthalpy'] = round(enthalpy, 3)
                    ind.set_calculator(None)
                    newStructs.append(ind)
        return newStructs

    def get_fp(self,pop):
        for ind in pop:
            X,y,w = self.get_data([ind])
            ind.info['image_fp']=X[0,1:]

"""
from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from .utils import SBCalculator
from amp.utilities import hash_images
class AMPmodel(MachineLearning):

    descriptors={'Gaussian':Gaussian()}
    models={'NN':NeuralNetwork}
    optimizers={'BFGS':BFGS,'FIRE':FIRE}

    def __init__(self,parameters):
        self.parameters=parameters
        self.descriptor=descriptors[parameters.descriptor]
        model=models[parameters.model](hiddenlayers=parameters.hiddenlayers)
        label=os.path.join(parameters.workDir,parameters.MLpath,'amp')
        self.calc=Amp(self.descriptor,model,label)
        self.calc.cores=12
        self.calc.model.lossfunction = LossFunction(convergence={'energy_rmse': 0.5,'force_rmse': 2})
        self.dataset=[]
        self.optimizer=optimizers[parameters.mloptimizer]
        
    def train(self):
        logging.info('OvO')
        for atoms in self.dataset:
            atoms.set_calculator(SBCalculator())
        self.calc.train(self.dataset)

    def updatedataset(self,images):
        self.dataset.extend(copy.deepcopy(images))

    def getloss(self,images):
        testset=copy.deepcopy(images)
        MSE_energy=0
        MSE_forces=0
        for atoms in testset:
            atoms.set_calculator(self.calc)
            MSE_energy+=(atoms.get_potential_energy()-atoms.info['energy'])**2
            MSE_forces+=np.mean((atoms.get_forces()-atoms.info['forces'])**2,axis=0)
        return MSE_energy/len(testset),MSE_forces/len(testset)

    def relax(self,structs):
        newStructs = []
        structs_=copy.deepcopy(structs)
        for ind in structs_:
            ind.set_calculator(self.calc)
            dyn = self.optimizer(ind,logfile="{}/MLrelax.log".format(self.parameters.MLpath))
            for j in range(self.parameters.mlrelaxNum):
                try:
                    dyn.run(fmax=self.parameters.epsArr[j], steps=self.parameters.stepArr[j])
                except Converged:
                    pass
                except TimeoutError:
                    logging.info("Timeout")
                    break
                except:
                    logging.debug("traceback.format_exc():\n{}".format(traceback.format_exc()))
                    logging.info("ML relax fail")
                    break
        else:
            ind.info['energy'] = ind.get_potential_energy()
            ind.info['forces'] = ind.get_forces()
            ind.info['stress'] = ind.get_stress()
            enthalpy = (ind.info['energy'] + self.parameters.pressure * ind.get_volume() * GPa)/len(ind)
            ind.info['enthalpy'] = round(enthalpy, 3)
            ind.set_calculator(None)
            newStructs.append(ind)
        return newStructs

    def get_fp(self,pop):
        hash_pop=hash_images(pop)
        self.descriptor.calculate_fingerprints(hash_pop)
        for ind in pop:
            h=list(hash_images([ind]).keys())[0]
            fps=np.array([np.array(self.descriptor.fingerprints[h][i][1]) for i in range(len(ind))])
            ind.info['image_fp']=np.mean(fps,axis=0)
"""