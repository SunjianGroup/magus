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

    def get_potential_energies(self, atoms=None, force_consistent=False):
        eFps, _, _ = self.cf.get_all_fingerprints(self.atoms)
        X=np.concatenate((np.ones((len(atoms),1)),eFps),axis=1)
        y=self.reg.predict(X)
        return y

    def calculate(self, atoms=None,properties=['energy'],system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        X,n=[],[]

        eFps, fFps, sFps = self.cf.get_all_fingerprints(self.atoms)
        fFps = np.sum(fFps,axis=0)
        sFps = np.sum(sFps,axis=0)
        X.append(np.mean(eFps,axis=0))
        X.extend(fFps.reshape(-1,eFps.shape[1]))
        X.extend(sFps.reshape(-1,eFps.shape[1]))
        n.extend([1.0]+[0.0]*len(atoms)*3+[0.0]*6)
        X=np.array(X)
        n=np.array(n)
        X=np.concatenate((n.reshape(-1,1),X),axis=1)

        y=self.reg.predict(X)
        self.results['energy'] = y[0]*len(self.atoms)
        self.results['free_energy'] = y[0]*len(self.atoms)
        self.results['forces'] = y[1:-6].reshape((len(self.atoms),3))
        self.results['stress'] = y[-6:]/self.atoms.get_volume()/2


optimizers={'BFGS':BFGS,'FIRE':FIRE}
class LRmodel(MachineLearning):
    def __init__(self,parameters):
        self.parameters=parameters
        cutoff = self.parameters.cutoff
        elems = [atomic_numbers[element] for element in parameters.symbols]
        nmax = self.parameters.ZernikeNmax
        ncut = self.parameters.ZernikeNcut
        diag = self.parameters.ZernikeDiag
        self.cf = ZernikeFp(cutoff, nmax, None, ncut, elems,diag=diag)
        self.w_energy = 30.0
        self.w_force = 1.0
        self.w_stress = 1.0
        self.X = None
        #self.optimizer=optimizers[parameters.mloptimizer]
        self.dataset = []

    def train(self):
        logging.info('{} in dataset,training begin!'.format(len(self.dataset)))
        self.reg = LinearRegression().fit(self.X, self.y, self.w)
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
                y.append(atoms.info['energy']/len(atoms))
                # try:
                #     y.append(atoms.info['energy']/len(atoms))
                # except:
                #     y.append(0.0)
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

    def relax(self,structs):
        calc = LRCalculator(self.reg,self.cf)
        newStructs = []
        structs_=copy.deepcopy(structs)
        for i,ind in enumerate(structs_):
            ind.set_calculator(calc)
            logging.info("Structure {}".format(i))
            for j in range(self.parameters.mlrelaxNum):
                ucf = ExpCellFilter(ind, scalar_pressure=self.parameters.pressure*GPa)
                # ucf = UnitCellFilter(ind, scalar_pressure=self.parameters.pressure*GPa)
                logfile = "{}/calcFold/MLrelax.log".format(self.parameters.workDir)
                if self.parameters.mloptimizer == 'cg':
                    gopt = SciPyFminCG(ucf, logfile=logfile,)
                elif self.parameters.mloptimizer == 'BFGS':
                    gopt = BFGS(ucf, logfile=logfile, maxstep=self.parameters.maxRelaxStep)
                elif self.parameters.mloptimizer == 'fire':
                    gopt = FIRE(ucf, logfile=logfile, maxmove=self.parameters.maxRelaxStep)

                try:
                    label=gopt.run(fmax=self.parameters.mlepsArr[j], steps=self.parameters.mlstepArr[j])
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
                    ind.info['stress'] = ind.get_stress()
                    enthalpy = (ind.info['energy'] + self.parameters.pressure * ind.get_volume() * GPa)/len(ind)
                    ind.info['enthalpy'] = round(enthalpy, 3)
                    ind.set_calculator(None)
                    newStructs.append(ind)
        return newStructs

    def scf(self,calcPop):
        calc = LRCalculator(self.reg,self.cf)
        scfPop = []
        for ind in calcPop:
            atoms=copy.deepcopy(ind)
            atoms.set_calculator(calc)
            try:
                atoms.info['energy'] = atoms.get_potential_energy()
                atoms.info['forces'] = atoms.get_forces()
                atoms.info['stress'] = atoms.get_stress()
                enthalpy = (atoms.info['energy'] + self.parameters.pressure * atoms.get_volume() * GPa)/len(atoms)
                atoms.info['enthalpy'] = round(enthalpy, 3)
                atoms.set_calculator(None)
                scfPop.append(atoms)
            except:
                pass
        return scfPop

    def get_fp(self,pop):
        for ind in pop:
            properties = []
            for s in ['energy', 'forces', 'stress']:
                if s in ind.info:
                    properties.append(s)
            X,_,_ = self.get_data([ind], implemented_properties=properties)
            ind.info['image_fp']=X[0,1:]

