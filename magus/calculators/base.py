import os, shutil
import numpy as np
import abc
from ase.atoms import Atoms
from magus.utils import *
from magus.population import Individual
from magus.writeresults import write_traj
from magus.queuemanage import JobManager


def split1(Njobs, Npara):
    Neach = int(np.ceil(Njobs / Npara))
    return [[i + j * Npara for j in range(Neach) if i + j * Npara < Njobs] for i in range(Npara)]

def split2(Njobs, Npara):
    Neach = int(np.ceil(Njobs / Npara))
    return [[i * Neach + j for j in range(Neach) if i * Neach + j < Njobs] for i in range(Npara)]

class Calculator(abc.ABC):
    def __init__(self, parameters):
        if not hasattr(self, 'p'):
            self.p = EmptyClass()
        Requirement = ['workDir']
        Default = {'pressure': 0}
        checkParameters(self.p, parameters, Requirement, Default)

    def cdcalcFold(self):
        os.chdir(self.p.workDir)
        if not os.path.exists('calcFold'):
            shutil.copytree('inputFold', 'calcFold')
        os.chdir('calcFold')

    def pre_processing(self, calcPop):
        if isinstance(calcPop[0], Individual):
            self.atomstype = 'Individual'
            self.Pop = calcPop
            return calcPop.frames
        elif isinstance(calcPop[0], Atoms):
            self.atomstype = 'Atoms'
            return calcPop

    def post_processing(self, pop):
        if self.atomstype == 'Atoms':
            return pop
        elif self.atomstype == 'Individual':
            return self.Pop(pop)

    @abc.abstractmethod
    def relax(self,calcPop):
        pass
    
    @abc.abstractmethod
    def scf(self,calcPop):
        pass


class ClusterCalculator(Calculator, abc.ABC):
    def __init__(self, parameters, prefix):
        super().__init__(parameters)
        Requirement = ['queueName', 'numCore', 'numParallel', 'calcNum']
        Default = {'Preprocessing':'', 'waitTime':200, 'verbose':False, 'killtime':10000000}
        checkParameters(self.p,parameters,Requirement,Default)
        self.J = JobManager(self.p.verbose, self.p.killtime)
        self.prefix = prefix

    def paralleljob(self, calcPop, runjob):
        numParallel = self.p.numParallel
        numJobs = len(calcPop)
        runArray = split1(numJobs, numParallel)
        calcDir = "{}/calcFold/{}".format(self.p.workDir, self.prefix)
        if not os.path.exists(calcDir):
            os.mkdir(calcDir)
        os.chdir(calcDir)
        self.prepare_for_calc()
        for i in range(numParallel):
            if len(runArray[i]) == 0:
                continue
            currdir = '{}/{:02d}'.format(calcDir, i)
            if os.path.exists(currdir):
                shutil.rmtree(currdir)
            os.mkdir(currdir)
            os.chdir(currdir)
            tmpPop = [calcPop[j] for j in runArray[i]]
            write_traj('initPop.traj', tmpPop)
            runjob()
        self.J.WaitJobsDone(self.p.waitTime)
        os.chdir(self.p.workDir)

    def scf_parallel(self, calcPop):
        self.cdcalcFold()
        self.paralleljob(calcPop, self.scfjob)
        scfPop = self.read_parallel_results()
        self.J.clear()
        return scfPop

    def relax_parallel(self, calcPop):
        self.cdcalcFold()
        self.paralleljob(calcPop, self.relaxjob)
        relaxPop = self.read_parallel_results()
        self.J.clear()
        return relaxPop

    def read_parallel_results(self):
        pop = []
        for job in self.J.jobs:
            try:
                pop.extend(ase.io.read("{}/optPop.traj".format(job['workDir']), format='traj', index=':'))
            except:
                logging.warning("ERROR in read results {}".format(job['workDir']))
        return pop

    @abc.abstractmethod
    def scfjob(self):
        pass

    @abc.abstractmethod
    def relaxjob(self, index):
        pass

    @abc.abstractmethod
    def prepare_for_calc(self):
        pass