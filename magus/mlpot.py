from __future__ import division, print_function
from itertools import product
import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from .dataset import PreProcess
import logging, time

class MLPot(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, fpSetup, symbols, gp, onlyDiagPress=False, xyPress=False, computeVar=False, **kwargs):
        Calculator.__init__(self, **kwargs)
        # self.cutoff, self.Gs = read_fp_setup(fpSetup, symbols)
        self.fpSetup = fpSetup
        self.symbols = symbols
        self.gp = gp
        self.onlyDiagPress = onlyDiagPress
        self.xyPress = xyPress
        self.computeVar = computeVar


    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'stress'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)
        # symbols = list(set(atoms.get_chemical_symbols()))
        # Gs = read_fp_setup(self.fpSetup, symbols)

        start = time.time()
        preProcess = PreProcess([atoms], self.fpSetup, self.symbols)
        data = preProcess.images_data(mod='predict', type='list')
        # logging.debug("PreProcess time: {}".format(time.time() - start))

        preEles, preInput, preTrans, _ = self.gp.prepare_data(data, mode='predict', virial=True)

        start = time.time()
        preRes, varRes = self.gp.predict(preEles, preInput, preTrans, computeVar=self.computeVar)

        # preRes = self.gp.predict(preEles, preInput, preTrans)
        energy = preRes[0].asnumpy()
        forces = preRes[1:3*natoms+1].reshape((-1,3)).asnumpy()
        stress = preRes[-6:].reshape((6,)).asnumpy()
        # logging.debug("Predict time: {}".format(time.time() - start))

        if self.computeVar:
            # logging.debug("varRes: {}".format(varRes.shape))
            eVar = varRes[0].asnumpy()
            fVar = varRes[1:3*natoms+1].reshape((-1,3)).asnumpy()
            sVar = varRes[-6:].reshape((6,)).asnumpy()
            self.results['eVar'] = eVar
            self.results['fVar'] = fVar
            self.results['sVar'] = sVar

        # logging.debug(stress[:3]*160.21766208)

        if self.onlyDiagPress:
            stress[3:] = 0
        if self.xyPress:
            stress[:2] = stress[:2].mean()

        #logging.debug("natoms: {}".format(natoms))
        #logging.debug("shape: {}".format(preRes.shape))





        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress





class ClusterMLPot(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, fpSetup, symbols, gps, cluster, **kwargs):
        """
        gps: a list of Gaussian Process Object
        cluster: a KMeans Object
        """
        Calculator.__init__(self, **kwargs)
        # self.cutoff, self.Gs = read_fp_setup(fpSetup, symbols)
        self.fpSetup = fpSetup
        self.symbols = symbols
        self.gps = gps
        self.cluster = cluster


    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'stress'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)
        # symbols = list(set(atoms.get_chemical_symbols()))
        # Gs = read_fp_setup(self.fpSetup, symbols)

        preProcess = PreProcess([atoms], self.fpSetup, self.symbols)
        data = preProcess.images_data(mod='predict', type='list')

        atFp = np.array([data[0][1]]).mean(axis=0)
        clusIndex = self.cluster.predict(atFp)
        atGp = self.gps[clusIndex[0]]

        preEles, preInput, preTrans = atGp.prepare_data(data, mode='predict', virial=True)
        # preEles, preInput, preTrans = self.gp.prepare_data(data, mode='predict', virial=True)

        preRes = atGp.predict(preEles, preInput, preTrans)
        # preRes = self.gp.predict(preEles, preInput, preTrans)
        energy = preRes[0].asnumpy()
        forces = preRes[1:3*natoms+1].reshape((-1,3)).asnumpy()
        stress = preRes[-6:].reshape((6,)).asnumpy()



        #logging.debug("natoms: {}".format(natoms))
        #logging.debug("shape: {}".format(preRes.shape))

        #logging.debug(stress[:3]*160)



        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress
