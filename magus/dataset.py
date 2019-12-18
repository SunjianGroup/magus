import numpy as np
from .descriptor import CalculateFingerprints
from ase.neighborlist import NeighborList
#from ase.calculators.dftd3 import DFTD3
from itertools import chain, product
import pickle
import logging

class PreProcess(object):
    def __init__(self, images, fpSetup, symbols):
        self.images = images
        self.fpSetup = fpSetup
        self.symbols = symbols
        self.read_fp_setup()

    def images_data(self, mod='predict',type='numpy',dftd3=False):
        """
        mode: 'normal' or 'train'
        type: type of data, 'numpy' or 'list'
        """
        if mod == 'predict':
            # data = [self.atoms_data(atoms) for atoms in self.images]
            data = []
            for atoms in self.images:
                enFps, fMats, vMat, nums = self.atoms_data(atoms)
                numAtom = len(enFps)
                fMat = np.concatenate(fMats, axis=1)
                fMat = fMat.T
                vMat = vMat.T

                enMat = np.ones((1, numAtom))

                # logging.debug("len of enFps: {}".format(len(enFps)))
                if type == 'list':
                    nums = nums.tolist()
                    # logging.debug("fp: {}".format(enFps[0].shape))
                    enFps = [fp.tolist() for fp in enFps]
                    enMat = enMat.tolist()
                    fMat = fMat.tolist()
                    vMat = vMat.tolist()
                data.append((nums, enFps, enMat, fMat, vMat))

        elif mod == 'trainNN':
            data = []
            for atoms in self.images:
                enFps, fMats, vMat, nums = self.atoms_data(atoms)
                fMat = np.concatenate(fMats, axis=1)
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                forces = np.reshape(forces, newshape=(3*len(atoms), 1))
                if type == 'list':
                    nums = nums.tolist()
                    enFps = [fp.tolist() for fp in enFps]
                    fMat = fMat.tolist()
                    vMat = vMat.tolist()
                    forces = forces.tolist()
                data.append((nums, enFps, fMat, vMat, energy, forces))
        elif mod == 'trainGP':
            data = []
            for atoms in self.images:
                enFps, fMats, vMat, nums = self.atoms_data(atoms)
                numAtom = len(enFps)
                fMat = np.concatenate(fMats, axis=1)
                fMat = fMat.T
                vMat = vMat.T

                # energy = atoms.get_potential_energy()
                # forces = atoms.get_forces()
                # stress = atoms.get_stress()

                energy = atoms.info['energy']
                forces = atoms.info['forces']
                stress = atoms.info['stress']

                # remove D3 effect
                # if dftd3:
                #     d3At = atoms.copy()
                #     d3 = DFTD3(command='dftd3')
                #     d3At.set_calculator(d3)
                #     d3Energy = d3At.get_potential_energy()
                #     d3Forces = d3At.get_forces()
                #     d3Stress = d3At.get_stress()

                #     energy -= d3Energy
                #     forces -= d3Forces
                #     stress -= d3Stress

                # print(d3Energy, d3Stress)



                energy = [energy]
                forces = np.reshape(forces, newshape=(3*len(atoms), 1))
                stress = np.reshape(stress, newshape=(6, 1))

                # special for GP
                # fMat
                # tmp = np.split(fMat, numAtom, axis=1)
                # for i, term in enumerate(tmp):
                #     tmp[i] = np.concatenate((np.zeros((term.shape[0], 1)), term), axis=1)
                # fMat = np.concatenate(tmp, axis=1)
                # forces
                # tmp = np.split(forces, numAtom, axis=0)
                # for i, term in enumerate(tmp):
                #     tmp[i] = np.concatenate((np.zeros((1, term.shape[1])), term), axis=0)
                # forces = np.concatenate(tmp, axis=0)
                # enMat
                # enMat = np.concatenate([[1]+[0 for i in range(len(self.Gs))] for j in range(numAtom)])
                enMat = np.ones((1, numAtom))



                if type == 'list':
                    nums = nums.tolist()
                    enFps = [fp.tolist() for fp in enFps]
                    enMat = enMat.tolist()
                    fMat = fMat.tolist()
                    vMat = vMat.tolist()
                    forces = forces.tolist()
                    stress = stress.tolist()
                data.append((nums, enFps, enMat, fMat, vMat, energy, forces, stress))

        return data


    def read_fp_setup(self):
        """
        read symmetry functions setup
        """
        fpSetup = self.fpSetup
        symbols = self.symbols

        cutoff = fpSetup['Rc']

        stpList = list()

        if 'sf2' in fpSetup.keys():
            sf2 = fpSetup['sf2']
            sf2Stp = [('G2', el, eta)
                    for el in symbols
                    for eta in sf2['eta']]

            for stp in sf2Stp:
                stpDict = dict(zip(['type', 'element', 'eta'], stp))
                stpList.append(stpDict)

        if 'sf4' in fpSetup.keys():
            sf4 = fpSetup['sf4']
            # all pairs, e.g. for TiO2, Ti-Ti, O-O, Ti-O
            pairs = [(i, j)
            for i in range(len(symbols))
            for j in range(len(symbols))
            if i <= j]

            sf4Stp = [('G4', [symbols[i], symbols[j]], eta, lam, zeta)
            for i, j in pairs
            for eta in sf4['eta']
            for lam in sf4['gamma']
            for zeta in sf4['zeta']
            ]

            for stp in sf4Stp:
                stpDict = dict(zip(['type', 'elements', 'eta', 'gamma', 'zeta'], stp))
                stpList.append(stpDict)

        # GsDict = dict()
        # for el in symbols:
        #     GsDict[el] = stpList[:]
        self.cutoff, self.Gs = cutoff, stpList


    def atoms_data(self, atoms):

        cutoff = self.cutoff
        Gs = self.Gs

        nl = NeighborList(cutoffs=([cutoff / 2.] * len(atoms)),
                            self_interaction=False,
                            bothways=True,
                            skin=0.)
        nl.update(atoms)
        cf = CalculateFingerprints(cutoff, Gs, atoms, nl, True)
        enFps, fMats, vMats = zip(*map(cf.index_fingerprint, range(len(atoms))))

        #debug
        self.cf = cf

        #with open('vMats.pkl', 'w') as f:
        #    pickle.dump(vMats, f, True)

        #with open('fMats.pkl', 'w') as f:
        #    pickle.dump(fMats, f, True)

        # vMat = sum(vMats)/atoms.get_volume()
        vMat = 0.5*sum(vMats)/atoms.get_volume()

        nums = atoms.numbers[:]

        return enFps, fMats, vMat, nums



