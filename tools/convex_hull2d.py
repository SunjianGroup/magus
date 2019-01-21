import numpy as np
from collections import Counter
from ase.phasediagram import PhaseDiagram
from ase.data import atomic_numbers

class Convex_2d:
    def __init__(self, symbols, formula, traj):
        """
        symbols: like ['H', 'O', 'He']
        formula: like [[2, 1, 0], [0, 0, 1]]
        len(formulas) should be 2
        traj: a list of atoms objects
        atoms.info should have key "enthalpy". It is enthalpy per atom.
        """
        self.symbols = symbols
        assert len(formula) == 2, "2D convex hull!"
        frml = np.array(formula)
        assert np.linalg.matrix_rank(frml) == 2, "Formula should be a full-rank matrix!"

        self.formula = frml
        self.invFrml = np.linalg.inv(np.dot(frml, frml.T))
        self.rawTraj = traj[:]
        self.filTraj = self.filter(traj)
        print('Exculde {} structures'.format(len(traj) - len(self.filTraj)))
        coefs = self.get_coef(self.filTraj)
        for coef, ats in zip(coefs, self.filTraj):
            ats.info['coef'] = coef

        Xats = [ats for ats in self.filTraj if ats.info['coef'][1] == 0]
        Yats = [ats for ats in self.filTraj if ats.info['coef'][0] == 0]

        Xsize = self.formula[0].sum()
        Ysize = self.formula[1].sum()

        XminEn = min([ats.info['enthalpy'] for ats in Xats]) * Xsize
        YminEn = min([ats.info['enthalpy'] for ats in Yats]) * Ysize

        refs = []
        for ats in self.filTraj:
            coef = ats.info['coef']
            # Total formation enthalpy
            formEnth = ats.info['enthalpy']*len(ats) - coef[0]*XminEn - coef[1]*YminEn
            ats.info['formEnth'] = formEnth
            name = self.parse_names([coef])[0]
            refs.append((name, formEnth))

        pd = PhaseDiagram(refs)
        self.pd = pd



    def filter(self, traj):
        symbols = self.symbols
        numbers = [atomic_numbers[s] for s in symbols]
        numSym = len(symbols)
        filTraj = []
        for ats in traj:
            nums = list(set(ats.numbers))
            if all(n in numbers for n in nums):
                ct = Counter(ats.numbers)
                frml = [ct[i] for i in numbers]
                mat = np.concatenate((self.formula, [frml]))
                ats.info['frml'] = np.array(frml)
                if np.linalg.matrix_rank(mat) == 2:
                    filTraj.append(ats)

        return filTraj

    def get_coef(self, traj):
        coefs = []
        for ats in traj:
            frml = ats.info['frml']
            coef = np.dot(np.dot(np.expand_dims(frml,axis=0), self.formula.T), self.invFrml)
            coef = np.rint(coef)
            tmp = [int(i) for i in coef[0].tolist()]
            coefs.append(tmp)

        return coefs

    def parse_names(self, coefs):
        # xy = ['X', 'Y']
        names = ["X{}Y{}".format(*coef) for coef in coefs]

        return names
