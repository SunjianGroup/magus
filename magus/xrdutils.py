"""
Created by Junjie Wang on Mar 2019
Modified by Qiuhan Jia on Jun 2022
"""
import numpy as np
import itertools
from magus.atomic_form_factors import ff

class XRD():
    def __init__(self, hkl, theta, Kh, atoms):
        self.hkl = hkl
        self.multi = 1
        self.theta = theta  # note that it is theta, not 2theta.
        self.Kh = Kh
        self.d = 2 * np.pi / Kh
        self.atoms = atoms
        self.scaled_positions = atoms.get_scaled_positions()

    def get_f(self, symbol):
        f = ff[symbol]
        f0 = f[-1]
        for i in np.arange(0, 8, 2):
            f0 += f[i] * np.exp(-f[i + 1] * (self.Kh / (4. * np.pi))**2)
        return f0

    def get_F(self):
        F = 0
        for i, atom in enumerate(self.atoms):
            t = 2.0 * np.pi * np.dot(self.hkl, self.scaled_positions[i])
            F += self.get_f(atom.symbol) * np.exp(1j * t)   # * np.exp(-0.01 * self.Kh**2 /(4.0*np.pi))
        return abs(F) ** 2

    def get_I(self):
        LP = 1 / np.sin(self.theta)**2 / np.cos(self.theta)
        P = 1 + np.cos(2 * self.theta)**2
        self.I = self.get_F() * LP * P * self.multi
        # self.I = self.get_F()*self.multi
        return self.I


class XrdStructure():
    def __init__(self, atoms, lamb, two_theta_range=[5, 175], threshold=0.1):
        """
        lamb: wave length in Angstrom.
        threshold: peaks lower than threshold*h_max will be remove.
        """
        self.atoms = atoms
        self.lattice = self.atoms.cell.cellpar()
        self.reciprocal_lattice = self.atoms.cell.reciprocal() * 2 * np.pi
        self.lamb = lamb
        self.thetamin, self.thetamax = two_theta_range[0] / 2, two_theta_range[1] / 2
        self.Khmax = 4 * np.pi * np.sin(self.thetamax / 180 * np.pi) / lamb
        self.Khmin = 4 * np.pi * np.sin(self.thetamin / 180 * np.pi) / lamb
        self.peaks = []
        self.getallhkl()

        Is = np.array([peak.get_I() for peak in self.peaks])
        Is = Is / np.max(Is)
        exist_peaks = [self.peaks[i] for i in range(len(Is)) if Is[i] > threshold]
        self.peaks = exist_peaks

        self.angles = np.array([peak.theta / np.pi * 360 for peak in self.peaks])  # 2 theta list
        self.Is = np.array([peak.get_I() for peak in self.peaks])
        self.Is = self.Is / np.max(self.Is)

    def getplotdata(self, function='Lorentzian', w=0.1, step=0.01, sigma=0.05):
        if function == 'Gaussian':
            def f(x, sigma):
                f = 0
                for h, mu in zip(self.Is, self.angles):
                    f += h / sigma / np.sqrt(2 * np.pi) * np.e**(-0.5 * (x - mu)**2 / sigma**2)
                return f

            angle = np.arange(2 * self.thetamin, 2 * self.thetamax, step)
            I = np.array([f(x, sigma) for x in angle])

        elif function == 'Lorentzian':
            def get_I(x, w=0.1):
                # y=I*w**(2*m)/(w**2+(2**(1/m)-1)*(x-5)**2)**m
                f = 0
                for h, mu in zip(self.Is, self.angles):
                    f += h * w**2 / (w**2 + (x - mu)**2)
                return f

            angle = np.arange(2 * self.thetamin, 2 * self.thetamax, step)
            I = np.array([get_I(x, w) for x in angle])
            I /= np.max(I)  # normalization
        return [angle, I]  # [2theta list, height list]

    def getpeakdata(self):
        sort = np.argsort(self.angles)
        return np.array([self.angles[sort], self.Is[sort]]).transpose(1, 0)  # [2theta, height] pairs

    def getallhkl(self):
        hklmax = (self.Khmax / np.sqrt(np.sum(self.reciprocal_lattice ** 2, axis=1)) + 1).astype(int)
        hrange, krange, lrange = [np.arange(i, -1 - i, -1) for i in hklmax]
        # hrange,krange,lrange=np.arange(0-hklmax[0],hklmax[0]+1),np.arange(0-hklmax[1],hklmax[1]+1),np.arange(0-hklmax[2],hklmax[2]+1)
        for hkl in itertools.product(hrange, krange, lrange):
            theta = self.gettheta(hkl)
            if theta:
                for peak in self.peaks:
                    if np.allclose(theta, peak.theta):
                        peak.multi += 1
                        theta = False
                        break
                if theta:
                    self.peaks.append(XRD(hkl, theta, self.getKh(hkl), self.atoms))

    def gettheta(self, hkl):
        if self.getKh(hkl) < self.Khmin or self.getKh(hkl) > self.Khmax:
            return False
        else:
            return np.arcsin(self.getKh(hkl) * self.lamb / 4 / np.pi)

    def getKh(self, hkl):
        Kh = np.dot(hkl, self.reciprocal_lattice)
        return np.sqrt(np.dot(Kh, Kh))


def loss(datath, datatar, match_tol=2, minimized_loss=False):
    """
    Parameters
    ----------
    datath : {angle_list,height_list} calculated according to the theory.
    datatar : {angle_list,height_list} target.
    match_tol : tolerance for matching the peaks in the theory and target. delta(2theta) < match_tol * tan(theta) for each peak.
    minimized_loss: bool
            scale the values of experimental data to minimize the loss.

    Returns
    -------
    F : the value of loss function.
        sum_matched((he-ht)^2) + sum_unmatched(he^2) + sum_unmatched(ht^2)

    PS
    ------
        One target peak may match multiple theory peaks
    """
    sortedth = datath[:, np.argsort(datath)[0]]  # [anglelist,hlist]
    sortedtar = datatar[:, np.argsort(np.array(datatar))[0]]  # sorted by angles
    sortedtar[1] /= max(sortedtar[1])
    ith, itar = 0, 0
    nth, ntar = len(sortedth[0]), len(sortedtar[0])
    mth_th, mth_tar = [], []  # matched indices for theory and target
    match_table = []  # the list of matched theoretical peaks for each target peak
    for itar in range(ntar):  # match peaks from left to right
        current_match = []
        for _ in range(nth - ith):
            if sortedth[0, ith] - sortedtar[0, itar] < -match_tol * np.tan(sortedtar[0, itar] * np.pi / 360):
                ith += 1
            elif sortedth[0, ith] - sortedtar[0, itar] < match_tol * np.tan(sortedtar[0, itar] * np.pi / 360):
                mth_th.append(ith)
                current_match.append(ith)
                ith += 1
            else:
                if current_match:
                    mth_tar.append(itar)
                    match_table.append(current_match)
                break
    if current_match:
        mth_tar.append(itar)
        match_table.append(current_match)

    unmth_th = np.setdiff1d(range(nth), mth_th, True)
    unmth_tar = np.setdiff1d(range(ntar), mth_tar, True)
    h_mth_th = np.array([np.sum(sortedth[1][i]) for i in match_table])
    '''heights of matched theory peaks. The peaks matching to the same target are merged.'''
    h_mth_tar = sortedtar[1][mth_tar]
    h_unmth_th = sortedth[1][unmth_th]
    h_unmth_tar = sortedtar[1][unmth_tar]

    alpha = 1
    if minimized_loss:
        alpha = sum(h_mth_th * h_mth_tar) / (sum(h_mth_th**2)+sum(h_unmth_th**2))
    F = sum((h_mth_tar - alpha * h_mth_th)**2) \
        + sum(h_unmth_tar**2) + alpha**2 * sum(h_unmth_th**2)
    return F
