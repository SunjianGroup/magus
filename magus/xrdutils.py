"""
Created by Junjie Wang on Mar 2019
Modified by Qiuhan Jia on Jun 2022
"""
import numpy as np
import itertools

'''
Element 	a1		b1		a2		b2		a3		b3		a4		b4		c
'''
ff = {
"H":		[0.489918,	20.6593,	0.262003,	7.74039,	0.196767,	49.5519,	0.049879,	2.20159,	0.001305],
"H1-":		[0.897661,	53.1368,	0.565616,	15.187,		0.415815,	186.576,	0.116973,	3.56709,	0.002389],
"He":		[0.8734,	9.1037,		0.6309,		3.3568,		0.3112,		22.9276,	0.178,		0.9821,		0.0064],
"Li":		[1.1282,	3.9546,		0.7508,		1.0524,		0.6175,		85.3905,	0.4653,		168.261,	0.0377],
"Li1+":		[0.6968,	4.6237,		0.7888,		1.9557,		0.3414,		0.6316,		0.1563,		10.0953,	0.0167],
"Be":		[1.5919,	43.6427,	1.1278,		1.8623,		0.5391,		103.483,	0.7029,		0.542,		0.0385],
"Be2+":		[6.2603,	0.0027,		0.8849,		0.8313,		0.7993,		2.2758,		0.1647,		5.1146,		-6.1092],
"B":		[2.0545,	23.2185,	1.3326,		1.021,		1.0979,		60.3498,	0.7068,		0.1403,		-0.1932],
"C":		[2.31,		20.8439,	1.02,		10.2075,	1.5886,		0.5687,		0.865,		51.6512,	0.2156],
"N":		[12.2126,	0.0057,		3.1322,		9.8933,		2.0125,		28.9975,	1.1663,		0.5826,		-11.529],
"O":		[3.0485,	13.2771,	2.2868,		5.7011,		1.5463,		0.3239,		0.867,		32.9089,	0.2508],
"O1-":		[4.1916,	12.8573,	1.63969,	4.17236,	1.52673,	47.0179,	-20.307,	-0.01404,	21.9412],
"F":		[3.5392,	10.2825,	2.6412,		4.2944,		1.517,		0.2615,		1.0243,		26.1476,	0.2776],
"F1-":		[3.6322,	5.27756,	3.51057,	14.7353,	1.26064,	0.442258,	0.940706,	47.3437,	0.653396],
"Ne":		[3.9553,	8.4042,		3.1125,		3.4262,		1.4546,		0.2306,		1.1251,		21.7184,	0.3515],
"Na":		[4.7626,	3.285,		3.1736,		8.8422,		1.2674,		0.3136,		1.1128,		129.424,	0.676],
"Na1+":		[3.2565,	2.6671,		3.9362,		6.1153,		1.3998,		0.2001,		1.0032,		14.039,		0.404],
"Mg":		[5.4204,	2.8275,		2.1735,		79.2611,	1.2269,		0.3808,		2.3073,		7.1937,		0.8584],
"Mg2+":		[3.4988,	2.1676,		3.8378,		4.7542,		1.3284,		0.185,		0.8497,		10.1411,	0.4853],
"Al":		[6.4202,	3.0387,		1.9002,		0.7426,		1.5936,		31.5472,	1.9646,		85.0886,	1.1151],
"Al3+":		[4.17448,	1.93816,	3.3876,		4.14553,	1.20296,	0.228753,	0.528137,	8.28524,	0.706786],
"Si":		[6.2915,	2.4386,		3.0353,		32.3337,	1.9891,		0.6785,		1.541,		81.6937,	1.1407],
"Si4+":		[4.43918,	1.64167,	3.20345,	3.43757,	1.19453,	0.2149,		0.41653,	6.65365,	0.746297],
"P":		[6.4345,	1.9067,		4.1791,		27.157,		1.78,		0.526,		1.4908,		68.1645,	1.1149],
"S":		[6.9053,	1.4679,		5.2034,		22.2151,	1.4379,		0.2536,		1.5863,		56.172,		0.8669],
"Cl":		[11.4604,	0.0104,		7.1964,		1.1662,		6.2556,		18.5194,	1.6455,		47.7784,	-9.5574],
"Cl1-":		[18.2915,	0.0066,		7.2084,		1.1717,		6.5337,		19.5424,	2.3386,		60.4486,	-16.378],
"Ar":		[7.4845,	0.9072,		6.7723,		14.8407,	0.6539,		43.8983,	1.6442,		33.3929,	1.4445],
"K":		[8.2186,	12.7949,	7.4398,		0.7748,		1.0519,		213.187,	0.8659,		41.6841,	1.4228],
"K1+":		[7.9578,	12.6331,	7.4917,		0.7674,		6.359,		-0.002,		1.1915,		31.9128,	-4.9978],
"Ca":		[8.6266,	10.4421,	7.3873,		0.6599,		1.5899,		85.7484,	1.0211,		178.437,	1.3751],
"Ca2+":		[15.6348,	-0.0074,	7.9518,		0.6089,		8.4372,		10.3116,	0.8537,		25.9905,	-14.875],
"Sc":		[9.189,		9.0213,		7.3679,		0.5729,		1.6409,		136.108,	1.468,		51.3531,	1.3329],
"Sc3+":		[13.4008,	0.29854,	8.0273,		7.9629,		1.65943,	-0.28604,	1.57936,	16.0662,	-6.6667],
"Ti":		[9.7595,	7.8508,		7.3558,		0.5,		1.6991,		35.6338,	1.9021,		116.105,	1.2807],
"Ti2+":		[9.11423,	7.5243,		7.62174,	0.457585,	2.2793,		19.5361,	0.087899,	61.6558,	0.897155],
"Ti3+":		[17.7344,	0.22061,	8.73816,	7.04716,	5.25691,	-0.15762,	1.92134,	15.9768,	-14.652],
"Ti4+":		[19.5114,	0.178847,	8.23473,	6.67018,	2.01341,	-0.29263,	1.5208,		12.9464,	-13.28],
"V":		[10.2971,	6.8657,		7.3511,		0.4385,		2.0703,		26.8938,	2.0571,		102.478,	1.2199],
"V2+":		[10.106,	6.8818,		7.3541,		0.4409,		2.2884,		20.3004,	0.0223,		115.122,	1.2298],
"V3+":		[9.43141,	6.39535,	7.7419,		0.383349,	2.15343,	15.1908,	0.016865,	63.969,		0.656565],
"V5+":		[15.6887,	0.679003,	8.14208,	5.40135,	2.03081,	9.97278,	-9.576,		0.940464,	1.7143],
"Cr":		[10.6406,	6.1038,		7.3537,		0.392,		3.324,		20.2626,	1.4922,		98.7399,	1.1832],
"Cr2+":		[9.54034,	5.66078,	7.7509,		0.344261,	3.58274,	13.3075,	0.509107,	32.4224,	0.616898],
"Cr3+":		[9.6809,	5.59463,	7.81136,	0.334393,	2.87603,	12.8288,	0.113575,	32.8761,	0.518275],
"Mn":		[11.2819,	5.3409,		7.3573,		0.3432,		3.0193,		17.8674,	2.2441,		83.7543,	1.0896],
"La":		[20.578,	2.94817,	19.599,		0.244475,	11.3727,	18.7726,	3.28719,	133.124,	2.14678],
}

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
