# Calculate fingerprints by aenet
from __future__ import print_function, division
from fractions import gcd
import os
import yaml
import logging
import numpy as np
import time
from sklearn import cluster
from ase.data import atomic_numbers
from ase.neighborlist import NeighborList
from . import fmodules
from .writeresults import write_xsf

def calc_all_fingerprints(inPop, parameters):
    """
    Calculate fingerprints of individuals in inPop.
    """
    fpDir = "%s/fpFold" %(parameters['workDir'])
    symbols = parameters["symbols"]
    workDir = parameters['workDir']

    cutoff, Gs = read_fp_setup(fpDir, symbols)
    fpPop = list()
    for fpInd in inPop:
        nl = NeighborList(cutoffs=([cutoff / 2.] * len(fpInd)),
                            self_interaction=False,
                            bothways=True,
                            skin=0.)
        nl.update(fpInd)
        cf = CalculateFingerprints(cutoff, Gs, fpInd, nl, True)
        fpInd.info['fingerprint'] = cf.average_fingerpinrt()
        fpPop.append(fpInd)

    return fpPop

def calc_one_fingerprint(atoms, parameters):
    """
    Calculate fingerprint of one Atoms Object.
    """
    fpDir = "%s/fpFold" %(parameters['workDir'])
    symbols = parameters["symbols"]
    workDir = parameters['workDir']

    cutoff, Gs = read_fp_setup(fpDir, symbols)
    nl = NeighborList(cutoffs=([cutoff / 2.] * len(atoms)),
                        self_interaction=False,
                        bothways=True,
                        skin=0.)
    nl.update(atoms)
    cf = CalculateFingerprints(cutoff, Gs, atoms, nl, True)
    fp = cf.average_fingerpinrt()
    return fp

def read_fp_setup(fpDir, symbols,):
    """
    read symmetry functions setup
    """
    fpSetup = yaml.load(open("%s/fpsetup.yaml"%(fpDir)))
    # fpSetup = yaml.load(open("%s/fpsetup.yaml"%(fpDir)), Loader=yaml.FullLoader)
    sf2 = fpSetup['sf2']
    # sf4 = fpSetup['sf4']
    cutoff = fpSetup['Rc']

    stpList = list()

    sf2Stp = [('G2', el, eta)
    for el in symbols
    for eta in sf2['eta']
    ]

    for stp in sf2Stp:
        stpDict = dict(zip(['type', 'element', 'eta'], stp))
        stpList.append(stpDict)

    # all pairs, e.g. for TiO2, Ti-Ti, O-O, Ti-O
    pairs = [(i, j)
    for i in range(len(symbols))
    for j in range(len(symbols))
    if i <= j]

    # sf4Stp = [('G4', [symbols[i], symbols[j]], eta, lam, zeta)
    # for i, j in pairs
    # for eta in sf4['eta']
    # for lam in sf4['gamma']
    # for zeta in sf4['zeta']
    # ]

    # for stp in sf4Stp:
    #     stpDict = dict(zip(['type', 'elements', 'eta', 'gamma', 'zeta'], stp))
    #     stpList.append(stpDict)

    # GsDict = dict()
    # for el in symbols:
    #     GsDict[el] = stpList[:]

    return cutoff, stpList


class CalculateFingerprints:

    """
    Class that calculates fingerprints.
    Inputs:
        cutoff: float (default 6.5 Angstroms)
                  Radius above which neighbor interactions are ignored.
        Gs:  a list of dictionaries for making
                    symmetry functions. Either auto-genetrated, or given
               in the following form:
                    Gs = [..., ...] where
                    ... = {"type":"G2", "element":"O", "eta":0.0009} or
                    ... = {"type":"G4", "elements":["O", "Au"], "eta":0.0001,
                           "gamma":0.1, "zeta":1.0}
        atoms: ASE atoms object
                   The initial atoms object on which the fingerprints will be
                   generated.
        _nl: ASE NeighborList object
        fortran: boolean
            If True, will use the fortran subroutines, else won't.
    """

    def __init__(self, cutoff, Gs, atoms, _nl, fortran):
        self.cutoff = cutoff
        self.Gs = Gs
        self.atoms = atoms
        self._nl = _nl
        self.fortran = fortran

    #########################################################################

    def index_fingerprint(self, index):
        """Returns the fingerprint of symmetry function values for atom
        specified by its index. Will automatically update if atom positions
        has changed; you don't need to call update() unless you are
        specifying a new atoms object."""

        neighbor_indices, neighbor_offsets = self._nl.get_neighbors(index)
        # for calculating fingerprints, summation runs over neighboring atoms
        # of type I (either inside or outside the main cell)
        neighbor_symbols = [self.atoms[n_index].symbol
                            for n_index in neighbor_indices]
        Rs = [self.atoms.positions[n_index] +
              np.dot(n_offset, self.atoms.get_cell()) for n_index, n_offset
              in zip(neighbor_indices, neighbor_offsets)]
        fingerprint = [self.process_G(G, index, neighbor_symbols, Rs)
                       for G in self.Gs]
        fingerprint = np.array(fingerprint)

        return fingerprint

    #########################################################################

    def get_der_fingerprint(self, symbol, index, mm, ii):
        """Returns the derivative of the fingerprint of symmetry function
        values for atom specified by its index with respect to coordinate
        x_{ii} of atom index mm."""

        Rindex = self.atoms.positions[index]
        neighbor_indices, neighbor_offsets = self._nl.get_neighbors(index)
        # for calculating derivatives of fingerprints, summation runs over
        # neighboring atoms of type I (either inside or outside the main cell)
        neighbor_symbols = [self.atoms[_index].symbol
                            for _index in neighbor_indices]
        Rs = [self.atoms.positions[_index] +
              np.dot(_offset, self.atoms.get_cell()) for _index, _offset
              in zip(neighbor_indices, neighbor_offsets)]
        der_fingerprint = [self.process_der_G(G,
                                              neighbor_indices,
                                              neighbor_symbols,
                                              Rs,
                                              index,
                                              Rindex,
                                              mm,
                                              ii) for G in self.Gs[symbol]]

        return der_fingerprint

    #########################################################################
    def average_fingerpinrt(self):
        sumFp = sum(map(self.index_fingerprint, range(len(self.atoms))))
        averageFp = sumFp/len(self.atoms)
        return averageFp


    #########################################################################

    def process_G(self, G, index, symbols, Rs):
        """Returns the value of G for atom at index. symbols and Rs are
        lists of neighbors' symbols and Cartesian positions, respectively.
        """
        home = self.atoms[index].position
        #print("process_G symbols: %s"%symbols)
        if G['type'] == 'G2':
            return calculate_G2(symbols, Rs, G['element'], G['eta'],
                                self.cutoff, home, self.fortran)
        elif G['type'] == 'G4':
            return calculate_G4(symbols, Rs, G['elements'], G['gamma'],
                                G['zeta'], G['eta'], self.cutoff, home,
                                self.fortran)
        else:
            raise NotImplementedError('Unknown G type: %s' % G['type'])

    #########################################################################
    def process_der_G(self, G, indices, symbols, Rs, a, Ra, m, i):
        """Returns the value of the derivative of G for atom at index a and
        position Ra with respect to coordinate x_{i} of atom index m.
        Symbols and Rs are lists of neighbors' symbols and Cartesian positions,
        respectively.
        """

        if G['type'] == 'G2':
            return calculate_der_G2(indices, symbols, Rs, G['element'],
                                    G['eta'], self.cutoff, a, Ra, m, i,
                                    self.fortran)
        elif G['type'] == 'G4':
            return calculate_der_G4(
                indices,
                symbols,
                Rs,
                G['elements'],
                G['gamma'],
                G['zeta'],
                G['eta'],
                self.cutoff,
                a,
                Ra,
                m,
                i,
                self.fortran)
        else:
            raise NotImplementedError('Unknown G type: %s' % G['type'])


##################################################################

def calculate_G2(symbols, Rs, G_element, eta, cutoff, home, fortran):
    """Calculate G2 symmetry function. Ideally this will not be used but
    will be a template for how to build the fortran version (and serves as
    a slow backup if the fortran one goes uncompiled)."""

    #    start = time.clock()
    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.simple_calculate_g2(numbers=numbers, rs=Rs,
                                          g_number=G_number, g_eta=eta,
                                          cutoff=cutoff, home=home)
        return ridge
    else:
        ridge = 0.  # One aspect of a fingerprint :)
        for symbol, R in zip(symbols, Rs):
            if symbol == G_element:
    #            print("%s == G_element"%symbol)
                Rij = np.linalg.norm(R - home)
                ridge += (np.exp(-eta * (Rij ** 2.) / (cutoff ** 2.)) *
                          cutoff_fxn(Rij, cutoff))
    # end = time.clock()
    #    print("calculate_G2() time: %s"%(end - start))
    return ridge


##############################################################################


def calculate_G4(symbols, Rs, G_elements, gamma, zeta, eta, cutoff, home,
                 fortran):
    """Calculate G4 symmetry function. Ideally this will not be used but
    will be a template for how to build the fortran version (and serves as
    a slow backup if the fortran one goes uncompiled)."""

    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_g4(numbers=numbers, rs=Rs,
                                          g_numbers=G_numbers, g_gamma=gamma,
                                          g_zeta=zeta, g_eta=eta,
                                          cutoff=cutoff, home=home)
        return ridge
    else:
        ridge = 0.
        counts = range(len(symbols))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([symbols[j], symbols[k]])
                if els != G_elements:
                    continue
                Rij_ = Rs[j] - home
                Rij = np.linalg.norm(Rij_)
                Rik_ = Rs[k] - home
                Rik = np.linalg.norm(Rik_)
                Rjk = np.linalg.norm(Rs[j] - Rs[k])
                cos_theta_ijk = np.dot(Rij_, Rik_) / Rij / Rik
                term = (1. + gamma * cos_theta_ijk) ** zeta
                term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                term *= (1. / 3.) * (cutoff_fxn(Rij, cutoff) +
                                     cutoff_fxn(Rik, cutoff) +
                                     cutoff_fxn(Rjk, cutoff))
                ridge += term
        ridge *= 2. ** (1. - zeta)
        return ridge
###############################################
def cutoff_fxn(Rij, Rc):
    """Cosine cutoff function in Parinello-Behler method."""
    if Rij > Rc:
        return 0.
    else:
        return 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)



##############################################################################


def der_cutoff_fxn(Rij, Rc):
    """Derivative of the Cosine cutoff function."""
    if Rij > Rc:
        return 0.
    else:
        return -0.5 * np.pi / Rc * np.sin(np.pi * Rij / Rc)

##############################################################################


def Kronecker_delta(i, j):
    """Kronecker delta function."""
    if i == j:
        return 1.
    else:
        return 0.

##############################################################################


def der_position_vector(a, b, m, i):
    """Returns the derivative of the position vector R_{ab} with respect to
        x_{i} of atomic index m."""
    der_position_vector = [None, None, None]
    der_position_vector[0] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(0, i)
    der_position_vector[1] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(1, i)
    der_position_vector[2] = (Kronecker_delta(m, a) - Kronecker_delta(m, b)) \
        * Kronecker_delta(2, i)

    return der_position_vector

##############################################################################


def der_position(m, n, Rm, Rn, l, i):
    """Returns the derivative of the norm of position vector R_{mn} with
        respect to x_{i} of atomic index l."""
    Rmn = np.linalg.norm(Rm - Rn)
    # mm != nn is necessary for periodic systems
    if l == m and m != n:
        der_position = (Rm[i] - Rn[i]) / Rmn
    elif l == n and m != n:
        der_position = -(Rm[i] - Rn[i]) / Rmn
    else:
        der_position = 0.
    return der_position

##############################################################################


def der_cos_theta(a, j, k, Ra, Rj, Rk, m, i):
    """Returns the derivative of Cos(theta_{ajk}) with respect to
        x_{i} of atomic index m."""
    Raj_ = Ra - Rj
    Raj = np.linalg.norm(Raj_)
    Rak_ = Ra - Rk
    Rak = np.linalg.norm(Rak_)
    der_cos_theta = 1. / \
        (Raj * Rak) * np.dot(der_position_vector(a, j, m, i), Rak_)
    der_cos_theta += +1. / \
        (Raj * Rak) * np.dot(Raj_, der_position_vector(a, k, m, i))
    der_cos_theta += -1. / \
        ((Raj ** 2.) * Rak) * np.dot(Raj_, Rak_) * \
        der_position(a, j, Ra, Rj, m, i)
    der_cos_theta += -1. / \
        (Raj * (Rak ** 2.)) * np.dot(Raj_, Rak_) * \
        der_position(a, k, Ra, Rk, m, i)
    return der_cos_theta

##############################################################################


def calculate_der_G2(n_indices, symbols, Rs, G_element, eta, cutoff, a, Ra,
                     m, i, fortran):
    """Calculate coordinate derivative of G2 symmetry function for atom at
    index a and position Ra with respect to coordinate x_{i} of atom index
    m."""

    if fortran:  # fortran version; faster
        G_number = [atomic_numbers[G_element]]
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        list_n_indices = [n_indices[_] for _ in range(len(n_indices))]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_der_g2(n_indices=list_n_indices,
                                              numbers=numbers, rs=Rs,
                                              g_number=G_number,
                                              g_eta=eta, cutoff=cutoff,
                                              aa=a, home=Ra, mm=m,
                                              ii=i)
    else:
        ridge = 0.  # One aspect of a fingerprint :)
        for symbol, Rj, n_index in zip(symbols, Rs, n_indices):
            if symbol == G_element:
                Raj = np.linalg.norm(Ra - Rj)
                term1 = (-2. * eta * Raj * cutoff_fxn(Raj, cutoff) /
                         (cutoff ** 2.) +
                         der_cutoff_fxn(Raj, cutoff))
                term2 = der_position(a, n_index, Ra, Rj, m, i)
                ridge += np.exp(- eta * (Raj ** 2.) / (cutoff ** 2.)) * \
                    term1 * term2
    return ridge

##############################################################################


def calculate_der_G4(n_indices, symbols, Rs, G_elements, gamma, zeta, eta,
                     cutoff, a, Ra, m, i, fortran):
    """Calculate coordinate derivative of G4 symmetry function for atom at
    index a and position Ra with respect to coordinate x_{i} of atom index
    m."""

    if fortran:  # fortran version; faster
        G_numbers = sorted([atomic_numbers[el] for el in G_elements])
        numbers = [atomic_numbers[symbol] for symbol in symbols]
        list_n_indices = [n_indices[_] for _ in range(len(n_indices))]
        if len(Rs) == 0:
            ridge = 0.
        else:
            ridge = fmodules.calculate_der_g4(n_indices=list_n_indices,
                                              numbers=numbers, rs=Rs,
                                              g_numbers=G_numbers,
                                              g_gamma=gamma,
                                              g_zeta=zeta, g_eta=eta,
                                              cutoff=cutoff, aa=a,
                                              home=Ra, mm=m,
                                              ii=i)
    else:
        ridge = 0.
        counts = range(len(symbols))
        for j in counts:
            for k in counts[(j + 1):]:
                els = sorted([symbols[j], symbols[k]])
                if els != G_elements:
                    continue
                Rj = Rs[j]
                Rk = Rs[k]
                Raj_ = Rs[j] - Ra
                Raj = np.linalg.norm(Raj_)
                Rak_ = Rs[k] - Ra
                Rak = np.linalg.norm(Rak_)
                Rjk_ = Rs[j] - Rs[k]
                Rjk = np.linalg.norm(Rjk_)
                cos_theta_ajk = np.dot(Raj_, Rak_) / Raj / Rak
                c1 = (1. + gamma * cos_theta_ajk)
                c2 = cutoff_fxn(Raj, cutoff)
                c3 = cutoff_fxn(Rak, cutoff)
                c4 = cutoff_fxn(Rjk, cutoff)
                if zeta == 1:
                    term1 = \
                        np.exp(- eta * (Raj ** 2. + Rak ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                else:
                    term1 = c1 ** (zeta - 1.) * \
                        np.exp(- eta * (Raj ** 2. + Rak ** 2. + Rjk ** 2.) /
                               (cutoff ** 2.))
                term2 = (1. / 3.) * (c2 + c3 + c4)
                term3 = der_cos_theta(a, n_indices[j], n_indices[k], Ra, Rj,
                                      Rk, m, i)
                term4 = gamma * zeta * term3
                term5 = der_position(a, n_indices[j], Ra, Rj, m, i)
                term4 += -2. * c1 * eta * Raj * term5 / (cutoff ** 2.)
                term6 = der_position(a, n_indices[k], Ra, Rk, m, i)
                term4 += -2. * c1 * eta * Rak * term6 / (cutoff ** 2.)
                term7 = der_position(n_indices[j], n_indices[k], Rj, Rk, m, i)
                term4 += -2. * c1 * eta * Rjk * term7 / (cutoff ** 2.)
                term2 = term2 * term4
                term8 = c1 * (1. / 3.) * der_cutoff_fxn(Raj, cutoff) * term5
                term9 = c1 * (1. / 3.) * der_cutoff_fxn(Rak, cutoff) * term6
                term10 = c1 * (1. / 3.) * der_cutoff_fxn(Rjk, cutoff) * term7
                term11 = term2 + term8 + term9 + term10
                term = term1 * term11
                ridge += term
        ridge *= 2. ** (1. - zeta)

    return ridge

##############################################################################

def clustering(inPop, numClusters, label=False):
    """
    clustering by fingerprints
    """
    if numClusters >= len(inPop):
        return [i for i in range(len(inPop))], inPop

    fpMat = np.array([ind.info['fingerprint'] for ind in inPop])
    km = cluster.KMeans(n_clusters=numClusters,)
    km.fit(fpMat)
    labels = km.labels_

    goodPop = [None]*numClusters
    for label, ind in zip(labels, inPop):
        curBest = goodPop[label]
        if curBest:
            if ind.info['dominators'] < curBest.info['dominators']:
                goodPop[label] = ind
        else:
            goodPop[label] = ind

    return labels, goodPop