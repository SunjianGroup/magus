######################################################################

import numpy as np
from ase.neighborlist import NeighborList
from ase.data import atomic_numbers
from ase import io
# import fmodules
from . import fmodules

##############################################################################

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
        Rs = self.atoms.positions[neighbor_indices] + np.dot(neighbor_offsets, self.atoms.get_cell())
        neighbor_symbols = [self.atoms[n_index].symbol
                            for n_index in neighbor_indices]
        # Rs = [self.atoms.positions[n_index] +
        #       np.dot(n_offset, self.atoms.get_cell()) for n_index, n_offset
        #       in zip(neighbor_indices, neighbor_offsets)]
        # fpG2 = [self.process_G(G, index, neighbor_symbols, Rs)
        #                for G in self.Gs if G['type'] == 'G2']
        # fpG4 = [self.process_G(G, index, neighbor_symbols, Rs)
        #                for G in self.Gs if G['type'] == 'G4']
        fingerprint, forceMat, virialMat = self.process_G(self.Gs, index, neighbor_symbols, neighbor_indices, Rs)


        return fingerprint, forceMat, virialMat

    #########################################################################


    #########################################################################

    def process_G(self, Gs, index, symbols, n_indices, Rs):
        """Returns the value of G for atom at index. symbols and Rs are
        lists of neighbors' symbols and Cartesian positions, respectively.
        """

        Gs_g2 = [G for G in Gs if G['type']=='G2']
        Gs_g4 = [G for G in Gs if G['type']=='G4']
        home = self.atoms[index].position

        # G2
        if len(Gs_g2) > 0:
            G2_elements = [G['element'] for G in Gs_g2]
            G2_etas = [G['eta'] for G in Gs_g2]

            G2_ridges, G2_fMat, G2_vMat = self.calculate_G2(symbols, Rs, G2_elements, G2_etas, n_indices,
                                self.cutoff, index, home, self.fortran)
            if len(Gs_g4) == 0:
                return G2_ridges, G2_fMat, G2_vMat

        # G4
        if len(Gs_g4) > 0:
            G4_elements = [G['elements'] for G in Gs_g4]
            G4_etas = [G['eta'] for G in Gs_g4]
            G4_zetas = [G['zeta'] for G in Gs_g4]

            G4_ridges, G4_fMat, G4_vMat = self.calculate_G4(symbols, Rs, G4_elements, G4_etas, G4_zetas,
                                    n_indices, self.cutoff, index, home, self.fortran)
            if len(Gs_g2) == 0:
                return G4_ridges, G4_fMat, G4_vMat

        return (np.concatenate((G2_ridges, G4_ridges), axis=0), np.concatenate((G2_fMat, G4_fMat), axis=0),
        np.concatenate((G2_vMat, G4_vMat), axis=0))

        # elif G['type'] == 'G4':
        #     return calculate_G4(symbols, Rs, G['elements'], G['gamma'],
        #                         G['zeta'], G['eta'], self.cutoff, home,
        #                         self.fortran)
        # else:
        #     raise NotImplementedError('Unknown G type: %s' % G['type'])

    #########################################################################



    ##################################################################

    def calculate_G2(self, symbols, Rs, G_elements, G_etas, n_indices, cutoff, index, home, fortran):
        """
        symbols: neighbor symbols
        Rs: neigbor positions
        G_elements: elements type for every G_i
        G_etas: eta values for every G_i
        n_indices: neighbor indices
        cutoff: cutoff radius
        index: centeral atom index
        home: centeral atom position
        fortran: use fortran module
        """

        if fortran:  # fortran version; faster
            G_numbers = [atomic_numbers[element] for element in G_elements]
            numbers = [atomic_numbers[symbol] for symbol in symbols]
            if len(Rs) == 0:
                ridges = np.zeros(len(G_numbers))
                forceMat = np.zeros([len(self.atoms) * len(G_numbers), 3])
                virialMat = np.zeros([len(self.atoms) * len(G_numbers), 6])
                # x_fp, y_fp, z_fp = [0.] * 3
            else:
                ridges, forceMat, virialMat = fmodules.calculate_g2(numbers=numbers, rs=Rs, g_numbers=G_numbers,
                                                g_etas=G_etas, indices=n_indices,
                                                cutoff=cutoff, c_index=index, home=home, atnum=len(self.atoms),)

                #debug
                self.oriFMat = forceMat

                length = forceMat.shape[0] * forceMat.shape[1]
                forceMat = np.reshape(forceMat, newshape=(length, 3), order='F')
                virialMat = np.reshape(virialMat, newshape=(length, 6), order='F')
                # ridge, x_fp, y_fp, z_fp = fmodules.calculate_g2(numbers=numbers, rs=Rs,
                #                               g_number=G_number, g_eta=eta,
                #                               cutoff=cutoff, home=home)
            return ridges, forceMat, virialMat
        # else:
        #     ridges = 0.
        #     # x_fp, y_fp, z_fp = 0., 0., 0.
        #     for symbol, R in zip(symbols, Rs):
        #         if symbol == G_element:
        #             # Rij = np.linalg.norm(R - home)
        #             Rij_ = R - home
        #             Rij = np.linalg.norm(Rij_)

        #             term = (np.exp(-eta * (Rij ** 2.) / (cutoff ** 2.)) *
        #                       cutoff_fxn(Rij, cutoff))
        #             ridge += term
        #     return ridges

    ##############################################################################


    def calculate_G4(self, symbols, Rs, G_elements, G_etas, G_zetas, n_indices, cutoff, index, home, fortran):
        """
            symbols: neighbor symbols
            Rs: neigbor positions
            G_elements: elements type for every G_i
            G_etas: eta values for every G_i
            G_zetas: zeta values for every G_i
            n_indices: neighbor indices
            cutoff: cutoff radius
            index: centeral atom index
            home: centeral atom position
            fortran: use fortran module
        """

        if fortran:  # fortran version; faster
            G_numbers = [(atomic_numbers[el0], atomic_numbers[el1]) for el0, el1 in G_elements]
            numbers = [atomic_numbers[symbol] for symbol in symbols]
            if len(Rs) == 0:
                ridges = 0.
                forceMat = np.zeros([len(self.atoms) * len(G_numbers), 3])
                virialMat = np.zeros([len(self.atoms) * len(G_numbers), 6])
            else:

                ridges, forceMat, virialMat = fmodules.calculate_g4(numbers=numbers, rs=Rs, g_numbers=G_numbers,
                                                g_etas=G_etas, g_zetas=G_zetas, indices=n_indices,
                                                cutoff=cutoff, c_index=index, home=home, atnum=len(self.atoms),)

                length = forceMat.shape[0] * forceMat.shape[1]
                forceMat = np.reshape(forceMat, newshape=(length, 3), order='F')
                virialMat = np.reshape(virialMat, newshape=(length, 6), order='F')
            return ridges, forceMat, virialMat
        # else:
        #     ridge = 0.
        #     counts = range(len(symbols))
        #     for j in counts:
        #         for k in counts[(j + 1):]:
        #             els = sorted([symbols[j], symbols[k]])
        #             if els != G_elements:
        #                 continue
        #             Rij_ = Rs[j] - home
        #             Rij = np.linalg.norm(Rij_)
        #             Rik_ = Rs[k] - home
        #             Rik = np.linalg.norm(Rik_)
        #             Rjk = np.linalg.norm(Rs[j] - Rs[k])
        #             cos_theta_ijk = np.dot(Rij_, Rik_) / Rij / Rik
        #             term = (1. + gamma * cos_theta_ijk) ** zeta
        #             term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
        #                         (cutoff ** 2.))
        #             term *= (1. / 3.) * (cutoff_fxn(Rij, cutoff) +
        #                                 cutoff_fxn(Rik, cutoff) +
        #                                 cutoff_fxn(Rjk, cutoff))
        #             ridge += term
        #     ridge *= 2. ** (1. - zeta)
        #     return ridge
###############################################
def cutoff_fxn(Rij, Rc):
    """Cosine cutoff function in Parinello-Behler method."""
    if Rij > Rc:
        return 0.
    else:
        return 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)
