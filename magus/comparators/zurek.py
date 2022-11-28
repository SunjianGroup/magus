"""Determine symmetry equivalence of two structures.
Based on the recipe from Comput. Phys. Commun. 183, 690-697 (2012)."""

from collections import Counter
from itertools import combinations, product
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.build.tools import niggli_reduce
import spglib
from magus.utils import COMPARATOR_PLUGIN


def normalize(cell):
    for i in range(3):
        cell[i] /= np.linalg.norm(cell[i])


@COMPARATOR_PLUGIN.register('zurek')
class ZurekComparator:
    def __init__(self, angle_tol=3.0, ltol=0.05, stol=0.05, vol_tol=0.05, symprec=0.1, to_primitive=True, **kwargs):
        self.angle_tol = angle_tol * np.pi / 180.0  # convert to radians
        self.stol = stol
        self.ltol = ltol
        self.vol_tol = vol_tol
        self.to_primitive = to_primitive
        self.symprec = symprec

    def _reduce_to_primitive(self, atoms):
        cell, scaled_pos, numbers = spglib.standardize_cell(atoms, symprec=self.symprec, to_primitive=True)
        atoms_ = atoms.__class__(scaled_positions=scaled_pos, numbers=numbers, cell=cell, pbc=True)
        atoms_.info = atoms.info
        return atoms_

    def _least_frequent_element_to_origin(self, atoms):
        atomic_numbers_count = Counter(atoms.numbers)
        least_freq_element = atomic_numbers_count.most_common()[-1][0]
        least_freq_pos = atoms.get_positions(wrap=True)[atoms.numbers == least_freq_element][0]
        cell_diag = np.sum(atoms.get_cell(), axis=0)
        atoms.positions -= least_freq_pos - 1e-6 * cell_diag
        atoms.wrap(pbc=[1, 1, 1])

    def prepare(self, atoms):
        """
        save some information to avoid duplicated calculation including:
            atomic_numbers_count
            least_freq_element
            standardize_form
            spglib dataset(in feature) 
        """
        if 'compare_info' in atoms.info:
            return atoms
        atoms.info['compare_info'] = {}
        atoms_ = atoms.copy()
        if self.to_primitive:
            atoms_ = self._reduce_to_primitive(atoms_)
        self._least_frequent_element_to_origin(atoms_)
        standard_atoms = self._get_standardize_form(atoms_)
        atomic_numbers_count = Counter(standard_atoms.numbers)

        least_freq_element = atomic_numbers_count.most_common()[-1][0]
        least_freq_struct = standard_atoms[standard_atoms.numbers == least_freq_element]

        least_freq_struct.wrap(pbc=[1, 1, 1])
        least_freq_positions = least_freq_struct.get_positions()
        # 3 * 3 * 3 supercell of only least frequent type at origin
        cell_diag = np.sum(least_freq_struct.cell[:], axis=0)
        delta_vec = 1E-6 * cell_diag
        
        sc_least_freq_struct = least_freq_struct * (3, 3, 3)
        sc_positions = sc_least_freq_struct.get_positions()
        sc_positions -= sc_positions[0] + cell_diag - delta_vec
        # reference vector lengths and angles
        ref_vec = standard_atoms.get_cell()
        ref_vec_lengths, ref_angles = ref_vec.cellpar(radians=True).reshape(2,3)
        ref_angles = np.where(ref_angles < np.pi / 2., ref_angles, np.pi - ref_angles)
        standardize_volume = standard_atoms.get_volume()
        volume_per_atom = standardize_volume / len(standard_atoms)
        positions_tolerance = (standardize_volume / len(standard_atoms)) ** (1 / 3)
        # expanded atoms
        expanded_atoms = self._expand(standard_atoms)

        atoms.info['compare_info'].update({
            'atomic_numbers_count': atomic_numbers_count,
            'least_freq_element': least_freq_element,
            'least_freq_struct': least_freq_struct,
            'least_freq_positions': least_freq_positions,
            'sc_positions': sc_positions,
            'ref_vec': ref_vec,
            'ref_vec_lengths': ref_vec_lengths,
            'ref_angles': ref_angles,
            'standardize_volume': standardize_volume,
            'volume_per_atom': volume_per_atom,
            'positions_tolerance': positions_tolerance,
            'expanded_atoms': expanded_atoms,
        })
        return atoms

    def _get_standardize_form(self, atoms):
        """
        standardized structure descriptions as much as possible by using the following procedure:
        1. Niggli reduction
        2. Standardize orientation
        3. Wrap atoms to cell
        """
        if 'standardize_form' not in atoms.info['compare_info']:
            atoms_ = atoms.copy()
            atoms_.info = {}
            niggli_reduce(atoms_)
            cell, rot_mat = atoms_.cell.standard_form()
            atoms_.set_cell(cell)
            atoms_.set_positions(atoms_.get_positions() @ rot_mat.T)
            atoms_.wrap(pbc=[1, 1, 1])
            atoms.info['compare_info']['standardize_form'] = atoms_
        return atoms.info['compare_info']['standardize_form']

    def _get_rotation_matrix(self, atoms1, atoms2):
        """Compute candidates for the transformation matrix."""
        angle_tol = self.angle_tol
        rtol = self.ltol / len(atoms1)

        sc_positions = atoms1.info['compare_info']['sc_positions']
        lengths = np.linalg.norm(sc_positions, axis=1)
        ref_vec  = atoms2.info['compare_info']['ref_vec']
        ref_vec_lengths = atoms2.info['compare_info']['ref_vec_lengths']
        ref_angles = atoms2.info['compare_info']['ref_angles']
        candidate_indices = []
        for k in range(3):
            correct_lengths_mask = np.isclose(lengths,
                                              ref_vec_lengths[k],
                                              rtol=rtol, atol=0)
            # The first vector is not interesting
            correct_lengths_mask[0] = False

            # If no trial vectors can be found (for any direction)
            # then the candidates are different and we return None
            if not np.any(correct_lengths_mask):
                return None

            candidate_indices.append(np.nonzero(correct_lengths_mask)[0])

        # Now we calculate all relevant angles in one step. The relevant angles
        # are the ones made by the current candidates. We will have to keep
        # track of the indices in the angles matrix and the indices in the
        # position and length arrays.

        # Get all candidate indices (aci), only unique values
        aci = np.sort(list(set().union(*candidate_indices)))

        # Make a dictionary from original positions and lengths index to
        # index in angle matrix
        i2ang = dict(zip(aci, range(len(aci))))

        # Calculate the dot product divided by the lengths:
        # cos(angle) = dot(vec1, vec2) / |vec1| |vec2|
        cosa = np.inner(sc_positions[aci],
                        sc_positions[aci]) / np.outer(lengths[aci],
                                                    lengths[aci])
        # Make sure the inverse cosine will work
        cosa[cosa > 1] = 1
        cosa[cosa < -1] = -1
        angles = np.arccos(cosa)
        # Do trick for enantiomorphic structures
        angles[angles > np.pi / 2] = np.pi - angles[angles > np.pi / 2]

        # Check which angles match the reference angles
        refined_candidate_list = []
        for i, j, k in product(*candidate_indices):
            if i != j and i != k and j!= k:
                a = np.array([angles[i2ang[j], i2ang[k]],
                            angles[i2ang[i], i2ang[k]],
                            angles[i2ang[i], i2ang[j]]])

                if np.allclose(a, ref_angles, atol=angle_tol, rtol=0):
                    refined_candidate_list.append(sc_positions[[i, j ,k]].T)

        # Get the rotation/reflection matrix [R] by:
        # [R] = [V][T]^-1, where [V] is the reference vectors and
        # [T] is the trial vectors
        # XXX What do we know about the length/shape of refined_candidate_list?
        if len(refined_candidate_list) == 0:
            return None
        else:
            inverted_trial = np.linalg.inv(refined_candidate_list)

        # Equivalent to np.matmul(ref_vec.T, inverted_trial)
        candidate_trans_mat = np.dot(ref_vec.T, inverted_trial.T).T
        return candidate_trans_mat

    def _has_same_elements(self, atoms1, atoms2):
        """Check if two structures have same elements."""
        elem1 = atoms1.info['compare_info']['atomic_numbers_count']
        elem2 = atoms2.info['compare_info']['atomic_numbers_count']
        return elem1 == elem2

    def _has_same_angles(self, atoms1, atoms2):
        """Check that the Niggli unit vectors has the same internal angles."""
        ang1 = np.sort(atoms1.info['compare_info']['ref_angles'])
        ang2 = np.sort(atoms2.info['compare_info']['ref_angles'])
        return np.allclose(ang1, ang2, rtol=0, atol=self.angle_tol)

    def _has_same_volume(self, atoms1, atoms2):
        vol1 = atoms1.info['compare_info']['volume_per_atom']
        vol2 = atoms2.info['compare_info']['volume_per_atom']
        return np.abs(vol1 - vol2) / vol1 < self.vol_tol

    def looks_like(self, ind1, ind2):
        self.prepare(ind1)
        if isinstance(ind2, Atoms):
            ind2 = [ind2]
        for ind in ind2:
            self.prepare(ind)
            # Compare chemical formulae
            if not self._has_same_elements(ind1, ind):
                continue
            # Compare angles
            if not self._has_same_angles(ind1, ind):
                continue
            # Compare volumes
            if not self._has_same_volume(ind1, ind):
                continue
            matrices = self._get_rotation_matrix(ind1, ind)
            if matrices is None:
                continue
            # Calculate tolerance on positions
            self.position_tolerance = self.stol * ind1.info['compare_info']['positions_tolerance']
            if self._positions_match(matrices, ind1, ind):
                return True
        return False

    def _positions_match(self, matrices, atoms1, atoms2):
        """Check if the position and elements match.

        Note that this function changes self.s1 and self.s2 to the rotation and
        translation that matches best. Hence, it is crucial that this function
        calls the element comparison, not the other way around.
        """
        translations = atoms1.info['compare_info']['least_freq_positions']
        standardize_atoms1 = atoms1.info['compare_info']['standardize_form'].copy()
        standardize_pos1 = standardize_atoms1.get_positions(wrap=True)
        # Get the expanded reference object
        exp2 = atoms2.info['compare_info']['expanded_atoms']
        # Build a KD tree to enable fast look-up of nearest neighbours
        tree = KDTree(exp2.get_positions())
        for translation in translations:
            # Translate
            pos1_trans = standardize_pos1 - translation
            for matrix in matrices:
                # Rotate
                pos1 = matrix.dot(pos1_trans.T).T
                # Update the atoms positions
                standardize_atoms1.set_positions(pos1)
                standardize_atoms1.wrap(pbc=[1, 1, 1])
                if self._elements_match(standardize_atoms1, exp2, tree):
                    return True
        return False

    def _expand(self, ref_atoms, tol=0.0001):
        """If an atom is closer to a boundary than tol it is repeated at the
        opposite boundaries.

        This ensures that atoms having crossed the cell boundaries due to
        numerical noise are properly detected.

        The distance between a position and cell boundary is calculated as:
        dot(position, (b_vec x c_vec) / (|b_vec| |c_vec|) ), where x is the
        cross product.
        """
        syms = ref_atoms.get_chemical_symbols()
        cell = ref_atoms.get_cell()
        positions = ref_atoms.get_positions(wrap=True)
        expanded_atoms = ref_atoms.copy()

        # Calculate normal vectors to the unit cell faces
        normal_vectors = np.array([np.cross(cell[1, :], cell[2, :]),
                                   np.cross(cell[0, :], cell[2, :]),
                                   np.cross(cell[0, :], cell[1, :])])
        normalize(normal_vectors)

        # Get the distance to the unit cell faces from each atomic position
        pos2faces = np.abs(positions.dot(normal_vectors.T))

        # And the opposite faces
        pos2oppofaces = np.abs(np.dot(positions - np.sum(cell, axis=0),
                                      normal_vectors.T))

        for i, i2face in enumerate(pos2faces):
            # Append indices for positions close to the other faces
            # and convert to boolean array signifying if the position at
            # index i is close to the faces bordering origo (0, 1, 2) or
            # the opposite faces (3, 4, 5)
            i_close2face = np.append(i2face, pos2oppofaces[i]) < tol
            # For each position i.e. row it holds that
            # 1 x True -> close to face -> 1 extra atom at opposite face
            # 2 x True -> close to edge -> 3 extra atoms at opposite edges
            # 3 x True -> close to corner -> 7 extra atoms opposite corners
            # E.g. to add atoms at all corners we need to use the cell
            # vectors: (a, b, c, a + b, a + c, b + c, a + b + c), we use
            # itertools.combinations to get them all
            for j in range(sum(i_close2face)):
                for c in combinations(np.nonzero(i_close2face)[0], j + 1):
                    # Get the displacement vectors by adding the corresponding
                    # cell vectors, if the atom is close to an opposite face
                    # i.e. k > 2 subtract the cell vector
                    disp_vec = np.zeros(3)
                    for k in c:
                        disp_vec += cell[k % 3] * (int(k < 3) * 2 - 1)
                    pos = positions[i] + disp_vec
                    expanded_atoms.append(Atom(syms[i], position=pos))
        return expanded_atoms

    def _elements_match(self, s1, s2, kdtree):
        """Check if all the elements in s1 match the corresponding position in s2

        NOTE: The unit cells may be in different octants
        Hence, try all cyclic permutations of x,y and z
        """
        pos1 = s1.get_positions()
        for order in range(1):  # Is the order still needed?
            pos_order = [order, (order + 1) % 3, (order + 2) % 3]
            pos = pos1[:, np.argsort(pos_order)]
            dists, closest_in_s2 = kdtree.query(pos)

            # Check if the elements are the same
            if not np.all(s2.numbers[closest_in_s2] == s1.numbers):
                return False

            # Check if any distance is too large
            if np.any(dists > self.position_tolerance):
                return False

            # Check for duplicates in what atom is closest
            if len(closest_in_s2) != len(set(closest_in_s2)):
                return False
        return True


@COMPARATOR_PLUGIN.register('ase-zurek')
class ASEComparator:
    def __init__(self, angle_tol=3.0, ltol=0.05, stol=0.05, vol_tol=0.05, symprec=0.1, to_primitive=True, **kwargs):
        self.comparator = SymmetryEquivalenceCheck(
            angle_tol=angle_tol, ltol=ltol, stol=stol, vol_tol=vol_tol, to_primitive=False)
        self.symprec = symprec
        self.to_primitive = to_primitive

    def looks_like(self, ind1, ind2):
        ind1_ = ind1.copy()
        if isinstance(ind2, Atoms):
            ind2 = [ind2]
        # we cannot control symprec in ase, so we convert atoms here
        if self.to_primitive:
            lattice, scaled_positions, numbers = spglib.find_primitive(ind1_, symprec=self.symprec)
            ind1_ = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers, pbc=ind1.pbc)
            ind2_ = []
            for ind in ind2:
                lattice, scaled_positions, numbers = spglib.find_primitive(ind, symprec=self.symprec)
                ind2_.append(Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers, pbc=ind.pbc))
        return self.comparator.compare(ind1_, ind2_)
