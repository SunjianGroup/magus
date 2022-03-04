import logging, itertools
from functools import reduce
from copy import deepcopy
import numpy as np
from numpy.linalg import matrix_rank
from scipy.spatial import ConvexHull
from math import gcd
from ase import Atoms
from magus.utils import get_units_formula, get_units_numlist


log = logging.getLogger(__name__)


def check_units(frames, units):
    for atoms in frames:
        count = get_units_numlist(atoms, units)
        if count is None:
            return False
    return True


# def get_basis(matrix):
#     """
#     get the simplest integer basis of a linear independent matrix using enumeration method
#     for example, the matrix [[5, 1, 3], [4, 2, 3]], 
#         basis [[1, 1, 1], [2, 0, 1]] is ok since:
#         [5, 1, 3] = [1, 1, 1] * 1 + [2, 0, 1] * 2
#         [4, 2, 3] = [1, 1, 1] * 2 + [2, 0, 1] * 1
#         but [[0, 2, 1], [2, 0, 1]] is wrong since:
#         [5, 1, 3] = [0, 2, 1] * 0.5 + [2, 0, 1] * 2.5 and the float in the decomposition matrix
#     """
#     dim = len(matrix)
#     eps = 1e-5
#     # max value of row i in the decomposition matrix is the max value of row i in the origin matrix
#     ranges = [np.arange(max(matrix[i]) + 1) for i in range(dim) for _ in range(dim)]
#     for i in itertools.product(*ranges):
#         M = np.array(i).reshape(dim, dim)
#         if matrix_rank(M) == dim and np.linalg.det(M) >= 1 and (M != np.eye(dim)).any():
#             new = np.linalg.inv(M) @ matrix
#             if (new >=0).all() and (np.abs(new - np.round(new)) < eps).all():
#                 break
#     else:
#         return matrix
#     return get_basis(np.round(new).astype(int))


# def get_all_basises(matrix, results, visited):
#     if {tuple(v) for v in matrix} in visited:
#         return results
#     visited.append({tuple(v) for v in matrix})
#     dim = len(matrix)
#     eps = 1e-5
#     simplest = True
#     # max value of row i in the decomposition matrix is the max value of row i in the origin matrix
#     ranges = [np.arange(max(matrix[i]) + 1) for i in range(dim) for _ in range(dim)]
#     for i in itertools.product(*ranges):
#         M = np.array(i).reshape(dim, dim)
#         if matrix_rank(M) == dim:
#             new = np.linalg.inv(M) @ matrix
#             if np.sum(new) < np.sum(matrix) and (new >=0).all() and (np.abs(new - np.round(new)) < eps).all():
#                 get_all_basises(np.round(new).astype(int), results, visited)
#                 simplest = False
#     if simplest:
#         results.append(tuple(matrix.tolist()))
#     return results


def get_all_basises(matrix):
    eps = 1e-5
    results = []
    ranges = [np.arange(i + 1) for i in matrix.reshape(-1)]
    for i in itertools.product(*ranges):
        M = np.array(i).reshape(*matrix.shape)
        if matrix_rank(M) < len(matrix):
            continue
        x = matrix @ np.linalg.pinv(M)
        if (x >= -eps).all() and (np.abs(x - np.round(x)) < eps).all() and (np.abs(x @ M - matrix) < eps).all():
            results.append(M)
    return results

# TODO reduce the numlists first to decrease the number of enumeration
def get_units(frames):
    """
    get units of given frames
    """
    # get all symbols and numlist of the symbols of all the structures
    symbols = set([s for atoms in frames
                   for s in atoms.get_chemical_symbols()])
    numlists = [[atoms.get_chemical_symbols().count(s) for s in symbols] for atoms in frames]
    numlists = np.unique(numlists, axis=0)
    dim = matrix_rank(numlists)
    if len(symbols) == dim:
        units_numlists = np.eye(dim, dtype=int)
        units = []
        for f in units_numlists:
            units.append(Atoms(symbols=[s for n, s in zip(f, symbols) for _ in range(n)]))
        return units
    else:
        units_numlists = []
        numlists = sorted(numlists, key=lambda x: np.prod(x + 1))
        for numlist in numlists:
            if matrix_rank([*units_numlists, numlist]) > matrix_rank(units_numlists):
                units_numlists.append(numlist)
            if matrix_rank(units_numlists) == dim:
                break
        units_numlists = np.array(units_numlists)
        candidate_basises = get_all_basises(units_numlists)
        for basis in candidate_basises:
            units = []
            for f in basis:
                units.append(Atoms(symbols=[s for n, s in zip(f, symbols) for _ in range(n)]))
            if check_units(frames, units):
                return units
        else:
            return None


class PhaseDiagram:
    """
    Similar to ase.phasediagram.PhaseDiagram, rewrite some parameters for Magus
    """
    def __init__(self, frames, boundary=None):
        self.frames = deepcopy(frames)
        if boundary is None:
            boundary = get_units(frames)
            assert boundary is not None, "Fail to find boundary, please give the boundary by '-b'"
        else:
            assert check_units(frames, boundary), "Given wrong boundary"
        self.boundary = boundary
        self.boundary_n_atoms = np.array([len(atoms) for atoms in self.boundary])
        self.points = []
        for atoms in frames:
            count = get_units_numlist(atoms, self.boundary) * self.boundary_n_atoms
            ratio = count / count.sum()
            self.points.append([*ratio, atoms.info['enthalpy']])
        self.points = np.array(self.points)
        if matrix_rank(self.points[:, :-1]) < len(self.boundary):
            log.warning("dim of frames smaller than number of boundary, will add artificial points. "
                        "It is unreasonable and may raise wrong convex hull!")
            for i, atoms in enumerate(self.boundary):
                points = np.zeros((1, len(self.boundary) + 1))
                points[0, i] = 1.
                points[0, -1] = 100.
                self.points = np.concatenate((self.points, points), axis=0)
        if len(self.points) <= len(self.boundary):
            # Simple case that qhull would choke on:
            self.simplices = np.arange(len(self.points)).reshape((1, len(self.points)))
            self.hull = np.ones(len(self.points), bool)
        else:
            hull = ConvexHull(self.points[:, 1:])
            # Find relevant simplices:
            ok = hull.equations[:, -2] < 0
            self.simplices = hull.simplices[ok]
            # Create a mask for those points that are on the convex hull:
            self.hull = np.zeros(len(self.points), bool)
            for simplex in self.simplices:
                self.hull[simplex] = True

    def append(self, atoms):
        self.extend([atoms])

    def extend(self, frames):
        self.frames.extend(frames)
        self.__init__(self.frames, self.boundary)

    def decompose(self, atoms):
        """
        adjust the method to calculate coef to avoid numerical fault an example is:
            Zn5O7 cannot be composed by Zn8O4 and Zn9
        """
        name = atoms.get_chemical_formula()
        count = get_units_numlist(atoms, self.boundary)
        if count is None:
            raise Exception('{} is not in the boundary'.format(name))
        count *= self.boundary_n_atoms
        point = count / count.sum()

        # Find coordinates within each simplex:
        for simplex in self.simplices:
            try:
                x = np.linalg.solve(self.points[simplex, :-1].T, point)
            except np.linalg.linalg.LinAlgError:
                continue
            if (x > -1e-5).all(): 
                energy = x @ self.points[simplex, -1]
                return energy
        else:
            raise Exception('connot break {}'.format(name))
    
    def plot(self, **plotkwargs):
        import matplotlib.pyplot as plt
        N = len(self.boundary)
        if N == 2:
            fig = plt.figure()
            ax = fig.gca()
            self.plot2d2(ax, **plotkwargs)
        elif N == 3:
            fig = plt.figure()
            ax = fig.gca()
            self.plot2d3(ax)
        elif N == 4:
            from mpl_toolkits.mplot3d import Axes3D
            Axes3D  # silence pyflakes
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            self.plot3d4(ax)
        else:
            raise ValueError('Cannot make plots for {} component systems!'.format(N))
        return ax

    def plot2d2(self, ax):
        x, e = self.points[:, 1:].T
        # make two end points to zero
        e1 = min(e[np.where(x==0)])
        e2 = min(e[np.where(x==1)])
        e = e - e1 * (1 - x) - e2 * x
        a = self.boundary_n_atoms[1] / self.boundary_n_atoms[0]
        x = x / (x + a - a * x)
        names = [get_units_formula(atoms, self.boundary) for atoms in self.frames]
        hull = self.hull
        simplices = self.simplices
        xlabel = self.boundary[1]
        ylabel = 'energy [eV/atom]'
        for i, j in simplices:
            ax.plot(x[[i, j]], e[[i, j]], '#5b5da5', linewidth=2.5)
        ax.scatter(x[~hull], e[~hull], c='#902424', s=80, marker="x", zorder=90)
        ax.scatter(x[hull], e[hull], c='#699872', s=80, marker="o", zorder=100)
        x = x[self.hull]
        e = e[self.hull]
        names = [name for name, h in zip(names, self.hull) if h]
        for a, b, name in zip(x, e, names):
            ax.text(a, b, name, ha='center', va='top', zorder=110)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # sometimes sick data like e=1000 may add to the pd, set ylim to avoid them
        bottom = min(min(e) * 1.2, -0.1)
        top = max(abs(bottom) * 0.3, 0.1)
        ax.set_ylim(bottom, top)
        return (x, e, names, hull, simplices, xlabel, ylabel)

    def plot2d3(self, ax):
        x, y = self.points[:, 1:-1].T.copy()
        x += y / 2
        y *= 3 ** 0.5 / 2
        names = [get_units_formula(atoms, self.boundary) for atoms in self.frames]
        hull = self.hull
        simplices = self.simplices

        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-b')
        ax.scatter(x[~hull], y[~hull], c='#902424', s=80, marker="x", zorder=90, alpha=0.5)
        ax.scatter(x[hull], y[hull], c='#699872', s=80, marker="o", zorder=100)
        # only label the structures on the hull
        for i in range(len(hull)):
            if hull[i]:
                ax.text(x[i], y[i], names[i], ha='center', va='top', zorder=110)
        return (x, y, names, hull, simplices)
