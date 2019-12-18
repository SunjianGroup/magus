import copy, time, logging, random, math
import numpy as np
from scipy import optimize
from ase.neighborlist import NeighborList
# try:
#     import mxnet as mx
# except ImportError:
#     pass

def mat_eye_symbol(n):
    """Creates n-by-n identity matrix, shape (n, n)

    :param n: Matrix size
    :param dtype: Data type (float32 or float64)
    :return: n-by-n identity matrix
    """

    index = mx.sym.arange(
        start=0, stop=n, step=1)
    # Shape: (n, n)
    return mx.sym.one_hot(index, depth=n)

def extract_diag_symbol(mat, n):
    """Extracts diagonal from square matrix

    :param mat: Square matrix, shape (n, n)
    :param n: Matrix size
    :return: Diagonal of mat, shape (n,)
    """

    index = mx.sym.arange(
        start=0, stop=n*n, step=n+1)
    return mx.sym.take(mx.sym.reshape(mat, shape=(n*n,)), index)

def extract_diag_nd(mat):
    """Extracts diagonal from square matrix

    :param mat: Square matrix, shape (n, n)
    :return: Diagonal of mat, shape (n,)
    """
    n = int(mat.shape[0])
    index = mx.nd.arange(
        start=0, stop=n*n, step=n+1)
    return mx.nd.take(mx.nd.reshape(mat, shape=(n*n,)), index)

def gpregr_criterion_symbol(kern_mat, targets, noise_var, num_cases):
    """Negative log likelihood criterion for Gaussian process regression

    Let n = num_cases.

    :param kern_mat: Kernel matrix, (n, n) symbol
    :param targets: Observed targets, (n,) symbol
    :param noise_var: Noise variance, (1,n) symbol (positive)
    :param num_cases: Number of cases n
    :param dtype: Data type (float32 or float64)
    :return: Learning criterion
    """

    # amat = kern_mat + mx.sym.broadcast_mul(
    #     mat_eye_symbol(num_cases), mx.sym.reshape(noise_var, shape=(1, 1)))
    amat = kern_mat + mx.sym.broadcast_mul(
        mat_eye_symbol(num_cases), mx.sym.tile(noise_var, reps=(num_cases,1)))
    chol_fact = mx.sym.linalg_potrf(amat)
    zvec = mx.sym.linalg_trsm(
        chol_fact, mx.sym.reshape(targets, shape=(num_cases, 1)), transpose=False,
        rightside=False)
    sqnorm_z = mx.sym.sum(mx.sym.square(zvec))
    # logdet_l = mx.sym.sum(mx.sym.log(mx.sym.abs(
    #     extract_diag_symbol(chol_fact, num_cases))))
    logdet_l = mx.sym.linalg.sumlogdiag(mx.sym.abs(chol_fact))


    return 0.5 * (sqnorm_z + (num_cases * np.log(2.0 * np.pi))) + logdet_l

def gpregr_prediction_symbol(kern_mat, targets, noise_var, num_cases):
    """Compute the symbols used for Gaussian process regression

    Let n = num_cases.

    :param kern_mat: Kernel matrix, (n, n) symbol
    :param targets: Observed targets, (n,) symbol
    :param noise_var: Noise variance, (1,n) symbol (positive)
    :param num_cases: Number of cases n
    :param dtype: Data type (float32 or float64)
    :return: MXNet symbol group
    """

    # amat = kern_mat + mx.sym.broadcast_mul(
    #     mat_eye_symbol(num_cases), mx.sym.reshape(noise_var, shape=(1, 1)))
    amat = kern_mat + mx.sym.broadcast_mul(
        mat_eye_symbol(num_cases), mx.sym.tile(noise_var, reps=(num_cases,1)))
    chol_fact = mx.sym.linalg_potrf(amat)
    zvec = mx.sym.linalg_trsm(
        chol_fact, mx.sym.reshape(targets, shape=(num_cases, 1)), transpose=False,
        rightside=False)
    sqnorm_z = mx.sym.sum(mx.sym.square(zvec))
    # logdet_l = mx.sym.sum(mx.sym.log(extract_diag_symbol(chol_fact, num_cases)))
    logdet_l = mx.sym.linalg.sumlogdiag(chol_fact)


    wv = mx.sym.linalg_trsm(A=chol_fact, B=zvec, transpose=True, rightside=False,name='woodbury_vector')
    post = mx.sym.Group([chol_fact, wv])
    return post

def criterion_symbol_VFE(Kuu, Kuf, Kff, targets, noise_var, num_inc,num_cases):
    """Negative log likelihood criterion for VFE approximation Gaussian process regression

    Let f = num_cases, u = num_inc.

    :param Kuu: Kernel matrix, (u, u) symbol
    :param Kuf: Kernel matrix, (u, f) symbol
    :param Kff: Kernel matrix, (f, f) symbol
    :param targets: Observed targets, (f,) symbol
    :param noise_var: Noise variance, (1,) symbol (positive)
    :param num_cases: Number of cases f
    :param num_inc: Number of including points u
    :return: Learning criterion
    """

    amat_uu = Kuu + mx.sym.broadcast_mul(
        mat_eye_symbol(num_inc), mx.sym.reshape(noise_var, shape=(1, 1)))
    # amat_uu = Kuu
    chol_uu = mx.sym.linalg_potrf(amat_uu)

    amat_uffu = amat_uu + mx.sym.broadcast_div(mx.sym.linalg.syrk(Kuf, transpose=False, alpha=1.), mx.sym.reshape(noise_var, shape=(1,1)))
    chol_uffu = mx.sym.linalg_potrf(amat_uffu)
    zvec = mx.sym.linalg_trsm(chol_uffu,
    mx.sym.dot(Kuf, mx.sym.reshape(targets, shape=(num_cases, 1))), transpose=False, rightside=False)
    zvec = mx.sym.broadcast_div(zvec, mx.sym.reshape(noise_var, shape=(1,1)))

    sqnorm_z = mx.sym.linalg.syrk(mx.sym.reshape(targets, shape=(num_cases, 1)), transpose=True)
    sqnorm_z = mx.sym.broadcast_div(sqnorm_z, mx.sym.reshape(noise_var, shape=(1,1)))
    sqnorm_z = mx.sym.reshape(sqnorm_z, shape=(1,)) - mx.sym.sum(mx.sym.square(zvec))

    logdet_l = mx.sym.linalg.sumlogdiag(mx.sym.abs(chol_uu))

    L_uf = mx.sym.linalg_trsm(chol_uu, Kuf, transpose=False, rightside=False)
    Qff = mx.sym.linalg.syrk(L_uf, transpose=True, alpha=1.)
    tr_term = mx.sym.sum(extract_diag_symbol(Kff, num_cases) - extract_diag_symbol(Qff, num_cases))
    tr_term = mx.sym.broadcast_div(tr_term, mx.sym.reshape(noise_var, shape=(1,1)))
    tr_term = mx.sym.reshape(tr_term, shape=(1,))

    return 0.5 * (sqnorm_z + (num_cases * np.log(2.0 * np.pi))) + logdet_l - 0.5 * tr_term


def prediction_symbol_VFE(Kuu, Kuf,  targets, noise_var, num_inc,num_cases,):
    """Negative log likelihood criterion for VFE approximation Gaussian process regression

    Let f = num_cases, u = num_inc.

    :param Kuu: Kernel matrix, (u, u) symbol
    :param Kuf: Kernel matrix, (u, f) symbol
    :param targets: Observed targets, (f,) symbol
    :param noise_var: Noise variance, (1,) symbol (positive)
    :param num_cases: Number of cases f
    :param num_inc: Number of including points u
    :return: Learning criterion
    """

    amat_uu = Kuu + mx.sym.broadcast_mul(mat_eye_symbol(num_inc), mx.sym.reshape(noise_var, shape=(1, 1)))
    # # amat_uu = Kuu
    # chol_uu = mx.sym.linalg_potrf(amat_uu)


    amat_uffu = amat_uu + mx.sym.broadcast_div(mx.sym.dot(Kuf, Kuf.transpose()), mx.sym.reshape(noise_var, shape=(1,1)))
    chol_uffu = mx.sym.linalg_potrf(amat_uffu)
    zvec = mx.sym.linalg_trsm(chol_uffu,
    mx.sym.dot(Kuf, mx.sym.reshape(targets, shape=(num_cases, 1))), transpose=False, rightside=False)
    zvec = mx.sym.broadcast_div(zvec, mx.sym.reshape(noise_var, shape=(1,1)))

    sqnorm_z = mx.sym.linalg.syrk(mx.sym.reshape(targets, shape=(num_cases, 1)), transpose=True)
    sqnorm_z = mx.sym.broadcast_div(sqnorm_z, mx.sym.reshape(noise_var, shape=(1,1)))
    sqnorm_z = mx.sym.reshape(sqnorm_z, shape=(1,)) - mx.sym.sum(mx.sym.square(zvec))

    # logdet_l = mx.sym.linalg.sumlogdiag(mx.sym.abs(chol_uu))

    wv = mx.sym.linalg_trsm(A=chol_uffu, B=zvec, transpose=True, rightside=False)
    # wv = mx.sym.broadcast_div(wv, mx.sym.reshape(noise_var, shape=(1,1)))
    # wv = mx.sym.broadcast_div(mx.sym.reshape(targets, shape=(num_cases, 1)), mx.sym.reshape(noise_var, shape=(1,1)))
    # - mx.sym.dot(Kuf.transpose(), wv)

    # L_uf = mx.sym.linalg_trsm(chol_uu, Kuf, transpose=False, rightside=False)
    # L_ff = mx.sym.linalg_trsm(chol_uffu, Kuf, transpose=False, rightside=False)
    # L_ff = mx.sym.broadcast_div(L_ff, mx.sym.reshape(noise_var, shape=(1,1)))

    # invKuu = mx.sym.linalg.potri(chol_uu)
    # tmp = mx.sym.dot(Kuf.transpose(), mx.sym.linalg.potri(chol_uffu))
    # tmp = mx.sym.dot(tmp, Kuf)
    # invKff = mx.sym.broadcast_div(mat_eye_symbol(num_cases), mx.sym.reshape(noise_var, shape=(1,1))) - tmp

    # wv = mx.sym.dot(invKff, targets)

    post = mx.sym.Group([wv])
    return post



# Kernel matrix computations for Gaussian kernel

def _sum_squares_symbol(x, axis=0):
    return mx.sym.sum(mx.sym.square(x), axis=axis, keepdims=True)

def _rescale_data(xmat, lengthscale):
    return mx.sym.broadcast_div(
        xmat, mx.sym.reshape(lengthscale, shape=(1, -1)))

def _kern_gaussian_pointwise(amat, variance):
    return mx.sym.broadcast_mul(
        mx.sym.exp(amat), mx.sym.reshape(variance, shape=(1, 1)))

def kern_gaussian_symm_symbol(xmat, variance, lengthscale, num_cases):
    """Symmetric kernel matrix K(xmat, xmat) for Gaussian (RBF) kernel

    Let n = num_cases, d = num_dim.

    :param xmat: Matrix of input points, shape (n, d)
    :param variance: Variance parameter, (1, 1) symbol
    :param lengthscale: Lengthscale parameter(s), size 1 or d symbol (positive)
    :param num_cases: Number of cases n
    :return: Kernel matrix K(xmat, xmat), shape (n, n)
    """

    xsc = _rescale_data(xmat, lengthscale)
    # Inner product matrix => amat
    amat = mx.sym.linalg.syrk(xsc, transpose=False, alpha=1.)
    # amat = mx.sym.linalg_gemm2(xsc, xsc, False, True, 1.)
    # Matrix of squared distances times (-1/2) => amat
    dg_a = (-0.5) * extract_diag_symbol(amat, num_cases)
    amat = mx.sym.broadcast_add(amat, mx.sym.reshape(dg_a, shape=(1, -1)))
    amat = mx.sym.broadcast_add(amat, mx.sym.reshape(dg_a, shape=(-1, 1)))
    return _kern_gaussian_pointwise(amat, variance)

def kern_gaussian_gen_symbol(x1mat, x2mat, variance, lengthscale):
    """General kernel matrix K(x1mat, x2mat) for Gaussian (RBF) kernel

    :param x1mat: Matrix of input points, shape (n1, d)
    :param x2mat: Matrix of input points, shape (n2, d)
    :param variance: Variance parameter, (1, 1) symbol
    :param lengthscale: Lengthscale parameter(s), size 1 or d symbol (positive)
    :return: Kernel matrix K(x1mat, x2mat), shape (n1, n2)
    """

    x1sc = _rescale_data(x1mat, lengthscale)
    x2sc = _rescale_data(x2mat, lengthscale)
    # Inner product matrix => amat
    amat = mx.sym.linalg_gemm2(
        x1sc, x2sc, transpose_a=False, transpose_b=True, alpha=1.)
    # Matrix of squared distances times (-1/2) => amat
    dg1 = (-0.5) * _sum_squares_symbol(x1sc, axis=1)
    amat = mx.sym.broadcast_add(amat, dg1)
    dg2 = mx.sym.reshape(
        (-0.5) * _sum_squares_symbol(x2sc, axis=1), shape=(1, -1))
    amat = mx.sym.broadcast_add(amat, dg2)
    return _kern_gaussian_pointwise(amat, variance)

# def gen_noise_symbol(x1mat, x2mat, variance, lengthscale):
#     """Use gaussian function to replace Delta function

#     :param x1mat: Matrix of input points, shape (n1, d)
#     :param x2mat: Matrix of input points, shape (n2, d)
#     :param variance: Variance parameter, (1, 1) symbol
#     :param lengthscale: Lengthscale parameter(s), size 1 or d symbol (positive)
#     :return: Noise matrix K(x1mat, x2mat), shape (n1, n2)
#     """

#     x1sc = _rescale_data(x1mat, lengthscale)
#     x2sc = _rescale_data(x2mat, lengthscale)
#     # Inner product matrix => amat
#     amat = mx.sym.linalg_gemm2(
#         x1sc, x2sc, transpose_a=False, transpose_b=True, alpha=1.)
#     # Matrix of squared distances times (-1/2) => amat
#     dg1 = (-0.5) * _sum_squares_symbol(x1sc, axis=1)
#     amat = mx.sym.broadcast_add(amat, dg1)
#     dg2 = mx.sym.reshape(
#         (-0.5) * _sum_squares_symbol(x2sc, axis=1), shape=(1, -1))
#     amat = mx.sym.broadcast_add(amat, dg2)
#     return _kern_gaussian_pointwise(-1*amat, variance)

def prepare_data(allData, input_dim):
    numStruct = len(allData)
    lens = [len(data[0]) for data in allData]
    allNum = sum(lens)
    newData = []

    for data in allData:

        enFps, enMat, fMat, vMat, energy, forces, stress = [np.array(term) for term in data]
        energy = np.array([energy])
        newData.append((enFps, enMat, fMat, vMat, energy, forces, stress))

    # input vectors
    allInput = np.concatenate([data[0] for data in newData], axis=0)
    # transform mat
    mats = [[None for i in range(numStruct)] for j in range(numStruct)]
    for i in range(numStruct):
        for j in range(numStruct):
            if i == j:
                mats[i][j] = newData[i][1]
            else:
                mats[i][j] = np.zeros((1, lens[j]))
    enTrans = np.concatenate([np.concatenate(mats[i], axis=1) for i in range(numStruct)], axis=0)

    for i in range(numStruct):
        for j in range(numStruct):
            if i == j:
                mats[i][j] = newData[i][2]
            else:
                mats[i][j] = np.zeros((3*lens[i], input_dim*lens[j]))
    fTrans = np.concatenate([np.concatenate(mats[i], axis=1) for i in range(numStruct)], axis=0)
    line1 = np.concatenate([enTrans, np.zeros((numStruct, input_dim*allNum))], axis=1)
    line2 = np.concatenate([np.zeros((3*allNum, allNum)), fTrans], axis=1)
    allTrans = np.concatenate([line1, line2], axis=0)

    # target vector
    enTarget = np.concatenate([data[4] for data in newData], axis=0)
    fTarget = np.concatenate([data[5] for data in newData], axis=0)
    allTarget = np.concatenate([enTarget, fTarget], axis=0)

    return allInput, allTrans, allTarget



class AtomGP(object):
    def __init__(self, input_dim, theta=1, delta=1, mode='Full', numInc=10, theta_bound=0.1):
        self.input_dim = input_dim
        self.numInc = numInc
        self.mode = mode
        self.params = dict()
        self.params['theta'] = mx.nd.array([[theta]])
        self.params['noise'] = mx.nd.array([[1,1,1]])
        if mode == 'Full':
            pass
        elif mode == 'VFE':
            self.params['noise'] = mx.nd.array([[1]])
            # self.params['inc_points'] = mx.nd.ones((numInc, input_dim))
        else:
            raise RuntimeError("Mode {} is not supported!".format(mode))
        # self.params['delta'] = mx.nd.array([[delta]])

        # self.grad_args['theta'] = mx.nd.zeros((1,1))
        # self.grad_args['delta'] = mx.nd.zeros((1,1))

        self.grad_args = dict()
        self.lower_bound = dict()
        self.vs = dict()
        self.sqrs = dict()
        for key, val in self.params.items():
            self.grad_args[key] = mx.nd.zeros_like(val)
            self.vs[key] = mx.nd.zeros_like(val)
            self.sqrs[key] = mx.nd.zeros_like(val)
            self.lower_bound[key] = mx.nd.zeros_like(val)
        self.lower_bound['theta'] = mx.nd.array([[theta_bound]])

        # remove delta from self.params
        self.delta = mx.nd.array([[delta]])


    def compute_Kuu(self,):
        """
        VFE mode
        Ref: Understanding Probabilistic Sparse Gaussian Process Approximations
        https://arxiv.org/abs/1606.04820
        """
        theta = mx.sym.var('theta')
        delta = mx.sym.var('delta')
        inc_points = mx.sym.var('inc_points')

        Kuu = kern_gaussian_symm_symbol(inc_points, delta, theta, self.numInc)
        self.Kuu = Kuu

    def compute_Kux(self, num_X, mode='train'):
        theta = mx.sym.var('theta')
        delta = mx.sym.var('delta')
        inc_points = mx.sym.var('inc_points') # left
        Xs = mx.sym.var('Xs') # right
        xTrans = mx.sym.var('xTrans')

        sclKux = kern_gaussian_gen_symbol(inc_points, Xs, delta, theta)
        leftTmp1 = mx.sym.tile(inc_points, reps=(1, num_X))
        rightTmp1 = mx.sym.repeat(mx.sym.reshape(Xs, shape=(1, self.input_dim*num_X)), self.numInc, axis=0)
        blockMat1 = leftTmp1 - rightTmp1
        dKMat1 = mx.sym.broadcast_div(blockMat1, mx.sym.square(theta))
        repeatKux = mx.sym.repeat(sclKux, self.input_dim, axis=1)
        dKMat1 = repeatKux * dKMat1

        oriKux = mx.sym.concat(sclKux, dKMat1, dim=1)
        Kux = mx.sym.dot(oriKux, xTrans.transpose(), name='Kux')
        if mode == 'train':
            self.Kux = Kux
        elif mode == 'predict':
            return Kux

    def compute_Kxx(self, num_X, mode='train'):
        theta = mx.sym.var('theta')
        delta = mx.sym.var('delta')
        Xs = mx.sym.var('Xs')
        Trans = mx.sym.var('Trans')
        eles = mx.sym.var('eles')
        sclNoi = mx.sym.var('sclNoi')
        noiWid = mx.sym.var('noiWid')


        # scalar Kxx
        sclKxx = kern_gaussian_symm_symbol(Xs, delta, theta, num_X)
        # sclKxx = sclKxx + kern_gaussian_symm_symbol(Xs, sclNoi, noiWid, num_X)
        sclKxx = sclKxx + mx.sym.broadcast_mul(sclNoi, mat_eye_symbol(num_X))

        # elements
        tmpEle = mx.sym.tile(eles, reps=(num_X, 1))
        eleMat = (tmpEle == tmpEle.transpose())
        sclKxx = sclKxx * eleMat

        # sclKxx = sclKxx + mx.sym.broadcast_mul(
        # mat_eye_symbol(num_X), mx.sym.reshape(noise, shape=(1, 1)))
        tmpMat1 = mx.sym.repeat(mx.sym.reshape(Xs, shape=(1, self.input_dim*num_X)), num_X, axis=0)
        tmpMat2 = mx.sym.concat(*[Xs]*num_X, dim=1)
        # block matrix element (ij) = Xi - Xj
        blockMat = tmpMat2 - tmpMat1
        dKMat = mx.sym.broadcast_div(blockMat, mx.sym.square(theta))
        repeatKxx = mx.sym.repeat(sclKxx, self.input_dim, axis=1)
        dKMat = repeatKxx * dKMat
        tmpMat3 = mx.sym.repeat(blockMat, self.input_dim, axis=0)
        d2KMat = mx.sym.broadcast_div(tmpMat3.transpose()*tmpMat3, mx.sym.square(theta))
        d2KMat = d2KMat + mx.sym.tile(mat_eye_symbol(self.input_dim), reps=(num_X, num_X))
        d2KMat = mx.sym.repeat(repeatKxx, self.input_dim, axis=0) * d2KMat
        d2KMat = mx.sym.broadcast_div(d2KMat, mx.sym.square(theta))

        line1 = mx.sym.concat(sclKxx, dKMat, dim=1)
        line2 = mx.sym.concat(dKMat.transpose(), d2KMat, dim=1)
        oriKxx = mx.sym.concat(line1, line2, dim=0)

        Kxx = mx.sym.dot(mx.sym.dot(Trans, oriKxx), mx.sym.swapaxes(Trans,0,1))
        # Kxx = (Kxx.transpose() + Kxx)/2

        # self.Kxx = mx.sym.Group([Kxx, oriKxx])
        if mode == 'train':
            self.Kxx = Kxx
        elif mode == 'predict':
            return Kxx

    def compute_Kxx_symbol(self, num_X, symInput, symTrans, mode='train'):
        theta = mx.sym.var('theta')
        delta = mx.sym.var('delta')
        # Xs = mx.sym.var('Xs')
        # Trans = mx.sym.var('Trans')
        eles = mx.sym.var('eles')
        sclNoi = mx.sym.var('sclNoi')
        noiWid = mx.sym.var('noiWid')


        # scalar Kxx
        sclKxx = kern_gaussian_symm_symbol(symInput, delta, theta, num_X)
        # sclKxx = sclKxx + kern_gaussian_symm_symbol(Xs, sclNoi, noiWid, num_X)
        sclKxx = sclKxx + mx.sym.broadcast_mul(sclNoi, mat_eye_symbol(num_X))

        # elements
        tmpEle = mx.sym.tile(eles, reps=(num_X, 1))
        eleMat = (tmpEle == tmpEle.transpose())
        sclKxx = sclKxx * eleMat

        # sclKxx = sclKxx + mx.sym.broadcast_mul(
        # mat_eye_symbol(num_X), mx.sym.reshape(noise, shape=(1, 1)))
        tmpMat1 = mx.sym.repeat(mx.sym.reshape(symInput, shape=(1, self.input_dim*num_X)), num_X, axis=0)
        tmpMat2 = mx.sym.concat(*[symInput]*num_X, dim=1)
        # block matrix element (ij) = Xi - Xj
        blockMat = tmpMat2 - tmpMat1
        dKMat = mx.sym.broadcast_div(blockMat, mx.sym.square(theta))
        repeatKxx = mx.sym.repeat(sclKxx, self.input_dim, axis=1)
        dKMat = repeatKxx * dKMat
        tmpMat3 = mx.sym.repeat(blockMat, self.input_dim, axis=0)
        d2KMat = mx.sym.broadcast_div(tmpMat3.transpose()*tmpMat3, mx.sym.square(theta))
        d2KMat = d2KMat + mx.sym.tile(mat_eye_symbol(self.input_dim), reps=(num_X, num_X))
        d2KMat = mx.sym.repeat(repeatKxx, self.input_dim, axis=0) * d2KMat
        d2KMat = mx.sym.broadcast_div(d2KMat, mx.sym.square(theta))

        line1 = mx.sym.concat(sclKxx, dKMat, dim=1)
        line2 = mx.sym.concat(dKMat.transpose(), d2KMat, dim=1)
        oriKxx = mx.sym.concat(line1, line2, dim=0)

        Kxx = mx.sym.dot(mx.sym.dot(symTrans, oriKxx), mx.sym.swapaxes(symTrans,0,1))
        # Kxx = (Kxx.transpose() + Kxx)/2

        # self.Kxx = mx.sym.Group([Kxx, oriKxx])
        if mode == 'train':
            self.Kxx = Kxx
        elif mode == 'predict':
            return Kxx

    def compute_Kxox(self, num_X1, num_X2, mode='train'):
        theta = mx.sym.var('theta')
        delta = mx.sym.var('delta')
        lXs = mx.sym.var('lXs')
        rXs = mx.sym.var('rXs')
        # left and right transform matrix
        lTrans = mx.sym.var('lTrans')
        rTrans = mx.sym.var('rTrans')
        #
        lEles = mx.sym.var('lEles')
        rEles = mx.sym.var('rEles')

        sclNoi = mx.sym.var('sclNoi')
        noiWid = mx.sym.var('noiWid')

        sclKxox = kern_gaussian_gen_symbol(lXs, rXs, delta, theta)
        # sclKxox = sclKxox + kern_gaussian_gen_symbol(lXs, rXs, sclNoi, noiWid)
        # elements
        tmpEle1 = mx.sym.tile(lEles, reps=(num_X2, 1))
        tmpEle2 = mx.sym.tile(rEles, reps=(num_X1, 1))
        eleMat = (tmpEle1.transpose() == tmpEle2)
        sclKxox = sclKxox * eleMat

        leftTmp1 = mx.sym.tile(lXs, reps=(1, num_X2))
        rightTmp1 = mx.sym.repeat(mx.sym.reshape(rXs, shape=(1, self.input_dim*num_X2)), num_X1, axis=0)
        blockMat1 = leftTmp1 - rightTmp1
        dKMat1 = mx.sym.broadcast_div(blockMat1, mx.sym.square(theta))
        repeatKxx = mx.sym.repeat(sclKxox, self.input_dim, axis=1)
        dKMat1 = repeatKxx * dKMat1

        rightTmp2 = mx.sym.tile(rXs, reps=(1, num_X1))
        leftTmp2 = mx.sym.repeat(mx.sym.reshape(lXs, shape=(1, self.input_dim*num_X1)), num_X2, axis=0)
        blockMat2 = rightTmp2 - leftTmp2
        blockMat2 = blockMat2.transpose()
        dKMat2 = mx.sym.broadcast_div(blockMat2, mx.sym.square(theta))
        dKMat2 = mx.sym.repeat(sclKxox, self.input_dim, axis=0) * dKMat2

        d2KMat = mx.sym.repeat(blockMat1, self.input_dim, axis=0) * mx.sym.repeat(blockMat2, self.input_dim, axis=1)
        d2KMat = mx.sym.broadcast_div(d2KMat, mx.sym.square(theta))
        d2KMat = d2KMat + mx.sym.tile(mat_eye_symbol(self.input_dim), reps=(num_X1, num_X2))
        d2KMat = mx.sym.repeat(repeatKxx, self.input_dim, axis=0) * d2KMat
        d2KMat = mx.sym.broadcast_div(d2KMat, mx.sym.square(theta))

        line1 = mx.sym.concat(sclKxox, dKMat1, dim=1)
        line2 = mx.sym.concat(dKMat2, d2KMat, dim=1)
        oriKxox = mx.sym.concat(line1, line2, dim=0)

        Kxox = mx.sym.dot(mx.sym.dot(lTrans, oriKxox), rTrans.transpose(), name='Kxox')

        # self.Kxox = mx.sym.Group([oriKxox, blockMat1, blockMat2])
        if mode == 'train':
            self.Kxox = Kxox
        elif mode == 'other':
            return Kxox

    def symm_kern(self, theta, delta, Xs, Trans, num_X):
        """
        theta, delta, Xs, Trans: Mxnet Symbol
        num_X: int
        """

        # scalar Kxx
        sclKxx = kern_gaussian_symm_symbol(Xs, delta, theta, num_X)
        # sclKxx = sclKxx + mx.sym.broadcast_mul(
        # mat_eye_symbol(num_X), mx.sym.reshape(noise, shape=(1, 1)))
        tmpMat1 = mx.sym.repeat(mx.sym.reshape(Xs, shape=(1, self.input_dim*num_X)), num_X, axis=0)
        tmpMat2 = mx.sym.concat(*[Xs]*num_X, dim=1)
        # block matrix element (ij) = Xi - Xj
        blockMat = tmpMat2 - tmpMat1
        dKMat = mx.sym.broadcast_div(blockMat, mx.sym.square(theta))
        repeatKxx = mx.sym.repeat(sclKxx, self.input_dim, axis=1)
        dKMat = repeatKxx * dKMat
        tmpMat3 = mx.sym.repeat(blockMat, self.input_dim, axis=0)
        d2KMat = mx.sym.broadcast_div(tmpMat3.transpose()*tmpMat3, mx.sym.square(theta))
        d2KMat = d2KMat + mx.sym.tile(mat_eye_symbol(self.input_dim), reps=(num_X, num_X))
        d2KMat = mx.sym.repeat(repeatKxx, self.input_dim, axis=0) * d2KMat
        d2KMat = mx.sym.broadcast_div(d2KMat, mx.sym.square(theta))

        line1 = mx.sym.concat(sclKxx, dKMat, dim=1)
        line2 = mx.sym.concat(dKMat.transpose(), d2KMat, dim=1)
        oriKxx = mx.sym.concat(line1, line2, dim=0)

        Kxx = mx.sym.dot(mx.sym.dot(Trans, oriKxx), mx.sym.swapaxes(Trans,0,1))
        # Kxx = (Kxx.transpose() + Kxx)/2

        # self.Kxx = mx.sym.Group([Kxx, oriKxx])

        return Kxx

    def gen_kern(self, theta, delta, lXs, rXs, lTrans, rTrans, num_X1, num_X2):
        """
        theta, delta, lXs, rXs, lTrans, rTrans: Mxnet symbol
        num_X1, num_X2: int
        """

        sclKx1x2 = kern_gaussian_gen_symbol(lXs, rXs, delta, theta)
        leftTmp1 = mx.sym.tile(lXs, reps=(1, num_X2))
        rightTmp1 = mx.sym.repeat(mx.sym.reshape(rXs, shape=(1, self.input_dim*num_X2)), num_X1, axis=0)
        blockMat1 = leftTmp1 - rightTmp1
        dKMat1 = mx.sym.broadcast_div(blockMat1, mx.sym.square(theta))
        repeatKx1x2 = mx.sym.repeat(sclKx1x2, self.input_dim, axis=1)
        dKMat1 = repeatKx1x2 * dKMat1

        rightTmp2 = mx.sym.tile(rXs, reps=(1, num_X1))
        leftTmp2 = mx.sym.repeat(mx.sym.reshape(lXs, shape=(1, self.input_dim*num_X1)), num_X2, axis=0)
        blockMat2 = rightTmp2 - leftTmp2
        blockMat2 = blockMat2.transpose()
        dKMat2 = mx.sym.broadcast_div(blockMat2, mx.sym.square(theta))
        dKMat2 = mx.sym.repeat(sclKx1x2, self.input_dim, axis=0) * dKMat2

        d2KMat = mx.sym.repeat(blockMat1, self.input_dim, axis=0) * mx.sym.repeat(blockMat2, self.input_dim, axis=1)
        d2KMat = mx.sym.broadcast_div(d2KMat, mx.sym.square(theta))
        d2KMat = d2KMat + mx.sym.tile(mat_eye_symbol(self.input_dim), reps=(num_X1, num_X2))
        d2KMat = mx.sym.repeat(repeatKx1x2, self.input_dim, axis=0) * d2KMat
        d2KMat = mx.sym.broadcast_div(d2KMat, mx.sym.square(theta))

        line1 = mx.sym.concat(sclKx1x2, dKMat1, dim=1)
        line2 = mx.sym.concat(dKMat2, d2KMat, dim=1)
        oriKx1x2 = mx.sym.concat(line1, line2, dim=0)

        Kx1x2 = mx.sym.dot(mx.sym.dot(lTrans, oriKx1x2), rTrans.transpose(), name='Kx1x2')

        return Kx1x2

    def prepare_data(self, allData, mode='predict', virial=True):
        numStruct = len(allData)
        lens = [len(data[0]) for data in allData]
        allNum = sum(lens)
        newData = []

        for i, data in enumerate(allData):
            if mode == 'predict':
                eles, enFps, enMat, fMat, vMat = [mx.nd.array(term) for term in data]
                newData.append((eles, enFps, enMat, fMat, vMat))
            elif mode == 'test' or mode == 'train':
                eles, enFps, enMat, fMat, vMat, energy, forces, stress = [mx.nd.array(term) for term in data]
                energy = mx.nd.expand_dims(energy, axis=0)
                newData.append((eles, enFps, enMat, fMat, vMat, energy, forces, stress))


        # input elements
        allEles = mx.nd.concat(*[data[0] for data in newData], dim=0)

        # input vectors
        allInput = mx.nd.concat(*[data[1] for data in newData], dim=0)
        # transform mat
        mats = [[None for i in range(numStruct)] for j in range(numStruct)]
        for i in range(numStruct):
            for j in range(numStruct):
                if i == j:
                    mats[i][j] = newData[i][2]
                else:
                    mats[i][j] = mx.nd.zeros((1, lens[j]))
        enTrans = mx.nd.concat(*[mx.nd.concat(*mats[i], dim=1) for i in range(numStruct)], dim=0)

        for i in range(numStruct):
            for j in range(numStruct):
                if i == j:
                    mats[i][j] = newData[i][3]
                else:
                    mats[i][j] = mx.nd.zeros((3*lens[i], self.input_dim*lens[j]))
        fTrans = mx.nd.concat(*[mx.nd.concat(*mats[i], dim=1) for i in range(numStruct)], dim=0)

        # logging.debug("vMat: {}".format(vMat.shape))
        for i in range(numStruct):
            for j in range(numStruct):
                if i == j:
                    mats[i][j] = newData[i][4]
                else:
                    mats[i][j] = mx.nd.zeros((6, self.input_dim*lens[j]))
        vTrans = mx.nd.concat(*[mx.nd.concat(*mats[i], dim=1) for i in range(numStruct)], dim=0)
        # logging.debug("vTrans: {}".format(vTrans.shape))

        f_and_vTrans = mx.nd.concat(fTrans, vTrans, dim=0)

        line1 = mx.nd.concat(enTrans, mx.nd.zeros((numStruct, self.input_dim*allNum)), dim=1)
        if virial:
            line2 = mx.nd.concat(mx.nd.zeros((3*allNum + 6*numStruct, allNum)), f_and_vTrans, dim=1)
        else:
            line2 = mx.nd.concat(mx.nd.zeros((3*allNum, allNum)), fTrans, dim=1)
        # logging.debug("line1: {}".format(line1.shape))
        # logging.debug("line2: {}".format(line2.shape))
        allTrans = mx.nd.concat(line1, line2, dim=0)

        # original target vector
        if mode == 'test' or mode == 'train':
            enTarget = mx.nd.concat(*[data[5] for data in newData], dim=0)
            fTarget = mx.nd.concat(*[data[6] for data in newData], dim=0)
            vTarget = mx.nd.concat(*[data[7] for data in newData], dim=0)

            # vDiag = mx.nd.concat(*[data[6][:3] for data in newData], dim=0)
            # vNonDiag = mx.nd.concat(*[data[6][3:] for data in newData], dim=0)

        if mode == 'train':
            # mean
            self.meanE = enTarget.sum()/allNum
            self.meanF = fTarget.mean()
            self.meanV = vTarget.mean()
            # self.meanVdiag = vDiag.mean()
            # self.meanVnonDiag = vNonDiag.mean()

            # vTarget = vTarget - self.meanV
            # tmpV = mx.nd.concat(vDiag.reshape((-1, 3)), vNonDiag.reshape((-1, 3)), dim=1)
            # tmpV = tmpV.reshape((-1, 1))
            # logging.debug("vTarget: {}, tmpV: {}".format(vTarget.shape, tmpV.shape))
            # logging.debug(vTarget + self.meanV - tmpV)

        if mode == 'test' or mode == 'train':
            if virial:
                allTarget = mx.nd.concat(enTarget, fTarget, vTarget, dim=0)
            else:
                allTarget = mx.nd.concat(enTarget, fTarget, dim=0)

        mx.nd.waitall()

        if mode == 'predict':
            return allEles, allInput, allTrans, lens
        elif mode == 'test' or mode == 'train':
            return allEles, allInput, allTrans, allTarget, lens

    def train(self, trainData, noise=[0.1, 0.1, 0.1], lr=0.01, opt=True, fminOptions={}, train_virial=True, noise_bound=[0.001, 0.01, 0.001], sclNoi=0.1, noiWid=1e-19):

        # start = time.time()
        # trainInput, trTrans, targets = [mx.nd.array(mat) for mat in  prepare_data(trainData, self.input_dim)]
        trainEles, trainInput, trTrans, targets, lens = self.prepare_data(trainData, mode='train', virial=train_virial)

        lens = mx.nd.array([lens])
        forceNum = lens.sum() * 3
        forceNum = int(forceNum.asscalar())
        virialNum = lens.shape[1] * 6

        # logging.debug(forceNum)

        # logging.debug("train-prepare: {}".format(time.time() - start))
        trainNum, _ = trainInput.shape
        numTarget, _ = targets.shape
        self.trainEles = trainEles
        self.trainNum, self.trainInput, self.trTrans = trainNum, trainInput, trTrans
        self.compute_Kxx(trainNum)
        self.noise = noise


        symTar = mx.sym.var('targets')
        symNoi = mx.sym.var('noise')
        symLens = mx.sym.var('lens')
        self.params['noise'] = mx.nd.array([noise])
        self.lower_bound['noise'] = mx.nd.array([noise_bound])
        self.sclNoi = sclNoi
        self.noiWid = noiWid

        # expand noise
        # expandNoi = mx.sym.tile(symNoi, reps=(1, numTarget))
        enNoi = mx.sym.broadcast_mul(symLens, mx.sym.slice(symNoi, begin=[0,0], end=[1,1]))
        enNoi = mx.sym.reshape(enNoi, shape=(1, -1))
        fNoi = mx.sym.tile(mx.sym.slice(symNoi, begin=[0,1], end=[1,2]), reps=(1, forceNum))
        vNoi = mx.sym.tile(mx.sym.slice(symNoi, begin=[0,2], end=[1,3]), reps=(1, virialNum))
        expandNoi = mx.sym.concat(enNoi, fNoi, vNoi, dim=1)

        # args = {
        #     'eles': trainEles,
        #     'Xs': trainInput,
        #     'Trans': trTrans,
        #     'targets': targets,
        #     'noise': self.params['noise'],
        #     'theta': self.params['theta'],
        #     'delta': self.delta,
        #     'lens': lens,
        # }
        # noiEx = expandNoi.bind(ctx=mx.cpu(), args=args)
        # noiEx.forward()
        # mx.nd.save('noise', noiEx.outputs[0])

        optRes = {}
        step = fminOptions['maxiter']
        if opt:
            log_N = gpregr_criterion_symbol(self.Kxx, symTar, expandNoi, numTarget)
            # log_N = gpregr_criterion_symbol(self.Kxx, symTar, symNoi, numTarget)
            self.log_N = log_N
            allGrad = copy.deepcopy(self.grad_args)

            for i in range(step):
                args = {
                    'eles': trainEles,
                    'Xs': trainInput,
                    'Trans': trTrans,
                    'targets': targets,
                    'noise': self.params['noise'],
                    'theta': self.params['theta'],
                    'delta': self.delta,
                    'lens': lens,
                    'sclNoi': mx.nd.array([self.sclNoi]),
                    # 'noiWid': mx.nd.array([[self.noiWid]]),
                }

                trainEx = log_N.bind(ctx=mx.cpu(), args=args, args_grad=allGrad)
                trainEx.forward()
                headGrad = [mx.nd.ones((1))]
                trainEx.backward(headGrad)
                logging.debug("theta: {}".format(self.params['theta'].asnumpy().tolist()[0]))
                logging.debug("noise: {}".format(self.params['noise'].asnumpy().tolist()[0]))
                logging.debug("step: {}, loss: {}".format(i, trainEx.outputs[0].asnumpy()))
                self.adam(allGrad, lr, i+1)


        optRes = copy.copy(self.params)
        self.optRes = optRes

        infer_post = gpregr_prediction_symbol(self.Kxx, symTar, expandNoi, numTarget)
        # infer_post = gpregr_prediction_symbol(self.Kxx, symTar, symNoi, numTarget)
        args = {
            'eles': trainEles,
            'Xs': trainInput,
            'Trans': trTrans,
            'targets': targets,
            'noise': self.params['noise'],
            'theta': self.params['theta'],
            'delta': self.delta,
            'lens': lens,
            'sclNoi': mx.nd.array([self.sclNoi]),
            # 'noiWid': mx.nd.array([[self.noiWid]]),
        }

        logging.debug("Shape of Args:")
        for key, val in args.items():
            logging.debug("{}:  {}".format(key, val.shape))

        postEx = infer_post.bind(ctx=mx.cpu(), args=args)
        postEx.forward()
        self.L, self.wv = postEx.outputs

        mx.nd.waitall()

        # mx.nd.save('train_old', [self.trainInput, self.trTrans, self.L, self.wv])

        # logging.debug('train')
        # logging.debug(self.wv)

    def compute_fps(self, dataset, afterTrain=False):
        """

        """
        setNeigh = DataSetNeighbors(self.cutoff, self.oriNumFps, dataset)
        setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, fpType, cutoff = setNeigh.prepare_input()
        lens = setNeigh.prepare_lens()
        if afterTrain:
            etas = self.etas
            fpType = self.fpType
        else:
            etas = setNeigh.prepare_etas(self.oriEtas)

        setInput, setTrans = descriptor_nd(etas, setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, cutoff, fpType)

        # symbolic
        # symInput, symTrans, fpArgs = descriptor(etas, setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, cutoff, fpType)
        # logging.debug("Shape of FpArgs:")
        # for key, val in fpArgs.items():
        #     logging.debug("{}:  {}".format(key, val.shape))
        # fpSyms = mx.sym.Group([symInput, symTrans])
        # fpEx = fpSyms.bind(ctx=mx.cpu(), args=fpArgs)
        # fpEx.forward()
        # setInput2, setTrans2 = fpEx.outputs

        # logging.debug(setInput == setInput2)
        # logging.debug(setTrans == setTrans2)

        # mx.nd.save('setTrans1.mx', setTrans)
        # mx.nd.save('setTrans2.mx', setTrans2)

        # setInput, setTrans = setInput2, setTrans2

        return setType, setInput, setTrans, lens


    def read_fpsetup(self, fpsetup):
        self.cutoff = fpsetup['Rc']
        self.oriNumFps = len(fpsetup['sf2']['eta'])
        self.oriEtas = fpsetup['sf2']['eta']

    def train_symbol(self, trainData, noise=[0.1, 0.1, 0.1], lr=0.01, opt=False, fminOptions={}, train_virial=True, noise_bound=[0.001, 0.01, 0.001], sclNoi=0.1, noiWid=1e-19):

        # start = time.time()
        # trainInput, trTrans, targets = [mx.nd.array(mat) for mat in  prepare_data(trainData, self.input_dim)]
        # trainEles, trainInput, trTrans, targets, lens = self.prepare_data(trainData, mode='train', virial=train_virial)
        setNeigh = DataSetNeighbors(self.cutoff, self.oriNumFps, trainData)
        # logging.debug(time.time())
        setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, fpType, cutoff = setNeigh.prepare_input()
        # logging.debug(time.time())
        etas = setNeigh.prepare_etas(self.oriEtas)
        lens = setNeigh.prepare_lens()
        symInput, symTrans, fpArgs = descriptor(etas, setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, cutoff, fpType)

        # test descriptor
        # testEx = mx.sym.Group([symInput, symTrans]).bind(ctx=mx.cpu(), args=fpArgs)
        # testEx.forward()
        # mx.nd.waitall()
        # logging.debug('test descriptor finish')

        targets = prepare_targets(trainData)


        numAtoms, numNeigh, _ = setIndices.shape
        numStruct, _ = setAtomsInd.shape
        forceNum = 3 * numAtoms
        virialNum = 6 * numStruct

        # logging.debug(forceNum)

        # logging.debug("train-prepare: {}".format(time.time() - start))
        trainNum = numAtoms
        numTarget = 3*numAtoms + 7*numStruct
        self.trainEles = setType
        self.etas = etas
        self.fpType = fpType
        self.trainNum = trainNum
        self.compute_Kxx_symbol(trainNum, symInput, symTrans,)
        self.noise = noise


        symTar = mx.sym.var('targets')
        symNoi = mx.sym.var('noise')
        symLens = mx.sym.var('lens')
        self.params['noise'] = mx.nd.array([noise])
        self.lower_bound['noise'] = mx.nd.array([noise_bound])
        self.sclNoi = sclNoi
        self.noiWid = noiWid

        # etas as parameters
        self.params['Etas'] = etas
        self.grad_args['Etas'] = mx.nd.zeros_like(etas)
        self.vs['Etas'] = mx.nd.zeros_like(etas)
        self.sqrs['Etas'] = mx.nd.zeros_like(etas)
        self.lower_bound['Etas'] = -10*mx.nd.ones_like(etas)


        # gamma, beta as parameter
        # ndGamma = fpArgs['Gamma'].copy()
        # self.params['Gamma'] = ndGamma
        # self.grad_args['Gamma'] = mx.nd.zeros_like(ndGamma)
        # self.vs['Gamma'] = mx.nd.zeros_like(ndGamma)
        # self.sqrs['Gamma'] = mx.nd.zeros_like(ndGamma)
        # self.lower_bound['Gamma'] = mx.nd.zeros_like(ndGamma)

        # ndBeta = fpArgs['Beta'].copy()
        # self.params['Beta'] = ndBeta
        # self.grad_args['Beta'] = mx.nd.zeros_like(ndBeta)
        # self.vs['Beta'] = mx.nd.zeros_like(ndBeta)
        # self.sqrs['Beta'] = mx.nd.zeros_like(ndBeta)
        # self.lower_bound['Beta'] = mx.nd.zeros_like(ndBeta)


        # expand noise
        # expandNoi = mx.sym.tile(symNoi, reps=(1, numTarget))
        enNoi = mx.sym.broadcast_mul(symLens, mx.sym.slice(symNoi, begin=[0,0], end=[1,1]))
        enNoi = mx.sym.reshape(enNoi, shape=(1, -1))
        fNoi = mx.sym.tile(mx.sym.slice(symNoi, begin=[0,1], end=[1,2]), reps=(1, forceNum))
        vNoi = mx.sym.tile(mx.sym.slice(symNoi, begin=[0,2], end=[1,3]), reps=(1, virialNum))
        expandNoi = mx.sym.concat(enNoi, fNoi, vNoi, dim=1)

        # args = {
        #     'targets': targets,
        #     'noise': self.params['noise'],
        #     'theta': self.params['theta'],
        #     'delta': self.delta,
        #     'lens': lens,
        # }
        # noiEx = expandNoi.bind(ctx=mx.cpu(), args=args)
        # noiEx.forward()
        # mx.nd.save('noise', noiEx.outputs[0])

        optRes = {}
        step = fminOptions['maxiter']
        if opt:
            log_N = gpregr_criterion_symbol(self.Kxx, symTar, expandNoi, numTarget)
            # log_N = gpregr_criterion_symbol(self.Kxx, symTar, symNoi, numTarget)
            self.log_N = log_N
            allGrad = copy.deepcopy(self.grad_args)

            for i in range(step):
                args = {
                    'eles': self.trainEles,
                    'targets': targets,
                    'noise': self.params['noise'],
                    'theta': self.params['theta'],
                    'delta': self.delta,
                    'lens': lens,
                    'sclNoi': mx.nd.array([self.sclNoi]),
                    # 'noiWid': mx.nd.array([[self.noiWid]]),
                }
                args.update(fpArgs)
                # args['Gamma'] = self.params['Gamma']
                # args['Beta'] = self.params['Beta']
                args['Etas'] = self.params['Etas']

                # logging.debug("Args:")
                # logging.debug("Etas: {}".format(args['Etas'].asnumpy().tolist()))

                trainEx = log_N.bind(ctx=mx.cpu(), args=args, args_grad=allGrad)
                trainEx.forward()
                headGrad = [mx.nd.ones((1))]
                trainEx.backward(headGrad)
                logging.debug("step: {}, loss: {}".format(i, trainEx.outputs[0].asnumpy()))
                logging.debug("theta: {}".format(args['theta'].asnumpy().tolist()[0]))
                logging.debug("noise: {}".format(args['noise'].asnumpy().tolist()[0]))
                logging.debug("Etas: {}".format(args['Etas'].asnumpy().tolist()))
                # logging.debug("Gamma: {}".format(args['Gamma'].asnumpy().tolist()))
                # logging.debug("Beta: {}".format(args['Beta'].asnumpy().tolist()))
                logging.debug("Grad theta: {}".format(allGrad['theta'].asnumpy().tolist()))
                logging.debug("Grad Etas: {}".format(allGrad['Etas'].asnumpy().tolist()))
                # logging.debug("Grad Gamma: {}".format(allGrad['Gamma'].asnumpy().tolist()))
                # logging.debug("Grad Beta: {}".format(allGrad['Beta'].asnumpy().tolist()))
                self.adam(allGrad, lr, i+1)


        optRes = copy.copy(self.params)
        self.optRes = optRes

        infer_post = gpregr_prediction_symbol(self.Kxx, symTar, expandNoi, numTarget)
        # infer_post = gpregr_prediction_symbol(self.Kxx, symTar, symNoi, numTarget)
        args = {
            'eles': self.trainEles,
            'targets': targets,
            'noise': self.params['noise'],
            'theta': self.params['theta'],
            'delta': self.delta,
            'lens': lens,
            'sclNoi': mx.nd.array([self.sclNoi]),
            # 'noiWid': mx.nd.array([[self.noiWid]]),
        }
        args.update(fpArgs)
        logging.debug("Args:")
        for key, val in args.items():
            logging.debug("{}:  {}".format(key, val.shape))
        postEx = infer_post.bind(ctx=mx.cpu(), args=args)
        postEx.forward()
        self.L, self.wv = postEx.outputs

        # compute fingerprints for prediction
        fpSyms = mx.sym.Group([symInput, symTrans])
        fpEx = fpSyms.bind(ctx=mx.cpu(), args=fpArgs)
        fpEx.forward()
        self.trainInput, self.trTrans = fpEx.outputs


        mx.nd.waitall()

        # mx.nd.save('train_sym', [self.trainInput, self.trTrans, self.L, self.wv])
        # logging.debug('train_symbol')
        # logging.debug(self.wv)

    def batch_train_symbol(self, trainData, noise=[0.1, 0.1, 0.1], lr=0.01, fminOptions={}, train_virial=True, noise_bound=[0.001, 0.01, 0.001], sclNoi=0.1, noiWid=1e-19, batchSize=10, numCycle=2):
        batchStep = int(math.floor(len(trainData)/batchSize))

        # etas as parameters
        # setNeigh = DataSetNeighbors(self.cutoff, self.oriNumFps, trainData[:1])
        etas = mx.nd.array(self.oriEtas)
        self.params['Etas'] = etas
        self.grad_args['Etas'] = mx.nd.zeros_like(etas)
        self.vs['Etas'] = mx.nd.zeros_like(etas)
        self.sqrs['Etas'] = mx.nd.zeros_like(etas)
        self.lower_bound['Etas'] = -10*mx.nd.ones_like(etas)

        # optRes = {}
        for cycle in range(numCycle):
            random.shuffle(trainData)
            logging.debug("Cycle {}".format(cycle))
            for step in range(batchStep):
                tmpData = trainData[step*batchSize:(step+1)*batchSize]
                setNeigh = DataSetNeighbors(self.cutoff, self.oriNumFps, tmpData)
                setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, fpType, cutoff = setNeigh.prepare_input()
                etas = setNeigh.prepare_etas(self.oriEtas)
                lens = setNeigh.prepare_lens()
                symInput, symTrans, fpArgs = descriptor(etas, setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, cutoff, fpType)


                targets = prepare_targets(tmpData)


                numAtoms, numNeigh, _ = setIndices.shape
                numStruct, _ = setAtomsInd.shape
                forceNum = 3 * numAtoms
                virialNum = 6 * numStruct

                trainNum = numAtoms
                numTarget = 3*numAtoms + 7*numStruct
                self.trainEles = setType
                # self.etas = etas
                self.fpType = fpType
                self.trainNum = trainNum
                self.compute_Kxx_symbol(trainNum, symInput, symTrans,)
                self.noise = noise


                symTar = mx.sym.var('targets')
                symNoi = mx.sym.var('noise')
                symLens = mx.sym.var('lens')
                self.params['noise'] = mx.nd.array([noise])
                self.lower_bound['noise'] = mx.nd.array([noise_bound])
                self.sclNoi = sclNoi
                self.noiWid = noiWid

                # expand noise
                # expandNoi = mx.sym.tile(symNoi, reps=(1, numTarget))
                enNoi = mx.sym.broadcast_mul(symLens, mx.sym.slice(symNoi, begin=[0,0], end=[1,1]))
                enNoi = mx.sym.reshape(enNoi, shape=(1, -1))
                fNoi = mx.sym.tile(mx.sym.slice(symNoi, begin=[0,1], end=[1,2]), reps=(1, forceNum))
                vNoi = mx.sym.tile(mx.sym.slice(symNoi, begin=[0,2], end=[1,3]), reps=(1, virialNum))
                expandNoi = mx.sym.concat(enNoi, fNoi, vNoi, dim=1)



                # step = fminOptions['maxiter']

                log_N = gpregr_criterion_symbol(self.Kxx, symTar, expandNoi, numTarget)
                # log_N = gpregr_criterion_symbol(self.Kxx, symTar, symNoi, numTarget)
                self.log_N = log_N
                allGrad = copy.deepcopy(self.grad_args)


                args = {
                    'eles': self.trainEles,
                    'targets': targets,
                    'noise': self.params['noise'],
                    'theta': self.params['theta'],
                    'delta': self.delta,
                    'lens': lens,
                    'sclNoi': mx.nd.array([self.sclNoi]),
                    # 'noiWid': mx.nd.array([[self.noiWid]]),
                }
                args.update(fpArgs)
                # args['Gamma'] = self.params['Gamma']
                # args['Beta'] = self.params['Beta']
                args['Etas'] = self.params['Etas']

                # logging.debug("Args:")
                # logging.debug("Etas: {}".format(args['Etas'].asnumpy().tolist()))

                trainEx = log_N.bind(ctx=mx.cpu(), args=args, args_grad=allGrad)
                trainEx.forward()
                headGrad = [mx.nd.ones((1))]
                trainEx.backward(headGrad)
                logging.debug("step: {}, loss: {}".format(step, trainEx.outputs[0].asnumpy()))
                logging.debug("theta: {}".format(args['theta'].asnumpy().tolist()[0]))
                logging.debug("noise: {}".format(args['noise'].asnumpy().tolist()[0]))
                logging.debug("Etas: {}".format(args['Etas'].asnumpy().tolist()))
                # logging.debug("Gamma: {}".format(args['Gamma'].asnumpy().tolist()))
                # logging.debug("Beta: {}".format(args['Beta'].asnumpy().tolist()))
                logging.debug("Grad theta: {}".format(allGrad['theta'].asnumpy().tolist()))
                logging.debug("Grad Etas: {}".format(allGrad['Etas'].asnumpy().tolist()))
                # logging.debug("Grad Gamma: {}".format(allGrad['Gamma'].asnumpy().tolist()))
                # logging.debug("Grad Beta: {}".format(allGrad['Beta'].asnumpy().tolist()))
                self.adam(allGrad, lr, step+1)

        optRes = copy.copy(self.params)
        self.optRes = optRes


        pass

    def train_precompute(self, trainData, noise=[0.1, 0.1, 0.1], lr=0.01, opt=False, fminOptions={}, train_virial=True, noise_bound=[0.001, 0.01, 0.001], sclNoi=0.1, noiWid=1e-19):
        """
        Compute fingerprints before training
        """

        logging.debug('precompute fps start')
        trainEles, trainInput, trTrans, lens = self.compute_fps(trainData)
        mx.nd.waitall()
        logging.debug('precompute fps finish')
        targets = prepare_targets(trainData)

        # mx.nd.save('targets.mx', targets)

        forceNum = lens.sum() * 3
        forceNum = int(forceNum.asscalar())
        virialNum = lens.shape[1] * 6
        trainNum, _ = trainInput.shape
        numTarget, _ = targets.shape

        # logging.debug(forceNum)

        # logging.debug("train-prepare: {}".format(time.time() - start))
        # trainNum, _ = trainInput.shape
        # numTarget, _ = targets.shape
        self.trainEles = trainEles
        self.trainNum, self.trainInput, self.trTrans = trainNum, trainInput, trTrans
        self.compute_Kxx(trainNum)
        self.noise = noise


        symTar = mx.sym.var('targets')
        symNoi = mx.sym.var('noise')
        symLens = mx.sym.var('lens')
        self.params['noise'] = mx.nd.array([noise])
        self.lower_bound['noise'] = mx.nd.array([noise_bound])
        self.sclNoi = sclNoi
        self.noiWid = noiWid

        # expand noise
        # expandNoi = mx.sym.tile(symNoi, reps=(1, numTarget))
        enNoi = mx.sym.broadcast_mul(symLens, mx.sym.slice(symNoi, begin=[0,0], end=[1,1]))
        enNoi = mx.sym.reshape(enNoi, shape=(1, -1))
        fNoi = mx.sym.tile(mx.sym.slice(symNoi, begin=[0,1], end=[1,2]), reps=(1, forceNum))
        vNoi = mx.sym.tile(mx.sym.slice(symNoi, begin=[0,2], end=[1,3]), reps=(1, virialNum))
        expandNoi = mx.sym.concat(enNoi, fNoi, vNoi, dim=1)

        # args = {
        #     'eles': trainEles,
        #     'Xs': trainInput,
        #     'Trans': trTrans,
        #     'targets': targets,
        #     'noise': self.params['noise'],
        #     'theta': self.params['theta'],
        #     'delta': self.delta,
        #     'lens': lens,
        # }
        # noiEx = expandNoi.bind(ctx=mx.cpu(), args=args)
        # noiEx.forward()
        # mx.nd.save('noise', noiEx.outputs[0])

        optRes = {}
        step = fminOptions['maxiter']
        if opt:
            log_N = gpregr_criterion_symbol(self.Kxx, symTar, expandNoi, numTarget)
            # log_N = gpregr_criterion_symbol(self.Kxx, symTar, symNoi, numTarget)
            self.log_N = log_N
            allGrad = copy.deepcopy(self.grad_args)

            for i in range(step):
                args = {
                    'eles': trainEles,
                    'Xs': trainInput,
                    'Trans': trTrans,
                    'targets': targets,
                    'noise': self.params['noise'],
                    'theta': self.params['theta'],
                    'delta': self.delta,
                    'lens': lens,
                    'sclNoi': mx.nd.array([self.sclNoi]),
                    # 'noiWid': mx.nd.array([[self.noiWid]]),
                }

                trainEx = log_N.bind(ctx=mx.cpu(), args=args, args_grad=allGrad)
                trainEx.forward()
                headGrad = [mx.nd.ones((1))]
                trainEx.backward(headGrad)
                logging.debug("theta: {}".format(self.params['theta'].asnumpy().tolist()[0]))
                logging.debug("noise: {}".format(self.params['noise'].asnumpy().tolist()[0]))
                logging.debug("step: {}, loss: {}".format(i, trainEx.outputs[0].asnumpy()))
                self.adam(allGrad, lr, i+1)


        optRes = copy.copy(self.params)
        self.optRes = optRes

        infer_post = gpregr_prediction_symbol(self.Kxx, symTar, expandNoi, numTarget)
        # infer_post = gpregr_prediction_symbol(self.Kxx, symTar, symNoi, numTarget)
        args = {
            'eles': trainEles,
            'Xs': trainInput,
            'Trans': trTrans,
            'targets': targets,
            'noise': self.params['noise'],
            'theta': self.params['theta'],
            'delta': self.delta,
            'lens': lens,
            'sclNoi': mx.nd.array([self.sclNoi]),
            # 'noiWid': mx.nd.array([[self.noiWid]]),
        }

        logging.debug("Shape of Args:")
        for key, val in args.items():
            logging.debug("{}:  {}".format(key, val.shape))

        postEx = infer_post.bind(ctx=mx.cpu(), args=args)
        postEx.forward()
        self.L, self.wv = postEx.outputs

        mx.nd.waitall()


    def predict(self, preEles, preInput, preTrans, computeVar=False):

        start = time.time()
        preNum, _ = preInput.shape
        numTarget, _ = preTrans.shape
        self.compute_Kxox(preNum, self.trainNum)
        args = {
            'lEles': preEles,
            'lXs': preInput,
            'lTrans': preTrans,
            'rEles': self.trainEles,
            'rXs': self.trainInput,
            'rTrans': self.trTrans,
            'theta': self.params['theta'],
            'delta': self.delta,
            # 'sclNoi': mx.nd.array([self.sclNoi]),
            # 'noiWid': mx.nd.array([[self.noiWid]]),
        }
        ktxEx = self.Kxox.bind(ctx=mx.cpu(), args=args)
        ktxEx.forward()
        Ktx = ktxEx.outputs[0]


        mx.nd.waitall()
        # logging.debug("Ktx {}".format(time.time()-start))

        # start = time.time()

        # logging.debug("Ktt {}".format(time.time()-start))

        self.Ktx = Ktx
        mu = mx.nd.linalg_gemm2(Ktx, self.wv)
        var = None


        if computeVar:
            symKtt = self.compute_Kxx(preNum, mode='predict')
            args = {
                'eles': preEles,
                'Xs': preInput,
                'Trans': preTrans,
                'theta': self.params['theta'],
                'delta': self.delta,
                'sclNoi': mx.nd.array([self.sclNoi]),
            }


            kttEx = symKtt.bind(ctx=mx.cpu(), args=args)
            kttEx.forward()
            Ktt = kttEx.outputs[0]
            tmp = mx.nd.linalg_trsm(A=self.L, B=Ktx.transpose(), transpose=False, rightside=False)
            var = extract_diag_nd(Ktt - mx.nd.sum(mx.nd.square(tmp), axis=0))

            # logging.debug("Ktt: {}".format(Ktt.shape))
            # logging.debug("self.L: {}".format(self.L.shape))
            # logging.debug("Ktx: {}".format(Ktx.shape))

        return mu, var

    def test(self, testData):
        numStruct = len(testData)

        # testInput, testTrans, testTar = [mx.nd.array(mat) for mat in prepare_data(testData, self.input_dim)]
        # testEles, testInput, testTrans, testTar, lens = self.prepare_data(testData, mode='test', virial=True)

        start = time.time()
        allPreE = []
        allPreF = []
        allPreV = []
        alltestE = []
        alltestF = []
        alltestV = []
        for oneData in testData:
            testEles, testInput, testTrans, testTar, lens = self.prepare_data([oneData], mode='test', virial=True)
            # mx.nd.save("test_old",[testEles, testInput, testTrans, mx.nd.array([lens]), testTar])
            preVal = self.predict(testEles, testInput, testTrans)
            # mx.nd.save('Ktx_old', self.Ktx)
            preE = preVal[:1]/lens[0]
            preF = preVal[1:1+3*len(testEles)]
            preV = preVal[-6:]
            testE = testTar[:1]/lens[0]
            testF = testTar[1:1+3*len(testEles)]
            testV = testTar[-6:]
            allPreE.append(preE)
            allPreF.append(preF)
            allPreV.append(preV)
            alltestE.append(testE)
            alltestF.append(testF)
            alltestV.append(testV)
            # logging.debug("predict V")
            # logging.debug(preV)
            # logging.debug("test V")
            # logging.debug(testV)

        allPreE = mx.nd.concat(*allPreE, dim=0)
        allPreF = mx.nd.concat(*allPreF, dim=0)
        allPreV = mx.nd.concat(*allPreV, dim=0)
        alltestE = mx.nd.concat(*alltestE, dim=0)
        alltestF = mx.nd.concat(*alltestF, dim=0)
        alltestV = mx.nd.concat(*alltestV, dim=0)

        self.allVal = [allPreE, allPreF, allPreV, alltestE, alltestF, alltestV]
        # mx.nd.save("allVal", [allPreE, allPreF, allPreV, alltestE, alltestF, alltestV])
        mx.nd.waitall()
        logging.debug("self.predict: {}".format(time.time() - start))

        # preVal[-6*numStruct:] = preVal[-6*numStruct:] + self.meanV
        # mx.nd.save('predictVal', preVal)
        # mx.nd.save('testVal', testTar)
        # preEn = preVal[:numStruct]
        # preForce = preVal[numStruct:]
        # realEn = testTar[:numStruct]
        # realForce = testTar[numStruct:]

    # @profile
    def test_symbol(self, testData):

        # testInput, testTrans, testTar = [mx.nd.array(mat) for mat in prepare_data(testData, self.input_dim)]
        # testEles, testInput, testTrans, testTar, lens = self.prepare_data(testData, mode='test', virial=True)

        start = time.time()
        allPreE = []
        allPreF = []
        allPreV = []
        alltestE = []
        alltestF = []
        alltestV = []
        for oneData in testData:
            testEles, testInput, testTrans, lens = self.compute_fps([oneData])
            testTar = prepare_targets([oneData])

            # mx.nd.save("test_sym",[testEles, testInput, testTrans, lens, testTar])

            preVal = self.predict(testEles, testInput, testTrans)
            # mx.nd.save('Ktx_sym', self.Ktx)
            preE = preVal[:1]/lens[0]
            preF = preVal[1:1+3*len(testEles)]
            preV = preVal[-6:]
            testE = testTar[:1]/lens[0]
            testF = testTar[1:1+3*len(testEles)]
            testV = testTar[-6:]
            allPreE.append(preE)
            allPreF.append(preF)
            allPreV.append(preV)
            alltestE.append(testE)
            alltestF.append(testF)
            alltestV.append(testV)
            # logging.debug("predict V")
            # logging.debug(preV)
            # logging.debug("test V")
            # logging.debug(testV)

        allPreE = mx.nd.concat(*allPreE, dim=0)
        allPreF = mx.nd.concat(*allPreF, dim=0)
        allPreV = mx.nd.concat(*allPreV, dim=0)
        alltestE = mx.nd.concat(*alltestE, dim=0)
        alltestF = mx.nd.concat(*alltestF, dim=0)
        alltestV = mx.nd.concat(*alltestV, dim=0)

        mx.nd.save("allVal", [allPreE, allPreF, allPreV, alltestE, alltestF, alltestV])
        logging.debug("self.predict: {}".format(time.time() - start))

        # preVal[-6*numStruct:] = preVal[-6*numStruct:] + self.meanV
        # mx.nd.save('predictVal', preVal)
        # mx.nd.save('testVal', testTar)
        # preEn = preVal[:numStruct]
        # preForce = preVal[numStruct:]
        # realEn = testTar[:numStruct]
        # realForce = testTar[numStruct:]

    # def train_VFE(self, trainData, noise=0.1, lr=0.01, opt=True, fminOptions={}):

    #     trainInput, trTrans, targets = self.prepare_data(trainData, mode='train')

    #     # logging.debug("train-prepare: {}".format(time.time() - start))
    #     trainNum, _ = trainInput.shape
    #     numTarget, _ = targets.shape
    #     self.trainNum, self.trainInput, self.trTrans = trainNum, trainInput, trTrans

    #     incData = random.sample(trainData, self.numInc)
    #     incInput, incTrans, incTar = self.prepare_data(incData)
    #     incNum, _ = incInput.shape
    #     incTarNum, _ = incTar.shape
    #     self.incNum, self.incInput, self.incTrans = incNum, incInput, incTrans


    #     symTheta = mx.sym.var('theta')
    #     symDelta = mx.sym.var('delta')
    #     Xs = mx.sym.var('Xs')
    #     xTrans = mx.sym.var('xTrans')
    #     Kxx = self.symm_kern(symTheta, symDelta, Xs, xTrans, trainNum)

    #     Us = mx.sym.var('Us')
    #     uTrans = mx.sym.var('uTrans')
    #     Kuu = self.symm_kern(symTheta, symDelta, Us, uTrans, incNum)

    #     Kux = self.gen_kern(symTheta, symDelta, Us, Xs, uTrans, xTrans, incNum, trainNum)




    #     # indices = random.sample(range(trainNum), self.numInc)
    #     # incPoints = [mx.nd.expand_dims(trainInput[i], axis=0) for i in indices]
    #     # incPoints = mx.nd.concat(*incPoints, dim=0)
    #     # self.params['inc_points'] = incPoints.copy()
    #     # Kuu = self.compute_Kxx(incNum, mode='predict')
    #     # Kux = self.compute_Kxox(incNum, trainNum, mode='other')
    #     self.noise = noise

    #     symTar = mx.sym.var('targets')
    #     symNoi = mx.sym.var('noise')
    #     self.params['noise'] = mx.nd.array([[noise]])

    #     optRes = {}
    #     step = fminOptions['maxiter']
    #     if opt:
    #         log_N = criterion_symbol_VFE(Kuu, Kux, Kxx, symTar, symNoi, incTarNum, numTarget)
    #         self.log_N = log_N
    #         allGrad = copy.deepcopy(self.grad_args)

    #         for i in range(step):
    #             args = {
    #                 'Xs': trainInput,
    #                 'xTrans': trTrans,
    #                 'Us': incInput,
    #                 'uTrans': incTrans,
    #                 'targets': targets,
    #                 'delta': self.delta,
    #                 # parameters:
    #                 'noise': self.params['noise'],
    #                 'theta': self.params['theta'],
    #             }

    #             # trainEx = self.Kxx.bind(ctx=mx.cpu(), args=args, args_grad=allGrad)
    #             # logging.debug("Symbol Kxx")
    #             # trainEx = self.Kuu.bind(ctx=mx.cpu(), args=args, args_grad=allGrad)
    #             # logging.debug("Symbol Kuu")
    #             # trainEx = self.Kux.bind(ctx=mx.cpu(), args=args, args_grad=allGrad)
    #             # logging.debug("Symbol Kux")

    #             trainEx = log_N.bind(ctx=mx.cpu(), args=args, args_grad=allGrad)
    #             trainEx.forward()
    #             headGrad = [mx.nd.ones((1))]
    #             trainEx.backward(headGrad)
    #             logging.debug("theta: {}".format(self.params['theta'].asnumpy()))
    #             logging.debug("noise: {}".format(self.params['noise'].asnumpy().tolist()[0]))
    #             logging.debug("step: {}, loss: {}".format(i, trainEx.outputs[0].asnumpy()))
    #             self.adam(allGrad, lr, i+1)


    #     optRes = copy.copy(self.params)
    #     self.optRes = optRes

    #     logging.debug("start infer_post")

    #     infer_post = prediction_symbol_VFE(Kuu, Kux, symTar, symNoi, incTarNum, numTarget)
    #     args = {
    #         'Xs': trainInput,
    #         'xTrans': trTrans,
    #         'Us': incInput,
    #         'uTrans': incTrans,
    #         'targets': targets,
    #         'delta': self.delta,
    #         # parameters:
    #         'noise': self.params['noise'],
    #         'theta': self.params['theta'],
    #             }
    #     postEx = infer_post.bind(ctx=mx.cpu(), args=args)
    #     postEx.forward()
    #     self.wv = postEx.outputs[0]
    #     print(self.wv)
    #     logging.debug("end infer_post")
        # self.L_uu, self.L_uf, self.wv = postEx.outputs

    # def predict_VFE(self, preInput, preTrans):

    #     # start = time.time()
    #     symTheta = mx.sym.var('theta')
    #     symDelta = mx.sym.var('delta')
    #     Ts = mx.sym.var('Ts')
    #     tTrans = mx.sym.var('tTrans')
    #     Us = mx.sym.var('Us')
    #     uTrans = mx.sym.var('uTrans')

    #     preNum, _ = preInput.shape
    #     numTarget, _ = preTrans.shape
    #     sym_Kut = self.gen_kern(symTheta, symDelta, Us, Ts, uTrans, tTrans, self.incNum, preNum)
    #     args = {
    #         'Ts': preInput,
    #         'tTrans': preTrans,
    #         'Us': self.incInput,
    #         'uTrans': self.incTrans,
    #         'delta': self.delta,
    #         # parameters:
    #         'theta': self.params['theta'],
    #         'noise': self.params['noise'],
    #             }
    #     kutEx = sym_Kut.bind(ctx=mx.cpu(), args=args)
    #     kutEx.forward()
    #     Kut = kutEx.outputs[0]
    #     Ktu = Kut.transpose()
    #     # tmp1 = mx.nd.dot(Ktu, self.invKuu)
    #     # Qtf = mx.nd.dot(tmp1, Kut)
    #     # tmp1 = mx.nd.linalg_trsm(B=Ktu, A=self.L_uu, transpose=True, rightside=True)
    #     # Qtf = mx.nd.linalg_gemm2(tmp1, self.L_uf)
    #     self.Kut = Kut



    #     # logging.debug("Ktx {}".format(time.time()-start))

    #     # start = time.time()
    #     # symKtt = self.compute_Kxx(preNum, mode='predict')
    #     # args = {
    #     #     'Xs': preInput,
    #     #     'Trans': preTrans,
    #     #     'noise': mx.nd.array([[self.noise]]),
    #     #     'theta': self.params['theta'],
    #     #     'delta': self.params['delta'],
    #     # }
    #     # kttEx = symKtt.bind(ctx=mx.cpu(), args=args)
    #     # kttEx.forward()
    #     # Ktt = kttEx.outputs[0]

    #     # logging.debug("Ktt {}".format(time.time()-start))

    #     mu = mx.nd.linalg_gemm2(Ktu, self.wv)
    #     # tmp = mx.nd.linalg_trsm(A=self.L, B=Ktx.transpose(), transpose=False, rightside=False)
    #     var = None

    #     # mx.nd.waitall()

    #     return mu

    # def test_VFE(self, testData):
    #     numStruct = len(testData)

    #     # testInput, testTrans, testTar = [mx.nd.array(mat) for mat in prepare_data(testData, self.input_dim)]
    #     testInput, testTrans, testTar = self.prepare_data(testData)

    #     # logging.debug("test - train")
    #     # logging.debug("input: {}".format(testInput - self.trainInput))
    #     # logging.debug("Trans: {}".format(testTrans - self.trTrans))

    #     # start = time.time()
    #     preVal = self.predict_VFE(testInput, testTrans)
    #     # logging.debug("self.predict: {}".format(time.time() - start))
    #     mx.nd.save('predictVal', preVal)
    #     mx.nd.save('testVal', testTar)
    #     preEn = preVal[:numStruct]
    #     preForce = preVal[numStruct:]
    #     realEn = testTar[:numStruct]
    #     realForce = testTar[numStruct:]


    def SGD(self, grad, lr):
        for key, val in self.params.items():
            self.params[key] = val - lr * grad[key]

    def adam(self, grad, lr, t):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8
        for key, val in self.params.items():
            v = self.vs[key]
            sqr = self.sqrs[key]
            self.vs[key] = beta1 * v + (1. - beta1) * grad[key]
            self.sqrs[key] = beta2 * sqr + (1. - beta2) * mx.nd.square(grad[key])
            v_bias_corr = self.vs[key] / (1. - beta1 ** t)
            sqr_bias_corr = self.sqrs[key] / (1. - beta2 ** t)
            div = lr * v_bias_corr / (mx.nd.sqrt(sqr_bias_corr) + eps_stable)
            nextVal = val - div
            if key in self.lower_bound.keys():
                self.params[key] = val - div*(nextVal >= self.lower_bound[key])
            else:
                self.params[key] = nextVal





class AtomsNeighbors:
    """
    Neighbors of one structure(ase Atoms object)
    """
    def __init__(self, cutoff, atoms):
        self.cutoff = cutoff
        self.atoms = atoms
        nl = NeighborList(cutoffs=([cutoff / 2.]*len(atoms)),
        self_interaction=False,
        bothways=True,
        skin=0.)
        nl.update(atoms)
        self._nl = nl

    # @profile
    def index_neighbor(self, index):
        neighbor_indices, neighbor_offsets = self._nl.get_neighbors(index)
        # Rs = [self.atoms.positions[n_index] +
        #       np.dot(n_offset, self.atoms.get_cell()) for n_index, n_offset
        #       in zip(neighbor_indices, neighbor_offsets)]
        # neighbor_numbers = [self.atoms.numbers[n_index] for n_index in neighbor_indices]
        Rs = self.atoms.positions[neighbor_indices] + np.dot(neighbor_offsets, self.atoms.get_cell())
        neighbor_numbers = self.atoms.numbers[neighbor_indices]

        Rs = Rs - self.atoms.positions[index]
        # neighbor_numbers = np.array(neighbor_numbers)


        return Rs, neighbor_indices

    # @profile
    def all_neighbor(self):
        allRs = []
        allIndices = []
        maxNeigh = 0
        for i in range(len(self.atoms)):
            Rs, indices = self.index_neighbor(i)
            lenNeigh = len(Rs)
            allRs.append(Rs)
            allIndices.append(indices)
            if maxNeigh < lenNeigh:
                maxNeigh = lenNeigh

        return allRs, allIndices, maxNeigh

class DataSetNeighbors:
    """
    Neighbors of dataset(a list of ase Atoms objects).
    All of the neighbor list of atoms will be extented in order to be used as input data.
    """
    def __init__(self, cutoff, oriNumFps, dataset):
        self.cutoff = cutoff
        self.oriNumFps = oriNumFps
        self.dataset = dataset
        self.numStruct = len(dataset)
        self.lenList = [len(atoms) for atoms in self.dataset]
        self.numAtom = sum(self.lenList)

    def prepare_etas(self, oriEtas):
        """
        Called after prepare_input()
        """
        etas = mx.nd.array(oriEtas)
        etas = etas.repeat(self.numType)
        return etas

    def prepare_lens(self):
        return mx.nd.array([self.lenList])

    # @profile
    def prepare_input(self):
        setRs = []
        setIndices = []
        setMaxNei = []
        setType = []
        setHot = []
        setVol = []
        setAtomsInd = np.zeros((self.numAtom, self.numStruct))
        for i, atoms in enumerate(self.dataset):
            atNeighbors = AtomsNeighbors(self.cutoff, atoms)
            atRs, atIndices, atMaxNei = atNeighbors.all_neighbor()
            # indices offset
            offset = sum(self.lenList[:i])
            for j, indices in enumerate(atIndices):
                atIndices[j] += offset

            setRs.extend(atRs)
            setIndices.extend(atIndices)
            setMaxNei.append(atMaxNei)
            setType.extend(atoms.numbers.tolist())
            setVol.append(atoms.get_volume())
            setAtomsInd[offset:offset+len(atoms),i] = 1

        maxNeigh = max(setMaxNei)
        # print(maxNeigh)
        fpType = list(set(setType))
        fpType.sort()
        numType = len(fpType)
        self.numType = numType
        fpType = mx.nd.array(fpType)
        fpType = fpType.repeat(self.oriNumFps)

        for k in range(self.numAtom):
            tmpRs = setRs[k]
            tmpIndices = setIndices[k]
            lenNeigh = tmpRs.shape[0]
            # print(tmpRs.shape)
            # print(np.concatenate((tmpRs, np.zeros((maxNeigh-lenNeigh, 3))), axis=0))
            setRs[k] = np.concatenate((tmpRs, np.zeros((maxNeigh-lenNeigh, 3))), axis=0)
            setIndices[k] = np.concatenate((tmpIndices, -1*np.ones((maxNeigh-lenNeigh,))), axis=0)
            setHot.append(np.concatenate((np.ones((lenNeigh,)), np.zeros((maxNeigh-lenNeigh,))), axis=0))

        setIndices = mx.nd.array(setIndices)
        setIndices = mx.nd.one_hot(setIndices, self.numAtom)
        # setForceInd = setIndices.asnumpy()

        # self interaction
        setHot = mx.nd.array(setHot)
        tmp1 = mx.nd.arange(self.numAtom).expand_dims(-1).repeat(repeats=maxNeigh, axis=1)
        tmpHot = mx.nd.one_hot(tmp1, self.numAtom)
        setForceInd = setIndices + tmpHot * setHot.expand_dims(-1)

        # Virial
        stackRs = np.stack([setRs]*3, axis=2)
        tmpVir = stackRs*stackRs.swapaxes(2,3)
        tmpVir = tmpVir.reshape(self.numAtom,maxNeigh,9)
        setVir = tmpVir[:,:,[0,4,8,5,2,1]]
        setVir = mx.nd.array(setVir)


        # for k in range(self.numAtom):
        #     for l in range(maxNeigh):
        #         if setForceInd[k, l].sum()>0:
        #             setForceInd[k, l, k] += 1

        # setForceInd = mx.nd.array(setForceInd)

        setRs = mx.nd.array(setRs)
        setType = mx.nd.array(setType)
        setAtomsInd = mx.nd.array(setAtomsInd)
        setAtomsInd = setAtomsInd.transpose()
        setVol = mx.nd.array(setVol)
        cutoff = mx.nd.array([self.cutoff])

        return setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, fpType, cutoff

def cutoff_fxn(Rij, cutoff):
    return 0.5*(mx.sym.cos(mx.sym.broadcast_div(math.pi * Rij.expand_dims(-1), cutoff))+1)

def der_cutoff_fxn(Rij, cutoff):
    return mx.sym.broadcast_div(-0.5*math.pi*mx.sym.sin(mx.sym.broadcast_div(math.pi * Rij.expand_dims(-1), cutoff)), cutoff)

def cutoff_fxn_nd(Rij, cutoff):
    return 0.5*(mx.nd.cos(mx.nd.broadcast_div(math.pi * Rij.expand_dims(-1), cutoff))+1)

def der_cutoff_fxn_nd(Rij, cutoff):
    return mx.nd.broadcast_div(-0.5*math.pi*mx.nd.sin(mx.nd.broadcast_div(math.pi * Rij.expand_dims(-1), cutoff)), cutoff)

def descriptor(etas, setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, cutoff, fpType):
    symEtas = mx.sym.var('Etas', )
    symRs = mx.sym.var('Rs', )
    symType = mx.sym.var('Type', )
    symHot = mx.sym.var('Hot', )
    symVir = mx.sym.var('Vir', )
    symVol = mx.sym.var('Vols', )
    symCut = mx.sym.var('cutoff', )
    symFpType = mx.sym.var('FpType', )

    symIndices = mx.sym.var('Indices', )
    symForceInd = mx.sym.var('ForceInd',)
    symAtomsInd = mx.sym.var('AtomsInd', )

    symGamma = mx.sym.var('Gamma')
    symBeta = mx.sym.var('Beta')

    ndGamma = mx.nd.ones([1,])
    ndBeta = mx.nd.zeros([1,])

    # cutoff = mx.nd.array([cutoff])
    # etas = mx.nd.array(etas)

    # constants
    numAtoms, numNeigh, _ = setIndices.shape
    numFps = len(etas)
    numStruct, _ = setAtomsInd.shape

    # scale etas using gamma
    sclEtas =  mx.sym.broadcast_add(mx.sym.broadcast_mul(symGamma, symEtas), symBeta)
    # sclEtas = symEtas

    sclR2 = (symRs**2).sum(axis=-1)
    sclR = sclR2.sqrt()
    tmp1 = mx.sym.dot(-1*sclR2.expand_dims(-1), mx.sym.broadcast_div((-2*sclEtas).exp(), symCut**2).expand_dims(0)).exp()
    # tmp1 = mx.sym.dot(-1*sclR2.expand_dims(-1), mx.sym.broadcast_div((-2*symEtas).exp(), symCut**2).expand_dims(0)).exp()
    # tmp1 = mx.sym.dot(-1*sclR2.expand_dims(-1), mx.sym.broadcast_div(symEtas, symCut**2).expand_dims(0)).exp()
    # tmp1 = mx.sym.dot(mx.sym.broadcast_div(sclR2,symCut**2).expand_dims(-1), -1*symEtas.expand_dims(0)).exp()
    tmp1 = mx.sym.broadcast_mul(tmp1, symHot.expand_dims(-1))
    typeFilter = mx.sym.broadcast_equal(mx.sym.dot(symIndices, symType.expand_dims(-1)), symFpType.expand_dims(0).expand_dims(0))
    tmp1 = tmp1*typeFilter
    term = mx.sym.broadcast_mul(tmp1, cutoff_fxn(sclR, symCut))
    symFps = term.sum(1)

    tmp3 = sclR.expand_dims(-1) * mx.sym.broadcast_div(cutoff_fxn(sclR, symCut), symCut**2)
    tmp4 = 2*mx.sym.broadcast_mul(tmp3.tile(reps=(1,1,numFps)), (-2*sclEtas).exp().expand_dims(0).expand_dims(0))
    # tmp4 = 2*mx.sym.broadcast_mul(tmp3.tile(reps=(1,1,numFps)), (-2*symEtas).exp().expand_dims(0).expand_dims(0))
    tmp5 = mx.sym.broadcast_sub(der_cutoff_fxn(sclR, symCut), tmp4)
    der_term = tmp1 * tmp5

    # force matrix
    tmp6 = symForceInd.expand_dims(-1).expand_dims(-1).tile(reps=(1,1,1,numFps,3))
    tmp7 = mx.sym.broadcast_div(symRs,(sclR+(sclR<=0)).expand_dims(-1)).expand_dims(-2).tile(reps=(1,1,numFps,1))
    tmp8 = der_term.expand_dims(-1).tile((1,1,1,3))
    tmp9 = tmp7*tmp8
    tmp10 = tmp9.expand_dims(2).tile((1,1,numAtoms,1,1))*tmp6
    tmp10 = mx.sym.cast_storage(tmp10, stype='row_sparse')

    tmp11 = tmp10.sum(1)
    # tmp11 = mx.sym.sparse.sum(tmp10, axis=1)
    tmp11 = tmp11.swapaxes(1,2)
    tmp11 = tmp11.swapaxes(1,3)
    fMat = tmp11.reshape(shape=(numAtoms*3, numAtoms*numFps))

    # virial matrix
    # should divided by 2*atoms volume
    tmp12 = symForceInd.expand_dims(-1).expand_dims(-1).tile(reps=(1,1,1,numFps,6))
    tmp13 = mx.sym.broadcast_div(symVir,(sclR+(sclR<=0)).expand_dims(-1)).expand_dims(-2).tile(reps=(1,1,numFps,1))
    tmp14 = der_term.expand_dims(-1).tile((1,1,1,6))
    tmp15 = tmp13*tmp14
    tmp16 = tmp15.expand_dims(2).tile((1,1,numAtoms,1,1))*tmp12
    tmp16 = mx.sym.cast_storage(tmp16, stype='row_sparse')

    tmp17 = tmp16.sum(1)
    # tmp17 = mx.sym.sparse.sum(tmp16, axis=1)
    tmp17 = tmp17.swapaxes(1,2)
    tmp17 = tmp17.swapaxes(1,3)
    vMat = mx.sym.dot(symAtomsInd, tmp17).reshape(shape=(numStruct*6, numAtoms*numFps))
    vMat = 0.5*mx.sym.broadcast_div(vMat,symVol.repeat(6).expand_dims(-1))

    # whole transform matrix
    blank1 = mx.sym.zeros((numStruct, numAtoms*numFps))
    blank2 = mx.sym.zeros((3*numAtoms+6*numStruct, numAtoms))
    tmp18 = mx.sym.concat(symAtomsInd, blank1, dim=1)
    fvTrans = mx.sym.concat(fMat, vMat, dim=0)
    tmp19 = mx.sym.concat(blank2, fvTrans, dim=1)
    symTrans = mx.sym.concat(tmp18, tmp19, dim=0)

    # res = mx.sym.Group([symFps, fMat, vMat])

    # sparse matrix
    # setAtomsInd = setAtomsInd.tostype('row_sparse')
    # setIndices = setIndices.tostype('row_sparse')
    # setForceInd = setForceInd.tostype('row_sparse')

    # etas = etas.tostype('row_sparse')
    # setRs = setRs.tostype('row_sparse')
    # setType = setType.tostype('row_sparse')
    # setHot = setHot.tostype('row_sparse')
    # cutoff = cutoff.tostype('row_sparse')
    # setVol = setVol.tostype('row_sparse')
    # setVir = setVir.tostype('row_sparse')
    # fpType = fpType.tostype('row_sparse')

    # args = dict()
    # args['Etas'] = etas
    # args['Rs'] = setRs
    # args['Indices'] = setIndices
    # args['Type'] = setType
    # args['Hot'] = setHot
    # args['cutoff'] = cutoff
    fpArgs = {
        'Etas': etas,
        'Rs': setRs,
        'Indices': setIndices,
        'Type': setType,
        'Hot': setHot,
        'cutoff': cutoff,
        'AtomsInd': setAtomsInd,
        'Vols': setVol,
        'Vir': setVir,
        'ForceInd': setForceInd,
        'FpType': fpType,
        'Gamma': ndGamma,
        'Beta': ndBeta,
    }
    # ex = symTrans.bind(ctx=mx.cpu(), args=args)
    # ex.forward()
    # out = ex.outputs

    return symFps, symTrans, fpArgs

def descriptor_nd(etas, setRs, setIndices, setType, setHot, setForceInd, setVir, setAtomsInd, setVol, cutoff, fpType):


    # sparse matrix
    # setAtomsInd = setAtomsInd.tostype('row_sparse')
    # setIndices = setIndices.tostype('row_sparse')
    # setForceInd = setForceInd.tostype('row_sparse')

    # etas = etas.tostype('row_sparse')
    # setRs = setRs.tostype('row_sparse')
    # setType = setType.tostype('row_sparse')
    # setHot = setHot.tostype('row_sparse')
    # cutoff = cutoff.tostype('row_sparse')
    # setVol = setVol.tostype('row_sparse')
    # setVir = setVir.tostype('row_sparse')
    # fpType = fpType.tostype('row_sparse')

    # ndGamma = mx.nd.ones([1,])
    # ndGamma = ndGamma.tostype('row_sparse')

    # cutoff = mx.nd.array([cutoff])
    # etas = mx.nd.array(etas)

    # constants
    numAtoms, numNeigh, _ = setIndices.shape
    numFps = len(etas)
    numStruct, _ = setAtomsInd.shape

    # scale etas using gamma
    # sclEtas = mx.nd.broadcast_mul(ndGamma, etas)
    sclEtas = etas

    sclR2 = (setRs**2).sum(axis=-1)
    sclR = sclR2.sqrt()
    tmp1 = mx.nd.dot(-1*sclR2.expand_dims(-1), mx.nd.broadcast_div((-2*sclEtas).exp(), cutoff**2).expand_dims(0)).exp()
    # tmp1 = mx.nd.dot(-1*sclR2.expand_dims(-1), mx.nd.broadcast_div((-2*etas).exp(), cutoff**2).expand_dims(0)).exp()
    # tmp1 = mx.nd.dot(-1*sclR2.expand_dims(-1), mx.nd.broadcast_div(etas, cutoff**2).expand_dims(0)).exp()
    # tmp1 = mx.nd.dot(mx.nd.broadcast_div(sclR2,cutoff**2).expand_dims(-1), -1*etas.expand_dims(0)).exp()
    tmp1 = mx.nd.broadcast_mul(tmp1, setHot.expand_dims(-1))
    typeFilter = mx.nd.broadcast_equal(mx.nd.dot(setIndices, setType.expand_dims(-1)), fpType.expand_dims(0).expand_dims(0))
    tmp1 = tmp1*typeFilter
    term = mx.nd.broadcast_mul(tmp1, cutoff_fxn_nd(sclR, cutoff))
    setFps = term.sum(1)
    # logging.debug("setFps: {}".format(setFps.shape))

    tmp3 = sclR.expand_dims(-1) * mx.nd.broadcast_div(cutoff_fxn_nd(sclR, cutoff), cutoff**2)
    # logging.debug("tmp3: {}".format(tmp3.shape))
    tmp4 = 2*mx.nd.broadcast_mul(tmp3.tile(reps=(1,1,numFps)), (-2*sclEtas).exp().expand_dims(0).expand_dims(0))
    # logging.debug("tmp4: {}".format(tmp4.shape))
    # tmp4 = 2*mx.nd.broadcast_mul(tmp3.tile(reps=(1,1,numFps)), (-2*etas).exp().expand_dims(0).expand_dims(0))
    tmp5 = mx.nd.broadcast_sub(der_cutoff_fxn_nd(sclR, cutoff), tmp4)
    # logging.debug("tmp5: {}".format(tmp5.shape))
    der_term = tmp1 * tmp5
    # logging.debug("der_term: {}".format(der_term))


    # force matrix
    tmp6 = setForceInd.expand_dims(-1).expand_dims(-1).tile(reps=(1,1,1,numFps,3))
    # logging.debug("tmp6: {}".format(tmp6))
    tmp7 = mx.nd.broadcast_div(setRs,(sclR+(sclR<=0)).expand_dims(-1)).expand_dims(-2).tile(reps=(1,1,numFps,1))
    # logging.debug("tmp7: {}".format(tmp7))
    tmp8 = der_term.expand_dims(-1).tile((1,1,1,3))
    # logging.debug("tmp8: {}".format(tmp8))
    tmp9 = tmp7*tmp8
    # logging.debug("tmp9: {}".format(tmp9))
    tmp10 = tmp9.expand_dims(2).tile((1,1,numAtoms,1,1))*tmp6
    logging.debug("tmp10: {}".format(tmp10.shape))

    mx.nd.waitall()

    tmp10 = tmp10.tostype('row_sparse')
    # mx.nd.save('tmp10.mx', tmp10)

    tmp11 = tmp10.sum(1)
    # logging.debug("tmp11: {}".format(tmp11))
    tmp11 = tmp11.swapaxes(1,2)
    tmp11 = tmp11.swapaxes(1,3)
    # logging.debug("tmp11: {}".format(tmp11))
    fMat = tmp11.reshape(shape=(numAtoms*3, numAtoms*numFps))
    # logging.debug("fMat: {}".format(fMat))
    # mx.nd.save('fMat.mx', fMat)

    # virial matrix
    # should divided by 2*atoms volume
    tmp12 = setForceInd.expand_dims(-1).expand_dims(-1).tile(reps=(1,1,1,numFps,6))
    # logging.debug("tmp12: {}".format(tmp12.shape))
    tmp13 = mx.nd.broadcast_div(setVir,(sclR+(sclR<=0)).expand_dims(-1)).expand_dims(-2).tile(reps=(1,1,numFps,1))
    # logging.debug("tmp13: {}".format(tmp13.shape))
    tmp14 = der_term.expand_dims(-1).tile((1,1,1,6))
    # logging.debug("tmp14: {}".format(tmp14.shape))
    tmp15 = tmp13*tmp14
    # logging.debug("tmp15: {}".format(tmp15.shape))
    tmp16 = tmp15.expand_dims(2).tile((1,1,numAtoms,1,1))*tmp12
    logging.debug("tmp16: {}".format(tmp16.shape))

    # tmp16 = tmp16.tostype('row_sparse')
    tmp16 = mx.nd.sparse.cast_storage(tmp16, 'row_sparse')
    # mx.nd.save('tmp16.mx', tmp16)

    tmp17 = tmp16.sum(1)
    tmp17 = tmp17.swapaxes(1,2)
    tmp17 = tmp17.swapaxes(1,3)
    # mx.nd.save('tmp17.mx', tmp17)
    logging.debug("tmp17: {}".format(tmp17.shape))
    mx.nd.waitall()
    logging.debug("setAtomsInd: {}".format(setAtomsInd.shape))
    vMat = mx.nd.dot(setAtomsInd, tmp17).reshape(shape=(numStruct*6, numAtoms*numFps))
    vMat = 0.5*mx.nd.broadcast_div(vMat,setVol.repeat(6).expand_dims(-1))
    # logging.debug("vMat: {}".format(vMat))
    # mx.nd.save('vMat.mx', vMat)

    # whole transform matrix
    blank1 = mx.nd.zeros((numStruct, numAtoms*numFps))
    blank2 = mx.nd.zeros((3*numAtoms+6*numStruct, numAtoms))
    tmp18 = mx.nd.concat(setAtomsInd, blank1, dim=1)
    fvTrans = mx.nd.concat(fMat, vMat, dim=0)
    tmp19 = mx.nd.concat(blank2, fvTrans, dim=1)
    setTrans = mx.nd.concat(tmp18, tmp19, dim=0)
    # logging.debug("setTrans: {}".format(setTrans))

    logging.debug("blank1: {}".format(blank1.shape))
    logging.debug("blank2: {}".format(blank2.shape))
    logging.debug("fvTrans: {}".format(fvTrans.shape))

    # mx.nd.waitall()

    return setFps, setTrans

def prepare_targets(dataset):
    enArr = []
    fArr = []
    vArr = []
    for atoms in dataset:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces().reshape((-1))
        virial = atoms.get_stress()
        enArr.append(energy)
        fArr.append(mx.nd.array(forces))
        vArr.append(mx.nd.array(virial))
    enTarget = mx.nd.array(enArr)
    fTarget = mx.nd.concat(*fArr, dim=0)
    vTarget = mx.nd.concat(*vArr, dim=0)
    allTarget = mx.nd.concat(enTarget, fTarget, vTarget, dim=0)
    allTarget = allTarget.expand_dims(-1)
    return allTarget



