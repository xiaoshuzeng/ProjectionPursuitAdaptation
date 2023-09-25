
__all__ = ['BasisAdapt', 'ActiveSubspace', 'ProjectionPursuitAdaptation', 'PolyBasis', 'KDE',
           'Hermite1d', 'Legendre1d', 'GradientPolyBasis', 'least_squares', 'weighted_least_squares']


import numpy as np
import math
import time
from scipy import special
from scipy.linalg import lstsq
from sklearn.linear_model import RidgeCV
from scipy import stats


class BasisAdapt:

    def __init__(self, name='Adapted PC expansion'):
        self.name = name

    def gauss_adaptation(self, c_k, nfun, ndim, method=0):
        '''
        Input:
            c_k: N0 dimensional numpy array of first order coefficients,
                    where N0 is the # of pc terms of 1st order expansion
            ndim: dimension of the problem
            nfun: number of QoI's
            method: The method to compute the isometry (Rotation matrix), from set {0,1,2,3}. Returns isometry.
                0 : (default) By Gram-Schmidt procedure on matrix A with Gaussian coeffs (normalized) at its
                    first row, and ones along diagonal zeros elsewhere for other rows.
                1 : Via a Gram-Schmidt procedure on the matrix A with Gaussian coeffs (normalized) at its first
                    row, and, starting from 2th row, put the ith largest Gaussian coeff on the column corresponding
                    to its variable xi at (i+1) row zeros elsewhere.

        Output:
               ndim by ndim numpy array, the rotation matrix
        '''
        assert c_k.shape[0] == ndim
        if nfun == 1:
            coeff = c_k.reshape((1, c_k.shape[0]))
        else:
            coeff = c_k.reshape((c_k.shape[0], c_k.shape[1]))

        if method == 0:   # Gram-Schmidt
            A = np.zeros((ndim*nfun, ndim))
            for k in range(nfun):
                C = np.eye(ndim)
                C[0, :] = coeff[k, :]
                [q, r] = np.linalg.qr(C.T)
                q = np.array(q)
                A[k*ndim:(k+1)*ndim, :] = q.T
            if np.dot(A[0, :], c_k) < 0:
                A = -A
            return A
        elif method == 1:  # Sort by importance, recommended method
            A = np.zeros((ndim*nfun, ndim))
            for k in range(nfun):
                c3 = np.argsort(np.abs(coeff[k, :]))[::-1]
                C = np.zeros((ndim, ndim))
                C[0, :] = coeff[k, :]
                loc = 0
                for i in range(0, ndim-1):
                    C[i+1, c3[loc]] = coeff[k, c3[loc]]
                    loc += 1
                [q, r] = np.linalg.qr(C.T)
                q = np.array(q)
                A[k*ndim:(k+1)*ndim, :] = q.T
                if np.dot(A[0, :], c_k) < 0:
                    A = -A
            return A
        else:
            raise ValueError('Method parameter must be in {0,1}')

    def eta_to_xi_mapping(self, eta, nfun, ndim, A, zeta=None):
        '''
        Maps points from lower dimensional eta space to the xi space.
        A is isometry which serves as the rotation matrix (xi = A' [eta, zeta])
        Input:
            eta : N by d0 numpy array, eta space points, N can be # of quarature points or MC points
                  N0 is the number of adapted dimensions of all QoI's
            A   : n*d by d numpy array, Rotation matrix or isometry
            note that the adaptated dimension is the same for each QoI
        Output:
            N by d numpy array, xi's mapped from eta's
        '''

        assert A.shape[1] == ndim

        d0 = eta.shape[1]//nfun
        N = eta.shape[0]

        if nfun == 1 or eta.shape[1] < ndim:
            if zeta == None:
                zeta = np.zeros((N, ndim-d0))
            else:
                assert eta.shape[0] == zeta.shape[0]
                assert eta.shape[1]/nfun + zeta.shape[1] == A.shape[0]
            eta_full = eta[:, :d0]
            for k in range(nfun):
                eta_full = np.hstack([eta_full, zeta])  # Augment eta with zeros
                if k < nfun:
                    eta_full = np.hstack((eta_full, eta[:, (k+1)*d0:(k+2)*d0]))
            xi = 1.0/nfun*np.dot(A.T, eta_full.T)

        else:
            for i in range(nfun):
                if i == 0:
                    print(d0)
                    A_p = A[:d0, :]
                else:
                    A_p = np.vstack((A_p, A[i*ndim:i*ndim+d0, :]))
            u, s, vh = np.linalg.svd(A_p, full_matrices=True)
            mu = np.dot(np.diag(1.0/s), u[:, :ndim].T)
            mu = np.dot(mu, eta.T)
            xi = np.dot(vh.T, mu)

        return xi.T

    def transform_coeffs(self, coeffs, deg, A, eta_dim=1, method=0, num_MC=100000, xi=None, w=None):
        """
        Only work for single QoI now!!
        Given the coefficients of a (low dimensional) chaos expansion with respect to the
        eta basis, where eta = A * xi, transforms the coefficients to those that correspond
        to an expansion with respect to the original xi basis.
        Ref: Tipireddy, R. and Ghanem, R., 2014. Basis adaptation in homogeneous chaos spaces.
        Journal of Computational Physics, 259, pp.304-317.

        :type coeffs: array
        :param coeffs: The coefficients of the PCE wrt the eta basis

        :type A: array
        :param A: array(d,d) where d is the dimensionality of xi.
               The (unitary) matrix that relates eta = A * xi.

        :type eta_dim: integer
        :param eta_dim: The number of eta components used in the current PCE (eta_dim <= d).

        :type num_MC: integer
        :param num_MC: The number of Monte Carlo samples to be used for estimating the
                inner products <psi_{beta}(A * xi), psi_{alpha}(xi)>
        """

        assert isinstance(eta_dim, int)
        assert isinstance(deg, int)
        assert eta_dim <= A.shape[0]
        pol_eta = PolyBasis(eta_dim, deg)
        assert len(pol_eta._MI_terms) == coeffs.shape[0]
        if method == 0:
            xi = np.random.normal(0, 1, size=(num_MC, A.shape[1]))
            eta = np.dot(A, xi.T)[:eta_dim, :].T
            pol = PolyBasis(A.shape[0], deg)
            return np.dot(coeffs.T, np.dot(pol_eta(eta).T, pol(xi))) / num_MC
        elif method == 1:
            eta = np.dot(A, xi.T)[:eta_dim, :].T
            pol = PolyBasis(A.shape[0], deg)
            return np.dot(coeffs.T, np.dot(pol_eta(eta).T * w, pol(xi)))
        else:
            raise ValueError(
                'For integration with Monte Carlo or quadrature rule choose 0 or 1 respectively.')

    def transform_coeffs_to_eta(self, coeffs, deg, A, eta_dim=1, method=0, num_MC=100000, xi=None, w=None):
        """
        Only work for single QoI now!!
        Transfer coefficients from xi space to eta space

        :type coeffs: array
        :param coeffs: The coefficients wrt the xi basis

        :type A: array
        :param A: array(d,d) where d is the dimensionality of xi.
               The (unitary) matrix that relates eta = A * xi.

        :type eta_dim: integer
        :param eta_dim: The number of eta components used in the current PCE (eta_dim <= d).

        :type num_MC: integer
        :param num_MC: The number of Monte Carlo samples to be used for estimating the
                inner products <psi_{beta}(A * xi), psi_{alpha}(xi)>
        """
        pol_eta = PolyBasis(eta_dim, deg)
        if method == 0:
            xi = np.random.normal(0, 1, size=(num_MC, A.shape[1]))
            eta = np.dot(A, xi.T)[:eta_dim, :].T
            pol = PolyBasis(A.shape[0], deg)
            output = np.dot(pol(xi).T, pol_eta(eta))
            output = np.dot(coeffs.T, output) / num_MC
            return output
        elif method == 1:
            eta = np.dot(A, xi.T)[:eta_dim, :].T
            pol = PolyBasis(A.shape[0], deg)
            return np.dot(coeffs.T, np.dot(pol(xi).T * w, pol_eta(eta)))
        else:
            raise ValueError(
                'For integration with Monte Carlo or quadrature rule choose 0 or 1 respectively.')

    def l2_error(self, q_1, q_2, d1, d2, nord):
        """
        l2-norm relative error function
        Return relative l2-error ||q_1 - q_2|| / ||q_2||

        :type q_1: array
        :param q_1: coefficients corresponding to eta_{i}

        :type q_2: array
        :param q_2: coefficients corresonding to eta_{i+1}
        """
        import math
        assert np.shape(q_2)[0] == math.factorial(
            d2+nord) / (math.factorial(nord) * math.factorial(d2))
        assert np.shape(q_1)[0] < np.shape(q_2)[0]
        Q1 = np.zeros(q_2.shape[0])
        Q1[PolyBasis().mi_terms_loc(d1, d2, nord)] = q_1
        return (np.linalg.norm((Q1 - q_2), 2) / np.linalg.norm(q_2, 2))


class ActiveSubspace:

    def __init__(self, name='Active subspace'):
        self.name = name

    def AS_buid_rotation(self, c_k, ndim, method=0):
        grad = c_k
        C = np.outer(grad, grad)
        u, s, vh = np.linalg.svd(C)
        evals, evecs = np.linalg.eigh(np.dot(C, C.T))
        return u.T

    def reduced_to_xi_mapping(self, eta, A):
        d0 = eta.shape[1]
        W = A[:d0, :].T
        return np.dot(W, eta.T).T


class ProjectionPursuitAdaptation:

    def __init__(self, tol_pce=2e-2, PPA_method=1, PPA_dim=None, recover_run=False, ndim_iteration=None, list_vec_a=None, mat_A_new=None, list_c_k=None, list_pce_evals=None, main_verbose=True, name='Projection Pursuit Adaptation'):
        tol_pce_coeffs = 2.5e-2
        tol_adapt_dire = 2.5e-2
        self._name = name
        self._tol_c_k = tol_pce_coeffs
        self._tol_vec_a = tol_adapt_dire
        self._tol_pce = tol_pce
        self._PPA_method = PPA_method
        self._PPA_dim = PPA_dim
        self._recover_run = recover_run
        self._main_verbose = main_verbose
        self._ndim_iteration = ndim_iteration
        self._list_vec_a = list_vec_a
        self._mat_A_new = mat_A_new
        self._list_c_k = list_c_k
        self._list_pce_evals = list_pce_evals

    def projection_pursuit_adaptation(self, nord, pc_type, xi, Q_evals):
        '''
        function to forroem projections pursuit regression
        N_data : the number of data
        ndim   : the dimension of the input parameters
        nord   : the order of PCE
        pc_type: polynomial chaos family (HG, or LU)
        xi     :      are the gaussian inputs
        Q_evals: is the QoI
        self._PPA_method=0: is the additive model where there are no restrictions no the prjections
            (the projections can be dependent)
        self._PPA_method=1: is the multivariate model where the projections are orthonormal
        '''
        # compute rotation matrix that serves as initial guess of projections
        self._nord = nord
        self._pc_type = pc_type
        N_data = xi.shape[0]
        ndim = xi.shape[1]
        psi_xi = PolyBasis(ndim, 1, pc_type)(xi)
        c_k0, _, _, _ = lstsq(psi_xi, Q_evals)
        gauss_rotation = BasisAdapt().gauss_adaptation(c_k0[1:], 1, ndim, method=1)

        list_vec_a = []             # list of the projections
        list_pce_evals = []         # list of PCE evaluations with increasing dimensions
        list_c_k = []               # list of PCE coefficients with increasing dimensons

        if not self._recover_run:
            ndim_iteration = 1
        else:
            ndim_iteration = self._ndim_iteration + 1
            list_vec_a = self._list_vec_a
            list_c_k = self._list_c_k
            list_pce_evals = self._list_pce_evals
            mat_A = self._mat_A_new
        if self._PPA_method == 0:
            while True:
                # initial guess of the current projection
                if self._main_verbose:
                    print('\nPerforming %d-d PP-adaptation ...' % ndim_iteration)
                if pc_type == 'HG':
                    if ndim_iteration <= ndim:
                        vec_a = gauss_rotation[ndim_iteration-1, :]
                    else:
                        vec_a = np.random.uniform(-1, 1, ndim)
                        vec_a = vec_a/np.linalg.norm(vec_a)
                elif pc_type == 'LU':
                    vec_a = np.random.rand(ndim)
                    vec_a = vec_a/np.linalg.norm(vec_a)
                eta = np.dot(vec_a, xi.T)
                eta = np.reshape(eta, (eta.size, 1))
                psi_eta = PolyBasis(1, nord, pc_type)(eta)
                c_k, _, _, _ = lstsq(psi_eta, Q_evals) if ndim_iteration == 1 else lstsq(
                    psi_eta, Q_evals - list_pce_evals[ndim_iteration-2])
                pce_evals = np.dot(psi_eta, c_k)

                # Iteration to compute the current projections
                k = 1
                while True:
                    if self._main_verbose:
                        if k % 10 == 0:
                            print('Current iteration is %d' % k)
                    # Start to search adapted directions from the firsr-order coefficients
                    grad_psi_eta = GradientPolyBasis(1, nord, pc_type)(eta)
                    grad_pce = np.array([np.dot(c_k, grad_psi_eta[i]) for i in range(N_data)])
                    b_hat = (eta[:, 0] + (Q_evals-pce_evals)/grad_pce[:, 0]) if ndim_iteration == 1 else (
                        eta[:, 0] + (Q_evals-list_pce_evals[ndim_iteration-2]-pce_evals)/grad_pce[:, 0])
                    w = grad_pce[:, 0]**2
                    W = np.diag(w)
                    # vec_a_new = weighted_least_squares(xi, W, b_hat)
                    vec_a_new, _, _, _ = lstsq((np.sqrt(w)*xi.T).T, np.sqrt(w)*b_hat)
                    vec_a_new = vec_a_new/np.linalg.norm(vec_a_new)

                    # update the PCE model with new projection
                    eta = np.dot(vec_a_new, xi.T)
                    eta = np.reshape(eta, (eta.size, 1))
                    psi_eta = PolyBasis(1, nord, pc_type)(eta)
                    c_k_new, _, _, _ = lstsq(psi_eta, Q_evals) if ndim_iteration == 1 else lstsq(
                        psi_eta, Q_evals-list_pce_evals[ndim_iteration-2])
                    pce_evals_new = np.dot(psi_eta, c_k_new)

                    # stopping criterion check
                    err_vec_a = np.linalg.norm(vec_a_new - vec_a)/np.linalg.norm(vec_a)
                    err_c_k = np.linalg.norm(c_k_new[1:] - c_k[1:])/np.linalg.norm(c_k[1:])
                    num_stop_it = 40
                    if (err_vec_a < self._tol_vec_a) and (err_c_k < self._tol_c_k):
                        break
                    elif k > num_stop_it:
                        if self._main_verbose:
                            print('Single iteration exceed %d, exit from current iteration !!!' % (num_stop_it))
                        break
                    else:
                        vec_a = np.copy(vec_a_new)
                        c_k = np.copy(c_k_new)
                        pce_evals = np.copy(pce_evals_new)
                        k += 1
                if self._main_verbose:
                    print('Number of iteration is :%d' % k)
                list_vec_a.append(vec_a_new)
                list_pce_evals.append(pce_evals) if ndim_iteration == 1 else list_pce_evals.append(
                    list_pce_evals[ndim_iteration-2]+pce_evals)
                list_c_k.append(c_k_new)
                if self._PPA_dim is not None:
                    if (ndim_iteration > 1):
                        err_pce = np.linalg.norm(
                            list_pce_evals[ndim_iteration-1] - list_pce_evals[ndim_iteration-2])/np.linalg.norm(list_pce_evals[ndim_iteration-2])
                        if self._main_verbose:
                            print('err_pce = %.6f' % err_pce)
                    if ndim_iteration == self._PPA_dim:
                        break
                elif ndim_iteration > 1:
                    err_pce = np.linalg.norm(
                        list_pce_evals[ndim_iteration-1] - list_pce_evals[ndim_iteration-2])/np.linalg.norm(list_pce_evals[ndim_iteration-2])
                    if self._main_verbose:
                        print('err_pce = %.6f' % err_pce)
                    if err_pce < self._tol_pce:
                        break
                ndim_iteration += 1
            mat_A_new = np.array(list_vec_a)

        elif self._PPA_method == 1:
            while True:
                if self._main_verbose:
                    print('\nPerforming %d-d PP-adaptation ...' % ndim_iteration)
                assert pc_type in [
                    'HG'], 'Only Hermite polynomials are currently supported! Please choose ' + str(['HG']) + '!'
                # mat_A = gauss_rotation[:ndim_iteration, :]
                vec_a = gauss_rotation[ndim_iteration-1, :]
                mat_A = np.reshape(
                    vec_a, (1, ndim)) if ndim_iteration == 1 else np.vstack((mat_A, vec_a))
                eta = np.dot(mat_A, xi.T).T
                psi_eta = PolyBasis(ndim_iteration, nord, pc_type)(eta)

                # c_k, _, _, _ = lstsq(psi_eta, Q_evals)
                clfCV = RidgeCV(alphas=[1e-2, 1e-1, 1, 10], fit_intercept=False)
                clfCV.fit(psi_eta, Q_evals)
                c_k = clfCV.coef_

                pce_evals = np.dot(psi_eta, c_k)

                k = 1
                while True:
                    if self._main_verbose:
                        if k % 10 == 0:
                            print('Current iteration is %d' % k)
                    # Start to search adapted directions from the firsr-order coefficients
                    grad_psi_eta = GradientPolyBasis(ndim_iteration, nord, pc_type)(eta)
                    grad_pce = np.array([np.dot(c_k, grad_psi_eta[i]) for i in range(N_data)])
                    b_hat = eta[:, -1] + (Q_evals-pce_evals)/grad_pce[:, -1]
                    w = grad_pce[:, -1]**2
                    W = np.diag(w)
                    # vec_a_new = weighted_least_squares(xi, W, b_hat)
                    vec_a_new, _, _, _ = lstsq((np.sqrt(w)*xi.T).T, np.sqrt(w)*b_hat)
                    mat_A_new = np.copy(mat_A)
                    mat_A_new[-1, :] = vec_a_new
                    [q, r] = np.linalg.qr(mat_A_new.T)
                    q = np.array(q).T
                    if q[0, 0]*mat_A_new[0, 0] < 0:
                        q = -q
                    vec_a_new = q[-1, :]
                    mat_A_new = q

                    eta = np.dot(mat_A_new, xi.T).T
                    psi_eta = PolyBasis(ndim_iteration, nord, pc_type)(eta)
                    # update the PCE model with new projection (we can use ridge regression
                    # to reduce overfitting)
                    # c_k_new, _, _, _ = lstsq(psi_eta, Q_evals)
                    clfCV = RidgeCV(alphas=[1e-2, 1e-1, 1, 10], fit_intercept=False)
                    clfCV.fit(psi_eta, Q_evals)
                    c_k_new = clfCV.coef_

                    pce_evals_new = np.dot(psi_eta, c_k_new)

                    err_vec_a = np.linalg.norm(vec_a_new - vec_a)/np.linalg.norm(vec_a)
                    err_c_k = np.linalg.norm(c_k_new[1:] - c_k[1:])/np.linalg.norm(c_k[1:])
                    num_stop_it = 40
                    if (err_vec_a < self._tol_vec_a) and (err_c_k < self._tol_c_k):
                        break
                    elif k > num_stop_it:
                        if self._main_verbose:
                            print('Single iteration exceed %d, exit from current iteration !!!' % num_stop_it)
                        break
                    else:
                        mat_A = np.copy(mat_A_new)
                        vec_a = np.copy(vec_a_new)
                        c_k = np.copy(c_k_new)
                        pce_evals = np.copy(pce_evals_new)
                        k += 1
                if self._main_verbose:
                    print('Number of iteration is :%d' % k)
                list_vec_a.append(vec_a_new)
                list_pce_evals.append(pce_evals)
                list_c_k.append(c_k_new)
                if self._PPA_dim is not None:
                    if (ndim_iteration > 1):
                        err_pce = np.linalg.norm(
                            list_pce_evals[ndim_iteration-1] - list_pce_evals[ndim_iteration-2])/np.linalg.norm(list_pce_evals[ndim_iteration-2])
                        if self._main_verbose:
                            print('err_pce = %.6f' % err_pce)
                    if ndim_iteration == self._PPA_dim:
                        break
                elif ndim_iteration > 1:
                    err_pce = np.linalg.norm(
                        list_pce_evals[ndim_iteration-1] - list_pce_evals[ndim_iteration-2])/np.linalg.norm(list_pce_evals[ndim_iteration-2])
                    if self._main_verbose:
                        print('err_pce = %.6f' % err_pce)
                    if err_pce < self._tol_pce:
                        break

                if ndim_iteration == ndim:
                    break

                ndim_iteration += 1

        if self._main_verbose:
            print('\n\nPP-adaptation dimension is: ', ndim_iteration, '\n\n')
        self._ndim_iteration = ndim_iteration
        self._list_vec_a = list_vec_a
        self._mat_A_new = mat_A_new
        self._list_c_k = list_c_k
        self._list_pce_evals = list_pce_evals

    def evaluation(self, germ_samples):
        ndim = self._mat_A_new.shape[1]
        ndim_samples = germ_samples.shape[1]
        n_MC = germ_samples.shape[0]

        if ndim_samples == ndim:
            eta = np.dot(self._mat_A_new, germ_samples.T).T
        else:
            eta = germ_samples

        if self._PPA_method == 0:
            list_psi_eta = [PolyBasis(1, self._nord, self._pc_type)(eta[:, i].reshape(n_MC, 1))
                            for i in range(self._ndim_iteration)]
            list_pce_evals = [np.dot(list_psi_eta[i], self._list_c_k[i]) for i in range(self._ndim_iteration)]
            pce_evals = sum(list_pce_evals)
        elif self._PPA_method == 1:
            psi_eta = PolyBasis(self._ndim_iteration, self._nord, self._pc_type)(eta)
            pce_evals = np.dot(psi_eta, self._list_c_k[-1])
        return pce_evals


class Hermite1d:
    """
    Construct 1-dimensional normalized Hermite polynomials.
    """
    _nord = None

    def __init__(self, nord=1):
        """
        Ininializes the object
        """
        self._nord = nord

    def eval(self, x):
        H = np.zeros(self._nord + 1)
        H[0], H[1] = 1.0, x
        for i in range(2, H.shape[0]):
            H[i] = (x * H[i-1] - (i-1) * H[i-2])
        # normalization
        H = H / [math.sqrt(math.factorial(i)) for i in range(H.shape[0])]
        return H

    def __call__(self, x):
        N = x.shape[0]
        H = np.zeros((N, self._nord + 1))
        for i in range(N):
            H[i, :] = self.eval(x[i])
        return H


class Legendre1d:
    """
    Construct 1-dimensional Legendre polynomials.
    """
    _nord = None

    def __init__(self, nord=1):
        """
        Initializes the object
        """
        self._nord = nord

    def eval(self, x):
        H = np.zeros(self._nord + 1)
        H[0], H[1] = 1.0, x
        for i in range(2, H.shape[0]):
            H[i] = ((2*i-1) * x * H[i-1] - (i-1) * H[i-2]) / i
        # H = H / [math.sqrt(2 / (2*i+1))
        #          for i in range(H.shape[0])]  # normalized
        return H

    def __call__(self, x):
        N = x.shape[0]
        H = np.zeros((N, self._nord + 1))
        for i in range(N):
            H[i, :] = self.eval(x[i])
        return H


class PolyBasis:
    """
    Construct PCE basis terms
    """
    _nord = None
    _ndim = None
    _MI_terms = None
    _type = None

    def __init__(self, ndim=1, nord=1, pol_type='HG'):

        # Ininializes the object
        assert pol_type in [
            'HG', 'LU'], 'Only Hermite and Legendre polynomials are currently supported! Please choose among ' + str(['HG', 'LU']) + '!'
        self._nord = nord
        self._ndim = ndim
        self._MI_terms = self.mi_terms(self._ndim, self._nord)
        self._type = pol_type

    def __call__(self, xi):
        assert xi.shape[1] == self._ndim

        if self._type == 'HG':
            H = [Hermite1d(nord=self._nord)(xi[:, i]) for i in range(self._ndim)]
            psi_xi = np.ones((xi.shape[0], self._MI_terms.shape[0]))
            for i in range(self._MI_terms.shape[0]):
                for j in range(self._ndim):
                    psi_xi[:, i] *= H[j][:, self._MI_terms[i, j]]

        elif self._type == 'LU':
            H = [Legendre1d(nord=self._nord)(xi[:, i])
                 for i in range(self._ndim)]
            psi_xi = np.ones((xi.shape[0], self._MI_terms.shape[0]))
            for i in range(self._MI_terms.shape[0]):
                for j in range(self._ndim):
                    psi_xi[:, i] *= H[j][:, self._MI_terms[i, j]]

        return psi_xi

    def mi_terms(self, ndim, nord):
        """
        Multiindex matrix

        ndim: integer

        nord: PCE order
        """
        q_num = [int(special.comb(ndim+i-1, i)) for i in range(nord+1)]
        mul_ind = np.array(np.zeros(ndim, dtype=int), dtype=int)
        mul_ind = np.vstack([mul_ind, np.eye(ndim, dtype=int)])
        I = np.eye(ndim, dtype=int)
        ind = [1] * ndim
        for j in range(1, nord):
            ind_new = []
            for i in range(ndim):
                a0 = np.copy(I[int(np.sum(ind[:i])):, :])
                a0[:, i] += 1
                mul_ind = np.vstack([mul_ind, a0])
                ind_new += [a0.shape[0]]
            ind = ind_new
            I = np.copy(mul_ind[np.sum(q_num[:j+1]):, :])
        return mul_ind

    def mi_terms_loc(self, d1, d2, nord):
        """
        Locate basis terms in Multi-index matrix
        """
        assert d1 < d2
        MI2 = self.mi_terms(d2, nord)
        if d2 == d1 + 1:
            return np.where(MI2[:, -1] == 0)[0]
        else:
            TFs = (MI2[:, d1:] == [0]*(d2-d1))
            locs = []
            for i in range(TFs.shape[0]):
                if TFs[i, :].all():
                    locs.append(i)
            return locs


class GradientPolyBasis:
    """
    Compute gradient of PCE basis terms
    """
    _nord = None
    _ndim = None
    _MI_terms = None
    _type = None

    def __init__(self, ndim=1, nord=1, pol_type='HG'):

        # Ininializes the object
        assert pol_type in [
            'HG', 'LU'], 'Gradient option support only Hermite and Legendre polynomials! Please choose ' + str(['HG', 'LU']) + '!'
        self._nord = nord
        self._ndim = ndim
        self._MI_terms = PolyBasis(self._ndim, self._nord).mi_terms(self._ndim, self._nord)
        self._type = pol_type

    def __call__(self, xi):
        assert xi.shape[1] == self._ndim

        if self._type == 'HG':
            PB = PolyBasis(self._ndim, self._nord)
            npce = PB._MI_terms.shape[0]
            H = [Hermite1d(nord=self._nord)(xi[:, i]) for i in range(self._ndim)]
            grad_psi_xi = np.ones((xi.shape[0], npce, self._ndim))
            # grad_psi_xi2 = np.ones((xi.shape[0], npce, self._ndim))
            psi_xi = np.ones((xi.shape[0], npce))
            for i in range(npce):
                for j in range(self._ndim):
                    psi_xi[:, i] *= H[j][:, PB._MI_terms[i, j]]
                # for k in range(self._ndim):
                #     eval_1d_deriv = self.derivative_1d_Hermite(xi[:, k])[:, PB._MI_terms[i, k]]
                #     grad_psi_xi[:, i, k] = psi_xi[:, i]/H[k][:, PB._MI_terms[i, k]] * eval_1d_deriv
            for k in range(self._ndim):
                grad_psi_xi[:, :, k] = psi_xi/H[k][:, PB._MI_terms[:, k]] * \
                    self.derivative_1d_Hermite(xi[:, k])[:, PB._MI_terms[:, k]]
        elif self._type == 'LU':
            PB = PolyBasis(self._ndim, self._nord)
            npce = PB._MI_terms.shape[0]
            L = [Legendre1d(nord=self._nord)(xi[:, i]) for i in range(self._ndim)]
            grad_psi_xi = np.ones((xi.shape[0], npce, self._ndim))
            psi_xi = np.ones((xi.shape[0], npce))
            for i in range(npce):
                for j in range(self._ndim):
                    psi_xi[:, i] *= L[j][:, PB._MI_terms[i, j]]
                # for k in range(self._ndim):
                #     eval_1d_deriv = self.derivative_1d_Legendre(xi[:, k])[:, PB._MI_terms[i, k]]
                #     grad_psi_xi[:, i, k] = psi_xi[:, i]/L[k][:, PB._MI_terms[i, k]] * eval_1d_deriv
            for k in range(self._ndim):
                grad_psi_xi[:, :, k] = psi_xi/L[k][:, PB._MI_terms[:, k]] * \
                    self.derivative_1d_Legendre(xi[:, k])[:, PB._MI_terms[:, k]]

        return grad_psi_xi

    def derivative_1d_Hermite(self, x):
        x = np.reshape(x, (x.size, 1))
        dH = np.zeros((x.shape[0], self._nord + 1))
        H = Hermite1d(self._nord)(x)
        for i in range(1, dH.shape[1]):
            dH[:, i] = i*H[:, i-1]
            # normalization
            dH[:, i] = dH[:, i]*np.sqrt(math.factorial(i-1))/np.sqrt(math.factorial(i))
        return dH

    def derivative_1d_Legendre(self, x):
        x = np.reshape(x, (x.size, 1))
        dL = np.zeros((x.shape[0], self._nord + 1))
        L = Legendre1d(self._nord)(x)
        for i in range(1, dL.shape[1]):
            dL[:, i] = (x[:, 0]*L[:, i]-L[:, i-1])*i/(x[:, 0]**2-1)
        return dL


def least_squares(X, y):
    '''
    X: (N, p) array, where N is the number of samples, and p is the number
       of features. X denote control variables
    y: (N, ) array, denoting output or quantity of interest.
    output: (p, ), coefficients associated with control variables
    '''
    output = np.dot(X.T, y)
    output = np.dot(np.linalg.inv(np.dot(X.T, X)), output)
    return output


def weighted_least_squares(X, W, y):
    '''
    X: (N, p) array, where N is the number of samples, and p is the number
       of features. X denote control variables
    W: (N, N) array, denoting the weight matrix
    y: (N, ) array, denoting output or quantity of interest.
    output: (p, ), coefficients associated with control variables
    '''
    # mat_W = np.diag(w)
    output = np.dot(X.T, np.dot(W, y))
    output = np.dot(np.linalg.inv(np.dot(np.dot(X.T, W), X)), output)
    return output


def KDE(fcn_evals, npts=400):
    """
    Performs kernel density estimation
    Input:
        fcn_evals: numpy array of evaluations of the forward model (values of heat flux Q)
    Output:
        xpts_pce: numpy array of points at which the PDF is estimated.
        PDF_data_pce: numpy array of estimated PDF values.
    """
    # Perform KDE on fcn_evals
    kern_pce = stats.kde.gaussian_kde(fcn_evals)
    # Generate points at which to evaluate the PDF
    L = fcn_evals.max()-fcn_evals.min()
    vec_x_left = fcn_evals.min()-np.arange(1, 101)/100*L
    vec_x_right = fcn_evals.max()+np.arange(1, 101)/100*L
    xleft = vec_x_left[np.argwhere(kern_pce(vec_x_left) < 1e-7)[0]]
    xright = vec_x_right[np.argwhere(kern_pce(vec_x_right) < 1e-7)[0]]
    xpts = np.linspace(xleft[0], xright[0], npts)
    # Evaluate the estimated PDF at these points
    PDF_data = kern_pce(xpts)
    return xpts, PDF_data
