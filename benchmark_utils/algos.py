import numpy as np
from scipy.sparse import issparse
from skglm.solvers.base import BaseSolver
from skglm.solvers.common import construct_grad, construct_grad_sparse
from .utils import _prox_vec


class ProxGD(BaseSolver):
    r"""ProxGD solver.
    """

    def __init__(self, max_iter=100, tol=1e-4, opt_strategy="fixpoint", verbose=0, fit_intercept=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.opt_strategy = opt_strategy
        self.fit_intercept = fit_intercept   # needed to be passed to GeneralizedLinearEstimator
        self.warm_start = False

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        p_objs_out = []
        n_samples, n_features = X.shape
        all_features = np.arange(n_features)
        X_is_sparse = issparse(X)
        t_new = 1.

        w = w_init.copy() if w_init is not None else np.zeros(n_features)
        Xw = Xw_init.copy() if Xw_init is not None else np.zeros(n_samples)

        if X_is_sparse:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
            lipschitz = datafit.get_global_lipschitz_sparse(X.data, X.indptr, X.indices, y)
        else:
            datafit.initialize(X, y)
            lipschitz = datafit.get_global_lipschitz(X, y)        
        

        # old_obj = np.inf

        for n_iter in range(self.max_iter):
            # t_old = t_new
            # t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            # w_old = w.copy()

            if X_is_sparse:
                if hasattr(datafit, "gradient_sparse"):
                    grad = datafit.gradient_sparse(
                        X.data, X.indptr, X.indices, y, X @ w)
                else:
                    grad = construct_grad_sparse(
                        X.data, X.indptr, X.indices, y, w, X @ w, datafit, all_features)
            else:
                if hasattr(datafit, "gradient"):
                    grad = datafit.gradient(X, y, X @ w)
                else:
                    grad = construct_grad(X, y, w, X @ w, datafit, all_features)

            step = 0.99 * 1 / lipschitz
            w -= step * grad
            w = _prox_vec(w.copy(), step, penalty)  # we copy just in case
            Xw = X @ w
            # z = w
            # z = w + (t_old - 1.) / t_new * (w - w_old)

            if self.opt_strategy == "subdiff":
                opt = penalty.subdiff_distance(w, grad, all_features)
            elif self.opt_strategy == "fixpoint":
                opt = np.abs(w - penalty.prox_vec(w - grad *step , step))
            else:
                raise ValueError(
                    "Unknown error optimality strategy. Expected "
                    f"`subdiff` or `fixpoint`. Got {self.opt_strategy}")

            stop_crit = np.max(opt)

            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            p_objs_out.append(p_obj)

            # opt = np.abs(old_obj - p_obj) / p_obj
            # old_obj = p_obj



            # stop_crit = opt

            if self.verbose:
                print(
                    f"Iteration {n_iter+1}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

            if stop_crit < self.tol:
                if self.verbose:
                    print(f"Stopping criterion max violation: {stop_crit:.2e}")
                # print("Early stopping since criterion is matched.")
                break
            if n_iter == self.max_iter - 1: 
                print("Stopped by max_iter, criterion not matched")
        return w, np.array(p_objs_out), stop_crit
