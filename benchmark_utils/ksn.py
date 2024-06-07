import numpy as np
from numba import njit
from numba import float64, int32

from skglm.penalties.base import BasePenalty
# from skglm.utils.prox_funcs import prox_SLOPE   


# License: BSD 3 clause

import warnings
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
from numbers import Integral, Real
from skglm.solvers import ProxNewton, LBFGS

from sklearn.utils.validation import (check_is_fitted, check_array,
                                      check_consistent_length)
from sklearn.linear_model._base import (
    LinearModel, RegressorMixin,
    LinearClassifierMixin, SparseCoefMixin, BaseEstimator
)
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.multiclass import OneVsRestClassifier, check_classification_targets

from skglm.utils.jit_compilation import compiled_clone
from skglm.solvers import AndersonCD, MultiTaskBCD, GroupBCD
from skglm.datafits import (Cox, Quadratic, Logistic, QuadraticSVC,
                            QuadraticMultiTask, QuadraticGroup)
from skglm.penalties import (L1, WeightedL1, L1_plus_L2, L2, WeightedGroupL2,
                             MCPenalty, WeightedMCPenalty, IndicatorBox, L2_1)
from skglm.utils.data import grp_converter

from .utils import _binary_search, _compute_theta, _find_alpha, _find_q, _hard_threshold, _interpolate, _ksncost, _op_method, prox_ksn

# from modopt.opt.proximity import KSupportNorm
import sys

# This contains some copy-paste from the code from here: https://cea-cosmic.github.io/ModOpt/_modules/modopt/opt/proximity.html#KSupportNorm
# in order to compile it in numba





class KSN(BasePenalty):
    """Nonconvex penalty based on the k-support norm.

    Attributes
    ----------
    alpha : float, 
        regularization level
    """

    def __init__(self, alpha, k):
        self.alpha = alpha
        self.k = k

    def get_spec(self):
        spec = (
            ('alpha', float64),
            ('k', int32)
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha, k=self.k)

    def value(self, w):
        """Compute the value of KSN at w."""
        # print('computing a cost')
        k = self.k
        alpha = self.alpha
        return alpha * (1/2 * _ksncost(w, k, alpha) - 1/2 * np.linalg.norm(w)**2)

    def prox_vec(self, x, stepsize):
        alpha = self.alpha
        coef = stepsize * alpha
        k = self.k
        data_shape = x.shape
        # k_max = np.product(data_shape)
        # if k > k_max:
        #     raise(
        #         'K value of the K-support norm is greater than the input '
        #         + 'dimension, its value will be set to {0}'.format(k_max),
        #     )
        #     k = k_max
        return prox_ksn(x, coef, k)


    def prox_1d(self, value, stepsize, j):
        raise ValueError(
            "No coordinate-wise proximal operator for SLOPE. Use `prox_vec` instead."
        )

    def subdiff_distance(self, w, grad, ws):
        return ValueError(
            "No subdifferential distance for SLOPE. Use `opt_strategy='fixpoint'`"
        )
    