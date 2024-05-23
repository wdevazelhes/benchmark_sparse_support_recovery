from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion
from benchmark_utils.ksn import KSN
from benchmark_utils.algos import ProxGD

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.linalg import lstsq
    from skglm import Lasso, ElasticNet, MCPRegression, GeneralizedLinearEstimator
    from skglm.datafits import Quadratic

class Solver(BaseSolver):
    name = "skglm"
    # stopping_criterion = RunOnGridCriterion(grid=np.linspace(0, 0.3, 10))
    stopping_criterion = SingleRunCriterion(1)
    parameters = {
        "estimator": ["lasso", "enet", "mcp", "ksn_optim"],
        # "estimator": ["lasso", "enet", "mcp", "ksn_optim", "ksn_0_1", "ksn_0_3", "ksn_0_6"],
        "max_iter": [1_0000000],
        "alphaNum": [1_000],
        "alphaRatio": [1e-10],
        "debiasing_step": [False, True],
    }
    install_cmd = "conda"
    requirements = ["pip:skglm", "scipy"]

    def set_objective(self, X, y, w_true):
        self.X = X
        self.y = y
        self.w_true = w_true
        if self.estimator in ["ksn_optim", "ksn_0_1", "ksn_0_3", "ksn_0_6"]:
            self.alphaMax = np.linalg.norm(self.X, ord=2) ** 2 / len(self.y)
        else:
            self.alphaMax = np.linalg.norm(self.X.T @ self.y, np.inf) / y.size
        self.alphaMin = self.alphaRatio * self.alphaMax
        self.alphaGrid = np.logspace(
            np.log10(self.alphaMax),
            np.log10(self.alphaMin),
            self.alphaNum,
        )

    def run(self, n_iter):
        # The grid_value parameter is the current entry in
        # self.stopping_criterion.grid which is the amount of sparsity we
        # target in the solution, i.e., the fraction of non-zero entries.
        # k = int(np.floor(grid_value * self.X.shape[1]))
        k = np.count_nonzero(self.w_true)

        if self.estimator == "lasso":
            solver_class = Lasso
        elif self.estimator == "enet":
            solver_class = ElasticNet
            self.alphaGrid *= 2.0
        elif self.estimator == "mcp":
            solver_class = MCPRegression
        elif self.estimator in ["ksn_optim", "ksn_0_1", "ksn_0_3", "ksn_0_6"]:
            solver_class = GeneralizedLinearEstimator
        else:
            raise ValueError(f"Unknown estimator {self.estimator}")

        w = np.zeros(self.X.shape[1])
        for alpha in self.alphaGrid:
            w_old = w
            if self.estimator == "ksn_optim":
                solver = solver_class(penalty=KSN(np.count_nonzero(self.w_true), alpha), datafit=Quadratic, solver=ProxGD)
            # elif self.estimator == "ksn_0_1":
            #     solver = solver_class(penalty=KSN(int(0.1 * self.X.shape[1]), alpha), datafit=Quadratic, solver=ProxGD)
            # elif self.estimator == "ksn_0_3":
            #     solver = solver_class(penalty=KSN(int(0.3 * self.X.shape[1]), alpha), datafit=Quadratic, solver=ProxGD)
            # elif self.estimator == "ksn_0_6":
            #     solver = solver_class(penalty=KSN(int(0.6 * self.X.shape[1]), alpha), datafit=Quadratic, solver=ProxGD)
            else:
                solver = solver_class(
                    alpha=alpha, max_iter=self.max_iter, fit_intercept=False
                )
            solver.fit(self.X, self.y)
            w = solver.coef_.flatten()
            if np.sum(w != 0) > k:
                w = w_old
                break

        if self.debiasing_step:
            if sum(w != 0) > 0:
                XX = self.X[:, w != 0]
                ww = lstsq(XX, self.y)
                ww = ww[0]
                w[w != 0] = ww

        self.w = w

    def get_result(self):
        return dict(w=self.w)
