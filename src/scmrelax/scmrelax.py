import numpy as np
import cvxpy as cp
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_X_y, check_array
from sklearn.base import BaseEstimator, RegressorMixin


class RelaxedBalancingCV(BaseEstimator, RegressorMixin):
    """Base class for relaxed-balancing estimators with cross-validation.

    Parameters
    ----------
    taus : array-like, optional
        Tuning parameters for CV. If None, automatically generated.

    n_taus: int, default=10
        Number of tuning parameters to generate.
        - If `taus` is None, this parameter is used to generate a grid of taus.
          The generated taus are logarithmically spaced between the lower and upper limits.
          The lower limit is the minimum value such that the problem is feasible,
          and the upper limit is the minimum value such that the equal weights are feasible.
        - The taus are generated in decreasing order.
        - Note that this is not necessarily the number of taus used in the cross-validation,
          as the taus are truncated to the minimum length across folds.

    cv : int, cross-validation generator or iterable, default=5
        Cross-validation strategy.

    nonneg : bool, default=True
        If True, the weights are constrained to be non-negative.

    Attributes
    ----------
    coef_ : array
        Coefficients of the fitted model.

    taus_ : array
        Effective tuning parameters used in the cross-validation.

    tau_ : float
        Best tuning parameter found by cross-validation.

    cv_mean_mse_ : array
        Mean cross-validated mean squared error for each tuning parameter.
    """

    _relaxation_min_obj = None     # Placeholder for the relaxation minimization objective
    _implicit_nonneg = False       # Placeholder for implicit non-negativity constraint

    # Params to tell MOSEK to solve the dual problem
    # Remark: cvxpy would dualize all continuous problems automatically.
    #         In some cases (though rarely), this can lead MOSEK to unable
    #         to solve the primal for entropy and log objectives.
    #         The reason is probably that the solution is quite close to
    #         the boundary of the simplex, making the objective value
    #         numerically unstable. In this case, we can force it to solve
    #         the dual (dual's dual, so actually primal) by setting this
    #         parameter. See https://www.cvxpy.org/tutorial/solvers/index.html
    #         for more details.
    _mosek_params = {'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'}

    def __init__(self, *, taus=None, n_taus=10, cv=5, nonneg=True):
        self.taus = taus
        self.n_taus = len(taus) if taus is not None else n_taus
        self.cv = cv
        self.nonneg = nonneg

    def _get_tau_lower_limit(self, X, y):
        """Calculate the lower limit for tau based on feasibility when simplex constraint."""
        T, J = X.shape
        w = cp.Variable(J)
        gam = cp.Variable()
        constraints = [cp.sum(w) == 1, w >= 0]
        objective = cp.Minimize(
            cp.pnorm(X.T @ (X @ w - y) / T + gam * np.ones(J), p='inf')
        )
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=False)
        return problem.value * 1.01     # 1% tolerance for numerical stability

    def _get_tau_upper_limit(self, X, y):
        """Calculate the upper limit for tau as the minimum where simple averages are feasible."""
        T, J = X.shape
        w = np.ones(J) / J
        gam = cp.Variable()
        objective = cp.Minimize(
            cp.pnorm(X.T @ (X @ w - y) / T + gam * np.ones(J), p='inf')
        )
        problem = cp.Problem(objective)
        problem.solve(solver=cp.MOSEK, verbose=False)
        return problem.value * 1.01     # 1% tolerance to make sure simple averages are feasible

    def _check_feas_for_taus(self, X, y):
        """Check feasibility for the given taus."""
        if self.nonneg:
            tau_max = np.max(self.taus)
            lower_limit = self._get_tau_lower_limit(X, y)
            if lower_limit <= tau_max:
                # Keep those taus that are larger than the lower limit
                # Sorted in decreasing order
                self.taus_ = np.sort(self.taus[self.taus >= lower_limit])[::-1]
            else:
                raise ValueError(
                    "Your taus cannot be smaller than the lower limit for feasibility."
                )
        else:
            # No positivity constraint, so no need to check feasibility
            pass

    def _generate_taus(self, X, y):
        """Generate a grid of taus based on the lower and upper limits."""
        # Even if no positive constraint, still use the limits for when there is
        # positive constraint for generating the taus
        lower_limit = self._get_tau_lower_limit(X, y)
        upper_limit = self._get_tau_upper_limit(X, y)
        # Generate taus in decreasing order
        self.taus_ = np.geomspace(upper_limit, lower_limit, self.n_taus)

    def _process_tau_grid(self, X, y):
        """Process the tau grid for cross-validation."""
        if self.taus is None:
            self._generate_taus(X, y)
        else:
            self._check_feas_for_taus(X, y)

    def _cross_validate(self, X, y):
        cv = check_cv(self.cv)
        fold_splits = list(cv.split(X))

        cv_errors = []
        for train_idx, test_idx in fold_splits:
            fold_errors = self._fit_fold(X, y, train_idx, test_idx)
            if fold_errors is None:
                continue
            cv_errors.append(fold_errors)

        if len(cv_errors) == 0:
            raise ValueError("No feasible taus for any fold. Please check your data and taus.")

        # Truncate cv_errors to the minimum length across folds
        # The truncation is conducted forward because the taus are sorted in decreasing order
        min_length = np.min([len(errors) for errors in cv_errors])
        cv_errors = np.array([errors[:min_length] for errors in cv_errors])
        # Keep the taus that are feasible for all folds
        self.taus_ = self.taus_[:min_length]

        self.cv_mean_mse_ = np.mean(cv_errors, axis=0)
        self.tau_ = self.taus_[np.argmin(self.cv_mean_mse_)]

    def _fit_fold(self, X, y, train_idx, test_idx):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        T, J = X_train.shape

        # If positive constraint, we need to check taus for this fold
        # If not, we can use the taus from the grid
        if self.nonneg:
            lower_limit = self._get_tau_lower_limit(X_train, y_train)
            taus_for_this_fold = self.taus_[self.taus_ >= lower_limit]
        else:
            taus_for_this_fold = self.taus_

        # Check if taus_for_this_fold is empty
        # If empty, do not proceed with the fold
        if len(taus_for_this_fold) == 0:
            return None

        w = cp.Variable(J)
        gam = cp.Variable()
        objective = cp.Minimize(cp.sum(self._relaxation_min_obj(w)))
        tau = cp.Parameter(nonneg=True)
        constraints = [
            cp.sum(w) == 1,
            cp.pnorm(X_train.T @ (X_train @ w - y_train) / T + gam * np.ones(J), p='inf') <= tau
        ]
        if self.nonneg and not self._implicit_nonneg:
            constraints.append(w >= 0)
        problem = cp.Problem(objective, constraints)

        w_estimates = []
        for val in taus_for_this_fold:
            tau.value = val
            try:
                problem.solve(solver=cp.MOSEK, verbose=False)
            except cp.error.SolverError:
                # If MOSEK fails, we can try forcing it to solve the dual
                # (which is actually the primal in this case)
                problem.solve(solver=cp.MOSEK, mosek_params=self._mosek_params, verbose=False)
            w_estimates.append(w.value)
        # shape (J, n_taus_for_this_fold)
        w_estimates = np.array(w_estimates).T

        # shape (T_test, n_taus_for_this_fold)
        y_pred = X_test @ w_estimates
        # shape (1, n_taus_for_this_fold)
        fold_errors = np.mean((y_test[:, None] - y_pred) ** 2, axis=0)

        return fold_errors

    def _fit_full_data(self, X, y):
        T, J = X.shape
        w = cp.Variable(J)
        gam = cp.Variable()
        objective = cp.Minimize(self._relaxation_min_obj(w))
        constraints = [
            cp.sum(w) == 1,
            cp.pnorm(X.T @ (X @ w - y) / T + gam * np.ones(J), p='inf') <= self.tau_
        ]
        if self.nonneg and not self._implicit_nonneg:
            constraints.append(w >= 0)
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
        except cp.error.SolverError:
            # If MOSEK fails, we can try forcing it to solve the dual
            # (which is actually the primal in this case)
            problem.solve(solver=cp.MOSEK, mosek_params=self._mosek_params, verbose=False)
        self.coef_ = w.value

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._process_tau_grid(X, y)
        self._cross_validate(X, y)
        self._fit_full_data(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        return X @ self.coef_


class L2RelaxationCV(RelaxedBalancingCV):
    """L2 Relaxation with cross-validation."""

    @staticmethod
    def _relaxation_min_obj(w):
        return cp.sum(cp.square(w))

    def __init__(self, *, taus=None, n_taus=10, cv=5, nonneg=True):
        super().__init__(taus=taus, n_taus=n_taus, cv=cv, nonneg=nonneg)


class EntropyRelaxationCV(RelaxedBalancingCV):
    """Entropy Relaxation with cross-validation."""

    _implicit_nonneg = True

    @staticmethod
    def _relaxation_min_obj(w):
        return cp.sum(-cp.entr(w))

    def __init__(self, *, taus=None, n_taus=10, cv=5):
        super().__init__(taus=taus, n_taus=n_taus, cv=cv, nonneg=True)


class ELRelaxationCV(RelaxedBalancingCV):
    """EL Relaxation with cross-validation."""

    _implicit_nonneg = True

    @staticmethod
    def _relaxation_min_obj(w):
        return cp.sum(-cp.log(w))

    def __init__(self, *, taus=None, n_taus=10, cv=5):
        super().__init__(taus=taus, n_taus=n_taus, cv=cv, nonneg=True)

def synthetic_control(X, y):
    """
    Implement the synthetic control method using convex optimization.
    
    Args:
        X: Control unit data
        y: Target unit data
    
    Returns:
        Optimal weights for control units
    """
    n, k = X.shape
    w = cp.Variable(k)
    objective = cp.Minimize(cp.norm(y-X@w, 2))
    constraints = [cp.sum(w)==1, w>=0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)
    return w.value

## Weight Estimation Using Different Methods
def fit(X_pre, y_pre, X):
    """
    Estimate weights using different relaxation methods.

    Parameters
    ----------
    X_pre : ndarray of shape (n_pre_samples, n_features)
        Pre-treatment input data matrix.

    y_pre : ndarray of shape (n_pre_samples,)
        Pre-treatment target values.

    X : ndarray of shape (n_samples, n_features)
        Post-treatment input data matrix.

    Returns
    -------
    dict
        Dictionary containing the estimated weights and predictions for each method:
        - 'scm': Synthetic control method.
        - 'EL': Exponentially Logarithmic relaxation.
        - 'entropy': Entropy relaxation.
        - 'l2': L2 relaxation.
    """
    results = {}
    
    # SCM weights
    w_scm = synthetic_control(X_pre, y_pre)
    results['scm'] = {
        'weights': w_scm,
        'predictions': X @ w_scm
    }
    
    # Log relaxation weights
    model_EL = ELRelaxationCV(cv=3, n_taus=10).fit(X_pre, y_pre)
    results['EL'] = {
        'weights': model_EL.coef_,
        'predictions': model_EL.predict(X)
    }
    
    # Entropy relaxation weights
    model_entropy = EntropyRelaxationCV(cv=3, n_taus=10).fit(X_pre, y_pre)
    results['entropy'] = {
        'weights': model_entropy.coef_,
        'predictions': model_entropy.predict(X)
    }
    
    # L2 relaxation weights
    model_l2 = L2RelaxationCV(cv=3, n_taus=10, nonneg=True).fit(X_pre, y_pre)
    results['l2'] = {
        'weights': model_l2.coef_,
        'predictions': model_l2.predict(X)
    }
    
    return results