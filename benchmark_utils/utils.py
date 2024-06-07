import numpy as np
import sys
from numba import njit

EPS = sys.float_info.epsilon

# Similarly to the content of ksn.py, this code contains copy pasting from the code from here: https://cea-cosmic.github.io/ModOpt/_modules/modopt/opt/proximity.html#KSupportNorm in order to compile it in numba


@njit
def _ksncost(w, k, beta):
    data_abs = np.abs(w)
    ix = np.argsort(data_abs)[::-1]
    data_abs = data_abs[ix]  # Sorted absolute value of the data
    q_val = _find_q(data_abs, k)
    assert k != q_val
    cost_val = (
        (
            np.sum(data_abs[:q_val] ** 2)
            # np.sum(data_abs[:q_val] ** 2) * 0.5
            + np.sum(data_abs[q_val:]) ** 2
            / (k - q_val)
        ) * beta
    )

    return cost_val

@njit
def _hard_threshold(arr, k):
    top_k_indices = np.argpartition(np.abs(arr), -k)[-k:]
    thresholded_arr = np.zeros_like(arr)
    thresholded_arr[top_k_indices] = arr[top_k_indices]
    return thresholded_arr


@njit
def _compute_theta(beta, input_data, alpha):
        extra_factor=1.0
        alpha_input = alpha * np.expand_dims(input_data, 0)
        theta = np.zeros(alpha_input.shape)
        alpha_beta = alpha_input - beta * extra_factor
        theta = alpha_beta * ((alpha_beta <= 1) & (alpha_beta >= 0))
        theta_notnan = np.nan_to_num(theta)
        theta_notnan += (alpha_input > (beta * extra_factor + 1))
        return theta_notnan

@njit
def _interpolate(k, alpha0, alpha1, sum0, sum1):
        if sum0 == k:
            return alpha0

        elif sum1 == k:
            return alpha1

        slope = (sum1 - sum0) / (alpha1 - alpha0)
        b_val = sum0 - slope * alpha0

        return (k - b_val) / slope

@njit
def _binary_search(beta, k, input_data, alpha):
        extra_factor=1.0
        first_idx = 0
        data_abs = np.abs(input_data)
        last_idx = alpha.shape[0] - 1
        found = False
        prev_midpoint = 0
        cnt = 0  # Avoid infinite looops
        tolerance = 1e-4

        # Checking particular to be sure that the solution is in the array
        sum0 = _compute_theta(beta, data_abs, alpha[0]).sum()
        sum1 = _compute_theta(beta, data_abs, alpha[-1]).sum()

        if sum1 <= k:
            midpoint = alpha.shape[0] - 2
            found = True

        if sum0 >= k:
            found = True
            midpoint = 0

        while (first_idx <= last_idx) and not found and (cnt < alpha.shape[0]):

            midpoint = (first_idx + last_idx) // 2
            cnt += 1

            if prev_midpoint == midpoint:

                # Particular case
                sum0 = _compute_theta(
                    beta,
                    data_abs,
                    alpha[first_idx]
                ).sum()
                sum1 = _compute_theta(
                    beta,
                    data_abs,
                    alpha[last_idx]
                ).sum()

                if (np.abs(sum0 - k) <= tolerance):
                    found = True
                    midpoint = first_idx

                if (np.abs(sum1 - k) <= tolerance):
                    found = True
                    midpoint = last_idx - 1
                    # -1 because output is index such that
                    # `sum(theta(alpha[index])) <= k`

                if (first_idx - last_idx) in {1, 2}:
                    sum0 = _compute_theta(
                        beta,
                        data_abs,
                        alpha[first_idx]
                    ).sum()
                    sum1 = _compute_theta(
                        beta,
                        data_abs,
                        alpha[last_idx]
                    ).sum()

                    if (sum0 <= k) & (sum1 >= k):
                        found = True

            sum0 = _compute_theta(
                beta,
                data_abs,
                alpha[midpoint]
            ).sum()
            sum1 = _compute_theta(
                beta,
                data_abs,
                alpha[midpoint + 1]
            ).sum()

            if sum0 <= k <= sum1:
                found = True

            elif sum1 < k:
                first_idx = midpoint

            elif sum0 > k:
                last_idx = midpoint

            prev_midpoint = midpoint

        if found:
            return midpoint, alpha[midpoint], alpha[midpoint + 1], sum0, sum1
        
        else:
            raise ValueError(
                'Cannot find the coordinate of alpha (i) such '
                + 'that sum(theta(alpha[i])) =< k and '
                + 'sum(theta(alpha[i+1])) >= k ',
            )

@njit
def _find_alpha(beta, k, input_data):
    """Find alpha value to compute theta.

    This method aim at finding alpha such that sum(theta(alpha)) = k.

    Parameters
    ----------
    input_data: numpy.ndarray
        Input data
    extra_factor: float
        Potential extra factor for the weights (default is ``1.0``)

    Returns
    -------
    float
        An interpolation of alpha such that sum(theta(alpha)) = k

    """
    extra_factor=1.0
    data_size = input_data.shape[0]

    # Computes the alpha^i points line 1 in Algorithm 1.
    alpha = np.zeros((data_size * 2))
    data_abs = np.abs(input_data)
    alpha[:data_size] = (
        (beta * extra_factor)
        / (data_abs + EPS)
    )
    alpha[data_size:] = (
        (beta * extra_factor + 1)
        / (data_abs + EPS)
    )
    alpha = np.sort(np.unique(alpha))

    # Identify points alpha^i and alpha^{i+1} line 2. Algorithm 1
    useless_variable, alpha_midpoint, alpha_midpoint_p_1, sum0, sum1 = _binary_search(
        beta,
        k,
        input_data,
        alpha,
    )

    # Interpolate alpha^\star such that its sum is equal to k
    return _interpolate(k, alpha_midpoint, alpha_midpoint_p_1, sum0, sum1)

@njit
def _op_method(input_data, beta, k):
    # return 1.0
    extra_factor = 1.0
    data_shape = input_data.shape

    # Computes line 1., 2. and 3. in Algorithm 1
    # print('finding alpha')
    alpha = _find_alpha(beta, k, np.abs(input_data.flatten()))

    # Computes line 4. in Algorithm 1
    # print('computing theta')
    theta = _compute_theta(beta, np.abs(input_data.flatten()), alpha)

    # Computes line 5. in Algorithm 1.
    rslt = np.nan_to_num(
        (input_data.flatten() * theta)
        / (theta + beta * extra_factor),
    )
    # rslt = (input_data.flatten() * theta)/ (theta + beta * extra_factor)
    return rslt.reshape(data_shape)


@njit
def _find_q(sorted_data, k):
    first_idx = 0
    last_idx = k - 1
    found = False
    q_val = (first_idx + last_idx) // 2
    cnt = 0

    # Particular case
    if (sorted_data.sum() / k) >= sorted_data[0]:
        found = True
        q_val = 0

    elif (
        (sorted_data[k - 1:].sum())
        <= sorted_data[k - 1]
    ):
        found = True
        q_val = k - 1

    while (
        not found and not cnt == k
        and (first_idx <= last_idx < k)
    ):

        q_val = (first_idx + last_idx) // 2
        cnt += 1
        assert k != q_val
        l1_part = sorted_data[q_val:].sum() / (k - q_val)

        if (
            sorted_data[q_val + 1] <= l1_part <= sorted_data[q_val]
        ):
            found = True

        else:
            if sorted_data[q_val] <= l1_part:
                last_idx = q_val
            if l1_part <= sorted_data[q_val + 1]:
                first_idx = q_val

    return q_val



@njit
def prox_ksn(x, coef, k):
        if coef < 1:
            return _op_method(x/(1-coef), coef/(1 - coef), k)
        else: 
            return _hard_threshold(x, k)
        

@njit
def _prox_vec(w, step, penalty):
    return penalty.prox_vec(w, step)