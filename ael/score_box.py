import numpy as np
from scipy.special import log_ndtr

# These three methods (compute_log_D_1D_interval, compute_log_D_1D_complement_interval, compute_log_N_and_sign_1D_interval) are dimension-agnostic, in that there can be any number of desired batch dimensions, including broadcasting dimensions. The result will be whatever you would get by broadcasting subtraction across the batch dimension.


def compute_log_D_1D_interval(x, x0, x1, sigma):
    log_D1 = log_ndtr((x1 - x) / sigma)
    log_D2 = log_ndtr((x0 - x) / sigma)
    base_D = np.maximum(log_D1, log_D2)
    log_D = np.log(np.exp(log_D1 - base_D) - np.exp(log_D2 - base_D)) + base_D

    log_D1_alt = log_ndtr((x - x1) / sigma)
    log_D2_alt = log_ndtr((x - x0) / sigma)
    base_D_alt = np.maximum(log_D1_alt, log_D2_alt)
    log_D_alt = (
        np.log(np.exp(log_D2_alt - base_D_alt) - np.exp(log_D1_alt - base_D_alt))
        + base_D_alt
    )
    log_D__B = np.where(x > (x0 + x1) / 2, log_D, log_D_alt)

    return log_D__B


def compute_log_D_1D_complement_interval(x, x0, x1, sigma):
    log_D1 = log_ndtr((x0 - x) / sigma)
    log_D2 = log_ndtr((x - x1) / sigma)
    base_D = np.maximum(log_D1, log_D2)
    log_D_complement = (
        np.log(np.exp(log_D1 - base_D) + np.exp(log_D2 - base_D)) + base_D
    )
    return log_D_complement


def compute_log_N_and_sign_1D_interval(x, x0, x1, sigma):
    log_N_upper = -0.5 * (((x1 - x) / sigma) ** 2) - 0.5 * np.log(2 * np.pi)
    log_N_lower = -0.5 * (((x0 - x) / sigma) ** 2) - 0.5 * np.log(2 * np.pi)
    base_N = np.maximum(log_N_upper, log_N_lower)
    log_N = (
        np.log(np.abs(np.exp(log_N_upper - base_N) - np.exp(log_N_lower - base_N)))
        + base_N
    )
    sign = -np.sign(log_N_upper - log_N_lower)

    return log_N, sign


# In this case, the last dimension for x must be D, and the last two dimensions for box must be (D, 2), representing min/max in each dimension.
# The lowerbase b represents an 'abstract batch' and the return result will have a shape (b) where b is the *broadcasted* shape of the x and box batch dimensions.


def box_complement_log_denominator(x_b_D, box_b_2_D, sigma):
    # Draw a picture in the 2D case and use induction to extrapolate to higher dimensions.
    # Output shape will be (b, D)
    log_D_dim0__b = compute_log_D_1D_complement_interval(
        x_b_D[..., 0], box_b_2_D[..., 0, 0], box_b_2_D[..., 1, 0], sigma
    )
    if box_b_2_D.shape[-1] > 1:
        log_D_remaining_dims__b = box_complement_log_denominator(
            x_b_D[..., 1:], box_b_2_D[..., :, 1:], sigma
        ) + compute_log_D_1D_interval(
            x_b_D[..., 0], box_b_2_D[..., 0, 0], box_b_2_D[..., 1, 0], sigma
        )
        base = np.maximum(log_D_dim0__b, log_D_remaining_dims__b)
        log_D_total__b = (
            np.log(
                np.exp(log_D_dim0__b - base) + np.exp(log_D_remaining_dims__b - base)
            )
            + base
        )
        return log_D_total__b
    return log_D_dim0__b


def box_inclusion_score_and_likelihood(x_b_D, box_b_2_D, sigma):
    arrays = []
    log_D_per_dim_D_B = np.array(
        [
            compute_log_D_1D_interval(
                x_b_D[..., dim], box_b_2_D[..., 0, dim], box_b_2_D[..., 1, dim], sigma
            )
            for dim in range(x_b_D.shape[-1])
        ]
    )
    for dim in range(x_b_D.shape[-1]):
        log_N__b, sign_b = compute_log_N_and_sign_1D_interval(
            x_b_D[..., dim], box_b_2_D[..., 0, dim], box_b_2_D[..., 1, dim], sigma
        )
        arrays.append(np.exp(log_N__b - log_D_per_dim_D_B[dim]) * sign_b)
    return np.stack(arrays, axis=-1), log_D_per_dim_D_B.sum(axis=0)


def box_exclusion_score_and_likelihood(x_b_D, box_b_2_D, sigma):
    log_D__B = box_complement_log_denominator(x_b_D, box_b_2_D, sigma)
    log_D_per_dim_for_inclusion_D_B = np.array(
        [
            compute_log_D_1D_interval(
                x_b_D[..., dim], box_b_2_D[..., 0, dim], box_b_2_D[..., 1, dim], sigma
            )
            for dim in range(x_b_D.shape[-1])
        ]
    )
    log_D_including_box_B = log_D_per_dim_for_inclusion_D_B.sum(axis=0)
    score_per_dim = []
    for dim in range(x_b_D.shape[-1]):
        # this is signed for the 'inclusion' case. negate sign for exclusion.
        log_N__b, sign_b = compute_log_N_and_sign_1D_interval(
            x_b_D[..., dim], box_b_2_D[..., 0, dim], box_b_2_D[..., 1, dim], sigma
        )
        score_this_dim = (
            np.exp(
                log_N__b
                - log_D__B
                + (log_D_including_box_B - log_D_per_dim_for_inclusion_D_B[dim])
            )
            * -sign_b
        )
        score_per_dim.append(score_this_dim)
    return np.stack(score_per_dim, axis=-1), log_D__B
