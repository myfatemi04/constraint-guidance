import numpy as np


# TODO: Remove in favor of the batched version.
def compute_agent_obstacle_score(
    agent_x,
    agent_y,
    obs_x,
    obs_y,
    obs_rad,
    sigma,
    n_integral=10,
    numerator_magnitude_skip_cutoff=-1,
    denominator_computation_cutoff=2,
):
    """
    Allows computing the score (displacement direction) on an agent at (agent_x, agent_y)
    due to an obstacle at (obs_x, obs_y) with radius obs_rad. The score is computed using numerical integration with n_integral
    points.

    The numerator of the integral is known to be bounded above in magnitude, which limits the potential impact on the final result.
    If the upper bound is below numerator_magnitude_skip_cutoff, then the score is returned as zero.

    Additionally, if the denominator is below denominator_computation_cutoff, then a more accurate computation of the denominator is performed. However, given that the denominator is less than or equal to 1, and the integrand is positive, if e.g. the denominator
    is known to be very close to 1 already, then the more accurate computation is not necessary. The error in the denominator integral
    is bounded above by (1 - denominator).
    """

    d_a_o = np.sqrt((agent_x - obs_x) ** 2 + (agent_y - obs_y) ** 2)
    R = np.array(
        [
            [obs_x - agent_x, agent_y - obs_y],
            [obs_y - agent_y, obs_x - agent_x],
        ]
    )
    R = R / d_a_o

    # Compute numerator.
    r1 = min(abs(d_a_o - obs_rad), abs(d_a_o + obs_rad))
    r2 = max(abs(d_a_o - obs_rad), abs(d_a_o + obs_rad))
    r_values = np.linspace(r1, r2, n_integral, endpoint=False)
    dr = r_values[1] - r_values[0]
    r_values = r_values + dr / 2

    denominator_first_int = 1 - np.exp(-0.5 * (r1**2) / (sigma**2))
    denominator_third_int = -np.exp(-0.5 * (r2**2) / (sigma**2))

    if d_a_o < obs_rad:
        denominator_first_int = 0

    denominator = denominator_first_int + denominator_third_int
    numerator = 0.0
    numerator_magnitude_upper_bound = (
        2
        * (r2 - r1)
        * np.exp(-0.5 * (r1**2) / (sigma**2))
        * (r1**2)
        / (2 * np.pi * sigma**2)
    )
    if numerator_magnitude_upper_bound < numerator_magnitude_skip_cutoff:
        return np.array([0.0, 0.0])

    for r in r_values:
        intersection_eps_x = -(obs_rad**2 - r**2 - d_a_o**2) / (2 * d_a_o)

        if intersection_eps_x**2 > r**2:
            raise ValueError(f"Invalid intersection computation. {obs_rad} {r} {d_a_o}")

        intersection_eps_y = np.sqrt(r**2 - intersection_eps_x**2)
        numerator_integrand = (
            r
            * np.exp(-0.5 * (r**2) / (sigma**2))
            * (np.pi - intersection_eps_y)
            / (2 * np.pi * sigma**2)
        )
        numerator += numerator_integrand * dr

        Theta = np.arccos(intersection_eps_x / r)

        if denominator < denominator_computation_cutoff:
            denominator += (
                2
                * r
                * (np.exp(-0.5 * (r**2) / (sigma**2)) / (2 * np.pi * sigma**2))
                * dr
                * Theta
            )

    numerator = R[:, 0] * numerator
    score = -1 / (sigma**2) * numerator / denominator

    return score


def compute_agent_obstacle_distance_batched(
    agent_x_B: np.ndarray,
    agent_y_B: np.ndarray,
    obs_x_B: np.ndarray,
    obs_y_B: np.ndarray,
):
    d_a_o_B = np.sqrt((agent_x_B - obs_x_B) ** 2 + (agent_y_B - obs_y_B) ** 2)
    return d_a_o_B


def compute_r1_r2_batched(obs_rad_B: np.ndarray, d_a_o_B: np.ndarray):
    temp0 = np.abs(d_a_o_B - obs_rad_B)
    temp1 = np.abs(d_a_o_B + obs_rad_B)
    r1_B = np.minimum(temp0, temp1)
    r2_B = np.maximum(temp0, temp1)
    return (r1_B, r2_B)


def compute_agent_obstacle_score_batched(
    agent_x_B: np.ndarray,
    agent_y_B: np.ndarray,
    obs_x_B: np.ndarray,
    obs_y_B: np.ndarray,
    obs_rad_B: np.ndarray,
    sigma_B: np.ndarray,
    r1_B: np.ndarray,
    r2_B: np.ndarray,
    d_a_o_B: np.ndarray,
    n_integral=10,
):
    """
    Here, 'B' indicates the batch dimension, 'D' indicates the spatial dimension, and 'T' indicates
    the *integral* dimension, not time.

    Values that may be precomputed to truncate the computation space (r1_B, r2_B, d_a_o_B) are passed in
    so that they do not need to be recomputed.
    """

    # Batch on last dimension to make broadcasting easier.
    r_values_T_B = (
        np.linspace(0, 1, n_integral, endpoint=False)[:, None] * (r2_B - r1_B)[None, :]
        + r1_B[None, :]
    )
    dr_B = r_values_T_B[1, :] - r_values_T_B[0, :]
    r_values_T_B = r_values_T_B + dr_B / 2

    denominator_first_int_B = 1 - np.exp(-0.5 * (r1_B**2) / (sigma_B**2))
    denominator_third_int_B = -np.exp(-0.5 * (r2_B**2) / (sigma_B**2))
    denominator_first_int_B[d_a_o_B < obs_rad_B] = 0

    denominator_B = denominator_first_int_B + denominator_third_int_B
    numerator_B = np.zeros_like(denominator_B)

    intersection_eps_x_T_B = -(obs_rad_B**2 - r_values_T_B**2 - d_a_o_B**2) / (
        2 * d_a_o_B
    )
    intersection_eps_y_T_B = np.sqrt(r_values_T_B**2 - intersection_eps_x_T_B**2)
    numerator_integrand_T_B = (
        r_values_T_B
        * np.exp(-0.5 * (r_values_T_B**2) / (sigma_B**2))
        * (np.pi - intersection_eps_y_T_B)
    )
    numerator_B += numerator_integrand_T_B.sum(axis=0) * (
        dr_B / (2 * np.pi * sigma_B**2)
    )

    Theta_T_B = np.arccos(intersection_eps_x_T_B / r_values_T_B)

    denominator_B += (
        r_values_T_B
        * (np.exp(-0.5 * (r_values_T_B**2) / (sigma_B**2)) / (2 * np.pi * sigma_B**2))
        * Theta_T_B
    ).sum(axis=0) * (2 * dr_B)

    # Avoid creating R array.
    numerator_D_B = (
        np.stack([obs_x_B - agent_x_B, obs_y_B - agent_y_B], axis=0)
        / d_a_o_B
        * numerator_B[None, :]
    )
    score_D_B = -1 / (sigma_B**2) * numerator_D_B / denominator_B

    return score_D_B.T
