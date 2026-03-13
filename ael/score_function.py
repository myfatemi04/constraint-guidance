from functools import lru_cache

import numpy as np

from ael.problem import Problem
from ael.score_box import box_exclusion_score_and_likelihood


def clip_magnitude(vector, max_magnitude):
    magnitude = np.linalg.norm(vector)
    if magnitude > max_magnitude:
        return vector / magnitude * max_magnitude
    else:
        return vector


# TODO: Remove in favor of the batched version.
def compute_agent_obstacle_score_unbatched(
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
    denominator_third_int = np.exp(-0.5 * (r2**2) / (sigma**2))

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
            -r
            * np.exp(-0.5 * (r**2) / (sigma**2))
            * intersection_eps_y
            / (np.pi * sigma**2)
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
    score = 1 / (sigma**2) * numerator / denominator

    return score


def compute_agent_obstacle_distance(
    agent_x_B: np.ndarray,
    agent_y_B: np.ndarray,
    obs_x_B: np.ndarray,
    obs_y_B: np.ndarray,
):
    d_a_o_B = np.sqrt((agent_x_B - obs_x_B) ** 2 + (agent_y_B - obs_y_B) ** 2)
    return d_a_o_B


def compute_r1_r2(obs_rad_B: np.ndarray, d_a_o_B: np.ndarray):
    temp0 = np.abs(d_a_o_B - obs_rad_B)
    temp1 = np.abs(d_a_o_B + obs_rad_B)
    r1_B = np.minimum(temp0, temp1)
    r2_B = np.maximum(temp0, temp1)
    return (r1_B, r2_B)


def compute_agent_obstacle_score_batched_helper(
    agent_x_B: np.ndarray,
    agent_y_B: np.ndarray,
    obs_x_B: np.ndarray,
    obs_y_B: np.ndarray,
    obs_rad_B: np.ndarray,
    sigma_B: np.ndarray,
    n_integral=10,
    denominator_threshold=0.01,
    numerator_threshold=0.01,
):
    """
    Here, 'B' indicates the batch dimension, 'D' indicates the spatial dimension, and 'T' indicates
    the *integral* dimension, not time.

    Values that may be precomputed to truncate the computation space (r1_B, r2_B, d_a_o_B) are passed in
    so that they do not need to be recomputed.
    """

    d_a_o_B = compute_agent_obstacle_distance(agent_x_B, agent_y_B, obs_x_B, obs_y_B)
    r1_B, r2_B = compute_r1_r2(obs_rad_B, d_a_o_B)

    # Batch on last dimension to make broadcasting easier.
    r_values_T_B = (
        np.linspace(0, 1, n_integral, endpoint=False)[:, None] * (r2_B - r1_B)[None, :]
        + r1_B[None, :]
    )
    dr_B = r_values_T_B[1, :] - r_values_T_B[0, :]
    r_values_T_B = r_values_T_B + dr_B / 2

    intersection_eps_x_T_B = -(obs_rad_B**2 - r_values_T_B**2 - d_a_o_B**2) / (
        2 * d_a_o_B
    )
    Theta_T_B = np.arccos(intersection_eps_x_T_B / r_values_T_B)

    denominator_first_int_B = 1 - np.exp(-0.5 * (r1_B**2) / (sigma_B**2))
    denominator_third_int_B = np.exp(-0.5 * (r2_B**2) / (sigma_B**2))
    denominator_first_int_B[d_a_o_B < obs_rad_B] = 0

    denominator_B = denominator_first_int_B + denominator_third_int_B
    denominator_mask = denominator_B < (1 - denominator_threshold)

    # Save compute by only computing denominator where it is needed.
    denominator_B[denominator_mask] += (
        r_values_T_B[:, denominator_mask]
        * (
            np.exp(
                -0.5
                * (r_values_T_B[:, denominator_mask] ** 2)
                / (sigma_B[denominator_mask] ** 2)
            )
            / (2 * np.pi * sigma_B[denominator_mask] ** 2)
        )
        * Theta_T_B[:, denominator_mask]
    ).sum(axis=0) * (2 * dr_B[denominator_mask])

    # Save compute by only computing numerator where it is needed.
    z1_B = 0.5 * (r1_B / sigma_B) ** 2
    # z2_B = 0.5 * (r2_B / sigma_B) ** 2
    numerator_bound_gt_1 = z1_B * np.exp(-z1_B)
    numerator_bound_lt_1 = 1 / np.e
    numerator_magnitude_bound_B = numerator_bound_gt_1 * (
        z1_B > 1
    ) + numerator_bound_lt_1 * (z1_B <= 1)
    numerator_magnitude_bound_B = (
        (r2_B - r1_B) * 2 / np.pi * numerator_magnitude_bound_B
    )
    numerator_mask = (
        numerator_magnitude_bound_B / (denominator_B + 1e-8)
    ) > numerator_threshold
    numerator_B = np.zeros_like(denominator_B)

    intersection_eps_y_T_B = np.sqrt(
        r_values_T_B[:, numerator_mask] ** 2
        - intersection_eps_x_T_B[:, numerator_mask] ** 2
    )
    numerator_integrand_T_B = (
        -r_values_T_B[:, numerator_mask]
        / (np.pi * sigma_B[numerator_mask] ** 2)
        * np.exp(
            -0.5
            * (r_values_T_B[:, numerator_mask] ** 2)
            / (sigma_B[numerator_mask] ** 2)
        )
        * intersection_eps_y_T_B
    )
    # Multiply by the prefactor and dr
    numerator_B[numerator_mask] = (
        numerator_integrand_T_B.sum(axis=0) * dr_B[numerator_mask]
    )

    # print("denominator proportion:", denominator_mask.sum() / denominator_B.shape[0])
    # print("numerator proportion:", numerator_mask.sum() / denominator_B.shape[0])

    # Multiplies by the component vector for epsilon'_x.
    numerator_D_B = (
        np.stack([obs_x_B - agent_x_B, obs_y_B - agent_y_B], axis=0)
        / d_a_o_B
        * numerator_B[None, :]
    )
    score_D_B = 1 / (sigma_B**2) * numerator_D_B / denominator_B

    return score_D_B.T


def compute_velocity_score_batched_helper(
    xy_T_B_D: np.ndarray, max_speed_B: np.ndarray, sigma_B: np.ndarray, n_integral=10
):
    # Here, S = 0 corresponds to the delta between T = 0 and T = 1.
    x_T_B = xy_T_B_D[..., 0]
    y_T_B = xy_T_B_D[..., 1]
    dx_S_B = x_T_B[1:, :] - x_T_B[:-1, :]
    dy_S_B = y_T_B[1:, :] - y_T_B[:-1, :]
    v2_S_B = dx_S_B**2 + dy_S_B**2
    v_S_B = np.sqrt(v2_S_B)

    temp0 = np.abs(v_S_B - max_speed_B)
    temp1 = np.abs(v_S_B + max_speed_B)
    r1_S_B = np.minimum(temp0, temp1)
    r2_S_B = np.maximum(temp0, temp1)

    r_values_N_S_B = np.linspace(0, 1, n_integral, endpoint=False)[:, None, None] * (
        r2_S_B - r1_S_B
    ) + (r1_S_B + 1e-8)

    dr_S_B = r_values_N_S_B[1, :] - r_values_N_S_B[0, :]
    r_values_N_S_B = r_values_N_S_B  # + dr_S_B / 2

    intersection_eps_x_S_B = -(max_speed_B**2 - r_values_N_S_B**2 - v_S_B**2) / (
        2 * v_S_B + 1e-6
    )

    theta_max_N_S_B = np.arccos(intersection_eps_x_S_B / r_values_N_S_B)

    # Select from r_values_N_S_B because it includes the dr offset.
    base_exp_arg_S_B = -0.5 * (r_values_N_S_B[0] / sigma_B) ** 2

    denominator_S_B = (
        # 2r
        (2 * r_values_N_S_B)
        # Gaussian PDF
        * (
            np.exp(-0.5 * (r_values_N_S_B / sigma_B) ** 2 - base_exp_arg_S_B)
            / (2 * np.pi * sigma_B**2)
        )
        # theta_{max}
        * theta_max_N_S_B
    ).sum(axis=0) * dr_S_B
    denominator_S_B += np.where(
        v_S_B < max_speed_B,
        (1 - np.exp(-r_values_N_S_B[0])) * np.exp(-base_exp_arg_S_B),
        0,
    )

    # print((-base_exp_arg_S_B, -r_values_N_S_B[0] - base_exp_arg_S_B))
    # print((np.exp(-base_exp_arg_S_B) - np.exp(-r_values_N_S_B[0] - base_exp_arg_S_B)))

    intersection_eps_y_S_B = np.sqrt(r_values_N_S_B**2 - intersection_eps_x_S_B**2)

    numerator_S_B = (
        # 2r^2 sin theta_{max}(r)
        (2 * r_values_N_S_B * intersection_eps_y_S_B)
        # Gaussian PDF
        * (
            np.exp(-0.5 * (r_values_N_S_B / sigma_B) ** 2 - base_exp_arg_S_B)
            / (2 * np.pi * sigma_B**2)
        )
    ).sum(axis=0) * dr_S_B

    # print(intersection_eps_y_S_B)
    # print(r_values_N_S_B)
    # exp = np.exp(-0.5 * (r_values_N_S_B / sigma_B) ** 2 - base_exp_arg_S_B)
    # exp_arg = -0.5 * (r_values_N_S_B / sigma_B) ** 2 - base_exp_arg_S_B
    # print(exp, exp_arg, r1_S_B, r_values_N_S_B)
    # print(numerator_S_B)
    # print(denominator_S_B)

    # Multiply by the component vector for epsilon'_x. D = |{x, y}| = 2
    numerator_S_B_D = (
        -np.stack([dx_S_B, dy_S_B], axis=-1)
        / (v_S_B[..., None] + 1e-6)
        * numerator_S_B[..., None]
    )
    score_S_B_D = (
        1 / (sigma_B[:, None] ** 2) * numerator_S_B_D / denominator_S_B[..., None]
    )

    # Assign to T coordinates via chain rule.
    score_T_B_D = np.zeros_like(xy_T_B_D)
    score_T_B_D[:-1, :, :] -= score_S_B_D
    score_T_B_D[1:, :, :] += score_S_B_D

    return score_T_B_D


@lru_cache(maxsize=16)
def get_kinetic_energy_kernel(N: int, sigma: float):
    velocity_kernel = np.zeros((N - 1, N))
    velocity_kernel[:, 1:] = np.eye(N - 1)
    velocity_kernel -= np.eye(N - 1, N)
    VTV = velocity_kernel.T @ velocity_kernel
    score_kernel = -np.linalg.inv(VTV + np.eye(N) / (sigma**2)) @ VTV / (sigma**2)
    return score_kernel


def compute_kinetic_energy_score(xy_T_B_D: np.ndarray, sigma):
    return np.einsum(
        "it,tad->iad", get_kinetic_energy_kernel(xy_T_B_D.shape[0], sigma), xy_T_B_D
    )


def compute_agent_circular_obstacle_score_from_problem(
    problem: Problem, trajectory: np.ndarray, sigma: float, n_integral: int
):
    """
    Returns scores (T, A, O, D) and (T, A1, A2, D) for agent-obstacle and agent-agent interactions respectively.
    This should be the main function used, because it offers the highest level of abstraction. Batching operations are handled internally.
    """

    agent_x_T_A = trajectory[:, :, 0]
    agent_y_T_A = trajectory[:, :, 1]
    obstacle_x_O = problem.circular_obstacle_positions[:, 0]
    obstacle_y_O = problem.circular_obstacle_positions[:, 1]
    obstacle_rad_O = problem.circular_obstacle_radii
    sigma_OA = sigma * np.ones(
        problem.circular_obstacle_radii.shape[0] * trajectory.shape[1]
    )

    agent_x_T_A_O = np.repeat(
        agent_x_T_A[:, :, None],
        problem.circular_obstacle_positions.shape[0],
        axis=2,
    )
    agent_y_T_A_O = np.repeat(
        agent_y_T_A[:, :, None],
        problem.circular_obstacle_positions.shape[0],
        axis=2,
    )
    obstacle_x_T_A_O = np.repeat(
        obstacle_x_O[None, None, :], trajectory.shape[0], axis=0
    )
    obstacle_x_T_A_O = np.repeat(obstacle_x_T_A_O, trajectory.shape[1], axis=1)
    obstacle_y_T_A_O = np.repeat(
        obstacle_y_O[None, None, :], trajectory.shape[0], axis=0
    )
    obstacle_y_T_A_O = np.repeat(obstacle_y_T_A_O, trajectory.shape[1], axis=1)
    obstacle_rad_T_A_O = np.repeat(
        (problem.agent_radii[:, None] + obstacle_rad_O[None, :])[None, :, :],
        trajectory.shape[0],
        axis=0,
    )
    sigma_T_A_O = np.repeat(sigma_OA[None, :], trajectory.shape[0], axis=0)

    T, A, O = agent_x_T_A_O.shape  # noqa: E741

    score = compute_agent_obstacle_score_batched_helper(
        agent_x_T_A_O.reshape(-1),
        agent_y_T_A_O.reshape(-1),
        obstacle_x_T_A_O.reshape(-1),
        obstacle_y_T_A_O.reshape(-1),
        obstacle_rad_T_A_O.reshape(-1),
        sigma_T_A_O.reshape(-1),
        n_integral=n_integral,
    ).reshape(T, A, O, 2)
    score[np.isnan(score)] = 0.0
    return score


def compute_agent_agent_score_from_problem(
    problem: Problem, trajectory: np.ndarray, sigma: float, n_integral: int
):
    agent_x_T_A = trajectory[:, :, 0]
    agent_y_T_A = trajectory[:, :, 1]
    agent_x_T_A1_A2 = np.repeat(agent_x_T_A[:, :, None], trajectory.shape[1], axis=2)
    agent_x_T_A2_A1 = np.repeat(agent_x_T_A[:, None, :], trajectory.shape[1], axis=1)
    agent_y_T_A1_A2 = np.repeat(agent_y_T_A[:, :, None], trajectory.shape[1], axis=2)
    agent_y_T_A2_A1 = np.repeat(agent_y_T_A[:, None, :], trajectory.shape[1], axis=1)
    sigma_T_A1_A2 = (
        np.sqrt(2)
        * sigma
        * np.ones((trajectory.shape[0], trajectory.shape[1], trajectory.shape[1]))
    )
    obstacle_rad_T_A1_A2 = np.repeat(
        (problem.agent_radii[:, None] + problem.agent_radii[None, :])[None, :, :],
        trajectory.shape[0],
        axis=0,
    )
    score_T_A1_A2_D = compute_agent_obstacle_score_batched_helper(
        agent_x_T_A1_A2.reshape(-1),
        agent_y_T_A1_A2.reshape(-1),
        agent_x_T_A2_A1.reshape(-1),
        agent_y_T_A2_A1.reshape(-1),
        obstacle_rad_T_A1_A2.reshape(-1),
        sigma_T_A1_A2.reshape(-1),
        n_integral=n_integral,
    ).reshape(trajectory.shape[0], trajectory.shape[1], trajectory.shape[1], 2)
    score_T_A1_A2_D[np.isnan(score_T_A1_A2_D)] = 0.0

    # Remove self-intersections.
    inds = np.arange(score_T_A1_A2_D.shape[1])
    score_T_A1_A2_D[:, inds, inds, :] = 0

    return score_T_A1_A2_D


def compute_agent_obstacle_score_rectangular_obstacles(
    xy_T_A_D, boxes_O_2_D, agent_radii, sigma
):
    T, A, D = xy_T_A_D.shape
    xy_T_A_1_D = xy_T_A_D.reshape(T, A, 1, D)
    boxes_A_O_2_D = np.repeat(boxes_O_2_D[None], A, axis=0)
    boxes_A_O_2_D[..., 0, :] -= 2 * agent_radii[:, None, None]
    boxes_A_O_2_D[..., 1, :] += 2 * agent_radii[:, None, None]
    score_T_A_O_D, likelihood_T_A_O = box_exclusion_score_and_likelihood(
        xy_T_A_1_D, boxes_A_O_2_D, sigma
    )
    score_T_A_D = score_T_A_O_D.sum(axis=-2)
    return score_T_A_D


def compute_score(
    xy_T_B_D,
    problem: Problem[np.ndarray],
    sigma,
    kinetic_weight,
    n_integral,
):
    """
    Batches across agents and obstacles.
    """

    score_T_B_D = np.zeros_like(xy_T_B_D)

    if problem.num_circular_obstacles > 0:
        score_T_B_D += compute_agent_circular_obstacle_score_from_problem(
            problem, xy_T_B_D, sigma, n_integral=n_integral
        ).sum(axis=2)

    if problem.num_axis_aligned_box_obstacles > 0:
        score_T_B_D += compute_agent_obstacle_score_rectangular_obstacles(
            xy_T_B_D,
            problem.axis_aligned_box_obstacle_bounds,
            problem.agent_radii,
            sigma,
        )

    score_T_B_D += compute_agent_agent_score_from_problem(
        problem, xy_T_B_D, sigma, n_integral=n_integral
    ).sum(axis=2)

    score_T_B_D += (
        compute_velocity_score_batched_helper(
            xy_T_B_D,
            problem.agent_max_speeds,
            np.ones(problem.num_agents) * sigma,
            n_integral,
        )
        * 0.1
    )

    score_T_B_D = score_T_B_D + kinetic_weight * compute_kinetic_energy_score(
        xy_T_B_D, sigma
    )

    return score_T_B_D
