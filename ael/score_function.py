from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ael.constraint_evaluation import (
    compute_agent_agent_constraint_residuals,
    compute_agent_obstacle_constraint_residuals,
    compute_constraint_residuals,
)
from ael.problem import Problem


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


def compute_agent_obstacle_score_from_problem(
    problem: Problem, trajectory: np.ndarray, sigma: float, n_integral: int
):
    """
    Returns scores (T, A, O, D) and (T, A1, A2, D) for agent-obstacle and agent-agent interactions respectively.
    This should be the main function used, because it offers the highest level of abstraction. Batching operations are handled internally.
    """

    # Create batch for agent-obstacle interactions.
    agent_x_T_A = trajectory[:, :, 0]
    agent_y_T_A = trajectory[:, :, 1]
    obstacle_x_O = problem.obstacle_positions[:, 0]
    obstacle_y_O = problem.obstacle_positions[:, 1]
    obstacle_rad_O = problem.obstacle_radii
    sigma_OA = sigma * np.ones(problem.obstacle_radii.shape[0] * trajectory.shape[1])

    agent_x_T_A_O = np.repeat(
        agent_x_T_A[:, :, None], problem.obstacle_positions.shape[0], axis=2
    )
    agent_y_T_A_O = np.repeat(
        agent_y_T_A[:, :, None], problem.obstacle_positions.shape[0], axis=2
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

    agent_x_flat = np.concatenate(
        [agent_x_T_A_O.reshape(-1), agent_x_T_A1_A2.reshape(-1)]
    )
    agent_y_flat = np.concatenate(
        [agent_y_T_A_O.reshape(-1), agent_y_T_A1_A2.reshape(-1)]
    )
    obstacle_x_flat = np.concatenate(
        [obstacle_x_T_A_O.reshape(-1), agent_x_T_A2_A1.reshape(-1)]
    )
    obstacle_y_flat = np.concatenate(
        [obstacle_y_T_A_O.reshape(-1), agent_y_T_A2_A1.reshape(-1)]
    )
    obstacle_rad_flat = np.concatenate(
        [obstacle_rad_T_A_O.reshape(-1), obstacle_rad_T_A1_A2.reshape(-1)]
    )
    sigma_flat = np.concatenate([sigma_T_A_O.reshape(-1), sigma_T_A1_A2.reshape(-1)])
    score_flat = compute_agent_obstacle_score_batched_helper(
        agent_x_flat,
        agent_y_flat,
        obstacle_x_flat,
        obstacle_y_flat,
        obstacle_rad_flat,
        sigma_flat,
        n_integral=n_integral,
    )

    # clip norm
    # norm = np.linalg.norm(score_flat, axis=-1, keepdims=True)
    # norm_clipped = np.clip(norm, 0, 1.0)
    # score_flat = score_flat * (norm_clipped / (1e-8 + norm))
    score_flat[np.isnan(score_flat)] = 0.0

    # unpack scores.
    score_T_A_O_D = score_flat[: T * A * O, :].reshape(
        trajectory.shape[0],
        trajectory.shape[1],
        problem.obstacle_positions.shape[0],
        2,
    )
    score_T_A1_A2_D = score_flat[T * A * O :, :].reshape(
        trajectory.shape[0], trajectory.shape[1], trajectory.shape[1], 2
    )
    # Remove self-intersections.
    inds = np.arange(score_T_A1_A2_D.shape[1])
    score_T_A1_A2_D[:, inds, inds, :] = 0

    return score_T_A_O_D, score_T_A1_A2_D


def compute_score(
    xy_T_B_D,
    problem: Problem[np.ndarray],
    sigma,
    include_obstacles,
    kinetic_weight,
    magnitude_clip,
    n_integral,
):
    """
    Batches across agents and obstacles.
    """

    score_T_B_D = np.zeros_like(xy_T_B_D)

    if include_obstacles:
        # Don't compute self-interactions.
        score_T_A_O_D, score_T_A1_A2_D = compute_agent_obstacle_score_from_problem(
            problem, xy_T_B_D, sigma, n_integral=n_integral
        )
        score_T_B_D += score_T_A1_A2_D.sum(axis=2) + score_T_A_O_D.sum(axis=2)

    score_T_B_D += compute_velocity_score_batched_helper(
        xy_T_B_D,
        problem.agent_max_speeds,
        np.ones(problem.num_agents) * sigma,
        n_integral,
    )

    score_T_B_D = score_T_B_D + kinetic_weight * compute_kinetic_energy_score(
        xy_T_B_D, sigma
    )

    return score_T_B_D


def evaluate_trajectory_unscaled_probabilities(
    trajectory_batch: np.ndarray,
    problem: Problem[np.ndarray],
    agent_agent_constraint_tolerance: float,
    agent_obstacle_constraint_tolerance: float,
    velocity_constraint_tolerance: float,
):
    """Computes an unbiased estimate of trajectory unscaled probabilities."""
    constraint_result = compute_constraint_residuals(problem, trajectory_batch)
    agent_agent_ok = ~(
        constraint_result.agent_agent_constraint_residuals
        > agent_agent_constraint_tolerance
    ).reshape(trajectory_batch.shape[0], -1).any(axis=-1)
    agent_obstacle_ok = ~(
        constraint_result.agent_obstacle_constraint_residuals
        > agent_obstacle_constraint_tolerance
    ).reshape(trajectory_batch.shape[0], -1).any(axis=-1)
    velocity_ok = ~(
        constraint_result.velocity_constraint_residuals > velocity_constraint_tolerance
    ).reshape(trajectory_batch.shape[0], -1).any(axis=-1)
    ok = agent_agent_ok & agent_obstacle_ok & velocity_ok

    # Evaluate kinetic energy.
    # (b, t-1, a)
    velocities_squared = (
        (trajectory_batch[:, 1:] - trajectory_batch[:, :-1]) ** 2
    ).sum(axis=-1)
    kinetic_energies = 0.5 * velocities_squared.reshape(
        trajectory_batch.shape[0], -1
    ).sum(axis=-1)
    # Compute kinetic energy likelihood. Stabilize by subtracting mean of K trajectories.
    kinetic_energies = kinetic_energies - (
        kinetic_energies[ok].mean() if ok.any() else 0.0
    )
    unscaled_probabilities = ok.astype(np.float32) * np.exp(-kinetic_energies)
    return unscaled_probabilities


def compute_score_mppi_unfactorized(
    trajectory: np.ndarray,
    problem: Problem[np.ndarray],
    sigma: float,
    num_samples: int,
    agent_agent_constraint_tolerance: float,
    agent_obstacle_constraint_tolerance: float,
    velocity_constraint_tolerance: float,
):
    trajectory_batch = trajectory[np.newaxis, ...].repeat(repeats=num_samples, axis=0)  # ty:ignore[no-matching-overload]
    noise = np.random.normal(size=trajectory_batch.shape)
    trajectory_weights = evaluate_trajectory_unscaled_probabilities(
        trajectory_batch + noise * sigma,
        problem,
        agent_agent_constraint_tolerance,
        agent_obstacle_constraint_tolerance,
        velocity_constraint_tolerance,
    )
    score = (
        np.sum(noise * trajectory_weights[:, None, None, None], axis=0)
        / (np.sum(trajectory_weights) + 1e-8)
        / (sigma**2)
    )
    return score


@dataclass
class MPPITrajectoryEvaluation:
    """
    Keys correspond to different factors that affect the overall unscaled probability. Each array
    has shape (b, t, a), where b is the number of noise samples, t is the number of time steps, and
    a is the number of agents, representing the unscaled probabilities per time step and agent.
    Separating into these factors allows for more detailed analysis of which components are contributing
    to the overall likelihood.
    """

    agent_agent: np.ndarray
    agent_obstacle: np.ndarray
    velocity: np.ndarray
    kinetic_energy: np.ndarray
    overall: np.ndarray


def evaluate_trajectory_unscaled_probabilities_factorized(
    trajectory: np.ndarray,
    noise_batch: np.ndarray,
    problem: Problem[np.ndarray],
    agent_agent_constraint_tolerance: float,
    agent_obstacle_constraint_tolerance: float,
    velocity_constraint_tolerance: float,
    use_velocity_baseline: bool = False,
    kinetic_energy_lambda: float = 5,
) -> MPPITrajectoryEvaluation:
    """
    Estimates trajectory likelihoods by adding noise to each coordinate separately.
    The constraint residuals for noisy trajectories are first computed. These preserve
    temporal information about when the constraints were violated. Then, relative
    kinetic energy likelihoods for each point are computed according to the resulting
    delta. The normalization factor cancels out, which is why the delta alone is sufficient.
    """

    b = noise_batch.shape[0]
    t = trajectory.shape[0]
    a = trajectory.shape[1]
    result = {
        "agent_agent": np.ones((b, t, a), dtype=np.float32),
        "agent_obstacle": np.ones((b, t, a), dtype=np.float32),
        "velocity": np.ones((b, t, a), dtype=np.float32),
        "kinetic_energy": np.ones((b, t, a), dtype=np.float32),
        "overall": np.ones((b, t, a), dtype=np.float32),
    }

    # (b, t, a, a)
    agent_agent_ok = (
        compute_agent_agent_constraint_residuals(problem, noise_batch + trajectory)
        <= agent_agent_constraint_tolerance
    )
    # (b, t, a, o)
    agent_obstacle_ok = (
        compute_agent_obstacle_constraint_residuals(problem, noise_batch + trajectory)
        <= agent_obstacle_constraint_tolerance
    )

    velocity_squared_baseline = ((trajectory[1:] - trajectory[:-1]) ** 2).sum(axis=-1)
    # Same endpoint, but adding noise to starting point. The start point doesn't matter because it's fixed
    # to the agent's current position.
    velocity_squared_per_deviation_reverse = (
        ((trajectory[1:] + noise_batch[:, 1:]) - trajectory[:-1]) ** 2
    ).sum(axis=-1)
    velocity_squared_per_deviation_forward = (
        ((trajectory[1:]) - (trajectory[:-1] + noise_batch[:, :-1])) ** 2
    ).sum(axis=-1)

    # (b, t, a)
    velocity_constraint_residual_baseline = (
        np.sqrt(velocity_squared_baseline) - problem.agent_max_speeds
    )
    velocity_constraint_residual_per_deviation_reverse = (
        np.sqrt(velocity_squared_per_deviation_reverse) - problem.agent_max_speeds
    )
    velocity_constraint_residual_per_deviation_forward = (
        np.sqrt(velocity_squared_per_deviation_forward) - problem.agent_max_speeds
    )

    result["agent_agent"] *= agent_agent_ok.all(axis=-1).astype(np.float32)
    result["agent_obstacle"] *= agent_obstacle_ok.all(axis=-1).astype(np.float32)

    velocity_ok_reverse = (
        velocity_constraint_residual_per_deviation_reverse
        < velocity_constraint_tolerance
    )
    velocity_ok_forward = (
        velocity_constraint_residual_per_deviation_forward
        < velocity_constraint_tolerance
    )
    velocity_ok_baseline = (
        velocity_constraint_residual_baseline < velocity_constraint_tolerance
    )

    if use_velocity_baseline:
        # When using the 'baseline', don't disregard trajectories for which the original case was invalid.
        velocity_ok_forward = velocity_ok_forward | (
            ~velocity_ok_forward & ~velocity_ok_baseline
        )
        velocity_ok_reverse = velocity_ok_reverse | (
            ~velocity_ok_reverse & ~velocity_ok_baseline
        )

    # Only apply velocity constraint effects to points after the starting point.
    result["velocity"][:, 1:] *= velocity_ok_forward.astype(np.float32)
    result["velocity"][:, :-1] *= velocity_ok_reverse.astype(np.float32)

    # Evaluate kinetic energy change.
    delta_kinetic_energy_per_deviation_reverse = 0.5 * (
        velocity_squared_per_deviation_reverse - velocity_squared_baseline
    )
    delta_kinetic_energy_per_deviation_reverse -= np.mean(
        delta_kinetic_energy_per_deviation_reverse
    )
    delta_kinetic_energy_per_deviation_reverse /= (
        np.std(delta_kinetic_energy_per_deviation_reverse) + 1e-8
    )
    delta_kinetic_energy_per_deviation_forward = 0.5 * (
        velocity_squared_per_deviation_forward - velocity_squared_baseline
    )
    delta_kinetic_energy_per_deviation_forward -= np.mean(
        delta_kinetic_energy_per_deviation_forward
    )
    delta_kinetic_energy_per_deviation_forward /= (
        np.std(delta_kinetic_energy_per_deviation_forward) + 1e-8
    )

    result["kinetic_energy"][:, 1:] *= np.exp(
        -delta_kinetic_energy_per_deviation_reverse * kinetic_energy_lambda
    )
    result["kinetic_energy"][:, :-1] *= np.exp(
        -delta_kinetic_energy_per_deviation_forward * kinetic_energy_lambda
    )

    return MPPITrajectoryEvaluation(
        agent_agent=result["agent_agent"],
        agent_obstacle=result["agent_obstacle"],
        velocity=result["velocity"],
        kinetic_energy=result["kinetic_energy"],
        overall=(
            result["kinetic_energy"] * result["agent_agent"] * result["agent_obstacle"]
            # * result["velocity"]
        ),
    )


def compute_score_mppi_factorized(
    trajectory: np.ndarray,
    problem: Problem[np.ndarray],
    sigma: float,
    num_samples: int,
    agent_agent_constraint_tolerance: float,
    agent_obstacle_constraint_tolerance: float,
    velocity_constraint_tolerance: float,
):
    noise = np.random.normal(size=(num_samples, *trajectory.shape)) * sigma
    evaluation = evaluate_trajectory_unscaled_probabilities_factorized(
        trajectory,
        noise,
        problem,
        agent_agent_constraint_tolerance,
        agent_obstacle_constraint_tolerance,
        velocity_constraint_tolerance,
    )
    eps = 1e-8
    weights = evaluation.overall / (
        # divide over batch dimension
        np.sum(evaluation.overall, axis=0) + eps
    )
    # acceptance = evaluation.overall > 0
    # print(acceptance.sum() / np.prod(evaluation.overall.shape))
    # score = 1 / sigma**2 * np.sum(noise * weights[:, :, :, None], axis=0)
    score = np.sum(noise * weights[:, :, :, None], axis=0)
    return score


ARC_CENTER_X_DIM = 0
ARC_CENTER_Y_DIM = 1
ARC_RADIUS_DIM = 2
ARC_THETA1_DIM = 3
ARC_THETA2_DIM = 4
ARC_SIGN_DIM = 5

DISK_CENTER_X_DIM = 0
DISK_CENTER_Y_DIM = 1
DISK_RADIUS_DIM = 2
DISK_SIGN_DIM = 3


def compute_feasibility_score_numerator(
    xy: np.ndarray, sigma: float, boundaries: np.ndarray, n_points: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    An improved approach for computing the score using line integrals.

    Args:
    - xy: (N, 2) array of points at which to evaluate the score.
    - sigma: stddev for Gaussian.
    - surfaces: (N, S, 6) array of (center_x, center_y, radius, theta1, theta2, sign). The first dimension may be 1 if the surfaces are shared across all inputs.

    The sign indicates whether it is an "exclusion" or "inclusion" constraint. These constraint
    surfaces can be created by computing the intersection points between each of the obstacles.
    Here, it's assumed that the constraint surfaces are piecewise arcs, because the feasible set
    is created through the intersection of disks, which have boundaries which are arcs. The
    denominator is estimated by sampling several points around the current location and checking
    for feasibility. For now, if it's found to be too low, the numerator will be normalized to
    have a magnitude of 1.
    """

    # (N, S). S is the number of surfaces.
    center_x = boundaries[:, :, ARC_CENTER_X_DIM]
    center_y = boundaries[:, :, ARC_CENTER_Y_DIM]
    radius = boundaries[:, :, ARC_RADIUS_DIM]
    theta1 = boundaries[:, :, ARC_THETA1_DIM]
    theta2 = boundaries[:, :, ARC_THETA2_DIM]
    sign = boundaries[:, :, ARC_SIGN_DIM]

    # (N, T, S). T is the number of points sampled along each surface.
    theta = np.linspace(theta1, theta2, num=n_points, endpoint=False, axis=1)
    n_x = np.cos(theta)
    n_y = np.sin(theta)
    # (N, S)
    dtheta = theta[:, 1, :] - theta[:, 0, :]
    # (N, T, S) <- (N, :, S) + (N, :, S) * (N, T, S)
    x = center_x[:, np.newaxis, :] + radius[:, np.newaxis, :] * n_x
    y = center_y[:, np.newaxis, :] + radius[:, np.newaxis, :] * n_y
    # (N, S) <- (N, S) * (N, S)
    ds = dtheta * radius

    # (N, T, S) <- (N, T, S) - (N, :, :). N is the number of points at which the score is being evaluated.
    delta_x = x - xy[:, 0, np.newaxis, np.newaxis]
    delta_y = y - xy[:, 1, np.newaxis, np.newaxis]
    exp_arg = -0.5 * (delta_x**2 + delta_y**2) / (sigma**2)
    log_scaler = exp_arg.max(axis=-1).max(axis=-1)
    exp_arg_adjusted = exp_arg - log_scaler[:, np.newaxis, np.newaxis]

    gaussian_pdf = np.exp(exp_arg_adjusted) / (2 * np.pi * sigma**2)

    # (N, T, S) <- (N, :, S) * (N, T, S) * (N, :, S) * (N, T, S)
    integrand_x = sign[:, np.newaxis, :] * gaussian_pdf * ds[:, np.newaxis, :] * n_x
    integrand_y = sign[:, np.newaxis, :] * gaussian_pdf * ds[:, np.newaxis, :] * n_y

    # (N,) <- sum over T and S of (N, T, S)
    integral_x = np.sum(np.sum(integrand_x, axis=-1), axis=-1)
    integral_y = np.sum(np.sum(integrand_y, axis=-1), axis=-1)

    """
    delta_min can be used to compare with the denominator.
    If the denominator is large, the numerator can essentially be discarded.
    If the denominator is small, the delta_min can be discarded and the numerator
    can be normalized to have a magnitude of 1, since the direction is still
    informative.
    """

    return np.stack([integral_x, integral_y], axis=-1), log_scaler


def compute_feasibility_score_denominator(
    xy: np.ndarray, sigma: float, disks: np.ndarray, n_samples: int = 100
):
    """
    Computes feasibility by sampling around the current point and checking whether the results
    lie inside the disks or not. The disks are (center_x, center_y, radius, sign) tuples. There
    is no theta 1 or theta 2 in this case.

    xy: (N, 2) - N is the number of points for which the score is being evaluated.
    """

    # (S,). S is the number of disks.
    center_x = disks[:, 0]
    center_y = disks[:, 1]
    radius = disks[:, 2]
    sign = disks[:, 3]

    # (B, N, 2). B is the number of samples, N is the number of points, 2 is for x and y.
    xy_noise = (
        xy[np.newaxis, :, :] + np.random.normal(size=(n_samples, *xy.shape)) * sigma
    )
    # (B, N, S) <- (B, N, :) - (S)
    xy_distances = np.sqrt(
        (xy_noise[:, :, np.newaxis, 0] - center_x) ** 2
        + (xy_noise[:, :, np.newaxis, 1] - center_y) ** 2
    )

    # (B, N, S) <- (B, N, S) compared to (S)
    inside = xy_distances <= radius
    outside = xy_distances >= radius

    # (B, N) <- all(axis=-1) of (B, N, S) <- (B, N, S) & (S)
    ok = np.all(outside & (sign == 1) | inside & (sign == -1), axis=-1)
    # (N,) <- mean(axis=0) of (B, N)
    ok_fraction = ok.mean(axis=0)

    return ok_fraction


def compute_score_from_boundary_integrals(
    trajectory: np.ndarray,
    problem: Problem[np.ndarray],
    sigma: float,
    obstacle_boundaries: np.ndarray,
    include_kinetic: bool,
):
    score = np.zeros_like(trajectory)
    for agent_i in range(trajectory.shape[1]):
        num_other_agents = trajectory.shape[1] - 1

        # Initialize surfaces with obstacles.
        boundaries = np.zeros(
            (trajectory.shape[0], obstacle_boundaries.shape[0] + num_other_agents, 6)
        )
        boundaries[:, : obstacle_boundaries.shape[0], :] = obstacle_boundaries
        boundaries[:, : obstacle_boundaries.shape[0], ARC_RADIUS_DIM] += (
            problem.agent_radii[agent_i]
        )

        # Initialize disks with obstacles. We must select indices carefully here because we're translating between two array formats.
        disks = np.zeros(
            (trajectory.shape[0], obstacle_boundaries.shape[0] + num_other_agents, 4)
        )
        disks[
            :,
            : obstacle_boundaries.shape[0],
            [DISK_CENTER_X_DIM, DISK_CENTER_Y_DIM, DISK_RADIUS_DIM, DISK_SIGN_DIM],
        ] = obstacle_boundaries[
            :, [ARC_CENTER_X_DIM, ARC_CENTER_Y_DIM, ARC_RADIUS_DIM, ARC_SIGN_DIM]
        ]
        disks[:, : obstacle_boundaries.shape[0], DISK_RADIUS_DIM] += (
            problem.agent_radii[agent_i]
        )

        # Include other agents as surfaces and disks.
        target_indices = []
        source_indices = []
        index = obstacle_boundaries.shape[0]
        for other_agent_i in range(trajectory.shape[1]):
            if other_agent_i == agent_i:
                continue

            source_indices.append(other_agent_i)
            target_indices.append(index)
            index += 1

        # Create new surfaces for agent-agent interactions.
        # These are circles around the other agents with radius equal to the sum of the agent radii.
        other_agent_trajectories = trajectory[:, source_indices, :]
        r = problem.agent_radii[agent_i] + problem.agent_radii[source_indices]
        boundaries[:, target_indices, ARC_CENTER_X_DIM] = other_agent_trajectories[
            :, :, 0
        ]
        boundaries[:, target_indices, ARC_CENTER_Y_DIM] = other_agent_trajectories[
            :, :, 1
        ]
        boundaries[:, target_indices, ARC_RADIUS_DIM] = r
        boundaries[:, target_indices, ARC_THETA1_DIM] = 0
        boundaries[:, target_indices, ARC_THETA2_DIM] = 2 * np.pi
        boundaries[:, target_indices, ARC_SIGN_DIM] = 1.0

        disks[:, target_indices, DISK_CENTER_X_DIM] = other_agent_trajectories[:, :, 0]
        disks[:, target_indices, DISK_CENTER_Y_DIM] = other_agent_trajectories[:, :, 1]
        disks[:, target_indices, DISK_RADIUS_DIM] = r
        disks[:, target_indices, DISK_SIGN_DIM] = 1.0

        xy = trajectory[:, agent_i, :]

        eps = 1e-12
        numer, log_numer_scaler = compute_feasibility_score_numerator(
            xy, sigma, boundaries
        )
        denom = compute_feasibility_score_denominator(xy, sigma, disks)

        numer_scaler = np.maximum(np.exp(log_numer_scaler), eps * (denom < 0.5))
        numer_scaler = np.exp(log_numer_scaler)

        obs_feas_score = (
            numer * numer_scaler[:, np.newaxis] / (denom[:, np.newaxis] + eps)
        )
        obs_feas_score_mag = np.linalg.norm(obs_feas_score, axis=-1, keepdims=True)
        obs_feas_score = obs_feas_score / np.maximum(obs_feas_score_mag, 1.0)

        # Object feasibility component.
        score[:, agent_i, :] = obs_feas_score

        # Kinetic energy component.
        if include_kinetic:
            mu = (trajectory[2:, agent_i, :] + trajectory[:-2, agent_i, :]) / 2
            score[1:-1, agent_i, :] += (
                1 / (1 + sigma**2) * (mu - trajectory[1:-1, agent_i, :])
            ) * 0.5

    return score
