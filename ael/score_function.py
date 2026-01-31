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
    numerator_mask = (numerator_magnitude_bound_B / denominator_B) > numerator_threshold
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


@lru_cache(maxsize=16)
def get_velocity_kernel(N: int, sigma: float):
    velocity_kernel = np.zeros((N - 1, N))
    velocity_kernel[:, 1:] = np.eye(N - 1)
    velocity_kernel -= np.eye(N - 1, N)
    VTV = velocity_kernel.T @ velocity_kernel
    score_kernel = -np.linalg.inv(VTV + np.eye(N) / (sigma**2)) @ VTV / (sigma**2)
    return score_kernel


def compute_kinetic_energy_score(trajectory: np.ndarray, sigma):
    return np.einsum(
        "it,tad->iad", get_velocity_kernel(trajectory.shape[0], sigma), trajectory
    )


# TODO: Remove.
def compute_score_unbatched(
    trajectory,
    problem: Problem[np.ndarray],
    sigma,
    include_obstacles=True,
    kinetic_weight=10.0,
    magnitude_clip=1.0,
):
    score = np.zeros_like(trajectory)

    if include_obstacles:
        for t in range(trajectory.shape[0]):
            for agent in range(trajectory.shape[1]):
                agent_x = trajectory[t, agent, 0]
                agent_y = trajectory[t, agent, 1]

                for obs_idx in range(problem.obstacle_positions.shape[0]):
                    obs_x = problem.obstacle_positions[obs_idx, 0]
                    obs_y = problem.obstacle_positions[obs_idx, 1]
                    obs_rad = problem.obstacle_radii[obs_idx]

                    score[t, agent] += clip_magnitude(
                        compute_agent_obstacle_score_unbatched(
                            agent_x,
                            agent_y,
                            obs_x,
                            obs_y,
                            obs_rad + problem.agent_radii[agent],
                            sigma,
                        ),
                        magnitude_clip,
                    )

                for other_agent in range(trajectory.shape[1]):
                    if other_agent == agent:
                        continue
                    other_agent_x = trajectory[t, other_agent, 0]
                    other_agent_y = trajectory[t, other_agent, 1]

                    score[t, agent] += clip_magnitude(
                        compute_agent_obstacle_score_unbatched(
                            agent_x,
                            agent_y,
                            other_agent_x,
                            other_agent_y,
                            problem.agent_radii[agent]
                            + problem.agent_radii[other_agent],
                            np.sqrt(2) * sigma,  # sum the variances
                        ),
                        magnitude_clip,
                    )

    score = score + kinetic_weight * compute_kinetic_energy_score(trajectory, sigma)

    return score


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
    norm = np.linalg.norm(score_flat, axis=-1, keepdims=True)
    norm_clipped = np.clip(norm, 0, 1.0)
    score_flat = score_flat * (norm_clipped / (1e-8 + norm))
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
    trajectory,
    problem: Problem[np.ndarray],
    sigma,
    include_obstacles=True,
    kinetic_weight=10.0,
    magnitude_clip=1.0,
    n_integral=200,
):
    """
    Batches across agents and obstacles.
    """

    score = np.zeros_like(trajectory)

    if include_obstacles:
        # Don't compute self-interactions.
        score_T_A_O_D, score_T_A1_A2_D = compute_agent_obstacle_score_from_problem(
            problem, trajectory, sigma, n_integral=n_integral
        )
        score += score_T_A1_A2_D.sum(axis=2) + score_T_A_O_D.sum(axis=2)

    score = score + kinetic_weight * compute_kinetic_energy_score(trajectory, sigma)

    return score


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


def compute_score_mppi(
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
    velocity_squared_per_deviation = (
        ((trajectory[1:] + noise_batch[:, 1:]) - trajectory[:-1]) ** 2
    ).sum(axis=-1)

    # (b, t, a)
    velocity_constraint_residual_baseline = (
        np.sqrt(velocity_squared_baseline) - problem.agent_max_speeds
    )
    velocity_constraint_residual_per_deviation = (
        np.sqrt(velocity_squared_per_deviation) - problem.agent_max_speeds
    )

    result["agent_agent"] *= agent_agent_ok.all(axis=-1).astype(np.float32)
    result["agent_obstacle"] *= agent_obstacle_ok.all(axis=-1).astype(np.float32)

    if use_velocity_baseline:
        velocity_ok = (
            velocity_constraint_residual_per_deviation < velocity_constraint_tolerance
        ) < (velocity_constraint_residual_baseline < velocity_constraint_tolerance)
    else:
        velocity_ok = (
            velocity_constraint_residual_per_deviation < velocity_constraint_tolerance
        )

    # Only apply velocity constraint effects to points after the starting point.
    result["velocity"][:, 1:] *= velocity_ok.astype(np.float32)

    # Evaluate kinetic energy change.
    delta_kinetic_energy_per_deviation = 0.5 * (
        velocity_squared_per_deviation - velocity_squared_baseline
    )
    delta_kinetic_energy_per_deviation -= np.mean(delta_kinetic_energy_per_deviation)
    delta_kinetic_energy_per_deviation /= (
        np.std(delta_kinetic_energy_per_deviation) + 1e-8
    )
    result["kinetic_energy"][:, 1:] *= np.exp(-delta_kinetic_energy_per_deviation)

    return MPPITrajectoryEvaluation(
        agent_agent=result["agent_agent"],
        agent_obstacle=result["agent_obstacle"],
        velocity=result["velocity"],
        kinetic_energy=result["kinetic_energy"],
        overall=(
            result["agent_agent"]
            * result["agent_obstacle"]
            * result["velocity"]
            * result["kinetic_energy"]
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
    trajectory_batch = trajectory[np.newaxis, ...].repeat(repeats=num_samples, axis=0)  # ty:ignore[no-matching-overload]
    noise = np.random.normal(size=trajectory_batch.shape) * sigma
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
        np.sum(evaluation.overall, axis=1, keepdims=True) + eps
    )
    score = 1 / sigma**2 * np.sum(noise * weights[:, :, :, None], axis=0)
    return score
