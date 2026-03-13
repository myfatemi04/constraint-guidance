from dataclasses import dataclass

import numpy as np

from ael.constraint_evaluation import (
    compute_agent_agent_constraint_residuals,
    compute_agent_circular_obstacle_constraint_residuals,
    compute_constraint_residuals,
)
from ael.problem import Problem


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
        constraint_result.agent_circular_obstacle_constraint_residuals
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


def evaluate_trajectory_unscaled_probabilities_factorized(
    trajectory_T_A_D: np.ndarray,
    noise_B_T_A_D: np.ndarray,
    problem: Problem[np.ndarray],
    agent_agent_constraint_tolerance: float,
    agent_obstacle_constraint_tolerance: float,
    velocity_constraint_tolerance: float,
    use_velocity_baseline: bool,
    kinetic_weight: float,
) -> MPPITrajectoryEvaluation:
    """
    Estimates trajectory likelihoods by adding noise to each coordinate separately.
    The constraint residuals for noisy trajectories are first computed. These preserve
    temporal information about when the constraints were violated. Then, relative
    kinetic energy likelihoods for each point are computed according to the resulting
    delta. The normalization factor cancels out, which is why the delta alone is sufficient.
    """

    b = noise_B_T_A_D.shape[0]
    t = trajectory_T_A_D.shape[0]
    a = trajectory_T_A_D.shape[1]
    o = problem.num_circular_obstacles
    result = {
        "agent_agent": np.ones((b, t, a, a), dtype=np.float32),
        "agent_obstacle": np.ones((b, t, a, o), dtype=np.float32),
        "velocity": np.ones((b, t, a), dtype=np.float32),
        "kinetic_energy": np.ones((b, t, a), dtype=np.float32),
    }

    # (b, t, a, a)
    agent_agent_ok = (
        compute_agent_agent_constraint_residuals(
            problem, noise_B_T_A_D + trajectory_T_A_D
        )
        <= agent_agent_constraint_tolerance
    )
    # (b, t, a, o)
    agent_obstacle_ok = (
        compute_agent_circular_obstacle_constraint_residuals(
            problem, noise_B_T_A_D + trajectory_T_A_D
        )
        <= agent_obstacle_constraint_tolerance
    )

    velocity_squared_baseline = (
        (trajectory_T_A_D[1:] - trajectory_T_A_D[:-1]) ** 2
    ).sum(axis=-1)
    # Same endpoint, but adding noise to starting point. The start point doesn't matter because it's fixed
    # to the agent's current position.
    velocity_squared_per_deviation_reverse = (
        ((trajectory_T_A_D[1:] + noise_B_T_A_D[:, 1:]) - trajectory_T_A_D[:-1]) ** 2
    ).sum(axis=-1)
    velocity_squared_per_deviation_forward = (
        ((trajectory_T_A_D[1:]) - (trajectory_T_A_D[:-1] + noise_B_T_A_D[:, :-1])) ** 2
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

    result["agent_agent"] = agent_agent_ok.astype(np.float32)
    result["agent_obstacle"] = agent_obstacle_ok.astype(np.float32)

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
        -delta_kinetic_energy_per_deviation_reverse * kinetic_weight
    )
    result["kinetic_energy"][:, :-1] *= np.exp(
        -delta_kinetic_energy_per_deviation_forward * kinetic_weight
    )

    return MPPITrajectoryEvaluation(
        agent_agent=result["agent_agent"],
        agent_obstacle=result["agent_obstacle"],
        velocity=result["velocity"],
        kinetic_energy=result["kinetic_energy"],
    )


def compute_score_mppi_factorized(
    trajectory_T_A_D: np.ndarray,
    problem: Problem[np.ndarray],
    sigma: float,
    num_samples: int,
    kinetic_weight: float,
    **kwargs,
):
    noise_B_T_A_D = (
        np.random.normal(size=(num_samples, *trajectory_T_A_D.shape)) * sigma
    )
    evaluation = evaluate_trajectory_unscaled_probabilities_factorized(
        trajectory_T_A_D,
        noise_B_T_A_D,
        problem,
        agent_agent_constraint_tolerance=0,
        agent_obstacle_constraint_tolerance=0,
        velocity_constraint_tolerance=0,
        use_velocity_baseline=True,
        kinetic_weight=kinetic_weight,
    )
    # Compute scores for each factor.
    eps = 1e-8
    agent_agent_weights = evaluation.agent_agent / (
        np.sum(evaluation.agent_agent, axis=0) + eps
    )
    agent_obstacle_weights = evaluation.agent_obstacle / (
        np.sum(evaluation.agent_obstacle, axis=0) + eps
    )
    velocity_weights = evaluation.velocity / (np.sum(evaluation.velocity, axis=0) + eps)
    kinetic_energy_weights = evaluation.kinetic_energy / (
        np.sum(evaluation.kinetic_energy, axis=0) + eps
    )
    total_weights = (
        agent_agent_weights.sum(axis=-1)
        + agent_obstacle_weights.sum(axis=-1)
        + velocity_weights
        + kinetic_energy_weights
    )
    score = np.sum(noise_B_T_A_D * total_weights[:, :, :, None], axis=0)
    return score
