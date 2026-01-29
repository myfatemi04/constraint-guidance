"""
The main solving algorithm.
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ael.problem import Problem
from ael.score_function import compute_score


@dataclass
class ConstraintSatisfaction:
    agent_obstacle_constraint_residuals: np.ndarray
    """ Residuals for agent-obstacle nonpenetration constraints. If positive, represents the amount by which the signed distance into the obstacle exceeds zero. Shape (t, num_agents, num_obstacles). """

    agent_agent_constraint_residuals: np.ndarray
    """ Residuals for agent-agent nonpenetration constraints. If positive, represents the amount by which the signed distance into another agent exceeds zero. Shape (t, num_agents, num_agents). """

    velocity_constraint_residuals: np.ndarray
    """ Residuals for velocity constraints. If positive, represents the amount by which the velocity exceeds the maximum allowed velocity. Shape (t-1, num_agents). """


@dataclass
class Result:
    solve_time: float
    """ Time taken to solve the problem in seconds. """

    trajectories: list[np.ndarray]
    """ List of trajectories at each optimization step. """

    identifier: str | None
    """ An identifier for the problem that was solved, if available. """

    constraint_satisfaction: ConstraintSatisfaction


@dataclass
class OptimizerOptions:
    optimization: Literal["adam", "sgd"] = "adam"
    """ The optimization algorithm to use. Adam is strongly recommended for stability."""

    beta1: float = 0.9
    """ Adam optimizer $\\beta_1$ parameter. """

    beta2: float = 0.999
    """ Adam optimizer $\\beta_2$ parameter. """

    eps: float = 1e-8
    """ Adam optimizer $\\epsilon$ parameter. """

    magnitude_clip: float = 1.0
    """ Represents the maximum L2 norm for score predictions of individual obstacles. """


@dataclass
class ScheduleEntry:
    sigma: float
    """ Noise level for computing the convolved score. """

    kinetic_weight: float = 50
    """ How much to weight the kinetic energy score term relative to the obstacle score. The reason to set this high is the agent may go out of bounds to avoid obstacles otherwise, despite the inferior kinetic energy penalty. """

    step_size: float = 0.5
    """ The size of the optimizer step. """

    num_steps: int = 60
    """ The number of steps to operate under these parameters. """

    num_integral: int = 20
    """ The number of integration points to use when computing the nonpenetration score functions. """


DEFAULT_SCHEDULE = [
    ScheduleEntry(sigma=0.01, step_size=0.5, num_steps=60, kinetic_weight=50),
    ScheduleEntry(sigma=0.01, step_size=0.5, num_steps=60, kinetic_weight=10),
    ScheduleEntry(sigma=0.001, step_size=0.5, num_steps=60, kinetic_weight=1),
    ScheduleEntry(sigma=0.001, step_size=0.5, num_steps=60, kinetic_weight=0.01),
]


def compute_constraint_residuals(
    problem: Problem[np.ndarray], trajectories: np.ndarray
) -> ConstraintSatisfaction:
    """Return the constraint residuals for agent-obstacle and agent-agent distances, respectively."""

    ### Agent-agent nonpenetration constraints ###
    agent_pairwise_distances = (
        # (t, num_agents, 1, 2) - (t, 1, num_agents, 2) -> (t, num_agents, num_agents, 2)
        trajectories[:, :, np.newaxis, :] - trajectories[:, np.newaxis, :, :]
    )
    agent_pairwise_distances = np.linalg.norm(agent_pairwise_distances, axis=-1)
    min_acceptable_distances = (
        problem.agent_radii[:, np.newaxis] + problem.agent_radii[np.newaxis, :]
    )
    agent_agent_constraint_residuals = (
        min_acceptable_distances - agent_pairwise_distances
    )
    agent_agent_constraint_residuals[agent_agent_constraint_residuals < 0] = 0.0
    diag_indices = np.diag_indices(agent_agent_constraint_residuals.shape[1])
    agent_agent_constraint_residuals[:, diag_indices[0], diag_indices[1]] = 0.0
    assert (np.diagonal(agent_agent_constraint_residuals, axis1=1, axis2=2) == 0).all()

    ### Agent-obstacle nonpenetration constraints ###
    agent_obstacle_distances = (
        # (t, num_agents, 1, 2) - (1, num_obstacles, 2) -> (t, num_agents, num_obstacles, 2)
        trajectories[:, :, np.newaxis, :] - problem.obstacle_positions[np.newaxis, :, :]
    )
    agent_obstacle_distances = np.linalg.norm(agent_obstacle_distances, axis=-1)
    min_acceptable_agent_obstacle_distances = (
        # (num_agents, 1) + (1, num_obstacles) -> (num_agents, num_obstacles)
        problem.agent_radii[:, np.newaxis] + problem.obstacle_radii[np.newaxis, :]
    )
    agent_obstacle_constraint_residuals = (
        min_acceptable_agent_obstacle_distances - agent_obstacle_distances
    )
    agent_obstacle_constraint_residuals[agent_obstacle_constraint_residuals < 0] = 0.0

    ### Velocity constraints ###
    velocities = trajectories[1:] - trajectories[:-1]
    speeds = np.linalg.norm(velocities, axis=-1)
    velocity_constraint_residuals = speeds - problem.agent_max_speeds[np.newaxis, :]
    velocity_constraint_residuals[velocity_constraint_residuals < 0] = 0.0

    return ConstraintSatisfaction(
        agent_obstacle_constraint_residuals,
        agent_agent_constraint_residuals,
        velocity_constraint_residuals,
    )


def solve(
    problem: Problem,
    optimizer_options: OptimizerOptions = OptimizerOptions(),
    schedule: list[ScheduleEntry] = DEFAULT_SCHEDULE,
    initial_trajectory: np.ndarray | None = None,
    identifier: str | None = None,
) -> Result:
    # TODO: Initialize from prior distribution based on energy.
    trajectory = (
        np.random.randn(64, problem.num_agents, 2) * 0.5
        if initial_trajectory is None
        else initial_trajectory.copy()
    )
    start_positions = problem.agent_start_positions
    end_positions = problem.agent_end_positions
    trajectory[0] = start_positions
    trajectory[-1] = end_positions

    # Adam parameters.
    score_m: np.ndarray = np.zeros_like(trajectory)
    score_v: np.ndarray = np.zeros_like(trajectory)
    beta1_t = 1.0
    beta2_t = 1.0

    trajectories = []

    t0 = time.time()

    for schedule_entry in schedule:
        for i in range(schedule_entry.num_steps):
            score = compute_score(
                trajectory,
                sigma=schedule_entry.sigma,
                problem=problem,
                include_obstacles=True,
                n_integral=schedule_entry.num_integral,
                kinetic_weight=schedule_entry.kinetic_weight,
                magnitude_clip=optimizer_options.magnitude_clip,
            )
            beta1_t *= optimizer_options.beta1
            beta2_t *= optimizer_options.beta2

            match optimizer_options.optimization:
                case "sgd":
                    trajectory += schedule_entry.step_size * score
                case "adam":
                    score_m = (
                        optimizer_options.beta1 * score_m
                        + (1 - optimizer_options.beta1) * score
                    )
                    score_v = optimizer_options.beta2 * score_v + (
                        1 - optimizer_options.beta2
                    ) * (score**2)
                    score_m_hat = score_m / (1 - beta1_t)
                    score_v_hat = score_v / (1 - beta2_t)
                    trajectory += (
                        schedule_entry.step_size
                        * score_m_hat
                        / (np.sqrt(score_v_hat) + optimizer_options.eps)
                    )

            trajectory[0] = start_positions
            trajectory[-1] = end_positions
            trajectories.append(trajectory.copy())

    solve_time = time.time() - t0

    constraint_satisfaction = compute_constraint_residuals(problem, trajectory)

    return Result(
        solve_time=solve_time,
        trajectories=trajectories,
        identifier=identifier,
        constraint_satisfaction=constraint_satisfaction,
    )
