"""
The main solving algorithm.
"""

import enum
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger

from ael.constraint_evaluation import (
    ConstraintSatisfaction,
    compute_constraint_residuals,
)
from ael.geometry import compute_obstacle_boundaries
from ael.initial_paths import get_initial_paths_by_agent
from ael.problem import Problem
from ael.score_function import (
    compute_score,
    compute_score_from_boundary_integrals,
    compute_score_mppi_factorized,
    compute_score_mppi_unfactorized,
)


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
    kind: Literal["adam", "sgd"] = "adam"
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

    step_size: float = 0.5
    """ The size of the optimizer step. """

    num_steps: int = 60
    """ The number of steps to operate under these parameters. """

    score_fn_kwargs: dict | None = None
    """ Any keyword arguments to pass to the score function. """


class ScoreComputationMethod(str, enum.Enum):
    APPROXIMATE_V0 = "approximate_v0"
    UNFACTORIZED_MPPI = "unfactorized_mppi"
    FACTORIZED_MPPI = "factorized_mppi"
    BOUNDARY_INTEGRALS = "boundary_integrals"
    VORONOI_GUIDANCE = "voronoi_guidance"
    NONE_BASELINE = "none_baseline"


STEPS = 500
DEFAULT_SCHEDULE_APPROXIMATE_V0 = [
    ScheduleEntry(
        sigma=0.1 * (0.01 / 0.1) ** (i / STEPS),
        step_size=0.03 * (0.01 / 0.03) ** (i / STEPS),
        num_steps=1,
        score_fn_kwargs=dict(
            kinetic_weight=10 * (1 / 10) ** (i / STEPS), n_integral=10
        ),
    )
    for i in range(STEPS)
]

DEFAULT_SCHEDULE_BOUNDARY_INTEGRALS = [
    ScheduleEntry(sigma=sigma, step_size=0.1, num_steps=200)
    # for sigma in [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
    for sigma in [0.01, 0.005]
]


DEFAULT_SCHEDULES: dict[ScoreComputationMethod, list[ScheduleEntry]] = {
    ScoreComputationMethod.APPROXIMATE_V0: DEFAULT_SCHEDULE_APPROXIMATE_V0,
    ScoreComputationMethod.UNFACTORIZED_MPPI: DEFAULT_SCHEDULE_APPROXIMATE_V0,
    ScoreComputationMethod.FACTORIZED_MPPI: [
        ScheduleEntry(
            **(
                s.__dict__
                | {"score_fn_kwargs": (s.score_fn_kwargs or {}) | {"num_samples": 256}}
            )
        )
        for s in DEFAULT_SCHEDULE_APPROXIMATE_V0
    ],
    ScoreComputationMethod.BOUNDARY_INTEGRALS: DEFAULT_SCHEDULE_BOUNDARY_INTEGRALS,
    ScoreComputationMethod.VORONOI_GUIDANCE: DEFAULT_SCHEDULE_APPROXIMATE_V0,
    ScoreComputationMethod.NONE_BASELINE: [],  # uses Voronoi as-is
}


def solve(
    problem: Problem,
    score_computation_method: ScoreComputationMethod,
    optimizer_options: OptimizerOptions = OptimizerOptions(),
    schedule: list[ScheduleEntry] | None = None,
    initial_paths: list[np.ndarray] | None = None,
) -> Result:
    t0 = time.time()
    if schedule is None:
        schedule = DEFAULT_SCHEDULES[score_computation_method]

    # TODO: Initialize from prior distribution based on energy.
    start_positions = problem.agent_start_positions
    end_positions = problem.agent_end_positions

    trajectory = np.linspace(start_positions, end_positions, num=64, axis=0)

    trajectory[0] = start_positions
    trajectory[-1] = end_positions

    # Adam parameters.
    score_m: np.ndarray = np.zeros_like(trajectory)
    score_v: np.ndarray = np.zeros_like(trajectory)
    beta1_t = 1.0
    beta2_t = 1.0

    if score_computation_method in [
        ScoreComputationMethod.VORONOI_GUIDANCE,
        ScoreComputationMethod.BOUNDARY_INTEGRALS,
    ]:
        obstacle_boundaries = compute_obstacle_boundaries(problem)

    # initialize trajectory to the first path for each agent
    if initial_paths is not None:
        for agent_index in range(problem.num_agents):
            trajectory[:, agent_index, :] = initial_paths[agent_index]
    else:
        target_paths_by_agent = get_initial_paths_by_agent(problem, dt=1.0)
        for agent_index in range(problem.num_agents):
            trajectory[:, agent_index, :] = target_paths_by_agent[agent_index]
    # target_paths_by_agent = None

    trajectories = [trajectory.copy()]

    for schedule_entry in schedule:
        for step in range(schedule_entry.num_steps):
            match score_computation_method:
                case ScoreComputationMethod.APPROXIMATE_V0:
                    score = compute_score(
                        trajectory,
                        sigma=schedule_entry.sigma,
                        problem=problem,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )
                case ScoreComputationMethod.UNFACTORIZED_MPPI:
                    score = compute_score_mppi_unfactorized(
                        trajectory,
                        problem=problem,
                        sigma=schedule_entry.sigma,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )
                case ScoreComputationMethod.FACTORIZED_MPPI:
                    score = compute_score_mppi_factorized(
                        trajectory,
                        problem=problem,
                        sigma=schedule_entry.sigma,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )
                case ScoreComputationMethod.BOUNDARY_INTEGRALS:
                    score = compute_score_from_boundary_integrals(
                        trajectory,
                        problem=problem,
                        sigma=schedule_entry.sigma,
                        obstacle_boundaries=obstacle_boundaries,
                        include_kinetic=True,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )
                case ScoreComputationMethod.VORONOI_GUIDANCE:
                    score = compute_score(
                        trajectory,
                        problem=problem,
                        sigma=schedule_entry.sigma,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )
                    # add Voronoi guidance
                    assert target_paths_by_agent is not None
                    for agent_index in range(problem.num_agents):
                        target_paths = target_paths_by_agent[agent_index]
                        logprobs = []
                        for path in target_paths:
                            # compute distance from current trajectory to path
                            l2 = np.linalg.norm(trajectory[:, agent_index, :] - path)
                            logprobs.append(
                                -(l2**2)
                                / (2.0 * schedule_entry.sigma**2 * trajectory.shape[0])
                            )
                        logprobs = np.array(logprobs)
                        # compute softmax over target trajectories
                        max_logprob = np.max(logprobs)
                        weights = np.exp(logprobs - max_logprob)
                        weights /= np.sum(weights)
                        # compute weighted average of paths
                        weighted_path = np.zeros_like(trajectory[:, agent_index, :])
                        for weight, path in zip(weights, target_paths):
                            weighted_path += weight * path
                        # compute score to move towards weighted path
                        score[:, agent_index, :] += (
                            (weighted_path - trajectory[:, agent_index, :])
                            * 0.01
                            / (schedule_entry.sigma**2 + 0.1**2)
                        )

            if np.isnan(score).any():
                print("nan in score", step)
                exit()

            beta1_t *= optimizer_options.beta1
            beta2_t *= optimizer_options.beta2

            match optimizer_options.kind:
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
                case "langevin":
                    ss = schedule_entry.step_size
                    score_norm = np.linalg.norm(score, axis=-1, keepdims=True)
                    score_multiplier = np.minimum(1, 100 / score_norm)
                    score = score * score_multiplier
                    trajectory += ss * score + np.sqrt(ss) * np.random.randn(
                        *score.shape
                    )

            trajectory[0] = start_positions
            trajectory[-1] = end_positions
            trajectories.append(trajectory.copy())

            if np.isnan(trajectory).any():
                logger.warning("Trajectory contains NaN values, stopping optimization.")
                break

    solve_time = time.time() - t0

    constraint_satisfaction = compute_constraint_residuals(problem, trajectory)

    return Result(
        solve_time=solve_time,
        trajectories=trajectories,
        identifier=problem.identifier,
        constraint_satisfaction=constraint_satisfaction,
    )
