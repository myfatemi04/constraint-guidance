"""
The main solving algorithm.
"""

import enum
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ael.constraint_evaluation import (
    ConstraintSatisfaction,
    compute_constraint_residuals,
)
from ael.geometry import compute_obstacle_boundaries
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


DEFAULT_SCHEDULE_APPROXIMATE_V0 = [
    ScheduleEntry(
        sigma=1.0, step_size=0.5, num_steps=60, score_fn_kwargs=dict(kinetic_weight=50)
    ),
    ScheduleEntry(
        sigma=0.1, step_size=0.5, num_steps=60, score_fn_kwargs=dict(kinetic_weight=50)
    ),
    ScheduleEntry(
        sigma=0.01, step_size=0.5, num_steps=60, score_fn_kwargs=dict(kinetic_weight=50)
    ),
    ScheduleEntry(
        sigma=0.01, step_size=0.5, num_steps=60, score_fn_kwargs=dict(kinetic_weight=10)
    ),
    ScheduleEntry(
        sigma=0.001, step_size=0.5, num_steps=60, score_fn_kwargs=dict(kinetic_weight=1)
    ),
    ScheduleEntry(
        sigma=0.001,
        step_size=0.5,
        num_steps=60,
        score_fn_kwargs=dict(kinetic_weight=0.2),
    ),
]

DEFAULT_SCHEDULE_MPPI = [
    ScheduleEntry(
        sigma=sigma,
        step_size=1.0,
        num_steps=20,
        score_fn_kwargs=dict(
            agent_agent_constraint_tolerance=0.0,
            agent_obstacle_constraint_tolerance=0.0,
            velocity_constraint_tolerance=0.0,
        ),
    )
    for sigma in [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
]

DEFAULT_SCHEDULE_BOUNDARY_INTEGRALS = [
    ScheduleEntry(sigma=sigma, step_size=0.1, num_steps=200)
    # for sigma in [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
    for sigma in [0.01, 0.005]
]


DEFAULT_SCHEDULES: dict[ScoreComputationMethod, list[ScheduleEntry]] = {
    ScoreComputationMethod.APPROXIMATE_V0: DEFAULT_SCHEDULE_APPROXIMATE_V0,
    ScoreComputationMethod.UNFACTORIZED_MPPI: DEFAULT_SCHEDULE_MPPI,
    ScoreComputationMethod.FACTORIZED_MPPI: DEFAULT_SCHEDULE_MPPI,
    ScoreComputationMethod.BOUNDARY_INTEGRALS: DEFAULT_SCHEDULE_BOUNDARY_INTEGRALS,
}


def solve(
    problem: Problem,
    score_computation_method: ScoreComputationMethod,
    optimizer_options: OptimizerOptions = OptimizerOptions(),
    schedule: list[ScheduleEntry] | None = None,
    initial_trajectory: np.ndarray | None = None,
    identifier: str | None = None,
) -> Result:
    if schedule is None:
        schedule = DEFAULT_SCHEDULES[score_computation_method]

    if score_computation_method in [
        ScoreComputationMethod.UNFACTORIZED_MPPI,
        ScoreComputationMethod.FACTORIZED_MPPI,
    ]:
        # step_size_ok = all(s.step_size == 1 for s in schedule)
        step_size_ok = True
        assert optimizer_options.kind == "sgd" and step_size_ok, (
            "MPPI computations require SGD with a step size of 1 for true equivalence."
        )

    # TODO: Initialize from prior distribution based on energy.
    start_positions = problem.agent_start_positions
    end_positions = problem.agent_end_positions

    # start_positions = np.zeros((problem.num_agents, 2))
    # end_positions = np.zeros((problem.num_agents, 2))
    # start_positions[:, 0] = -1.0
    # end_positions[:, 0] = -1.0
    # end_positions[:, 1] = 1.0

    trajectory = np.linspace(start_positions, end_positions, num=64, axis=0)
    # trajectory += np.random.randn(*trajectory.shape) * 0.1

    trajectory[0] = start_positions
    trajectory[-1] = end_positions

    # Adam parameters.
    score_m: np.ndarray = np.zeros_like(trajectory)
    score_v: np.ndarray = np.zeros_like(trajectory)
    beta1_t = 1.0
    beta2_t = 1.0

    obstacle_boundaries = compute_obstacle_boundaries(problem)

    trajectories = []

    t0 = time.time()

    for schedule_entry in schedule:
        print(schedule_entry)

        for i in range(schedule_entry.num_steps):
            match score_computation_method:
                case ScoreComputationMethod.APPROXIMATE_V0:
                    score = compute_score(
                        trajectory,
                        sigma=schedule_entry.sigma,
                        problem=problem,
                        include_obstacles=True,
                        magnitude_clip=optimizer_options.magnitude_clip,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )
                case ScoreComputationMethod.UNFACTORIZED_MPPI:
                    score = compute_score_mppi_unfactorized(
                        trajectory,
                        problem=problem,
                        sigma=schedule_entry.sigma,
                        num_samples=100,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )
                case ScoreComputationMethod.FACTORIZED_MPPI:
                    score = compute_score_mppi_factorized(
                        trajectory,
                        problem=problem,
                        sigma=schedule_entry.sigma,
                        num_samples=100,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )
                case ScoreComputationMethod.BOUNDARY_INTEGRALS:
                    score = compute_score_from_boundary_integrals(
                        trajectory,
                        problem=problem,
                        sigma=schedule_entry.sigma,
                        obstacle_boundaries=obstacle_boundaries,
                        **(schedule_entry.score_fn_kwargs or {}),
                    )

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
