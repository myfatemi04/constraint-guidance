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
from ael.problem import Problem
from ael.score_function import (
    compute_score,
    compute_score_mppi,
    compute_score_mppi_factorized,
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

    score_fn_kwargs: dict | None = None
    """ Any keyword arguments to pass to the score function. """


class ScoreComputationMethod(str, enum.Enum):
    APPROXIMATE_V0 = "approximate_v0"
    UNFACTORIZED_MPPI = "unfactorized_mppi"
    FACTORIZED_MPPI = "factorized_mppi"


DEFAULT_SCHEDULE = [
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

DEFAULT_SCHEDULE_UNFACTORIZED_MPPI = [
    ScheduleEntry(
        sigma=1.0,
        step_size=0.5,
        num_steps=100,
        score_fn_kwargs=dict(
            agent_agent_constraint_tolerance=1.0,
            agent_obstacle_constraint_tolerance=1.0,
            velocity_constraint_tolerance=1.0,
        ),
    ),
    ScheduleEntry(
        sigma=0.5,
        step_size=0.5,
        num_steps=100,
        score_fn_kwargs=dict(
            agent_agent_constraint_tolerance=0.5,
            agent_obstacle_constraint_tolerance=0.5,
            velocity_constraint_tolerance=0.5,
        ),
    ),
    ScheduleEntry(
        sigma=0.1,
        step_size=0.5,
        num_steps=100,
        score_fn_kwargs=dict(
            agent_agent_constraint_tolerance=0.1,
            agent_obstacle_constraint_tolerance=0.1,
            velocity_constraint_tolerance=0.1,
        ),
    ),
    ScheduleEntry(
        sigma=0.01,
        step_size=0.5,
        num_steps=100,
        score_fn_kwargs=dict(
            agent_agent_constraint_tolerance=0.01,
            agent_obstacle_constraint_tolerance=0.01,
            velocity_constraint_tolerance=0.01,
        ),
    ),
]


def solve(
    problem: Problem,
    score_computation_method: ScoreComputationMethod,
    optimizer_options: OptimizerOptions = OptimizerOptions(),
    schedule: list[ScheduleEntry] = DEFAULT_SCHEDULE,
    initial_trajectory: np.ndarray | None = None,
    identifier: str | None = None,
) -> Result:
    # TODO: Initialize from prior distribution based on energy.
    start_positions = problem.agent_start_positions
    end_positions = problem.agent_end_positions

    trajectory = np.linspace(start_positions, end_positions, num=64, axis=0)
    trajectory += np.random.randn(*trajectory.shape) * 0.1

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
                    score = compute_score_mppi(
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
