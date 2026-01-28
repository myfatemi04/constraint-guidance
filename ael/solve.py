# Implementing Langevin dynamics!
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ael.problem import Problem
from ael.score_function import compute_score


@dataclass
class Result:
    solve_time: float
    """ Time taken to solve the problem in seconds. """

    trajectories: list[np.ndarray]
    """ List of trajectories at each optimization step. """


@dataclass
class OptimizerOptions:
    optimization: Literal["adam", "sgd"] = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    magnitude_clip: float = 10000.0


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


DEFAULT_SCHEDULE = [
    ScheduleEntry(sigma=0.01, step_size=0.5, num_steps=60, kinetic_weight=50),
    ScheduleEntry(sigma=0.001, step_size=0.5, num_steps=60, kinetic_weight=50),
    ScheduleEntry(sigma=0.0001, step_size=0.5, num_steps=60, kinetic_weight=50),
]


def solve(
    problem: Problem,
    optimizer_options: OptimizerOptions = OptimizerOptions(),
    schedule: list[ScheduleEntry] = DEFAULT_SCHEDULE,
) -> Result:
    # TODO: Initialize from prior distribution based on energy.
    trajectory = np.random.randn(64, problem.num_agents, 2) * 0.5
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
                n_integral=20,
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

    return Result(solve_time=solve_time, trajectories=trajectories)
