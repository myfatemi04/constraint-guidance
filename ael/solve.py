"""
The main solving algorithm. If used with the CLI, allows running a demo script.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from ael.problem import Problem
from ael.score_function import compute_score
from ael.visualize import save_optimization_process_video, save_video


@dataclass
class Result:
    solve_time: float
    """ Time taken to solve the problem in seconds. """

    trajectories: list[np.ndarray]
    """ List of trajectories at each optimization step. """

    identifier: str | None
    """ An identifier for the problem that was solved, if available. """

    agent_obstacle_constraint_residuals: np.ndarray
    """ Residuals for agent-obstacle nonpenetration constraints. If positive, represents the amount by which the signed distance into the obstacle exceeds zero. Shape (t, num_agents, num_obstacles). """

    agent_agent_constraint_residuals: np.ndarray
    """ Residuals for agent-agent nonpenetration constraints. If positive, represents the amount by which the signed distance into another agent exceeds zero. Shape (t, num_agents, num_agents). """


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

    magnitude_clip: float = 10000.0
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
    ScheduleEntry(sigma=0.001, step_size=0.5, num_steps=60, kinetic_weight=10),
    ScheduleEntry(sigma=0.0001, step_size=0.5, num_steps=60, kinetic_weight=2),
]


def compute_constraint_residuals(
    problem: Problem[np.ndarray], trajectories: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
    residuals_agent_agent = min_acceptable_distances - agent_pairwise_distances
    residuals_agent_agent[residuals_agent_agent < 0] = 0.0
    diag_indices = np.diag_indices(residuals_agent_agent.shape[1])
    residuals_agent_agent[:, diag_indices[0], diag_indices[1]] = 0.0
    assert (np.diagonal(residuals_agent_agent, axis1=1, axis2=2) == 0).all()

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
    residuals_agent_obstacle = (
        min_acceptable_agent_obstacle_distances - agent_obstacle_distances
    )
    residuals_agent_obstacle[residuals_agent_obstacle < 0] = 0.0

    return residuals_agent_obstacle, residuals_agent_agent


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

    agent_obstacle_constraint_residuals, agent_agent_constraint_residuals = (
        compute_constraint_residuals(problem, trajectory)
    )

    return Result(
        solve_time=solve_time,
        trajectories=trajectories,
        identifier=identifier,
        agent_obstacle_constraint_residuals=agent_obstacle_constraint_residuals,
        agent_agent_constraint_residuals=agent_agent_constraint_residuals,
    )


@dataclass
class MainArgs:
    problem_set: str = "dense"
    """ Loads problems from `./instances_data/instances_{problem_set}.json`. """

    sample_index: int | None = None
    """ If specified, only solves the problem at this index in the problem set. If not specified, selects the first problem. """

    num_robots: int | None = None
    """ If specified, selects the first problem with this many robots. Cannot be simultaneously specified with `sample_index`. """

    optimizer: OptimizerOptions = field(default_factory=OptimizerOptions)
    """ Options for the optimizer to use. """

    schedule: str = "default"
    """ Path to a JSON file specifying the schedule to use, or 'default'. """

    save_dir: str = "./results/demo_{date}T{time}"
    """ Path to a directory in which to save results. Allows formatting with `date` and `time` variables, which are formatted as YYYY-mm-dd and HH-MM-SS, respectively. """


PROBLEM_SET_ROOT_DIR = Path(__file__).resolve().parents[1] / "instances_data"


def store_result(problem: Problem, result: Result, save_dir: Path):
    """Helper function that stores all tensors. Also stores high-level information like the time taken to solve, maximum constraint violations, and objective function values. Furthermore, visualizes the optimization process and final trajectories."""

    import json

    # Save tensors.
    np.savez_compressed(
        save_dir / "result.npz",
        trajectories=np.array(result.trajectories),
        agent_obstacle_constraint_residuals=result.agent_obstacle_constraint_residuals,
        agent_agent_constraint_residuals=result.agent_agent_constraint_residuals,
    )

    # Save high-level information.
    info_dict = {
        "solve_time": result.solve_time,
        "identifier": result.identifier,
        "max_agent_obstacle_constraint_residual": float(
            np.max(result.agent_obstacle_constraint_residuals)
        ),
        "max_agent_agent_constraint_residual": float(
            np.max(result.agent_agent_constraint_residuals)
        ),
    }
    with open(save_dir / "info.json", "w") as f:
        json.dump(info_dict, f, indent=4)

    # Visualize optimization process.
    save_optimization_process_video(
        problem, result.trajectories, save_dir / "optimization_process.mp4"
    )
    # Visualize final trajectory.
    save_video(problem, result.trajectories[-1], save_dir / "final_trajectory.mp4")


def main(args: MainArgs):
    import json

    problem_set_path = PROBLEM_SET_ROOT_DIR / f"instances_{args.problem_set}.json"
    if not problem_set_path.exists():
        raise ValueError(f"Problem set path {problem_set_path} does not exist.")

    with open(problem_set_path) as f:
        problems = json.load(f)

    if args.sample_index is not None and args.num_robots is not None:
        raise ValueError("Cannot simultaneously specify sample_index and num_robots.")

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(problems):
            raise ValueError("sample_index out of range.")

        problem = Problem.from_json(problems[args.sample_index], type="numpy")

    elif args.num_robots is not None:
        problem = None
        for pd in problems:
            if pd["num_agents"] == args.num_robots:
                problem = Problem.from_json(pd, type="numpy")
                break
        else:
            raise ValueError(f"No problem found with {args.num_robots} robots.")

    else:
        problem = Problem.from_json(problems[0], type="numpy")

    if args.schedule == "default":
        schedule = DEFAULT_SCHEDULE
    else:
        schedule_path = Path(args.schedule)
        if not schedule_path.exists():
            raise ValueError(f"Schedule path {schedule_path} does not exist.")

        with open(schedule_path) as f:
            schedule_json = json.load(f)

        schedule = [ScheduleEntry(**entry_dict) for entry_dict in schedule_json]

    result = solve(problem=problem, optimizer_options=args.optimizer, schedule=schedule)
    logger.info(f"Solved problem in {result.solve_time:.2f} seconds.")

    save_dir = Path(
        args.save_dir.format(
            date=time.strftime("%Y-%m-%d"), time=time.strftime("%H-%M-%S")
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    store_result(problem, result, save_dir)


if __name__ == "__main__":
    import tyro

    args = tyro.cli(MainArgs)
    main(args)
