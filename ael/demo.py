"""A CLI that demonstrates solving a problem from a problem set and saving the results."""

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

import ael.maps
from ael.problem import Problem
from ael.solve import (
    DEFAULT_SCHEDULES,
    OptimizerOptions,
    Result,
    ScheduleEntry,
    ScoreComputationMethod,
    solve,
)
from ael.visualize import save_optimization_process_video, save_video, visualize

PROBLEM_SET_ROOT_DIR = Path(__file__).resolve().parents[1] / "instances_data"


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

    save_dir: str = "./results/demo/{date}/{time}"
    """ Path to a directory in which to save results. Allows formatting with `date` and `time` variables, which are formatted as YYYY-mm-dd and HH-MM-SS, respectively."""

    score_computation_method: ScoreComputationMethod = (
        ScoreComputationMethod.APPROXIMATE_V0
        # ScoreComputationMethod.FACTORIZED_MPPI
    )


def store_result(
    problem: Problem,
    result: Result,
    schedule: list[ScheduleEntry],
    save_dir: Path,
    score_computation_method: ScoreComputationMethod,
):
    """Helper function that stores all tensors. Also stores high-level information like the time taken to solve, maximum constraint violations, and objective function values. Furthermore, visualizes the optimization process and final trajectories."""

    import json
    import subprocess

    import matplotlib.pyplot as plt

    # Save tensors.
    np.savez_compressed(
        save_dir / "result.npz",
        trajectories=np.array(result.trajectories),
        agent_circular_obstacle_constraint_residuals=result.constraint_satisfaction.agent_circular_obstacle_constraint_residuals,
        agent_rectangular_obstacle_constraint_residuals=result.constraint_satisfaction.agent_rectangular_obstacle_constraint_residuals,
        agent_agent_constraint_residuals=result.constraint_satisfaction.agent_agent_constraint_residuals,
        velocity_constraint_residuals=result.constraint_satisfaction.velocity_constraint_residuals,
    )

    agent_obs_resids = np.concatenate(
        [
            result.constraint_satisfaction.agent_circular_obstacle_constraint_residuals,
            result.constraint_satisfaction.agent_rectangular_obstacle_constraint_residuals,
        ],
        axis=-1,
    )
    if agent_obs_resids.size == 0:
        agent_obs_resids = np.zeros(1)

    # Save high-level information.
    info_dict = {
        "solve_time": result.solve_time,
        "identifier": result.identifier,
        "max_constraint_residuals": {
            "agent_obstacle": float(np.max(agent_obs_resids)),
            "agent_agent": float(
                np.max(result.constraint_satisfaction.agent_agent_constraint_residuals)
            ),
            "velocity": float(
                np.max(result.constraint_satisfaction.velocity_constraint_residuals)
            ),
        },
        "schedule": [entry.__dict__ for entry in schedule],
        "score_computation_method": score_computation_method.name,
    }
    with open(save_dir / "info.json", "w") as f:
        json.dump(info_dict, f, indent=4)

    logger.info(
        json.dumps({k: v for (k, v) in info_dict.items() if k != "schedule"}, indent=4)
    )

    # Visualize final trajectories.
    plt.figure(figsize=(6, 6))
    plt.title("Final Trajectory")
    visualize(problem, plt.gca(), result.trajectories[-1])
    plt.tight_layout()
    plt.savefig(save_dir / "final_trajectory.png")
    plt.clf()
    # Visualize optimization process.
    save_optimization_process_video(
        problem, result.trajectories[::8], save_dir / "optimization_process.mp4"
    )
    # Visualize final trajectory.
    save_video(problem, result.trajectories[-1], save_dir / "final_trajectory.mp4")
    # Also save GIFs.
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            str(save_dir / "optimization_process.mp4"),
            str(save_dir / "optimization_process.gif"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.call(
        [
            "ffmpeg",
            "-i",
            str(save_dir / "final_trajectory.mp4"),
            str(save_dir / "final_trajectory.gif"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def get_problem(args: MainArgs):
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
        problem.identifier = f"{args.problem_set}_{args.sample_index}"

    elif args.num_robots is not None:
        problem = None
        for i, pd in enumerate(problems):
            if len(pd["agents"]["start_positions"]) == args.num_robots:
                problem = Problem.from_json(pd, type="numpy")
                problem.identifier = f"{args.problem_set}_{i}"
                break
        else:
            raise ValueError(f"No problem found with {args.num_robots} robots.")

    else:
        problem = Problem.from_json(problems[0], type="numpy")
        problem.identifier = f"{args.problem_set}_0"

    return problem


def get_schedule(args: MainArgs):
    match args.schedule:
        case "default":
            schedule = DEFAULT_SCHEDULES[args.score_computation_method]
        case _:
            schedule_path = Path(args.schedule)
            if not schedule_path.exists():
                raise ValueError(f"Schedule path {schedule_path} does not exist.")

            with open(schedule_path) as f:
                schedule_json = json.load(f)

            schedule = [ScheduleEntry(**entry_dict) for entry_dict in schedule_json]

    return schedule


def main(args: MainArgs):
    schedule = get_schedule(args)
    problem = get_problem(args)

    with open("instances_data/larger/basic_maps.pkl", "rb") as f:
        data = pickle.load(f)

    problem = ael.maps.load_instance_from_pickled_format(data[1])
    problem.identifier = f"basic_maps__idx_1__num_robots_{problem.num_agents}"

    # problem = ael.maps.get_sample_problem(
    #     key="highways", num_agents=args.num_robots or 3, dist=1.8, num_timesteps=128
    # )

    result = solve(
        problem=problem,
        score_computation_method=args.score_computation_method,
        optimizer_options=args.optimizer,
        schedule=schedule,
    )
    save_dir = Path(
        args.save_dir.format(
            date=time.strftime("%Y-%m-%d"), time=time.strftime("%H-%M-%S")
        )
        + (f"_{result.identifier}" if result.identifier is not None else "")
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    store_result(
        problem,
        result,
        schedule,
        save_dir,
        score_computation_method=args.score_computation_method,
    )


if __name__ == "__main__":
    import tyro

    args = tyro.cli(MainArgs)
    main(args)
