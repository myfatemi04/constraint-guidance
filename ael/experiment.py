import enum
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import tqdm

from ael.maps import load_instance_from_pickled_format
from ael.problem import Problem
from ael.solve import (
    DEFAULT_SCHEDULES,
    OptimizerOptions,
    Result,
    ScheduleEntry,
    ScoreComputationMethod,
    solve,
)


class HighLevelSearchMethod(enum.Enum):
    none = "none"
    cbs_spatial_approximation = "cbs_spatial_approximation"


@dataclass
class MainArgs:
    score_computation_method: ScoreComputationMethod = (
        ScoreComputationMethod.APPROXIMATE_V0
    )

    highlevel_search: HighLevelSearchMethod = HighLevelSearchMethod.none

    problem_set: str = "dense"
    """ Loads problems from `./instances_data/instances_{problem_set}.json`. Can be comma-separated list. """

    num_robots: int | Literal["any"] = "any"
    """ If specified, selects only problems with this many robots. """

    optimizer: OptimizerOptions = field(default_factory=OptimizerOptions)
    """ Options for the optimizer to use. """

    schedule: str = "default"
    """ Path to a JSON file specifying the schedule to use, or 'default'. """

    save_dir: str = "./results/{date}/experiment_{time}_{score_computation_method}_{highlevel_search}"
    """ Path to a directory in which to save results. Allows formatting with `date`, `time`, `score_computation_method`, `highlevel_search` variables, which are formatted as YYYY-mm-dd and HH-MM-SS, respectively. """

    label: str | None = "{problem_set}"
    """ Label to attach to this experiment. Gets appended to the save directory if specified. Can use the {problem_set} and {num_robots} variables. If None, no label is used. """


def main(args: MainArgs):
    save_dir = Path(
        args.save_dir.format(
            date=time.strftime("%Y-%m-%d"),
            time=time.strftime("%H-%M-%S"),
            score_computation_method=args.score_computation_method.name,
            highlevel_search=args.highlevel_search.name,
        )
    )
    for problem_set in args.problem_set.split(","):
        args.problem_set = problem_set
        run_problem_set(args, problem_set, save_dir / problem_set)


def run_problem_set(args: MainArgs, problem_set: str, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    if problem_set.startswith("larger__"):
        tag = problem_set[len("larger__") :]
        with open(f"instances_data/larger/{tag}_maps.pkl", "rb") as f:
            data = pickle.load(f)

        problems = [load_instance_from_pickled_format(instance) for instance in data]
    else:
        with open(f"instances_data/instances_{problem_set}.json", "r") as f:
            data = json.load(f)

        problems = [Problem.from_json(d, type="numpy") for d in data]

    if args.schedule == "default":
        schedule = DEFAULT_SCHEDULES[args.score_computation_method]
    else:
        schedule_path = Path(args.schedule)
        if not schedule_path.exists():
            raise ValueError(f"Schedule path {schedule_path} does not exist.")

        with open(schedule_path) as f:
            schedule_json = json.load(f)

        schedule = [ScheduleEntry(**entry_dict) for entry_dict in schedule_json]

    data = {
        "energy": [],
        "solve_time": [],
        "agent_obstacle_max_residual": [],
        "agent_agent_max_residual": [],
        "velocity_max_residual": [],
        "identifier": [],
        "num_robots": [],
    }

    params = {
        "score_computation_method": args.score_computation_method.name,
        "highlevel_search": args.highlevel_search.name,
        "schedule": [entry.__dict__ for entry in schedule],
        "optimizer": args.optimizer.__dict__,
    }
    with open(save_dir / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    with ProcessPoolExecutor() as executor:
        futures = []

        for i, problem in enumerate(problems):
            problem.identifier = f"{problem_set}_{i}"
            if args.num_robots != "any" and problem.num_agents != args.num_robots:
                continue

            if args.highlevel_search == HighLevelSearchMethod.cbs_spatial_approximation:
                from ael.cbs_spatial_approximation import cbs_spatial_approximation

                future = executor.submit(
                    cbs_spatial_approximation,
                    problem,
                    partial(
                        solve,
                        score_computation_method=args.score_computation_method,
                        schedule=schedule,
                    ),
                )
                futures.append(future)
            elif args.highlevel_search == HighLevelSearchMethod.none:
                futures.append(
                    executor.submit(
                        solve,
                        problem,
                        args.score_computation_method,
                        args.optimizer,
                        schedule,
                    )
                )

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = cast(Result, future.result())
            if (
                result.constraint_satisfaction.agent_circular_obstacle_constraint_residuals.size
                == 0
                and result.constraint_satisfaction.agent_rectangular_obstacle_constraint_residuals.size
                == 0
            ):
                print(
                    f"Warning: No obstacle constraint residuals for problem {result.identifier}."
                )
                agent_obstacle_max_residual = 0
            elif (
                result.constraint_satisfaction.agent_circular_obstacle_constraint_residuals.size
                == 0
            ):
                print(
                    f"Warning: No circular obstacle constraint residuals for problem {result.identifier}."
                )
                agent_obstacle_max_residual = np.max(
                    result.constraint_satisfaction.agent_rectangular_obstacle_constraint_residuals
                )
            elif (
                result.constraint_satisfaction.agent_rectangular_obstacle_constraint_residuals.size
                == 0
            ):
                print(
                    f"Warning: No rectangular obstacle constraint residuals for problem {result.identifier}."
                )
                agent_obstacle_max_residual = np.max(
                    result.constraint_satisfaction.agent_circular_obstacle_constraint_residuals
                )
            else:
                resid = np.concatenate(
                    [
                        result.constraint_satisfaction.agent_circular_obstacle_constraint_residuals,
                        result.constraint_satisfaction.agent_rectangular_obstacle_constraint_residuals,
                    ]
                )
                agent_obstacle_max_residual = np.max(
                    resid if resid.size > 0 else np.zeros(1)
                )
            agent_agent_max_residual = np.max(
                result.constraint_satisfaction.agent_agent_constraint_residuals
            )
            velocity_max_residual = np.max(
                result.constraint_satisfaction.velocity_constraint_residuals
            )
            data["solve_time"].append(result.solve_time)
            data["agent_obstacle_max_residual"].append(agent_obstacle_max_residual)
            data["agent_agent_max_residual"].append(agent_agent_max_residual)
            data["velocity_max_residual"].append(velocity_max_residual)
            data["energy"].append((result.trajectories[-1] ** 2).sum(axis=-1).mean())
            data["num_robots"].append(result.trajectories[-1].shape[1])
            data["identifier"].append(result.identifier)

        executor.shutdown(wait=True)

        df = pd.DataFrame(data)
        df.to_csv(save_dir / "table.csv", index=False)


if __name__ == "__main__":
    import tyro

    args = tyro.cli(MainArgs)
    main(args)
