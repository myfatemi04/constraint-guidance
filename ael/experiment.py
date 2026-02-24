import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import tqdm

from ael.problem import Problem
from ael.solve import (
    DEFAULT_SCHEDULES,
    OptimizerOptions,
    Result,
    ScheduleEntry,
    ScoreComputationMethod,
    solve,
)


@dataclass
class MainArgs:
    score_computation_method: ScoreComputationMethod

    problem_set: str = "dense"
    """ Loads problems from `./instances_data/instances_{problem_set}.json`. """

    num_robots: int | Literal["any"] = "any"
    """ If specified, selects only problems with this many robots. """

    optimizer: OptimizerOptions = field(default_factory=OptimizerOptions)
    """ Options for the optimizer to use. """

    schedule: str = "default"
    """ Path to a JSON file specifying the schedule to use, or 'default'. """

    save_dir: str = "./results/{date}/experiment_{time}"
    """ Path to a directory in which to save results. Allows formatting with `date` and `time` variables, which are formatted as YYYY-mm-dd and HH-MM-SS, respectively. """

    label: str | None = "{problem_set}_num_robots={num_robots}"
    """ Label to attach to this experiment. Gets appended to the save directory if specified. Can use the {problem_set} and {num_robots} variables. If None, no label is used. """


def main(args: MainArgs):
    with open(f"instances_data/instances_{args.problem_set}.json", "r") as f:
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

    save_dir = Path(
        args.save_dir.format(
            date=time.strftime("%Y-%m-%d"), time=time.strftime("%H-%M-%S")
        )
        + (
            f"_{args.label.format(problem_set=args.problem_set, num_robots=args.num_robots)}"
            if args.label is not None
            else ""
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "energy": [],
        "solve_time": [],
        "agent_obstacle_max_residual": [],
        "agent_agent_max_residual": [],
        "velocity_max_residual": [],
        "identifier": [],
        "num_robots": [],
    }

    with ProcessPoolExecutor() as executor:
        futures = []

        with open(save_dir / "schedule.json", "w") as f:
            json.dump([entry.__dict__ for entry in schedule], f, indent=4)

        with open(save_dir / "optimizer.json", "w") as f:
            json.dump(args.optimizer.__dict__, f, indent=4)

        for problem in problems:
            if args.num_robots != "any" and problem.num_agents != args.num_robots:
                continue

            futures.append(
                executor.submit(
                    solve,
                    problem,
                    args.score_computation_method,
                    args.optimizer,
                    schedule=schedule,
                )
            )

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = cast(Result, future.result())
            agent_obstacle_max_residual = np.max(
                result.constraint_satisfaction.agent_obstacle_constraint_residuals
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
