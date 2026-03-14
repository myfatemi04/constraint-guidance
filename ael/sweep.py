"""
Sweep various parameters.
"""

import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import tqdm

import ael.maps
import ael.problem
import ael.solve


@dataclass
class MainArgs:
    problem_set: str
    optimizer: ael.solve.OptimizerOptions = field(
        default_factory=ael.solve.OptimizerOptions
    )
    score_computation_method: ael.solve.ScoreComputationMethod = (
        ael.solve.ScoreComputationMethod.APPROXIMATE_V0
    )
    save_dir: str = "./results/{date}/sweep_{time}_{label}"
    """ Path to a directory in which to save results. Allows formatting with `date`, `time`, `score_computation_method`, `highlevel_search` variables, which are formatted as YYYY-mm-dd and HH-MM-SS, respectively. """
    label: str = "no-label"


def run_problem_set(args: MainArgs):
    root_save_dir = Path(
        args.save_dir.format(
            date=time.strftime("%Y-%m-%d"),
            time=time.strftime("%H-%M-%S"),
            label=args.label,
        )
    )
    root_save_dir.mkdir(parents=True, exist_ok=True)

    assert args.problem_set == "larger__dense"

    problems: list[ael.problem.Problem] = []

    if args.problem_set.startswith("larger__"):
        tag = args.problem_set[len("larger__") :]
        with open(f"instances_data/larger/{tag}_maps.pkl", "rb") as f:
            data = pickle.load(f)

        problems = [
            ael.maps.load_instance_from_pickled_format(instance) for instance in data
        ]
    else:
        with open(f"instances_data/instances_{args.problem_set}.json", "r") as f:
            data = json.load(f)

        problems = [ael.problem.Problem.from_json(d, type="numpy") for d in data]

    for num_agents in range(10, 22, 2):
        for problem in problems:
            assert problem.num_agents >= num_agents, (
                f"Problem {problem.identifier} has only {problem.num_agents} agents, cannot use {num_agents} agents."
            )

        for steps in [100, 200, 500, 1000, 2000]:
            tag = f"num_agents_{num_agents}__steps_{steps}"
            save_dir = root_save_dir / tag
            save_dir.mkdir(parents=True, exist_ok=True)

            schedule_args = {
                "init_sigma": 0.3,
                "end_sigma": 0.001,
                "init_kinetic_weight": 10,
                "end_kinetic_weight": 1,
                "step_size": 0.8,
                "steps": steps,
                "n_integral": 10,
            }
            schedule = ael.solve.make_exponential_schedule(**schedule_args)
            data = {
                "energy": [],
                "length": [],
                "solve_time": [],
                "agent_obstacle_max_residual": [],
                "agent_agent_max_residual": [],
                "velocity_max_residual": [],
                "identifier": [],
                "num_robots": [],
            }
            params = {
                "score_computation_method": args.score_computation_method.name,
                "schedule": {"type": "exponential", "args": schedule_args},
                "optimizer": args.optimizer.__dict__,
            }
            with open(save_dir / "params.json", "w") as f:
                json.dump(params, f, indent=4)

            rng = np.random.default_rng(seed=0)

            with ProcessPoolExecutor() as executor:
                futures = []

                for i, problem in enumerate(problems):
                    problem.identifier = f"{args.problem_set}_{i}"

                    # Select a subset of the agents.
                    agent_indices = rng.choice(
                        np.arange(problem.num_agents), size=num_agents, replace=False
                    )
                    problem = problem.agent_subset(agent_indices)

                    future = executor.submit(
                        ael.solve.solve,
                        problem,
                        args.score_computation_method,
                        args.optimizer,
                        schedule,
                    )
                    futures.append(future)

                for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                    result = cast(ael.solve.Result, future.result())
                    (
                        agent_obstacle_max_residual,
                        agent_agent_max_residual,
                        velocity_max_residual,
                    ) = result.constraint_satisfaction.compute_max_residuals()

                    data["solve_time"].append(result.solve_time)
                    data["agent_obstacle_max_residual"].append(
                        agent_obstacle_max_residual
                    )
                    data["agent_agent_max_residual"].append(agent_agent_max_residual)
                    data["velocity_max_residual"].append(velocity_max_residual)
                    data["energy"].append(
                        (result.trajectories[-1] ** 2).sum(axis=-1).mean()
                    )
                    length = (
                        np.linalg.norm(
                            np.diff(result.trajectories[-1], axis=0), axis=-1
                        )
                        .sum(axis=-1)
                        .mean()
                    )
                    data["length"].append(length)
                    data["num_robots"].append(result.trajectories[-1].shape[1])
                    data["identifier"].append(result.identifier)

                executor.shutdown(wait=True)

                df = pd.DataFrame(data)
                df.to_csv(save_dir / "table.csv", index=False)


if __name__ == "__main__":
    import tyro

    run_problem_set(tyro.cli(MainArgs))
