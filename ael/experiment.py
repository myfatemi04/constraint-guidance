import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tqdm

from ael.problem import Problem
from ael.solve import ScheduleEntry, solve


def make_schedule(num_steps_per_sigma: int = 60):
    return [
        ScheduleEntry(
            sigma=0.01, step_size=0.5, num_steps=num_steps_per_sigma, kinetic_weight=50
        ),
        ScheduleEntry(
            sigma=0.001, step_size=0.5, num_steps=num_steps_per_sigma, kinetic_weight=10
        ),
        ScheduleEntry(
            sigma=0.0001,
            step_size=0.5,
            num_steps=num_steps_per_sigma,
            kinetic_weight=2,
        ),
    ]


def job(problem: Problem, schedule: list[ScheduleEntry]):
    result = solve(problem, schedule=schedule), problem
    return result


if __name__ == "__main__":
    with open("instances_data/instances_dense.json", "r") as f:
        data = json.load(f)

    problems = [Problem.from_json(d, type="numpy") for d in data]

    for num_steps_per_sigma in [120, 180, 240]:
        schedule = make_schedule(num_steps_per_sigma)
        for count in [3, 6, 9]:
            futures = []
            data = {
                "energy": [],
                "solve_time": [],
                "agent_obstacle_max_residual": [],
                "agent_agent_max_residual": [],
            }

            with ProcessPoolExecutor() as executor:
                for problem in problems:
                    if problem.num_agents != count:
                        continue

                    futures.append(executor.submit(job, problem, schedule))

                for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                    result, problem = future.result()
                    agent_obstacle_max_residual = np.max(
                        result.agent_obstacle_constraint_residuals
                    )
                    agent_agent_max_residual = np.max(
                        result.agent_agent_constraint_residuals
                    )
                    data["solve_time"].append(result.solve_time)
                    data["agent_obstacle_max_residual"].append(
                        agent_obstacle_max_residual
                    )
                    data["agent_agent_max_residual"].append(agent_agent_max_residual)
                    data["energy"].append(
                        (result.trajectories[-1] ** 2).sum(axis=-1).mean()
                    )

                executor.shutdown(wait=True)

            pd.DataFrame(data).to_csv(
                f"experiment_results_{count}_{num_steps_per_sigma=}_decrease_kinetic.csv",
                index=False,
            )
