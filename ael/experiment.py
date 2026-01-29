import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tqdm

from ael.problem import Problem
from ael.solve import ScheduleEntry, solve


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
                    residuals = compute_constraint_residuals(
                        problem, result.trajectories[-1]
                    )
                    agent_obstacle_max_residual = np.max(residuals[0])
                    agent_agent_max_residual = np.max(residuals[1])
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
