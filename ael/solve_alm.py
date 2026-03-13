"""
Uses an iterative augmented Lagrangian approach. Instead of solving exactly, follows a schedule where constraint violations are subject to an increasingly high penalty term. These constraint violations are
also used to update the Lagrange multipliers.
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from ael.constraint_evaluation import compute_constraint_residuals
from ael.initial_paths import get_initial_paths_by_agent
from ael.problem import Problem
from ael.solve import Result
from ael.visualize import visualize


def solve_alm(
    problem: Problem,
    rho_multiplier=1.05,
    num_steps=100,
    num_inner_optimization_steps=10,
):
    t0 = time.time()
    trajectory = np.zeros((problem.num_timesteps, problem.num_agents, 2))
    # for adam
    trajectory_m = trajectory.copy()
    trajectory_b = trajectory.copy()
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    lr = 0.01

    # initialize trajectory to the first path for each agent
    target_paths_by_agent = get_initial_paths_by_agent(problem, dt=1.0)
    for agent_index in range(problem.num_agents):
        trajectory[:, agent_index, :] = target_paths_by_agent[agent_index]

    agent_agent_nu = np.zeros(
        (problem.num_timesteps, problem.num_agents, problem.num_agents)
    )
    agent_obstacle_nu = np.zeros(
        (problem.num_timesteps, problem.num_agents, problem.num_obstacles)
    )
    velocity_nu = np.zeros((problem.num_timesteps - 1, problem.num_agents))
    rho = 0.05

    trajectory_history = [trajectory]
    total_gradient_steps = 0

    for i in range(num_steps):
        for j in range(num_inner_optimization_steps):
            total_gradient_steps += 1
            # gradient is in displacement direction, proportional
            agent_agent_displacements = (
                trajectory[:, :, None, :] - trajectory[:, None, :, :]
            )
            agent_obstacle_displacements = (
                trajectory[:, :, None, :] - problem.obstacle_positions[None, None, :, :]
            )
            agent_stepwise_displacements = trajectory[1:, :, :] - trajectory[:-1, :, :]
            agent_agent_distances = np.linalg.norm(agent_agent_displacements, axis=-1)
            agent_obstacle_distances = np.linalg.norm(
                agent_obstacle_displacements, axis=-1
            )
            agent_displacements = np.linalg.norm(agent_stepwise_displacements, axis=-1)
            agent_agent_constraint_functions = agent_agent_distances - (
                problem.agent_radii[None, :] + problem.agent_radii[:, None]
            )
            agent_obstacle_constraint_functions = agent_obstacle_distances - (
                problem.agent_radii[:, None] + problem.obstacle_radii[None, :]
            )
            agent_velocity_constraint_functions = -(
                agent_displacements - problem.agent_max_speeds
            )  # negative is important; requires distance < max speed

            grad = np.zeros_like(trajectory)

            # kinetic energy
            ke_weight = 50
            grad[1:] += agent_stepwise_displacements * ke_weight
            grad[:-1] -= agent_stepwise_displacements * ke_weight

            # agent-agent constraints
            mask = agent_agent_constraint_functions < 0
            update = np.zeros_like(agent_agent_displacements)
            update[mask] = -rho * agent_agent_displacements[mask] - agent_agent_nu[
                mask
            ][..., None] * (
                agent_agent_displacements[mask]
                / (
                    np.linalg.norm(
                        agent_agent_displacements[mask], axis=-1, keepdims=True
                    )
                    + epsilon
                )
            )
            grad += np.sum(update, axis=2)

            # agent-obstacle constraints
            mask = agent_obstacle_constraint_functions < 0
            update = np.zeros_like(agent_obstacle_displacements)
            update[mask] = -rho * agent_obstacle_displacements[
                mask
            ] - agent_obstacle_nu[mask][..., None] * (
                agent_obstacle_displacements[mask]
                / (
                    np.linalg.norm(
                        agent_obstacle_displacements[mask], axis=-1, keepdims=True
                    )
                    + epsilon
                )
            )
            grad += np.sum(update, axis=2)

            # velocity constraints
            mask = agent_velocity_constraint_functions < 0
            grad[1:][mask] += rho * agent_stepwise_displacements[mask] + velocity_nu[
                mask
            ][..., None] * (
                agent_stepwise_displacements[mask]
                / (
                    np.linalg.norm(
                        agent_stepwise_displacements[mask], axis=-1, keepdims=True
                    )
                    + epsilon
                )
            )
            grad[:-1][mask] += -rho * agent_stepwise_displacements[mask] - velocity_nu[
                mask
            ][..., None] * (
                agent_stepwise_displacements[mask]
                / (
                    np.linalg.norm(
                        agent_stepwise_displacements[mask], axis=-1, keepdims=True
                    )
                    + epsilon
                )
            )

            # adam step
            trajectory_m = beta1 * trajectory_m + (1 - beta1) * grad
            trajectory_b = beta2 * trajectory_b + (1 - beta2) * (grad**2)
            m_hat = trajectory_m / (1 - beta1**total_gradient_steps)
            b_hat = trajectory_b / (1 - beta2**total_gradient_steps)
            trajectory -= lr * m_hat / (np.sqrt(b_hat) + epsilon)

            # fix the start and end positions
            trajectory[0, :, :] = problem.agent_start_positions
            trajectory[-1, :, :] = problem.agent_end_positions

        agent_agent_nu += rho * np.maximum(agent_agent_constraint_functions, 0)
        agent_obstacle_nu += rho * np.maximum(agent_obstacle_constraint_functions, 0)
        velocity_nu += rho * np.maximum(agent_velocity_constraint_functions, 0)
        rho *= rho_multiplier

        trajectory_history.append(trajectory)
    t1 = time.time()

    constraint_satisfaction = compute_constraint_residuals(problem, trajectory)

    return Result(
        solve_time=t1 - t0,
        trajectories=trajectory_history,
        identifier=problem.identifier,
        constraint_satisfaction=constraint_satisfaction,
    )


def test():
    with open("instances_data/instances_connected_room.json") as f:
        data = json.load(f)

    t0 = time.time()
    problem = Problem.from_json(data[2])
    trajectories = solve_alm(problem)
    visualize(problem, plt.gca(), trajectories)
    t1 = time.time()
    print(f"Time taken: {t1 - t0}")

    plt.show()


if __name__ == "__main__":
    test()
