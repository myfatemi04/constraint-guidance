from dataclasses import dataclass

import numpy as np

from ael.problem import Problem


@dataclass
class ConstraintSatisfaction:
    agent_obstacle_constraint_residuals: np.ndarray
    """ Residuals for agent-obstacle nonpenetration constraints. If positive, represents the amount by which the signed distance into the obstacle exceeds zero. Shape ([b], t, num_agents, num_obstacles). """

    agent_agent_constraint_residuals: np.ndarray
    """ Residuals for agent-agent nonpenetration constraints. If positive, represents the amount by which the signed distance into another agent exceeds zero. Shape ([b], t, num_agents, num_agents). """

    velocity_constraint_residuals: np.ndarray
    """ Residuals for velocity constraints. If positive, represents the amount by which the velocity exceeds the maximum allowed velocity. Shape ([b], t-1, num_agents). """


def compute_agent_obstacle_constraint_residuals(
    problem: Problem[np.ndarray], trajectory: np.ndarray
) -> np.ndarray:
    ### Agent-obstacle nonpenetration constraints ###
    agent_obstacle_distances = (
        # ([b], t, num_agents, 1, 2) - (1, num_obstacles, 2) -> ([b], t, num_agents, num_obstacles, 2)
        trajectory[..., :, np.newaxis, :] - problem.obstacle_positions[np.newaxis, :, :]
    )
    agent_obstacle_distances = np.linalg.norm(agent_obstacle_distances, axis=-1)
    min_acceptable_agent_obstacle_distances = (
        # (num_agents, 1) + (1, num_obstacles) -> (num_agents, num_obstacles)
        problem.agent_radii[:, np.newaxis] + problem.obstacle_radii[np.newaxis, :]
    )
    agent_obstacle_constraint_residuals = (
        min_acceptable_agent_obstacle_distances - agent_obstacle_distances
    )
    agent_obstacle_constraint_residuals[agent_obstacle_constraint_residuals < 0] = 0.0

    return agent_obstacle_constraint_residuals


def compute_agent_agent_constraint_residuals(
    problem: Problem[np.ndarray], trajectory: np.ndarray
) -> np.ndarray:
    ### Agent-agent nonpenetration constraints ###
    agent_pairwise_distances = (
        # ([b], t, num_agents, 1, 2) - ([b], t, 1, num_agents, 2) -> ([b], t, num_agents, num_agents, 2)
        trajectory[..., :, np.newaxis, :] - trajectory[..., np.newaxis, :, :]
    )
    # ([b], t, num_agents, num_agents)
    agent_pairwise_distances = np.linalg.norm(agent_pairwise_distances, axis=-1)
    min_acceptable_distances = (
        problem.agent_radii[:, np.newaxis] + problem.agent_radii[np.newaxis, :]
    )
    agent_agent_constraint_residuals = (
        min_acceptable_distances - agent_pairwise_distances
    )
    agent_agent_constraint_residuals[agent_agent_constraint_residuals < 0] = 0.0
    diag_indices = np.diag_indices(agent_agent_constraint_residuals.shape[-2])
    agent_agent_constraint_residuals[..., diag_indices[0], diag_indices[1]] = 0.0
    assert (
        np.diagonal(agent_agent_constraint_residuals, axis1=-2, axis2=-1) == 0
    ).all()

    return agent_agent_constraint_residuals


def compute_velocity_constraint_residuals(
    problem: Problem[np.ndarray], trajectory: np.ndarray
) -> np.ndarray:
    ### Velocity constraints ###
    # ([b], t, num_agents, 2)
    velocities = trajectory[..., 1:, :, :] - trajectory[..., :-1, :, :]
    speeds = np.linalg.norm(velocities, axis=-1)
    velocity_constraint_residuals = speeds - problem.agent_max_speeds
    velocity_constraint_residuals[velocity_constraint_residuals < 0] = 0.0

    return velocity_constraint_residuals


def compute_constraint_residuals(
    problem: Problem[np.ndarray], trajectory: np.ndarray
) -> ConstraintSatisfaction:
    """
    Return the constraint residuals for agent-obstacle and agent-agent distances, respectively.
    Allows computation for single trajectories and/or batches of trajectories. A residual > 0
    indicates a constraint violation.
    """
    return ConstraintSatisfaction(
        agent_obstacle_constraint_residuals=compute_agent_obstacle_constraint_residuals(
            problem, trajectory
        ),
        agent_agent_constraint_residuals=compute_agent_agent_constraint_residuals(
            problem, trajectory
        ),
        velocity_constraint_residuals=compute_velocity_constraint_residuals(
            problem, trajectory
        ),
    )
