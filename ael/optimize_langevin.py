import numpy as np

from ael.problem import Problem
from ael.score_agent_obstacle import (
    compute_agent_obstacle_distance_batched,
    compute_agent_obstacle_score,
    compute_agent_obstacle_score_batched,
    compute_r1_r2_batched,
)
from ael.score_kinetic_energy import compute_kinetic_energy_score


def clip_magnitude(vector, max_magnitude):
    magnitude = np.linalg.norm(vector)
    if magnitude > max_magnitude:
        return vector / magnitude * max_magnitude
    else:
        return vector


def step_langevin(
    trajectory, sigma, step_size, problem: Problem[np.ndarray], include_obstacles=True
):
    score = np.zeros_like(trajectory)

    if include_obstacles:
        for t in range(trajectory.shape[0]):
            for agent in range(trajectory.shape[1]):
                agent_x = trajectory[t, agent, 0]
                agent_y = trajectory[t, agent, 1]

                for obs_idx in range(problem.obstacle_positions.shape[0]):
                    obs_x = problem.obstacle_positions[obs_idx, 0]
                    obs_y = problem.obstacle_positions[obs_idx, 1]
                    obs_rad = problem.obstacle_radii[obs_idx]

                    score[t, agent] += clip_magnitude(
                        compute_agent_obstacle_score(
                            agent_x,
                            agent_y,
                            obs_x,
                            obs_y,
                            obs_rad + problem.agent_radii[agent],
                            sigma,
                        ),
                        1.0,
                    )

                for other_agent in range(trajectory.shape[1]):
                    if other_agent == agent:
                        continue
                    other_agent_x = trajectory[t, other_agent, 0]
                    other_agent_y = trajectory[t, other_agent, 1]

                    score[t, agent] += clip_magnitude(
                        compute_agent_obstacle_score(
                            agent_x,
                            agent_y,
                            other_agent_x,
                            other_agent_y,
                            problem.agent_radii[agent]
                            + problem.agent_radii[other_agent],
                            np.sqrt(2) * sigma,  # sum the variances
                        ),
                        1.0,
                    )

    score = score + 10 * compute_kinetic_energy_score(trajectory, sigma)

    return (
        trajectory + step_size * score
        # + np.sqrt(2 * step_size) * np.random.randn(*trajectory.shape)
    )


def step_langevin_batched(
    trajectory,
    sigma,
    step_size,
    problem: Problem,
    include_obstacles=True,
    n_integral=50,
    include_noise_term=False,
):
    """
    Batches across agents and obstacles.
    """

    score = np.zeros_like(trajectory)

    if include_obstacles:
        # Create batch for agent-obstacle interactions.
        agent_x_T_A = trajectory[:, :, 0]
        agent_y_T_A = trajectory[:, :, 1]
        obstacle_x_O = problem.obstacle_positions[:, 0]
        obstacle_y_O = problem.obstacle_positions[:, 1]
        obstacle_rad_O = problem.obstacle_radii
        sigma_OA = sigma * np.ones(
            problem.obstacle_radii.shape[0] * trajectory.shape[1]
        )

        agent_x_T_A_O = np.repeat(
            agent_x_T_A[:, :, None], problem.obstacle_positions.shape[0], axis=2
        )
        agent_y_T_A_O = np.repeat(
            agent_y_T_A[:, :, None], problem.obstacle_positions.shape[0], axis=2
        )
        obstacle_x_T_A_O = np.repeat(
            obstacle_x_O[None, None, :], trajectory.shape[0], axis=0
        )
        obstacle_x_T_A_O = np.repeat(obstacle_x_T_A_O, trajectory.shape[1], axis=1)
        obstacle_y_T_A_O = np.repeat(
            obstacle_y_O[None, None, :], trajectory.shape[0], axis=0
        )
        obstacle_y_T_A_O = np.repeat(obstacle_y_T_A_O, trajectory.shape[1], axis=1)
        obstacle_rad_T_A_O = np.repeat(
            (problem.agent_radii[:, None] + obstacle_rad_O[None, :])[None, :, :],
            trajectory.shape[0],
            axis=0,
        )
        sigma_T_A_O = np.repeat(sigma_OA[None, :], trajectory.shape[0], axis=0)

        T, A, O = agent_x_T_A_O.shape  # noqa: E741

        agent_x_T_A1_A2 = np.repeat(
            agent_x_T_A[:, :, None], trajectory.shape[1], axis=2
        )
        agent_x_T_A2_A1 = np.repeat(
            agent_x_T_A[:, None, :], trajectory.shape[1], axis=1
        )
        agent_y_T_A1_A2 = np.repeat(
            agent_y_T_A[:, :, None], trajectory.shape[1], axis=2
        )
        agent_y_T_A2_A1 = np.repeat(
            agent_y_T_A[:, None, :], trajectory.shape[1], axis=1
        )
        sigma_T_A1_A2 = (
            np.sqrt(2)
            * sigma
            * np.ones((trajectory.shape[0], trajectory.shape[1], trajectory.shape[1]))
        )
        obstacle_rad_T_A1_A2 = np.repeat(
            (problem.agent_radii[:, None] + problem.agent_radii[None, :])[None, :, :],
            trajectory.shape[0],
            axis=0,
        )

        agent_x_flat = np.concatenate(
            [agent_x_T_A_O.reshape(-1), agent_x_T_A1_A2.reshape(-1)]
        )
        agent_y_flat = np.concatenate(
            [agent_y_T_A_O.reshape(-1), agent_y_T_A1_A2.reshape(-1)]
        )
        obstacle_x_flat = np.concatenate(
            [obstacle_x_T_A_O.reshape(-1), agent_x_T_A2_A1.reshape(-1)]
        )
        obstacle_y_flat = np.concatenate(
            [obstacle_y_T_A_O.reshape(-1), agent_y_T_A2_A1.reshape(-1)]
        )
        obstacle_rad_flat = np.concatenate(
            [obstacle_rad_T_A_O.reshape(-1), obstacle_rad_T_A1_A2.reshape(-1)]
        )
        sigma_flat = np.concatenate(
            [sigma_T_A_O.reshape(-1), sigma_T_A1_A2.reshape(-1)]
        )
        d_a_o_flat = compute_agent_obstacle_distance_batched(
            agent_x_flat, agent_y_flat, obstacle_x_flat, obstacle_y_flat
        )
        r1_flat, r2_flat = compute_r1_r2_batched(obstacle_rad_flat, d_a_o_flat)
        score_flat = compute_agent_obstacle_score_batched(
            agent_x_flat,
            agent_y_flat,
            obstacle_x_flat,
            obstacle_y_flat,
            obstacle_rad_flat,
            sigma_flat,
            r1_flat,
            r2_flat,
            d_a_o_flat,
            n_integral=n_integral,
        )

        # clip norm
        norm = np.linalg.norm(score_flat, axis=-1, keepdims=True)
        norm_clipped = np.clip(norm, 0, 1.0)
        score_flat = score_flat * (norm_clipped / (1e-8 + norm))

        # unpack scores.
        score_T_A_O_D = score_flat[: T * A * O, :].reshape(
            trajectory.shape[0],
            trajectory.shape[1],
            problem.obstacle_positions.shape[0],
            2,
        )
        score_T_A1_A2_D = score_flat[T * A * O :, :].reshape(
            trajectory.shape[0], trajectory.shape[1], trajectory.shape[1], 2
        )
        # Don't compute self-interactions.
        score_T_A1_A2_D[np.isnan(score_T_A1_A2_D)] = 0.0
        score += (
            score_T_A1_A2_D.sum(axis=2)
            - np.diagonal(score_T_A1_A2_D, axis1=1, axis2=2).transpose(0, 2, 1)
        ) + (score_T_A_O_D.sum(axis=2))

    score = score + 10 * compute_kinetic_energy_score(trajectory, sigma)

    return (
        trajectory
        + step_size * score
        + (
            np.sqrt(2 * step_size) * np.random.randn(*trajectory.shape)
            if include_noise_term
            else 0
        )
    )
