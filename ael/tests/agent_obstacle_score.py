import time

import numpy as np

from ael.agent_obstacle_score import (
    compute_agent_obstacle_distance_batched,
    compute_agent_obstacle_score,
    compute_agent_obstacle_score_batched,
    compute_r1_r2_batched,
)


def test_batching():
    # Verify that the batched version matches the unbatched version.
    displacement_directions_unbatched = {}
    displacement_directions_single_batched = {}

    for c_x in [2, 4, 6]:
        for R in [1, 3, 5]:
            for sigma in [0.1, 0.3, 0.5]:
                t0 = time.time()
                result = compute_agent_obstacle_score(
                    0,
                    0,
                    c_x,
                    0,
                    R,
                    sigma,
                    n_integral=200,
                )
                t1 = time.time()
                displacement_directions_unbatched[c_x, R, sigma] = (result, t1 - t0)

                t0 = time.time()
                agent_x_B = np.array([0])
                agent_y_B = np.array([0])
                obs_x_B = np.array([c_x])
                obs_y_B = np.array([0])
                obs_rad_B = np.array([R])
                sigma_B = np.array([sigma])
                d_a_o_B = compute_agent_obstacle_distance_batched(
                    agent_x_B, agent_y_B, obs_x_B, obs_y_B
                )
                r1_B, r2_B = compute_r1_r2_batched(obs_rad_B, d_a_o_B)
                result = compute_agent_obstacle_score_batched(
                    agent_x_B,
                    agent_y_B,
                    obs_x_B,
                    obs_y_B,
                    obs_rad_B,
                    sigma_B,
                    r1_B,
                    r2_B,
                    d_a_o_B,
                    n_integral=200,
                )
                t1 = time.time()
                displacement_directions_single_batched[c_x, R, sigma] = (
                    result,
                    t1 - t0,
                )

                result1 = displacement_directions_unbatched[c_x, R, sigma][0]
                result2 = displacement_directions_single_batched[c_x, R, sigma][0][0]

                assert np.allclose(result1, result2, atol=1e-6), (
                    f"Mismatch for {c_x}, {R}, {sigma}"
                )

    t0 = time.time()
    keys = list(displacement_directions_unbatched.keys())
    agent_x_B = np.zeros(len(keys))
    agent_y_B = np.zeros(len(keys))
    obs_x_B = np.array([k[0] for k in keys])
    obs_y_B = np.zeros(len(keys))
    obs_rad_B = np.array([k[1] for k in keys])
    sigma_B = np.array([k[2] for k in keys])
    d_a_o_B = compute_agent_obstacle_distance_batched(
        agent_x_B, agent_y_B, obs_x_B, obs_y_B
    )
    r1_B, r2_B = compute_r1_r2_batched(obs_rad_B, d_a_o_B)
    displacement_directions_batch = compute_agent_obstacle_score_batched(
        agent_x_B,
        agent_y_B,
        obs_x_B,
        obs_y_B,
        obs_rad_B,
        sigma_B,
        r1_B,
        r2_B,
        d_a_o_B,
        n_integral=200,
    )
    t1 = time.time()

    duration = t1 - t0
    average_duration_batched = duration / len(keys)
    average_duration_unbatched = 0

    for i, k in enumerate(keys):
        result1 = displacement_directions_unbatched[k][0]
        result2 = displacement_directions_batch[i]

        average_duration_unbatched += displacement_directions_unbatched[k][1]

        assert np.allclose(result1, result2, atol=1e-6), f"Mismatch for {k}"

    average_duration_unbatched /= len(keys)

    print("All tests passed.")
    print(
        f"{average_duration_batched=} {average_duration_unbatched=} ratio={average_duration_unbatched / average_duration_batched:.2f}"
    )
