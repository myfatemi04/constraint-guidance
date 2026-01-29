import io
from pathlib import Path

import av
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from matplotlib.axes import Axes

from ael.problem import Problem


def visualize(problem: Problem, ax: Axes, agent_positions: np.ndarray | None = None):
    # Plot the obstacles
    for obs_index in range(problem.num_obstacles):
        x, y = problem.obstacle_positions[obs_index].tolist()
        ax.add_patch(
            patches.Circle(
                (x, y), problem.obstacle_radii[obs_index].item(), color="r", alpha=0.5
            )
        )

    # Plot the agents' trajectories
    if agent_positions is not None:
        for agent_index in range(problem.num_agents):
            if agent_positions.shape[0] == 1:
                x, y = agent_positions[0, agent_index].tolist()
                ax.add_patch(
                    patches.Circle((x, y), problem.agent_radii[agent_index].item())
                )
            else:
                ax.plot(
                    agent_positions[:, agent_index, 0],
                    agent_positions[:, agent_index, 1],
                    marker="o",
                    label=f"Agent {agent_index}",
                    # set size to agent radius
                    markersize=problem.agent_radii[agent_index].item() * 10,
                )

    # Plot the agents' start and goal positions
    for agent_index in range(problem.num_agents):
        (sx, sy) = problem._as_numpy(problem.agent_start_positions[agent_index])
        (ex, ey) = problem._as_numpy(problem.agent_end_positions[agent_index])
        ax.plot(
            sx,
            sy,
            marker="o",
            color="green",
            markersize=10,
            label=f"Start {agent_index}",
        )
        ax.plot(
            ex,
            ey,
            marker="*",
            color="blue",
            markersize=10,
            label=f"Goal {agent_index}",
        )

    ax.set_aspect("equal")


def save_video(problem: Problem, agent_positions: np.ndarray, path: str | Path):
    buf = io.BytesIO()
    images = []

    for step in range(problem.num_timesteps):
        plt.clf()
        visualize(
            problem,
            plt.gca(),
            agent_positions[step : step + 1],
        )
        plt.title(f"Timestep {step}")
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = PIL.Image.open(buf).copy()
        images.append(image)
        buf.truncate(0)
        buf.seek(0)

    with av.open(path, "w") as container:
        stream = container.add_stream("h264", rate=4)
        for img in images:
            frame = av.VideoFrame.from_image(img)
            packet = stream.encode(frame)
            if packet:
                container.mux(packet)
        # Flush stream
        for packet in stream.encode(None):
            container.mux(packet)


def save_optimization_process_video(
    problem: Problem, agent_positions: np.ndarray | list[np.ndarray], path: str | Path
):
    buf = io.BytesIO()
    images = []

    for step in range(len(agent_positions)):
        plt.clf()
        visualize(problem, plt.gca(), agent_positions[step])
        plt.title(f"Timestep {step}")
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = PIL.Image.open(buf).copy()
        images.append(image)
        buf.truncate(0)
        buf.seek(0)

    with av.open(path, "w") as container:
        stream = container.add_stream("h264", rate=4)
        for img in images:
            frame = av.VideoFrame.from_image(img)
            packet = stream.encode(frame)
            if packet:
                container.mux(packet)
        # Flush stream
        for packet in stream.encode(None):
            container.mux(packet)
