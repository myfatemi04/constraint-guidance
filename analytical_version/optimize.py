import io
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, overload

import av
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import tqdm
from matplotlib.axes import Axes

TensorType = TypeVar("TensorType", torch.Tensor, np.ndarray)


@dataclass
class Obstacles:
    positions: torch.Tensor
    radii: torch.Tensor


@dataclass
class SolutionValue:
    agent_agent_distances: torch.Tensor
    agent_obstacle_distances: torch.Tensor
    agent_positions: torch.Tensor

    def get_batch_item(self, index: int):
        return SolutionValue(
            agent_positions=self.agent_positions[index],
            agent_agent_distances=self.agent_agent_distances[index],
            agent_obstacle_distances=self.agent_obstacle_distances[index],
        )


@overload
def _tensor(array, type: Literal["torch"]) -> torch.Tensor: ...


@overload
def _tensor(array, type: Literal["numpy"]) -> np.ndarray: ...


def _tensor(array, type: Literal["torch", "numpy"] = "torch"):
    match type:
        case "torch":
            return torch.tensor(array, dtype=torch.float32)
        case "numpy":
            return np.array(array, dtype=np.float32)
        case _:
            raise ValueError(f"Unknown type: {type}")


@dataclass
class Problem(Generic[TensorType]):
    num_timesteps: int
    agent_start_positions: TensorType
    agent_end_positions: TensorType
    agent_reference_trajectory: TensorType | None
    agent_radii: TensorType
    agent_max_speeds: TensorType
    obstacle_positions: TensorType
    obstacle_radii: TensorType

    @property
    def num_agents(self):
        return self.agent_start_positions.shape[0]

    @property
    def num_obstacles(self):
        return self.obstacle_positions.shape[0]

    @staticmethod
    def _as_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(array, np.ndarray):
            return array

        return array.detach().cpu().numpy()

    def visualize(self, ax: Axes, agent_positions: np.ndarray | None = None):
        # Plot the obstacles
        for obs_index in range(self.num_obstacles):
            x, y = self.obstacle_positions[obs_index].tolist()
            ax.add_patch(
                patches.Circle(
                    (x, y), self.obstacle_radii[obs_index].item(), color="r", alpha=0.5
                )
            )

        # Plot the agents' trajectories
        if agent_positions is not None:
            for agent_index in range(self.num_agents):
                if agent_positions.shape[0] == 1:
                    x, y = agent_positions[0, agent_index].tolist()
                    ax.add_patch(
                        patches.Circle((x, y), self.agent_radii[agent_index].item())
                    )
                else:
                    ax.plot(
                        agent_positions[:, agent_index, 0],
                        agent_positions[:, agent_index, 1],
                        marker="o",
                        label=f"Agent {agent_index}",
                        # set size to agent radius
                        markersize=self.agent_radii[agent_index].item() * 10,
                    )

        # Plot the agents' start and goal positions
        for agent_index in range(self.num_agents):
            (sx, sy) = self._as_numpy(self.agent_start_positions[agent_index])
            (ex, ey) = self._as_numpy(self.agent_end_positions[agent_index])
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

    @overload
    @classmethod
    def from_json(cls, entry) -> "Problem[torch.Tensor]": ...

    @overload
    @classmethod
    def from_json(cls, entry, type: Literal["torch"]) -> "Problem[torch.Tensor]": ...

    @overload
    @classmethod
    def from_json(cls, entry, type: Literal["numpy"]) -> "Problem[np.ndarray]": ...

    @classmethod
    def from_json(cls, entry, type: Literal["torch", "numpy"] = "torch") -> Any:
        return cls(
            num_timesteps=entry["num_timesteps"],
            agent_start_positions=_tensor(
                entry["agents"]["start_positions"], type=type
            ),
            agent_end_positions=_tensor(entry["agents"]["end_positions"], type=type),
            agent_radii=_tensor(entry["agents"]["radii"], type=type),
            agent_max_speeds=_tensor(entry["agents"]["max_speeds"], type=type),
            agent_reference_trajectory=None,
            obstacle_positions=_tensor(entry["obstacles"]["positions"], type=type),
            obstacle_radii=_tensor(entry["obstacles"]["radii"], type=type),
        )


def save_video(problem: Problem, agent_positions: np.ndarray, path: str | Path):
    buf = io.BytesIO()
    images = []

    for step in range(problem.num_timesteps):
        plt.clf()
        problem.visualize(plt.gca(), agent_positions[step : step + 1])
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


def main():
    with open("instances_data/instances_dense.json", "r") as f:
        data = json.load(f)

    for prob in data:
        problem = Problem.from_json(prob)
        base_dir = f"results/n_robots={problem.num_agents}_{prob['sample_idx']}"
        os.makedirs(base_dir, exist_ok=True)

        results = {True: [], False: []}
        for use_coarse_to_fine in [True, False]:
            print(f"use_coarse_to_fine: {use_coarse_to_fine}")
            i = 0
            batch_size = 1
            # Penalty terms
            rho_agent_obstacle = 1
            rho_agent_agent = 1
            rho_lowlevel_vel = 20
            # Lagrange multipliers
            horizon = 65
            nu_agent_agent = torch.zeros(
                (batch_size, horizon, problem.num_agents, problem.num_agents)
            )
            nu_agent_obstacle = torch.zeros(
                (batch_size, horizon, problem.num_agents, problem.num_obstacles)
            )
            nu_lowlevel_vel = torch.zeros((batch_size, horizon - 1, problem.num_agents))
            transition_by = 1000
            update_alm_every = 10
            update_alm_after = 1000
            update_alm_penalty_term_iterations = 100
            update_alm_penalty_terms_until = (
                update_alm_after + update_alm_penalty_term_iterations * update_alm_every
            )
            final_optimization_steps = 1000
            total_steps = update_alm_penalty_terms_until + final_optimization_steps
            energy_lowlevel_weight = 0  # to 10
            energy_highlevel_weight = 10.0  # to 0
            rate = 200 ** (1 / update_alm_penalty_term_iterations)
            lr = 0.1

            curves = {
                "agent_agent_penalties": [],
                "agent_obstacle_penalties": [],
                "lowlevel_vel_penalties": [],
                "energy_lowlevel": [],
            }

            sol = SolutionValue(
                agent_agent_distances=torch.zeros(
                    (batch_size, horizon, problem.num_agents, problem.num_agents)
                ),
                agent_obstacle_distances=torch.zeros(
                    (batch_size, horizon, problem.num_agents, problem.num_obstacles)
                ),
                agent_positions=torch.randn(
                    (batch_size, horizon, problem.num_agents, 2), requires_grad=True
                ),
            )

            opt = torch.optim.Adam([sol.agent_positions], lr=lr)

            for step in tqdm.tqdm(range(total_steps)):
                plan_highlevel = sol.agent_positions[:, ::4]
                highlevel_vel_sq = (
                    # (b, t, a, d)
                    (plan_highlevel[:, 1:, :, :] - plan_highlevel[:, :-1, :, :])
                    .pow(2)
                    .sum(-1)
                )
                energy_highlevel = highlevel_vel_sq.sum((1, 2))  # (b)
                lowlevel_vel_sq = (
                    (
                        # (b, t, a, d)
                        sol.agent_positions[:, 1:, :, :]
                        - sol.agent_positions[:, :-1, :, :]
                    )
                    .pow(2)
                    .sum(-1)
                )
                energy_lowlevel = lowlevel_vel_sq.sum((1, 2))  # (b)

                highlevel_vel_penalty = (  # noqa: F841
                    highlevel_vel_sq[highlevel_vel_sq > (0.05**2)] - (0.05**2)
                ).sum()
                lowlevel_vel_constraint = torch.relu(lowlevel_vel_sq - (0.05**2))

                # (t, a, o)
                obstacle_center_dist_sq = (
                    (
                        # (b, t, a, d) -> (b, t, a, 1, d)
                        sol.agent_positions.unsqueeze(-2)
                        # (o, d) -> (1, 1, 1, o, d)
                        - problem.obstacle_positions.unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    .pow(2)
                    .sum(-1)
                )
                obstacle_signed_distances = obstacle_center_dist_sq - (
                    # (a) -> (1, 1, a, 1)
                    # (o) -> (1, 1, 1, o)
                    problem.agent_radii.view(1, 1, -1, 1)
                    + problem.obstacle_radii.view(1, 1, 1, -1)
                ).pow(2)
                agent_obstacle_constraint = torch.relu(-obstacle_signed_distances)

                # (b, t, a1, a2)
                agent_center_dist_sq = (
                    (
                        # (b, t, a, d) -> (b, t, a1, 1, d)
                        # (b, t, a, d) -> (b, t, 1, a2, d)
                        sol.agent_positions.unsqueeze(-2)
                        - sol.agent_positions.unsqueeze(-3)
                    )
                    .pow(2)
                    .sum(-1)
                )
                agent_signed_distances = agent_center_dist_sq - (
                    # (a1) -> (1, 1, a1, 1)
                    # (a2) -> (1, 1, 1, a2)
                    problem.agent_radii.view(1, 1, -1, 1)
                    + problem.agent_radii.view(1, 1, 1, -1)
                ).pow(2)
                agent_signed_distances = (
                    agent_signed_distances
                    # to ignore self-interactions in the relu
                    + torch.eye(problem.num_agents).view(
                        1, problem.num_agents, problem.num_agents
                    )
                    * 1e6
                )
                agent_agent_constraint = torch.relu(-agent_signed_distances)

                # (b, t, a, o)
                obstacle_penalties = (
                    agent_obstacle_constraint.pow(2) * rho_agent_obstacle / 2
                    + agent_obstacle_constraint * nu_agent_obstacle
                ).sum((1, 2, 3))
                # (b, t, a1, a2)
                agent_penalties = (
                    agent_agent_constraint.pow(2) * rho_agent_agent / 2
                    + agent_agent_constraint * nu_agent_agent
                ).sum((1, 2, 3))
                # (b, t, a)
                lowlevel_vel_penalties = (
                    lowlevel_vel_constraint.pow(2) * rho_lowlevel_vel / 2
                    # + lowlevel_vel_constraint * nu_lowlevel_vel
                )

                if use_coarse_to_fine:
                    energy_lowlevel_weight = (
                        10 * min(step, transition_by) / transition_by
                    )
                    energy_highlevel_weight = 10 - energy_lowlevel_weight
                else:
                    energy_lowlevel_weight = 10
                    energy_highlevel_weight = 0

                loss = (
                    energy_highlevel.sum() * energy_highlevel_weight
                    + energy_lowlevel.sum() * energy_lowlevel_weight
                    + obstacle_penalties.sum()
                    + agent_penalties.sum()
                    + lowlevel_vel_penalties.sum()
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                if (step + 1) % update_alm_every == 0 and step > update_alm_after:
                    # Update Lagrange multipliers.
                    nu_agent_agent += rho_agent_agent * agent_agent_constraint.detach()
                    nu_agent_obstacle += (
                        rho_agent_obstacle * agent_obstacle_constraint.detach()
                    )
                    nu_lowlevel_vel += (
                        rho_lowlevel_vel * lowlevel_vel_constraint.detach()
                    )

                    if step < update_alm_penalty_terms_until:
                        rho_agent_obstacle *= rate
                        rho_agent_agent *= rate
                        rho_lowlevel_vel *= rate

                with torch.no_grad():
                    sol.agent_positions[:, 0, :, :] = (
                        problem.agent_start_positions.unsqueeze(0)
                    )
                    sol.agent_positions[:, -1, :, :] = (
                        problem.agent_end_positions.unsqueeze(0)
                    )

                curves["agent_agent_penalties"].append(agent_penalties.tolist())
                curves["agent_obstacle_penalties"].append(obstacle_penalties.tolist())
                curves["lowlevel_vel_penalties"].append(
                    lowlevel_vel_penalties.sum((-1, -2)).tolist()
                )
                curves["energy_lowlevel"].append(energy_lowlevel.tolist())

            plt.clf()

            plt.subplot(2, 3, 1)
            plt.plot(curves["agent_agent_penalties"], label="agent_agent_penalties")
            plt.title("AA")
            plt.yscale("log")

            plt.subplot(2, 3, 2)
            plt.plot(
                curves["agent_obstacle_penalties"], label="agent_obstacle_penalties"
            )
            plt.title("AO")
            plt.yscale("log")

            plt.subplot(2, 3, 3)
            plt.plot(curves["lowlevel_vel_penalties"], label="lowlevel_vel_penalties")
            plt.title("V")
            plt.yscale("log")

            plt.subplot(2, 3, 4)
            plt.plot(curves["energy_lowlevel"], label="energy_lowlevel")
            plt.title("E")
            plt.yscale("log")

            ax = plt.subplot(2, 3, 5)
            problem.visualize(ax, sol.agent_positions[0].detach().cpu().numpy())
            plt.tight_layout()
            plt.savefig(f"{base_dir}/optimization_curve_{use_coarse_to_fine}_{i}.png")

            save_video(
                problem,
                sol.agent_positions[0].detach().cpu().numpy(),
                f"{base_dir}/trajectory_{use_coarse_to_fine}_{i}.mp4",
            )

            results[use_coarse_to_fine].append(
                {
                    "agent_penalties": agent_penalties.tolist(),
                    "obstacle_penalties": obstacle_penalties.tolist(),
                    "energy_lowlevel": energy_lowlevel.tolist(),
                    "lowlevel_vel_penalties": lowlevel_vel_penalties.sum().tolist(),
                }
            )

            with open(f"{base_dir}/solution_{use_coarse_to_fine}_{i}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "agent_positions": sol.agent_positions.detach().cpu().numpy(),
                        "scores": results[use_coarse_to_fine][-1],
                    },
                    f,
                )

        # write results to JSON.
        with open(f"{base_dir}/summary.json", "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
