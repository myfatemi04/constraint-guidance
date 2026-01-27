"""
Visualizer to help debug the agent-obstacle score function. Allows you to click on a position on the map and modify the $\\sigma$ value to view the impact each obstacle has on the agent's overall score function.
"""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Circle
from matplotlib.quiver import Quiver
from matplotlib.widgets import Slider

from ael.agent_obstacle_score import (
    compute_agent_obstacle_distance_batched,
    compute_agent_obstacle_score_batched,
    compute_r1_r2_batched,
)
from ael.optimize import Problem
from ael.visualize import visualize


class AgentObstacleDebugger:
    def __init__(self, problem: Problem[np.ndarray], ax: Axes):
        self.problem = problem
        self.ax = ax
        self.x: float = 0.0
        self.y: float = 0.0
        self.sigma: float = 1.0
        self.quivers: list[tuple[Quiver, float, float]] = []
        self.position_circle: Circle | None = None
        self.agent_radius: float = problem.agent_radii[0]

        # Visualize the problem
        visualize(problem, ax)

        # Initialize quivers for each obstacle
        self._setup_quivers()
        self.render()

    def _setup_quivers(self):
        """Initialize quivers over each obstacle with random vectors."""
        for obstacle in range(self.problem.obstacle_positions.shape[0]):
            # Random initial vectors
            u, v = np.random.randn(2)

            quiver = self.ax.quiver(
                self.problem.obstacle_positions[obstacle, 0],
                self.problem.obstacle_positions[obstacle, 1],
                u,
                v,
                scale=5,
                color="red",
                alpha=0.7,
                width=0.005,
            )
            self.quivers.append(
                (
                    quiver,
                    self.problem.obstacle_positions[obstacle, 0],
                    self.problem.obstacle_positions[obstacle, 1],
                )
            )

    def render(self):
        """Update quivers based on current x, y, and sigma values."""
        # Update or create position circle
        if self.position_circle is None:
            self.position_circle = cast(
                Circle,
                self.ax.add_patch(
                    plt.Circle(
                        (self.x, self.y), self.sigma, color="blue", alpha=0.5, zorder=10
                    )
                ),
            )
        else:
            self.position_circle.center = (self.x, self.y)
            self.position_circle.radius = self.sigma

        sigma_batch = np.full((self.problem.obstacle_positions.shape[0],), self.sigma)
        agent_x_B = np.full((self.problem.obstacle_positions.shape[0],), self.x)
        agent_y_B = np.full((self.problem.obstacle_positions.shape[0],), self.y)
        obs_x_B = self.problem.obstacle_positions[:, 0]
        obs_y_B = self.problem.obstacle_positions[:, 1]
        obs_rad_B = self.problem.obstacle_radii + self.agent_radius

        d_a_o_B = compute_agent_obstacle_distance_batched(
            agent_x_B, agent_y_B, obs_x_B, obs_y_B
        )
        r1_B, r2_B = compute_r1_r2_batched(obs_rad_B, d_a_o_B)
        scores_B = compute_agent_obstacle_score_batched(
            agent_x_B,
            agent_y_B,
            obs_x_B,
            obs_y_B,
            obs_rad_B,
            sigma_batch,
            r1_B,
            r2_B,
            d_a_o_B,
            n_integral=20000,
        )

        print(
            agent_x_B[0],
            agent_y_B[0],
            obs_x_B[0],
            obs_y_B[0],
            sigma_batch[0],
            r1_B[0],
            r2_B[0],
            d_a_o_B[0],
            scores_B[0],
        )

        # Update each quiver with new random vectors scaled by sigma
        for score, (quiver, ox, oy) in zip(scores_B, self.quivers):
            # Update quiver by setting new U, V data
            quiver.set_UVC(score[0], score[1])

        self.ax.figure.canvas.draw_idle()

    def on_click(self, event):
        """Handle mouse click events."""
        if (
            event.button is MouseButton.LEFT
            and event.xdata is not None
            and event.ydata is not None
            and event.inaxes is self.ax
        ):
            self.x = event.xdata
            self.y = event.ydata
            print(f"Clicked at ({self.x:.2f}, {self.y:.2f})")
            self.render()

    def on_sigma_change(self, val):
        """Handle sigma slider changes."""
        self.sigma = val
        print(f"Sigma: {self.sigma:.2f}")
        self.render()


def main():
    # Load data and create problem
    with open("instances_data/instances_dense.json", "r") as f:
        data = json.load(f)

    problem = Problem.from_json(data[0], type="numpy")

    problem.obstacle_positions = np.array([[0.782, 0.793]])
    problem.obstacle_radii = np.array([0.05])

    # Create figure with space for slider
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    # Create debugger instance
    debugger = AgentObstacleDebugger(problem, ax)

    # Add sigma slider
    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
    sigma_slider = Slider(ax_slider, "Sigma", 0.1, 5.0, valinit=1.0)
    sigma_slider.on_changed(debugger.on_sigma_change)

    # Connect click event
    fig.canvas.mpl_connect("button_press_event", debugger.on_click)

    plt.show()


if __name__ == "__main__":
    main()
