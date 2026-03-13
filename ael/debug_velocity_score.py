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

from ael.optimize import Problem
from ael.score_function import compute_velocity_score_batched_helper
from ael.visualize import visualize


class VelocityDebugger:
    # Relatively a duplicate of debug_agent_obstacle_score.
    def __init__(
        self,
        problem: Problem[np.ndarray],
        ax: Axes,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_sigma: float = 1.0,
    ):
        self.problem = problem
        self.ax = ax
        self.x: float = init_x
        self.y: float = init_y
        self.sigma: float = init_sigma
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
        for obstacle in range(self.problem.circular_obstacle_positions.shape[0]):
            # Random initial vectors
            u, v = np.random.randn(2)

            quiver = self.ax.quiver(
                self.problem.circular_obstacle_positions[obstacle, 0],
                self.problem.circular_obstacle_positions[obstacle, 1],
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
                    self.problem.circular_obstacle_positions[obstacle, 0],
                    self.problem.circular_obstacle_positions[obstacle, 1],
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

        sigma_B = np.full((self.problem.num_agents,), self.sigma)
        xy_T_B_D = np.stack([self.x, self.y], axis=-1)[:, None, :]

        scores_T = compute_velocity_score_batched_helper(
            xy_T_B_D,
            self.problem.agent_max_speeds,
            sigma_B,
            n_integral=20,
        )[:, 0, :]

        # Update each quiver with new random vectors scaled by sigma
        for score, (quiver, ox, oy) in zip(scores_T, self.quivers):
            # Update quiver by setting new U, V data
            quiver.set_UVC(score[0], score[1])

        score_magnitudes = np.linalg.norm(scores_T, axis=-1)
        print("Maximum score magnitude:", score_magnitudes.max())

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

    # problem.obstacle_positions = np.array([[0.782, 0.793]])
    # problem.obstacle_radii = np.array([0.05])

    # Create figure with space for slider
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)

    # Create debugger instance
    init_sigma = 0.47005208333333
    debugger = VelocityDebugger(
        problem,
        ax,
        init_x=0.5054223744292239,
        init_y=0.2875570776255709,
        init_sigma=init_sigma,
    )

    # Add sigma slider
    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
    sigma_slider = Slider(ax_slider, "Sigma", 0.001, 1.0, valinit=init_sigma)
    sigma_slider.on_changed(debugger.on_sigma_change)

    # Connect click event
    fig.canvas.mpl_connect("button_press_event", debugger.on_click)

    plt.show()


if __name__ == "__main__":
    main()
