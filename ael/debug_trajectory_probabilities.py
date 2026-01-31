"""
Interactive visualizer to debug trajectory probabilities from evaluate_trajectory_unscaled_probabilities_factorized.
Allows dragging agent positions at each timestep to see how noise affects the different probability factors.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from ael.problem import Problem
from ael.score_function import evaluate_trajectory_unscaled_probabilities_factorized
from ael.visualize import visualize


class TrajectoryProbabilityDebugger:
    def __init__(
        self,
        problem: Problem[np.ndarray],
        n_timesteps: int = 64,
        agent_agent_constraint_tolerance: float = 0.0,
        agent_obstacle_constraint_tolerance: float = 0.0,
        velocity_constraint_tolerance: float = 0.0,
    ):
        self.problem = problem
        self.n_timesteps = n_timesteps
        self.agent_agent_constraint_tolerance = agent_agent_constraint_tolerance
        self.agent_obstacle_constraint_tolerance = agent_obstacle_constraint_tolerance
        self.velocity_constraint_tolerance = velocity_constraint_tolerance

        # Create reference trajectory (linear interpolation between start and goal)
        # Shape: (n_timesteps, n_agents, 2)
        self.reference_trajectory = np.zeros((n_timesteps, problem.num_agents, 2))
        for agent_idx in range(problem.num_agents):
            self.reference_trajectory[:, agent_idx, :] = np.linspace(
                problem.agent_start_positions[agent_idx],
                problem.agent_end_positions[agent_idx],
                n_timesteps,
            )

        # Current noise (drag displacements) - shape: (n_timesteps, n_agents, 2)
        self.noise = np.zeros_like(self.reference_trajectory)

        # Dragging state
        self.dragging = False
        self.drag_timestep: int | None = None
        self.drag_agent: int | None = None
        self.drag_start_pos: tuple[float, float] | None = None

        # Hover highlighting state
        self.hovered_agent: int | None = None
        self.hovered_timestep: int | None = None
        self.original_colors: dict = {}

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 8))
        self.fig.suptitle("Trajectory Probability Debugger", fontsize=16)

        # Flatten axes for easier indexing
        self.axes_flat = self.axes.flatten()

        # Setup axes
        self.trajectory_ax = self.axes_flat[0]
        self.agent_agent_ax = self.axes_flat[1]
        self.agent_obstacle_ax = self.axes_flat[2]
        self.velocity_ax = self.axes_flat[3]
        self.kinetic_ax = self.axes_flat[4]
        self.overall_ax = self.axes_flat[5]

        self.trajectory_ax.set_title("Trajectory (drag points)")
        self.agent_agent_ax.set_title("Agent-Agent Constraints")
        self.agent_obstacle_ax.set_title("Agent-Obstacle Constraints")
        self.velocity_ax.set_title("Velocity Constraints")
        self.kinetic_ax.set_title("Kinetic Energy")
        self.overall_ax.set_title("Overall Probability")

        # Store plot elements
        self.trajectory_circles: list[list[Circle]] = []
        self.trajectory_agent_size_circles: list[list[Circle]] = []
        self.trajectory_lines = []

        # Initialize visualization
        self._setup_trajectory_plot()
        self.render()

        # Connect mouse events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def _setup_trajectory_plot(self):
        """Initialize the trajectory visualization."""
        # Draw obstacles and environment
        visualize(self.problem, self.trajectory_ax)

        # Draw reference trajectories and create draggable circles
        for agent_idx in range(self.problem.num_agents):
            # Plot reference trajectory
            (line,) = self.trajectory_ax.plot(
                self.reference_trajectory[:, agent_idx, 0],
                self.reference_trajectory[:, agent_idx, 1],
                "o-",
                alpha=0.3,
                markersize=3,
                label=f"Agent {agent_idx}",
            )
            self.trajectory_lines.append(line)

            # Create draggable circles for each timestep
            agent_circles = []
            agent_size_circles = []
            for t in range(self.n_timesteps):
                # Faint circle showing real agent size
                agent_radius = self.problem.agent_radii[agent_idx]
                agent_size_circle = Circle(
                    self.reference_trajectory[t, agent_idx],
                    agent_radius,
                    color=line.get_color(),
                    alpha=0.15,
                    linestyle="--",
                    fill=False,
                    linewidth=1,
                )
                self.trajectory_ax.add_patch(agent_size_circle)
                agent_size_circles.append(agent_size_circle)

                # Small circle for dragging
                circle = Circle(
                    self.reference_trajectory[t, agent_idx],
                    0.02,
                    color=line.get_color(),
                    alpha=0.6,
                    picker=5,
                )
                self.trajectory_ax.add_patch(circle)
                agent_circles.append(circle)
            self.trajectory_circles.append(agent_circles)
            self.trajectory_agent_size_circles.append(agent_size_circles)
        self.trajectory_ax.set_aspect("equal")

    def render(self):
        """Update all visualizations based on current noise."""
        # Get current trajectory (reference + noise)
        current_trajectory = self.reference_trajectory + self.noise

        # Update trajectory plot
        for agent_idx in range(self.problem.num_agents):
            # Update line
            self.trajectory_lines[agent_idx].set_data(
                current_trajectory[:, agent_idx, 0], current_trajectory[:, agent_idx, 1]
            )

            # Update circles
            for t in range(self.n_timesteps):
                pos = current_trajectory[t, agent_idx]
                self.trajectory_circles[agent_idx][t].center = pos
                self.trajectory_agent_size_circles[agent_idx][t].center = pos

        # Mark trajectory axes as needing redraw
        self.trajectory_ax.figure.canvas.flush_events()

        # Evaluate trajectory probabilities
        # Create noise batch with single sample (just the current noise)
        noise_batch = self.noise[np.newaxis, ...]  # Shape: (1, t, a, 2)

        evaluation = evaluate_trajectory_unscaled_probabilities_factorized(
            self.reference_trajectory,
            noise_batch,
            self.problem,
            self.agent_agent_constraint_tolerance,
            self.agent_obstacle_constraint_tolerance,
            self.velocity_constraint_tolerance,
        )

        # Extract results (remove batch dimension since we only have 1 sample)
        # Shape: (t, a)
        agent_agent = evaluation.agent_agent[0]
        agent_obstacle = evaluation.agent_obstacle[0]
        velocity = evaluation.velocity[0]
        kinetic = evaluation.kinetic_energy[0]
        overall = evaluation.overall[0]

        # Plot heatmaps for each factor
        self._plot_heatmap(self.agent_agent_ax, agent_agent, "Agent-Agent")
        self._plot_heatmap(self.agent_obstacle_ax, agent_obstacle, "Agent-Obstacle")
        self._plot_heatmap(self.velocity_ax, velocity, "Velocity")
        self._plot_heatmap(self.kinetic_ax, kinetic, "Kinetic Energy")
        self._plot_heatmap(self.overall_ax, overall, "Overall")

        self.fig.canvas.draw_idle()

    def _plot_heatmap(self, ax, data, title):
        """Plot a heatmap of probability factors."""
        ax.clear()
        im = ax.imshow(
            data.T,  # Transpose to have agents on y-axis
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Agent")
        ax.set_title(title)
        ax.set_yticks(range(self.problem.num_agents))

        # Add colorbar if not present
        if not hasattr(ax, "_colorbar"):
            ax._colorbar = self.fig.colorbar(im, ax=ax)

    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.trajectory_ax:
            return

        # Find which circle was clicked
        for agent_idx in range(self.problem.num_agents):
            for t in range(self.n_timesteps):
                circle = self.trajectory_circles[agent_idx][t]
                if circle.contains(event)[0]:
                    self.dragging = True
                    self.drag_agent = agent_idx
                    self.drag_timestep = t
                    self.drag_start_pos = (event.xdata, event.ydata)
                    return

    def on_release(self, event):
        """Handle mouse release events."""
        self.dragging = False
        self.drag_agent = None
        self.drag_timestep = None
        self.drag_start_pos = None

    def on_motion(self, event):
        """Handle mouse motion events."""
        # Handle hover highlighting
        if not self.dragging and event.inaxes != self.trajectory_ax:
            self._update_hover_highlighting(event)

        # Handle dragging
        if not self.dragging or event.xdata is None or event.ydata is None:
            return

        if (
            self.drag_agent is None
            or self.drag_timestep is None
            or self.drag_start_pos is None
        ):
            return

        # Calculate displacement from reference position
        current_pos = np.array([event.xdata, event.ydata])
        reference_pos = self.reference_trajectory[self.drag_timestep, self.drag_agent]
        displacement = current_pos - reference_pos

        # Update noise
        self.noise[self.drag_timestep, self.drag_agent] = displacement

        # Re-render
        self.render()

    def _update_hover_highlighting(self, event):
        """Update hover highlighting for circles based on heatmap or trajectory hover."""
        # Find which circle or heatmap is being hovered
        hovered_agent = None
        hovered_timestep = None

        # Check if hovering over trajectory circles
        for agent_idx in range(self.problem.num_agents):
            for t in range(self.n_timesteps):
                circle = self.trajectory_circles[agent_idx][t]
                if circle.contains(event)[0]:
                    hovered_agent = agent_idx
                    hovered_timestep = t
                    break
            if hovered_agent is not None:
                break

        # If not hovering over trajectory, check if hovering over heatmaps
        if hovered_agent is None and hovered_timestep is None:
            # Check each heatmap axis
            heatmap_axes = [
                self.agent_agent_ax,
                self.agent_obstacle_ax,
                self.velocity_ax,
                self.kinetic_ax,
                self.overall_ax,
            ]

            for ax in heatmap_axes:
                if event.inaxes == ax:
                    # Get pixel coordinates in the image
                    if event.xdata is not None and event.ydata is not None:
                        # Convert data coordinates to image indices
                        x = int(np.round(event.xdata))
                        y = int(np.round(event.ydata))

                        # Clamp to valid range
                        if (
                            0 <= x < self.n_timesteps
                            and 0 <= y < self.problem.num_agents
                        ):
                            hovered_timestep = x
                            hovered_agent = y
                    break

        # If hover state changed, update highlighting
        if (hovered_agent, hovered_timestep) != (
            self.hovered_agent,
            self.hovered_timestep,
        ):
            # Clear previous highlighting
            if self.hovered_agent is not None and self.hovered_timestep is not None:
                self.trajectory_circles[self.hovered_agent][
                    self.hovered_timestep
                ].set_alpha(0.6)
                self.trajectory_circles[self.hovered_agent][
                    self.hovered_timestep
                ].set_edgecolor(self.trajectory_lines[self.hovered_agent].get_color())

            # Apply new highlighting
            self.hovered_agent = hovered_agent
            self.hovered_timestep = hovered_timestep

            if hovered_agent is not None and hovered_timestep is not None:
                # Highlight only this single circle
                self.trajectory_circles[hovered_agent][hovered_timestep].set_alpha(1.0)
                self.trajectory_circles[hovered_agent][hovered_timestep].set_edgecolor(
                    "yellow"
                )

            self.fig.canvas.draw_idle()


def main():
    # Load a problem instance
    with open("instances_data/instances_simple.json", "r") as f:
        data = json.load(f)

    problem = Problem.from_json(data[0], type="numpy")
    problem.agent_max_speeds[:] *= 2.0

    # Create debugger
    _debugger = TrajectoryProbabilityDebugger(
        problem,
        n_timesteps=16,
        agent_agent_constraint_tolerance=0.0,
        agent_obstacle_constraint_tolerance=0.0,
        velocity_constraint_tolerance=0.0,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
