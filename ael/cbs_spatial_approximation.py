import heapq
import json
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ael.problem import Problem
from ael.solve import (
    DEFAULT_SCHEDULES,
    Result,
    ScheduleEntry,
    ScoreComputationMethod,
    get_initial_paths_by_agent,
    solve,
)
from ael.visgraphprior import make_roadmap


@dataclass
class CBSConstraint:
    agent_i: int
    disallowed_location: np.ndarray
    disallowed_radius: float
    # time_step: int


@dataclass
class CBSNode:
    constraints: list[CBSConstraint]
    initial_paths: list[np.ndarray]


def make_constrained_graph(constraints: list[CBSConstraint], graph: nx.Graph):
    constrained_graph = graph.copy()
    for n in graph.nodes():
        if any(
            [
                np.linalg.norm(
                    constrained_graph.nodes[n]["pos"] - constraint.disallowed_location
                )
                < constraint.disallowed_radius
                for constraint in constraints
            ]
        ):
            constrained_graph.remove_node(n)
    return constrained_graph


def cbs_spatial_approximation(
    problem: Problem,
    score_computation_method: ScoreComputationMethod,
    schedule: list[ScheduleEntry],
) -> Result:
    graph = make_roadmap(problem)
    initial_paths = get_initial_paths_by_agent(problem, dt=1.0, graph=graph)
    initial_cost = (
        np.linalg.norm(initial_paths[1:] - initial_paths[:-1], axis=-1)
        # Sum across timesteps.
        .sum(axis=-1)
        # Sum across agents.
        .sum(axis=-1)
    )

    nodes = [
        (
            initial_cost,
            # this is so we don't need to implement tie-breaking logic in the heap, since CBSNodes aren't directly comparable
            np.random.rand(),
            CBSNode(constraints=[], initial_paths=initial_paths),
        )
    ]
    max_tries = 128
    tries = 0
    result: Result | None = None
    t0 = time.time()

    while len(nodes) > 0:
        tries += 1
        if tries > max_tries:
            print("Reached max tries, returning best effort solution.")
            # breaking out of the loop will return the best effort solution, which may still have constraint violations.
            break
        cost, _, node = heapq.heappop(nodes)
        # Try to solve this node.
        result = solve(
            problem,
            score_computation_method,
            schedule=schedule,
            initial_paths=node.initial_paths,
        )
        # Check for constraint violations. (t, a, a)
        aa_constraint_residuals_T_A_A = (
            result.constraint_satisfaction.agent_agent_constraint_residuals
        )
        # Find the earliest constraint violation.
        violation_indices = np.argwhere(aa_constraint_residuals_T_A_A > 1e-4)
        traj = result.trajectories[-1]

        # import matplotlib.pyplot as plt
        # print("visualizing")
        # Visualize result.
        # visualize(problem, plt.gca(), agent_positions=result.trajectories[-1])
        # # mark the first violation index
        # if violation_indices.shape[0] > 0:
        #     t, a1, a2 = violation_indices[0]
        #     plt.plot(traj[t, [a1, a2], 0], traj[t, [a1, a2], 1], "ro", markersize=10)
        # plt.show()

        if violation_indices.shape[0] == 0:
            # No constraint violations, return this solution.
            t1 = time.time()
            result.solve_time = t1 - t0
            return result
        else:
            t, a1_, a2_ = violation_indices[0]

            def add_cbs_subnodes(a1, a2):
                # Create a new constraint for a1 and add it to the list of constraints.
                r = problem.agent_radii[a1] + problem.agent_radii[a2]
                new_constraint_a1 = CBSConstraint(
                    agent_i=a1, disallowed_location=traj[t, a2], disallowed_radius=r
                )
                new_constraints_a1 = node.constraints + [new_constraint_a1]
                constrained_graph_a1 = make_constrained_graph(new_constraints_a1, graph)
                next_initial_paths = node.initial_paths.copy()
                next_initial_paths[a1] = get_initial_paths_by_agent(
                    problem, dt=1.0, graph=constrained_graph_a1
                )[a1]
                cost_a1 = (
                    np.linalg.norm(
                        next_initial_paths[a1][1:] - next_initial_paths[a1][:-1],
                        axis=-1,
                    )
                    # Sum across timesteps.
                    .sum(axis=-1)
                    # Sum across agents.
                    .sum(axis=-1)
                )
                node_a1 = CBSNode(
                    constraints=new_constraints_a1, initial_paths=next_initial_paths
                )
                heapq.heappush(nodes, (cost_a1, np.random.rand(), node_a1))

            add_cbs_subnodes(a1_, a2_)
            add_cbs_subnodes(a2_, a1_)

    assert result is not None

    t1 = time.time()
    result.solve_time = t1 - t0
    return result


if __name__ == "__main__":
    with open("instances_data/instances_dense.json", "r") as f:
        data = json.load(f)
    # a problem that was found not to converge originally
    hard_problem = Problem.from_json(data[284])
    result = cbs_spatial_approximation(
        hard_problem,
        score_computation_method=ScoreComputationMethod.APPROXIMATE_V0,
        schedule=DEFAULT_SCHEDULES[ScoreComputationMethod.APPROXIMATE_V0],
    )
