import copy
import time
from functools import partial

import numpy as np
import torch
from loguru import logger
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Set,
    SolverFactory,
    Var,
    minimize,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition


def circle_signed_distance(agent_x, agent_y, obs_x, obs_y, agent_rad, obs_rad):
    center_distance = (agent_x - obs_x) ** 2 + (agent_y - obs_y) ** 2
    min_allowed_distance = (agent_rad + obs_rad) ** 2
    return center_distance - min_allowed_distance


def objective_fn(model, start_idx, x, rho, horizons, num_dyn_obs):
    expr = 0.0

    # Minimize the distance between predicted path and the true path
    for i in model.I:
        if i < start_idx:
            continue  # Skip prefix in cost
        for j in model.J:
            expr += sum((model.P_out[i, j, k] - x[i, j, k]) ** 2 for k in model.Dim)

    # minimize total distance
    for i in model.I:
        if i < start_idx:
            continue  # Skip prefix in cost
        for j in model.J:
            if i < model.I.last():
                expr += sum(
                    (model.P_out[i + 1, j, k] - model.P_out[i, j, k]) ** 2
                    for k in model.Dim
                )

    for i in model.I:  # Horizon
        # Skip prefix in cost
        if i < start_idx:
            continue

        # Obstacle avoidance
        for j in model.J:  # Agents
            for k in model.K:  # Obstacles
                temp_expr = model.d_o[i, j, k] - circle_signed_distance(
                    agent_x=model.P_out[i, j, 0],
                    agent_y=model.P_out[i, j, 1],
                    obs_x=model.obs_pos_x[k],
                    obs_y=model.obs_pos_y[k],
                    agent_rad=model.agent_rads[j],
                    obs_rad=model.obs_rads[k],
                )
                expr += model.nu_o[i, j, k] * temp_expr + rho * temp_expr**2

        # Collision avoidance
        for j, k in model.AgentPairs:  # Pairs of agents
            temp_expr = model.d_a[i, (j, k)] - circle_signed_distance(
                agent_x=model.P_out[i, j, 0],
                agent_y=model.P_out[i, j, 1],
                obs_x=model.P_out[i, k, 0],
                obs_y=model.P_out[i, k, 1],
                agent_rad=model.agent_rads[j],
                obs_rad=model.agent_rads[k],
            )
            expr += model.nu_a[i, (j, k)] * temp_expr + rho * temp_expr**2

    # Dynamic obstacles
    if num_dyn_obs > 0:
        for j in model.J:
            for k in model.KDyn:
                t = model.dyn_obs_t[k]
                # Only consider if t is within horizon and >= start_idx
                if start_idx <= t < horizons:
                    temp_expr = model.d_dyn[j, k] - circle_signed_distance(
                        agent_x=model.P_out[t, j, 0],
                        agent_y=model.P_out[t, j, 1],
                        obs_x=model.dyn_obs_x[k],
                        obs_y=model.dyn_obs_y[k],
                        agent_rad=model.agent_rads[j],
                        obs_rad=model.dyn_obs_rad[k],
                    )
                    expr += model.nu_dyn[j, k] * temp_expr + rho * temp_expr**2

    return expr


def solve_L_aug(
    x,
    agent_starts,
    agent_goals,
    agent_rads,
    max_speeds,
    obs_pos,
    obs_rads,
    horizons,
    nu_o,
    nu_a,
    rho,
    init_traj4proj_array,
    init_d_o,
    init_d_a,
    start_idx=0,
    other_agents=None,
    nu_dyn=None,
    init_d_dyn=None,
):
    num_agents = init_traj4proj_array.shape[1]
    num_obs = obs_pos.shape[0]
    max_speeds = max_speeds * np.ones(num_agents)

    # Dynamic obstacles handling
    if other_agents is None:
        other_agents = []
    num_dyn_obs = len(other_agents)

    model = ConcreteModel()

    # Define sets
    model.I = RangeSet(0, horizons - 1)  # Time steps
    model.J = RangeSet(0, num_agents - 1)  # Agents
    model.K = RangeSet(0, num_obs - 1)  # Obstacles
    model.Dim = RangeSet(0, 1)  # Dimensions (x and y)
    model.AgentPairs = Set(
        initialize=[
            (j, k) for j in range(num_agents) for k in range(num_agents) if k > j
        ]
    )
    model.KDyn = RangeSet(0, max(0, num_dyn_obs - 1))  # Dynamic Obstacles indices

    initial_p_out = {
        (i, j, k): init_traj4proj_array[i, j, k]
        for i in range(horizons)
        for j in range(num_agents)
        for k in range(2)
    }
    initial_d_o = {
        (i, j, k): init_d_o[i, j, k] + 1e-8
        for i in range(horizons)
        for j in range(num_agents)
        for k in range(num_obs)
    }
    initial_d_a = {
        (i, (j, k)): init_d_a[i, j, k] + 1e-8
        for i in range(horizons)
        for j in range(num_agents)
        for k in range(num_agents)
        if k > j
    }

    # Initialize d_dyn
    initial_d_dyn = {}
    if num_dyn_obs > 0:
        for j in range(num_agents):
            for k in range(num_dyn_obs):
                initial_d_dyn[(j, k)] = init_d_dyn[j, k] + 1e-8

    model.P_out = Var(
        model.I, model.J, model.Dim, bounds=(-1, 1), initialize=initial_p_out
    )
    model.d_o = Var(
        model.I, model.J, model.K, within=NonNegativeReals, initialize=initial_d_o
    )
    model.d_a = Var(
        model.I, model.AgentPairs, within=NonNegativeReals, initialize=initial_d_a
    )

    if num_dyn_obs > 0:
        model.d_dyn = Var(
            model.J, model.KDyn, within=NonNegativeReals, initialize=initial_d_dyn
        )
    else:
        # Create dummy variable to avoid Pyomo errors if accessed in loops (though we guard loops)
        model.d_dyn = Var(model.J, model.KDyn, within=NonNegativeReals)

    # Parameters (converted from arrays to dictionaries for Pyomo)
    model.agent_rads = Param(
        model.J, initialize={j: agent_rads[j] for j in range(num_agents)}
    )
    model.max_speeds = Param(
        model.J, initialize={j: max_speeds[j] for j in range(num_agents)}
    )
    model.obs_rads = Param(model.K, initialize={k: obs_rads[k] for k in range(num_obs)})
    model.obs_pos_x = Param(
        model.K, initialize={k: obs_pos[k, 0] for k in range(num_obs)}
    )
    model.obs_pos_y = Param(
        model.K, initialize={k: obs_pos[k, 1] for k in range(num_obs)}
    )

    # Dynamic obstacles parameters
    dyn_obs_x_dict = {}
    dyn_obs_y_dict = {}
    dyn_obs_rad_dict = {}
    dyn_obs_t_dict = {}
    for k, obs in enumerate(other_agents):
        # obs: [pos, rad, time_step]
        dyn_obs_x_dict[k] = obs[0][0]
        dyn_obs_y_dict[k] = obs[0][1]
        dyn_obs_rad_dict[k] = float(obs[1])
        dyn_obs_t_dict[k] = int(obs[2])

    model.dyn_obs_x = Param(model.KDyn, initialize=dyn_obs_x_dict)
    model.dyn_obs_y = Param(model.KDyn, initialize=dyn_obs_y_dict)
    model.dyn_obs_rad = Param(model.KDyn, initialize=dyn_obs_rad_dict)
    model.dyn_obs_t = Param(model.KDyn, initialize=dyn_obs_t_dict)

    # Lagrangian multipliers
    model.nu_o = Param(
        model.I,
        model.J,
        model.K,
        initialize={
            (i, j, k): nu_o[i, j, k]
            for i in range(horizons)
            for j in range(num_agents)
            for k in range(num_obs)
        },
    )
    model.nu_a = Param(
        model.I,
        model.AgentPairs,
        initialize={
            (i, j, k): nu_a[i, j, k]
            for i in range(horizons)
            for j in range(num_agents)
            for k in range(num_agents)
            if k > j
        },
    )
    model.nu_dyn = Param(
        model.J,
        model.KDyn,
        initialize=(
            {
                (j, k): nu_dyn[j, k]
                for j in range(num_agents)
                for k in range(num_dyn_obs)
            }
            if num_dyn_obs > 0
            else {}
        ),
    )
    # Objective
    rule = partial(
        objective_fn,
        start_idx=start_idx,
        x=x,
        rho=rho,
        horizons=horizons,
        num_dyn_obs=num_dyn_obs,
    )
    model.obj = Objective(rule=rule, sense=minimize)

    # Start and end position constraints
    model.start_position_constraints_x = Constraint(
        model.J, rule=lambda model, j: model.P_out[0, j, 0] == agent_starts[j, 0]
    )
    model.start_position_constraints_y = Constraint(
        model.J, rule=lambda model, j: model.P_out[0, j, 1] == agent_starts[j, 1]
    )
    model.end_position_constraints_x = Constraint(
        model.J,
        rule=lambda model, j: model.P_out[horizons - 1, j, 0] == agent_goals[j, 0],
    )
    model.end_position_constraints_y = Constraint(
        model.J,
        rule=lambda model, j: model.P_out[horizons - 1, j, 1] == agent_goals[j, 1],
    )

    # Fix prefix positions logic removed - now handled by assignment outside solve_L_aug
    # if fixed_prefix_positions is not None and len(fixed_prefix_positions) > 1:
    #    ... (removed constraints) ...

    # Speed constraints
    def speed_constraint_rule(model, i, j):
        if i >= horizons - 1:
            return Constraint.Skip

        return (
            sum(
                (model.P_out[i, j, k] - model.P_out[i + 1, j, k]) ** 2
                for k in model.Dim
            )
            <= model.max_speeds[j] ** 2
        )

    model.speed_constraints = Constraint(model.I, model.J, rule=speed_constraint_rule)

    # Solve the model
    solver = SolverFactory("ipopt")
    results = solver.solve(model, tee=False)

    # Extract solution
    x_iteration = np.zeros((horizons, num_agents, 2))
    d_o_value = np.zeros((horizons, num_agents, num_obs))
    d_a_value = np.zeros((horizons, num_agents, num_agents))
    d_dyn_value = np.zeros((num_agents, max(1, num_dyn_obs)))  # Ensure at least size 1

    flag = int(
        (results.solver.status == SolverStatus.ok)
        and (results.solver.termination_condition == TerminationCondition.optimal)
    )
    if flag:
        for i in range(horizons):
            for j in range(num_agents):
                x_iteration[i, j, 0] = value(model.P_out[i, j, 0])
                x_iteration[i, j, 1] = value(model.P_out[i, j, 1])
                for k_obs in range(num_obs):
                    d_o_value[i, j, k_obs] = value(model.d_o[i, j, k_obs])
                for k_agent in range(num_agents):
                    if k_agent > j:
                        d_a_value[i, j, k_agent] = value(model.d_a[i, (j, k_agent)])

        if num_dyn_obs > 0:
            for j in range(num_agents):
                for k in range(num_dyn_obs):
                    d_dyn_value[j, k] = value(model.d_dyn[j, k])

    return x_iteration, d_o_value, d_a_value, d_dyn_value, flag


def compute_all_signed_distances(
    agent_pos, obs_pos, agent_rads, obs_rads, start_idx=0, other_agents=[]
):
    horizons = agent_pos.shape[0]
    num_agents = agent_pos.shape[1]
    num_obs = obs_pos.shape[0]
    num_dyn = len(other_agents)

    # initialize the gradient of nu_o and nu_a
    d_o = np.zeros((horizons, num_agents, num_obs))
    d_a = np.zeros((horizons, num_agents, num_agents))
    d_dyn = np.zeros((num_agents, max(1, num_dyn)))

    # calculate the gradient of nu_o and nu_a
    for i in range(start_idx, horizons):
        for j in range(num_agents):
            for k in range(num_obs):
                d_o[i, j, k] = circle_signed_distance(
                    agent_x=agent_pos[i, j, 0],
                    agent_y=agent_pos[i, j, 1],
                    obs_x=obs_pos[k, 0],
                    obs_y=obs_pos[k, 1],
                    agent_rad=agent_rads[j],
                    obs_rad=obs_rads[k],
                )
            for k in range(j + 1, num_agents):
                d_a[i, j, k] = circle_signed_distance(
                    agent_x=agent_pos[i, j, 0],
                    agent_y=agent_pos[i, j, 1],
                    obs_x=agent_pos[i, k, 0],
                    obs_y=agent_pos[i, k, 1],
                    agent_rad=agent_rads[j],
                    obs_rad=agent_rads[k],
                )

    if num_dyn > 0:
        for j in range(num_agents):
            for k in range(num_dyn):
                obs = other_agents[k]
                pos = obs[0]
                rad = obs[1]
                t = int(obs[2])
                if start_idx <= t < horizons:
                    d_dyn[j, k] = circle_signed_distance(
                        agent_x=agent_pos[t, j, 0],
                        agent_y=agent_pos[t, j, 1],
                        obs_x=pos[0],
                        obs_y=pos[1],
                        agent_rad=agent_rads[j],
                        obs_rad=rad,
                    )

    return d_o, d_a, d_dyn


def apply_projection_alm(
    x, projection_info, hard_conds, first_projection, init_traj4proj, proj_params
):
    """
    x: tensor of shape (batch_size, horizon, state_dim)
    """

    # rebuttal
    grad_nu_o_set = []
    grad_nu_a_set = []

    # Get hard conditions from projection_info
    agents_starts_states_normalized = hard_conds[0][0, :]
    agents_goals_states_normalized = hard_conds[63][0, :]

    # get the number of the agents
    num_agents = int(agents_starts_states_normalized.shape[0] / 4)

    # unnormalize the x
    agents_starts_states = projection_info.unnormalize_trajectories(
        agents_starts_states_normalized
    )
    agents_goals_states = projection_info.unnormalize_trajectories(
        agents_goals_states_normalized
    )
    x = projection_info.unnormalize_trajectories(x)

    # get the position of the agents starts and goals
    agents_starts_pos = []
    agents_goals_pos = []
    for i in range(num_agents):
        agents_starts_pos.append(
            [agents_starts_states[i * 2], agents_starts_states[i * 2 + 1]]
        )
        agents_goals_pos.append(
            [agents_goals_states[i * 2], agents_goals_states[i * 2 + 1]]
        )
    agents_starts_pos = torch.tensor(agents_starts_pos)
    agents_goals_pos = torch.tensor(agents_goals_pos)
    # get the velocity of the agents starts and goals
    agents_starts_v = torch.zeros((num_agents, 2))
    agents_goals_v = torch.zeros((num_agents, 2))

    # get the radius of the agents
    # agents_rads = projection_info.robot.radius*1
    agents_rads = 0.05
    # get an array of the agents' radius
    agents_rads = agents_rads * np.ones((num_agents))

    # get the obstacles
    obj_list = list(projection_info.env.obj_all_list)

    obs_pos_1 = obj_list[0].fields[0].centers
    obs_rads_1 = obj_list[0].fields[0].radii

    obs_pos_2 = obj_list[1].fields[0].centers
    obs_rads_2 = obj_list[1].fields[0].radii

    obs_pos = torch.cat([obs_pos_1, obs_pos_2], dim=0)
    obs_rads = torch.cat([obs_rads_1, obs_rads_2], dim=0)

    num_obs = obs_rads.shape[0]

    horizons = x.shape[1]
    traj_index = np.arange(horizons)

    agents_max_speeds = proj_params["agents_max_speeds"]

    x_candidate = x[0, :, :].clone()

    agents_starts_pos = agents_starts_pos.cpu().numpy()
    agents_goals_pos = agents_goals_pos.cpu().numpy()
    obs_pos = obs_pos.cpu().numpy()
    obs_rads = obs_rads.cpu().numpy()
    x_candidate = x_candidate.cpu().numpy()

    # Optional fixed prefix from proj_params (assumed for planned agent index 0)
    prefix_positions = None
    start_idx = 0
    if "prefix_traj" in proj_params:
        try:
            prefix_positions = np.asarray(proj_params["prefix_traj"], dtype=float)
            if prefix_positions is not None and len(prefix_positions) > 0:
                start_idx = min(len(prefix_positions), horizons)
        except Exception:
            prefix_positions = None
            start_idx = 0

    # Initialize the lagrange multipliers
    nu_o = np.zeros((horizons, num_agents, num_obs))
    nu_a = np.zeros((horizons, num_agents, num_agents))
    p_init = np.zeros((horizons, num_agents, 2))
    d_o_dummy = np.zeros((horizons, num_agents, num_obs))
    d_a_dummy = np.zeros((horizons, num_agents, num_agents))
    if not first_projection:
        for i in range(horizons):
            for j in range(num_agents):
                p_init[i, j, 0] = x_candidate[traj_index[i], j * 2]
                p_init[i, j, 1] = x_candidate[traj_index[i], j * 2 + 1]

    else:
        # use the interpolation between the start and end points
        for i in range(horizons):
            for j in range(num_agents):
                p_init[i, j, :] = agents_starts_pos[j, :] + (
                    agents_goals_pos[j, :] - agents_starts_pos[j, :]
                ) * (i / (horizons - 1))

    # change the shape of x_candidate
    x_candidate_alm = np.zeros((horizons, num_agents, 2))
    for i in range(horizons):
        for j in range(num_agents):
            x_candidate_alm[i, j, 0] = x_candidate[traj_index[i], j * 2]
            x_candidate_alm[i, j, 1] = x_candidate[traj_index[i], j * 2 + 1]

    # Dynamic obstacles from proj_params
    other_agents = []
    if "other_agents" in proj_params and proj_params["other_agents"] is not None:
        other_agents = proj_params["other_agents"]
    num_dyn_obs = len(other_agents)

    # Initialize dynamic obstacle variables
    nu_dyn = np.zeros((num_agents, max(1, num_dyn_obs)))

    # calculate the nu_o and nu_a
    # agent_rads, obs_pos, obs_rads, p_value, d_o, d_a
    init_traj4proj_array = np.stack(
        [init_traj4proj[k] for k in init_traj4proj.keys()], axis=1
    )
    init_d_o, init_d_a, init_d_dyn = compute_all_signed_distances(
        init_traj4proj_array,
        obs_pos,
        agents_rads,
        obs_rads,
        start_idx=start_idx,
        other_agents=other_agents,
    )

    # Set lagrange parameters
    rho = proj_params["rho"]
    rho_factor = proj_params["rho_factor"]

    grad_o = []
    grad_a = []

    t_begin = time.time()
    success = 0
    for steps in range(proj_params["alm_iteration"]):
        tt1 = time.time()

        # select only the agent index for what we're solving
        agent_idx = next(iter(init_traj4proj.keys()))
        agents_starts_pos_ = agents_starts_pos[agent_idx : agent_idx + 1, :]
        agents_goals_pos_ = agents_goals_pos[agent_idx : agent_idx + 1, :]
        agents_rads_ = agents_rads[agent_idx : agent_idx + 1]
        # agents_max_speeds = np.array([agents_max_speeds[agent_idx]])

        x_iteration, d_o_dummy, d_a_dummy, d_dyn_dummy, solver_flag = solve_L_aug(
            x_candidate_alm,
            agents_starts_pos_,
            agents_goals_pos_,
            agents_rads_,
            agents_max_speeds,
            obs_pos,
            obs_rads,
            horizons,
            nu_o,
            nu_a,
            rho,
            init_traj4proj_array,
            init_d_o,
            init_d_a,
            start_idx=start_idx,
            other_agents=other_agents,
            nu_dyn=nu_dyn,
            init_d_dyn=init_d_dyn,
        )

        # If solver failed, return original input without projection
        if solver_flag == 0:
            logger.warning(
                f"Solver failed at step {steps} - returning original input without projection"
            )
            return x, 0  # Return original x and success=0

        d_o_real, d_a_real, d_dyn_real = compute_all_signed_distances(
            x_iteration, obs_pos, agents_rads_, obs_rads, start_idx, other_agents
        )

        # Enforce prefix positions after optimization step
        if prefix_positions is not None and len(prefix_positions) > 1:
            # Assume planned agent is index 0
            # prefix_positions shape: (L, 2), where L <= horizons
            L = min(len(prefix_positions), x_iteration.shape[0])
            # Only overwrite the prefix part.
            # Note: x_iteration shape is (horizons, num_agents, 2)
            # Overwriting agent 0's path for the first L steps
            x_iteration[:L, 0, :] = prefix_positions[:L, :]

        grad_nu_o = d_o_dummy - d_o_real
        grad_nu_a = d_a_dummy - d_a_real
        grad_nu_dyn = d_dyn_dummy - d_dyn_real

        tolerance = proj_params["tolerance"]

        logger.info(f"{tolerance=} {grad_nu_o=} {grad_nu_a=} {grad_nu_dyn=}")

        # threshold at zero: 1 if feasible (grad ≥ 0), else 0
        fea_o_real = np.sum((d_o_real <= -tolerance).astype(int))
        # We only care about violations after start_idx, so we should zero out any potential 'noise' before start_idx
        # (though loop above already ensures zeros before start_idx)
        fea_a_real = np.sum((d_a_real <= -tolerance).astype(int))
        fea_dyn_real = (
            np.sum((d_dyn_real <= -tolerance).astype(int))
            if len(other_agents) > 0
            else 0
        )

        init_traj4proj_array = x_iteration
        init_d_o, init_d_a, init_d_dyn = d_o_dummy, d_a_dummy, d_dyn_dummy

        grad_nu_o_norm = np.mean(np.linalg.norm(grad_nu_o))
        grad_nu_a_norm = np.mean(np.linalg.norm(grad_nu_a))
        grad_nu_dyn_norm = (
            np.mean(np.linalg.norm(grad_nu_dyn)) if num_dyn_obs > 0 else 0
        )
        logger.info(
            f"ALM step {steps}: {grad_nu_o_norm=}, {grad_nu_a_norm=}, {grad_nu_dyn_norm=}, {fea_o_real=}, {fea_a_real=}, {fea_dyn_real=}"
        )

        grad_nu_o_set.append(grad_nu_o_norm)
        grad_nu_a_set.append(grad_nu_a_norm)

        # Visualize trajectory (and dynamic obstacles)
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.clf()
        ax = plt.gca()
        # Plot the obstacles
        for k in range(obs_pos.shape[0]):
            ax.add_patch(
                patches.Circle(
                    (obs_pos[k, 0], obs_pos[k, 1]), obs_rads[k], color="r", alpha=0.5
                )
            )
        # Plot the dynamic obstacles
        for obs in other_agents:
            pos = obs[0]
            rad = obs[1]
            ax.add_patch(
                patches.Circle((pos[0], pos[1]), rad, color="orange", alpha=0.5)
            )
        # Plot the agents' trajectories
        for j in range(x_iteration.shape[1]):
            plt.plot(
                x_iteration[:, j, 0],
                x_iteration[:, j, 1],
                marker="o",
                label=f"Agent {j}",
            )
        # Plot the agents' start and goal positions
        for j in range(num_agents):
            start_pos = agents_starts_pos[j]
            goal_pos = agents_goals_pos[j]
            plt.plot(
                start_pos[0],
                start_pos[1],
                marker="o",
                color="green",
                markersize=10,
                label=f"Start {j}",
            )
            plt.plot(
                goal_pos[0],
                goal_pos[1],
                marker="*",
                color="blue",
                markersize=10,
                label=f"Goal {j}",
            )

        plt.title(f"ALM Projection Step {steps}")
        plt.axis("equal")
        plt.pause(0.1)

        # Check convergence. Includes dynamic obstacles (norm_dyn, fea_dyn_real).
        grads_small = (
            grad_nu_o_norm <= 1e-3
            and grad_nu_a_norm <= 1e-3
            and grad_nu_dyn_norm <= 1e-3
        )
        constraints_ok = fea_o_real + fea_a_real + fea_dyn_real < 1
        if grads_small or constraints_ok:
            success = 1
            break

        # update the lagrange parameters
        rho = rho * rho_factor
        nu_o = nu_o + rho * grad_nu_o
        nu_a = nu_a + rho * grad_nu_a
        nu_dyn = nu_dyn + rho * grad_nu_dyn if num_dyn_obs > 0 else nu_dyn

        grad_o.append(grad_nu_o_norm)
        grad_a.append(grad_nu_a_norm)

        tt2 = time.time()

        logger.info(f"ALM step {steps} completed in {tt2 - tt1} seconds.")

    t_end = time.time()
    logger.info(f"Total time: {t_end - t_begin} seconds.")

    x_projected = copy.copy(x_candidate)
    for i in range(horizons):
        for j in range(x_iteration.shape[1]):
            x_projected[traj_index[i], j * 2] = x_iteration[i, j, 0]
            x_projected[traj_index[i], j * 2 + 1] = x_iteration[i, j, 1]

    for i in range(0, x.shape[0]):
        x_projected = torch.tensor(x_projected).to(x.dtype).to(x.device)
        x_projected = projection_info.normalize_trajectories(x_projected)
        x[i, :, :] = x_projected

    return x, success
