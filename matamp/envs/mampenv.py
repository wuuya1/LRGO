import time
import copy
import numpy as np
from math import sqrt
from matamp.configs import config
from matamp.envs.kd_tree import KdTree
from draw.plt2d import plt_visulazation
from matamp.tools.utils import pi_2_pi, l2norm


class MACAEnv(object):
    def __init__(self):
        self.num_agents = 0
        self.agents = None
        self.targets = []
        self.obstacles = []
        self.polyobs_vertices = []
        self.boundaries = []
        self.task_area = []
        self.exit_area = []
        self.kdTree = None
        self.ta_name = None
        self.solve_ta_cost = 0.0

    def set_agents(self, agents, targets=None, obstacles=None, obs_verts=None, ta_name='lsta'):
        self.agents = agents
        self.num_agents = len(self.agents)
        self.task_area = self.agents[0].task_area
        self.exit_area = self.agents[0].exit_area
        self.targets = targets
        self.obstacles = obstacles
        self.polyobs_vertices = obs_verts if obs_verts is not None else []
        self.ta_name = ta_name
        t1 = time.time()
        if self.ta_name == 'cbba':
            self.cal_assignment_scheme_cbba(scale=1000)
        else:
            self.cal_assignment_scheme_tas(scale=1000)
        t2 = time.time()
        self.solve_ta_cost = t2 - t1

        # Visualize the initial global trajectory points.
        paths_smooth = []
        for a in self.agents:
            a.solve_ta_cost = self.solve_ta_cost
            path_smooth = a.path
            paths_smooth.append(path_smooth)
        print(self.agents[0].all_agent_rewards, self.solve_ta_cost)
        plt_visulazation(paths_smooth, self.agents, self.targets, self.obstacles, self.task_area, self.exit_area)

        self.kdTree = KdTree(self.agents, self.obstacles, self.targets, self.polyobs_vertices)
        self.kdTree.buildObstacleTree()
        self.kdTree.buildPolyObstacleTree()
        self.kdTree.buildTargetTree()

    def step(self, actions):
        self._take_action(actions)
        which_agents_done = self.is_done()
        return which_agents_done

    def _take_action(self, actions):
        self.kdTree.buildAgentTree()

        num_actions_per_agent = 2  # speed, alpha
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        # Compute next action.
        for agent_index, agent in enumerate(self.agents):
            if agent.is_at_goal or agent.is_collision or agent.is_out_of_max_time or agent.is_run_done:
                t1 = time.time()
                all_actions[agent_index, :] = np.array([0., 0.])
                t2 = time.time()
            else:
                t1 = time.time()
                other_agents = copy.copy(self.agents)
                dict_comm = {'other_agents': other_agents, 'targets': self.targets, 'obstacles': self.obstacles}
                all_actions[agent_index, :] = agent.policy.find_next_action(dict_comm, agent, self.kdTree)
                t2 = time.time()
            solve_time_cost = t2 - t1
            agent.solve_time_cost += solve_time_cost

        # Update velocity and position.
        for i, agent in enumerate(self.agents):
            update_velocitie(agent, all_actions[i, :])
            if not agent.is_at_goal:
                agent.step_num += 1
            self.check_agent_state(agent)  # Check collision.

        # Check collision.
        for i, agent in enumerate(self.agents):
            if agent.is_collision:
                info = 'agent' + str(agent.id) + ': collision'
                print(info)

    def is_done(self):
        for ag in self.agents:
            ag_cleared_tar = 0
            for tar in ag.targets:
                if tar.is_cleared:
                    ag_cleared_tar += 1
            c1 = l2norm(ag.pos_global_frame, ag.goal_global_frame) <= ag.near_goal_threshold
            if c1:
                if not ag.is_at_goal:
                    ag.history_pos.append(np.array([ag.goal_global_frame[0], ag.goal_global_frame[1], 0]))
                    ag.path_node[-1] = np.array([ag.goal_global_frame[0], ag.goal_global_frame[1], 0])
                    ag.history_speed.append(ag.current_speed)
                    ag.path_speed.append(ag.current_speed)
                    ag.is_at_goal = True
            else:
                if ag.in_ending_area and not ag.is_at_goal and (ag.stuck_num > 20 or ag.ending_step > 320):
                    ag.history_pos.append(np.array([ag.goal_global_frame[0], ag.goal_global_frame[1], 0]))
                    ag.path_node[-1] = np.array([ag.goal_global_frame[0], ag.goal_global_frame[1], 0])
                    ag.history_speed.append(ag.current_speed)
                    ag.path_speed.append(ag.current_speed)
                    ag.is_at_goal = True
            if ag.is_at_goal or ag.is_collision or ag.is_out_of_max_time:
                ag.is_run_done = True
        is_done_condition = np.array([ag.is_run_done for ag in self.agents])
        check_is_done_condition = np.logical_and.reduce(is_done_condition)
        return check_is_done_condition

    def check_agent_state(self, agent):
        for ob in self.targets:
            is_agent_target = False
            if ob.is_sl and ob.is_cleared:
                continue
            else:
                for tar in agent.targets:
                    if ob.id == tar.id:
                        is_agent_target = True
                if is_agent_target:
                    continue
            dis_a_ob = l2norm(agent.pos_global_frame, ob.pos_global_frame)
            if dis_a_ob < (agent.radius + ob.radius):
                agent.is_collision = True
                info = 'collision: agent' + str(agent.id) + ' and ' 'target' + str(ob.id)
                print(info)

        for ob in self.obstacles:
            if not ob.is_poly:
                dis_a_ob = l2norm(agent.pos_global_frame, ob.pos_global_frame)
                if dis_a_ob <= (agent.radius + ob.radius):
                    agent.is_collision = True
                    info = 'collision: agent' + str(agent.id) + ' and ' 'circle obstacle' + str(ob.id)
                    print(info)
            else:
                dists = []
                for vertice in ob.vertices_:
                    dist = l2norm(agent.pos_global_frame, vertice.point_)
                    dists.append(dist)
                dis_a_ob = min(dists)
                if dis_a_ob <= agent.radius:
                    agent.is_collision = True
                    info = 'collision: agent' + str(agent.id) + ' and ' 'polygonal obstacle' + str(ob.id)
                    print(info)

        for ag in self.agents:
            if ag.id == agent.id: continue
            dis_a_agent = l2norm(agent.pos_global_frame, ag.pos_global_frame)
            c1 = dis_a_agent < (agent.radius + ag.radius)
            if c1:
                if not ag.is_at_goal:
                    ag.is_collision = True
                if not agent.is_at_goal:
                    agent.is_collision = True
                info = 'collision: agent' + str(agent.id) + ' and ' 'agent' + str(ag.id)
                print(info)
        if agent.travel_dist > agent.max_run_dist:
            agent.is_out_of_max_time = True
            info = ' agent' + str(agent.id) + ' is_out_of_max_time'
            print(info)

    def cal_assignment_scheme_tas(self, scale=1):
        tasks = []
        for i in range(len(self.targets)):
            if not self.targets[i].is_cleared:
                tasks.append(self.targets[i])
        if len(tasks) > 0:
            for agent in self.agents:
                agent.taPolicy.set_paras(agent, tasks, self.obstacles, scale=scale)    # The scale is determined by the distance; at the kilometer level, it is equal to 1e3.

            # phase1: Initialize the formation of the task sample set for the agent.
            for agent in self.agents:
                agent.taPolicy.form_task_sample()

            # phase2: Task assignment
            while True:
                # phase2.1: Update the agent's wa* and ja* values.
                for agent in self.agents:
                    agent.taPolicy.update_wa_ja()
                # phase2.2: The agent broadcasts its ID, wa*, and ja* information.
                message_pool = [agent.taPolicy.send_message() for agent in self.agents]
                # phase2.3: The agent receives the broadcasted ID, wa*, and ja* information.
                for agent in self.agents:
                    agent.taPolicy.receive_message(message_pool)
                # phase2.4: Resolve allocation conflicts with maximum consistency and perform task assignment.
                for agent in self.agents:
                    if self.ta_name == 'lrca':
                        a_star, j_star_a_star = agent.taPolicy.co_consensus()
                    else:
                        a_star, j_star_a_star = agent.taPolicy.max_consensus()
                    agent.taPolicy.allocation_task(a_star, j_star_a_star)

                assigned_num = 0
                converged = []
                for agent in self.agents:
                    assigned_num += len(agent.taPolicy.Ta)
                    if len(agent.taPolicy.Na) == 0:
                        converged.append(True)
                        assigend_tarID = agent.cleared_tarID + agent.taPolicy.p
                        agent.rewards = agent.taPolicy.reward_score(assigend_tarID, agent.initial_pos[:2], self.targets, scl=scale)
                if sum(converged) == len(self.agents):
                    all_agent_rewards = 0.0
                    assignment_results = {}
                    agent_ids = []
                    for agent in self.agents:
                        all_agent_rewards += agent.rewards
                        assignment_results[agent.id] = agent.cleared_tarID + agent.taPolicy.p
                        agent_ids.append(agent.id)
                    break
            print(assignment_results)
            for agent in self.agents:
                if agent.all_agent_rewards < all_agent_rewards:
                    if agent.all_agent_rewards == 0.0:
                        agent.first_all_agent_rewards = all_agent_rewards
                    agent.all_agent_rewards = all_agent_rewards
                    agent.tarId_list = agent.taPolicy.p
                    agent.assigend_tarID = agent.cleared_tarID + agent.taPolicy.p
                    agent.assignment_results = assignment_results
                    self.add_agent_path_by_allocation(agent)

    def cal_assignment_scheme_cbba(self, scale=1):
        robot_num, task_num = len(self.agents), len(self.targets)
        for agent in self.agents:
            agent.taPolicy.set_paras(agent, self.targets, self.obstacles, scale=scale)
        G = np.ones((robot_num, robot_num))  # Fully connected network

        t = 0  # Iteration number
        max_time = 600.
        t1 = time.time()
        while True:
            converged_list = []  # Converged List

            # Phase 1: Auction Process
            for agent in self.agents:
                # select task by local information
                agent.taPolicy.build_bundle()

            # Communicating: Send winning bid list to neighbors (depend on env)
            Y = None  # Store the y, z, and s data of the other agents, excluding oneself.
            message_pool = [agent.taPolicy.send_message() for agent in self.agents]
            for idx, agent in enumerate(self.agents):
                # Recieve winning bidlist from neighbors
                g = G[idx]

                connected, = np.where(g == 1)
                connected = list(connected)
                connected.remove(idx)

                if len(connected) > 0:
                    Y = {neighbor_id: message_pool[neighbor_id] for neighbor_id in connected}
                else:
                    Y = None

                agent.taPolicy.receive_message(Y)

            # Phase 2: Consensus Process
            for agent in self.agents:
                # Update local information and decision
                if Y is not None:
                    converged = agent.taPolicy.update_task()
                    converged_list.append(converged)
                    if converged:
                        assigend_tarID = agent.cleared_tarID + agent.taPolicy.p
                        agent.rewards = agent.taPolicy.reward_score(assigend_tarID, agent.initial_pos[:2], self.targets,
                                                                    scl=scale)
            t += 1
            t2 = time.time()
            if sum(converged_list) == robot_num or t2 - t1 > max_time:
                all_agent_rewards = 0.0
                assignment_results = []
                for agent in self.agents:
                    all_agent_rewards += agent.rewards
                    assignment_results.append(agent.cleared_tarID + agent.taPolicy.p)
                break
        for agent in self.agents:
            if agent.all_agent_rewards < all_agent_rewards:
                if agent.all_agent_rewards == 0.0:
                    agent.first_all_agent_rewards = all_agent_rewards
                agent.all_agent_rewards = all_agent_rewards
                agent.tarId_list = agent.taPolicy.p
                agent.assigend_tarID = agent.cleared_tarID + agent.taPolicy.p
                agent.assignment_results = assignment_results
                self.add_agent_path_by_allocation(agent)

    def add_agent_path_by_allocation(self, agent):
        agent.path = []
        agent.targets = []
        for cleared_tar_idx in agent.cleared_tarID:
            agent.targets.append(self.targets[cleared_tar_idx])
        if not agent.is_assign_again:
            agent.is_assign_again = True
            agent.path.insert(0, [agent.pos_global_frame[0], agent.pos_global_frame[1], 0])  # Insert the starting position at the beginning.
        for j in range(len(agent.tarId_list)):
            tar_idx = agent.tarId_list[j]
            target = self.targets[tar_idx]
            agent.targets.append(target)
            agent.path.append([target.pos_global_frame[0], target.pos_global_frame[1], 0])
        agent.path.append([agent.goal_global_frame[0], agent.goal_global_frame[1], 0])  # Add the endpoint.
        # The reverse of the original path goes from the starting point to the endpoint.
        agent.path.reverse()  # The order of the path after reversing goes from the endpoint to the starting point.
        update_goal_pos(agent)


def update_velocitie(agent, action):
    selected_speed = action[0]
    selected_heading = pi_2_pi(action[1] + agent.heading_global_frame)

    dx = selected_speed * agent.dt_nominal * np.cos(selected_heading)
    dy = selected_speed * agent.dt_nominal * np.sin(selected_heading)
    pos = agent.pos_global_frame + np.array([dx, dy])
    may_collision = False
    if agent.is_in_ending_area():
        agent.in_ending_area = True
        agent.ending_step += 1
        neigbor_ag = None
        for obj in agent.neighbors:
            obj = obj[0]
            if neigbor_ag is None and obj.is_agent:
                neigbor_ag = obj
            if neigbor_ag is not None:
                break
        if neigbor_ag is not None:
            agent_rad = agent.radius + 0.5
            ag_rad = neigbor_ag.radius + 0.5
            combinedRadius2 = agent_rad + ag_rad
            if l2norm(neigbor_ag.pos_global_frame, pos) < combinedRadius2:
                may_collision = True
    else:
        agent.in_ending_area = False
        agent.ending_step = 0

    if may_collision:
        selected_speed = 0.0
        selected_heading = agent.heading_global_frame
    else:
        agent.pos_global_frame += np.array([dx, dy])
    length = sqrt(dx ** 2 + dy ** 2)
    if agent.in_ending_area:        # Once in the termination area, travel time and distance are no longer calculated.
        travel_time = 0.0
        length = 0.0
    elif selected_speed < 1e-5 and not agent.in_ending_area:
        travel_time = agent.dt_nominal
    else:
        travel_time = length / selected_speed
    agent.travel_dist += length
    agent.travel_time += travel_time
    agent.max_yaw_rate = max(abs(action[1]), agent.max_yaw_rate)

    agent.delta_heading_global_frame = pi_2_pi(selected_heading - agent.heading_global_frame)
    agent.speed_global_frame = selected_speed
    agent.heading_global_frame = selected_heading
    agent.vel_global_frame[0] = round(selected_speed * np.cos(selected_heading), 5)
    agent.vel_global_frame[1] = round(selected_speed * np.sin(selected_heading), 5)
    agent.to_vector()


def update_goal_pos(agent):
    if len(agent.path) > 1:
        if agent.in_direction == 'west' or agent.in_direction == 'east':
            goal = np.array([agent.goal_global_frame[0], agent.path[1][1]])
            agent.goal_global_frame = np.array([goal[0], goal[1]])
        else:
            goal = np.array([agent.path[1][0], agent.goal_global_frame[1]])
            agent.goal_global_frame = np.array([goal[0], goal[1]])
    else:
        if agent.in_direction == 'west' or agent.in_direction == 'east':
            goal = np.array([agent.goal_global_frame[0], agent.pos_global_frame[1]])
            agent.goal_global_frame = np.array([goal[0], goal[1]])
        else:
            goal = np.array([agent.pos_global_frame[0], agent.goal_global_frame[1]])
            agent.goal_global_frame = np.array([goal[0], goal[1]])
    agent.path[0] = [agent.goal_global_frame[0], agent.goal_global_frame[1], 0]


