import numpy as np
import pandas as pd
from matamp.tools.utils import l2norm, get_boundaries, dist_sq_point_line_segment, l2normsq, sqr, takeSecond


class Agent(object):
    def __init__(self, start_pos, goal_pos, vel, radius, pref_speed, maxSpeed, minSpeed, maxAngular, turning_radius,
                 policy, taPolicy, id, tid, member_num, dt=0.1, near_goal_threshold=0.5, sampling_size=0.1, group=0,
                 capacity=5, expansion=0.5, task_area=None, exit_area=None, in_direction='west'):
        # agent's physical information
        self.is_agent = True
        self.is_poly = False
        self.is_obstacle = False
        self.is_sl = False
        self.targetLoad = capacity
        self.targets = []
        self.policy = policy()
        self.taPolicy = taPolicy()
        self.tarId_list = []         # The current task assignment scheme
        self.is_assign_again = False
        self.assigend_tarID = []     # The final task assignment scheme
        self.group = group
        self.member_num = member_num
        self.task_area = task_area
        self.exit_area = exit_area
        self.is_exit_area = True if len(exit_area) > 0 else False
        self.in_direction = in_direction
        self.min_x, self.max_x, self.min_y, self.max_y = get_boundaries(task_area)
        if len(exit_area) > 0:
            self.min_xe, self.max_xe, self.min_ye, self.max_ye = get_boundaries(exit_area)
        else:
            self.min_xe, self.max_xe, self.min_ye, self.max_ye = 0.0, 0.0, 0.0, 0.0

        self.goal_pos = np.array(goal_pos, dtype='float64')
        self.initial_pos = np.array(start_pos, dtype='float64')
        self.pos_global_frame = np.array(start_pos[:2], dtype='float64')
        self.goal_global_frame = np.array(goal_pos[:2], dtype='float64')

        self.heading_global_frame = np.pi
        self.goal_heading_frame = np.pi
        self.initial_heading = np.pi
        self.vel_global_unicycle = np.array([0.0, 0.0], dtype='float64')

        self.vel_global_frame = np.array(vel)
        self.turning_radius = turning_radius
        self.pref_speed = pref_speed
        self.radius = radius
        self.id = id
        self.near_goal_threshold = near_goal_threshold
        self.sampling_size = sampling_size

        self.tid = {self.id: tid}
        self.expansion = expansion
        self.startPointdis = 5*radius
        self.is_expansion = False
        self.first_all_agent_rewards = 0.0
        self.cleared_tarID = []
        self.view_path = []
        self.assignment_results = []

        self.timeStep = dt
        self.maxSpeed = maxSpeed
        self.minSpeed = minSpeed
        self.maxAngular = maxAngular
        self.maxAccel = 1.0
        self.neighbors = []         # [(obj, dis), (obj, dis)]
        self.poly_obs_neighbors = []    # [(obj, dis), (obj, dis)]
        self.dt_nominal = dt
        self.maxNeighbors = 15
        self.timeHorizon = 10.0
        self.timeHorizonObst = 10.0
        self.neighborDist = 1000.
        self.forwardDist = 1000.
        self.in_ending_area = False
        self.ending_step = 0

        self.path = []
        self.path_node = []         # Global sequence of path points
        self.path_speed = []

        self.travel_time = 0.0
        self.travel_dist = 0.0
        self.solve_ta_cost = 0.0
        self.all_agent_rewards = 0.0
        self.cleared_num = 0
        self.solve_time_cost = 0.0
        self.rewards = 0.0
        self.step_num = 0
        self.max_yaw_rate = 0.0
        self.max_path_plan_cost = 0.0
        self.max_path_replan_cost = 0.0
        self.max_init_cost = 0.0
        self.max_count = 0
        self.max_solve_time = 0.0

        self.straight_path_length = l2norm(start_pos[:2], goal_pos[:2])-0.5  # For computing Distance Rate.
        self.desire_steps = int(self.straight_path_length / (pref_speed*dt))  # For computing Time Rate.

        self.history_pos = []
        self.history_speed = []
        self.current_speed = 0.0
        self.is_at_goal = False
        self.is_run_done = False
        self.is_collision = False
        self.is_use_dubins = False
        self.is_out_of_max_time = False
        self.stuck_num = 0

        self.max_run_dist = 5.0 * l2norm(start_pos[:2], goal_pos[:2])
        self.ANIMATION_COLUMNS = ['pos_x', 'pos_y', 'alpha', 'vel_x', 'vel_y',
                                  'gol_x', 'gol_y', 'radius', 'speed']
        self.history_info = pd.DataFrame(columns=self.ANIMATION_COLUMNS)

    def insertAgentNeighbor(self, other_agent, rangeSq):
        if self.id != other_agent.id:
            distSq = l2normsq(self.pos_global_frame, other_agent.pos_global_frame)
            if distSq < sqr(self.radius + other_agent.radius) and distSq < rangeSq:     # COLLISION!
                if not self.is_collision:
                    self.is_collision = True
                    self.neighbors.clear()

                if len(self.neighbors) == self.maxNeighbors:
                    self.neighbors.pop()
                self.neighbors.append((other_agent, distSq))
                self.neighbors.sort(key=takeSecond)
                if len(self.neighbors) == self.maxNeighbors:
                    rangeSq = self.neighbors[-1][1]
            elif not self.is_collision and distSq < rangeSq:
                if len(self.neighbors) == self.maxNeighbors:
                    self.neighbors.pop()
                self.neighbors.append((other_agent, distSq))
                self.neighbors.sort(key=takeSecond)
                if len(self.neighbors) == self.maxNeighbors:
                    rangeSq = self.neighbors[-1][1]
        # return rangeSq

    def insertObstacleNeighbor(self, obstacle, rangeSq):
        # if not obstacle.is_poly:
        distSq = l2normsq(self.pos_global_frame, obstacle.pos_global_frame)
        # print()
        if distSq < sqr(self.radius + obstacle.radius) and distSq < rangeSq:  # COLLISION!
            if not self.is_collision and not obstacle.is_poly:
                self.is_collision = True
                self.neighbors.clear()

            if len(self.neighbors) == self.maxNeighbors:
                self.neighbors.pop()
            self.neighbors.append((obstacle, distSq))
            self.neighbors.sort(key=takeSecond)
            if len(self.neighbors) == self.maxNeighbors:
                rangeSq = self.neighbors[-1][1]
        elif not self.is_collision and distSq < rangeSq:
            if len(self.neighbors) == self.maxNeighbors:
                self.neighbors.pop()
            self.neighbors.append((obstacle, distSq))
            self.neighbors.sort(key=takeSecond)
            if len(self.neighbors) == self.maxNeighbors:
                rangeSq = self.neighbors[-1][1]
        # return rangeSq

    def insertPolyObstacleNeighbor(self, poly_vertice, rangeSq):
        """
        Inserts a static obstacle neighbor into the set of neighbors of this agent.

        Args:
            poly_vertice (Vertice): The number of the polygonal obstacle to be inserted.
            rangeSq (float): The squared range around this agent.
        """
        nextObstacle = poly_vertice.next_
        distSq = dist_sq_point_line_segment(poly_vertice.point_, nextObstacle.point_, self.pos_global_frame)

        if distSq < sqr(self.radius) and distSq < rangeSq:
            if not self.is_collision:
                self.is_collision = True
                self.poly_obs_neighbors.clear()

            self.poly_obs_neighbors.append((poly_vertice, distSq))
            i = len(self.poly_obs_neighbors) - 1
            while i != 0 and distSq < self.poly_obs_neighbors[i - 1][1]:
                self.poly_obs_neighbors[i] = self.poly_obs_neighbors[i - 1]
                i -= 1
            self.poly_obs_neighbors[i] = (poly_vertice, distSq)

        elif distSq < rangeSq:
            self.poly_obs_neighbors.append((poly_vertice, distSq))
            i = len(self.poly_obs_neighbors) - 1
            while i != 0 and distSq < self.poly_obs_neighbors[i - 1][1]:
                self.poly_obs_neighbors[i] = self.poly_obs_neighbors[i - 1]
                i -= 1
            self.poly_obs_neighbors[i] = (poly_vertice, distSq)

    def insertTargetNeighbor(self, target, rangeSq):
        if not target.is_cleared:
            distSq = l2normsq(self.pos_global_frame, target.pos_global_frame)

            if distSq < sqr(self.radius + target.radius) and distSq < rangeSq:  # COLLISION!
                if not self.is_collision and (target not in self.targets and not target.is_cleared):
                    self.is_collision = True
                    self.neighbors.clear()

                if len(self.neighbors) == self.maxNeighbors:
                    self.neighbors.pop()
                self.neighbors.append((target, distSq))
                self.neighbors.sort(key=takeSecond)
                if len(self.neighbors) == self.maxNeighbors:
                    rangeSq = self.neighbors[-1][1]
            elif not self.is_collision and distSq < rangeSq:
                if len(self.neighbors) == self.maxNeighbors:
                    self.neighbors.pop()
                self.neighbors.append((target, distSq))
                self.neighbors.sort(key=takeSecond)
                if len(self.neighbors) == self.maxNeighbors:
                    rangeSq = self.neighbors[-1][1]
        # return rangeSq

    def is_in_ending_area(self):
        is_finished = np.logical_and.reduce([tar.is_cleared for tar in self.targets])
        # simulation in large-scale scenarios
        # if self.is_exit_area:
        #     min_xe, max_xe = self.min_xe + 5 * self.radius, self.max_xe - 5 * self.radius
        #     min_ye, max_ye = self.min_ye + 2.5 * self.radius, self.max_ye - 2.5 * self.radius
        #     is_in = min_xe < self.pos_global_frame[0] < max_xe and min_ye < self.pos_global_frame[1] < max_ye

        # real experiment
        if self.is_exit_area:
            min_xe, max_xe = self.min_xe + 1.1 * self.radius, self.max_xe - 1.1 * self.radius
            min_ye, max_ye = self.min_ye + 1.1 * self.radius, self.max_ye - 1.1 * self.radius
            is_in = min_xe < self.pos_global_frame[0] < max_xe and min_ye < self.pos_global_frame[1] < max_ye
            is_in = is_finished and is_in
        else:
            is_in = False
        return is_in

    def to_vector(self):
        """ Convert the agent's attributes to a single global state vector. """
        global_state_dict = {
            'radius': self.radius,
            'pref_speed': self.pref_speed,
            'pos_x': self.pos_global_frame[0],
            'pos_y': self.pos_global_frame[1],
            'gol_x': self.goal_global_frame[0],
            'gol_y': self.goal_global_frame[1],
            'vel_x': self.vel_global_frame[0],
            'vel_y': self.vel_global_frame[1],
            'alpha': self.heading_global_frame,
            'speed': self.current_speed

        }
        global_state = np.array([val for val in global_state_dict.values()])
        animation_columns_dict = {}
        for key in self.ANIMATION_COLUMNS:
            animation_columns_dict.update({key: global_state_dict[key]})
        self.history_info = self.history_info.append([animation_columns_dict], ignore_index=True)