"""
@ Author: Gang Xu
@ Date: 2023.01
@ Function:
@ Paper:
"""
import copy
import numpy as np
from scipy import spatial
from matamp.tools.utils import get_boundaries, l2norm


class DSTAPolicy(object):
    def __init__(self):
        self.Na = []  # Set of task samples for agent A
        self.Ta = []  # Set of tasks assigned to agent A
        self.Wa = []  # Marginal earnings of all tasks in Na
        self.wa_star = 0.0  # Maximum marginal value of agent A
        self.ja_star = None  # Task id corresponding to the maximum marginal value
        self.vel = 1.0

        self.task_num = -1
        self.agent_num = -1
        self.tasks = None
        self.tasks_pos = None
        self.tasks_radius = None
        self.obstacles = None

        # Agent information
        self.id = -1
        self.pos = None
        self.exit_area = None
        self.task_area = None
        # list of targets' ID
        self.p = []
        # The allocation results of all normally functioning agents in the environment, stored in order of access, with the task IDs.
        self.assigned_result = None
        # Maximum Task Number
        self.L_t = -1
        # socre function parameters
        self.Lambda = 0.95
        self.c_bar = None
        # The IDs of other agents, their maximum marginal profit values, and the corresponding task IDs.
        self.receive_info = None
        self.ja_star_set = []
        # Distance information
        self.dis_u2t = None
        self.dis_t2t = None

    def set_paras(self, agent, tasks, obstacles, scale=1):
        self.task_num = len(tasks)
        self.agent_num = agent.member_num
        self.tasks = tasks
        self.tasks_pos = np.array([tasks[j].pos_global_frame[:2] for j in range(self.task_num)])
        self.tasks_radius = [tasks[j].radius for j in range(self.task_num)]
        self.obstacles = obstacles
        self.id = agent.id
        self.pos = agent.pos_global_frame
        self.exit_area = agent.exit_area
        self.task_area = agent.task_area
        self.L_t = agent.targetLoad
        self.assigned_result = [[] for _ in range(self.agent_num)]
        self.c_bar = np.ones(self.task_num)
        self.dis_u2t = self.dis_UT() / scale
        self.dis_t2t = self.dis_TT() / scale

    def send_message(self):
        return self.id, self.ja_star, self.wa_star

    def receive_message(self, receive_info):
        self.receive_info = receive_info

    def form_task_sample(self, p=1):
        """
        @param p: The sampling probability. It is important to note that in order to ensure
        consistent results for the same set of data, the probability should be set to 1. Otherwise,
        there may be a slight possibility of some tasks not being sampled.
        """
        # phase1: Initialize the formation of the task sample set for the agent.
        for j in range(self.task_num):
            if np.random.rand() <= p:
                self.Na.append(j)
        for j in self.Na:
            w_aj = self.score_scheme(j)
            self.Wa.append(w_aj)
        self.sort_Wa_Na()
        self.wa_star = self.Wa[0]
        self.ja_star = self.Na[0]

    def update_wa_ja(self):
        # phase2.1: Update the agent's wa* and ja* values.
        if len(self.Na) > 0 and len(self.Ta) < self.L_t:
            Wa = []
            for j in self.Na:
                w_aj = self.score_scheme(j)
                Wa.append(w_aj)
            Wa = np.array(Wa)
            max_idx = int(np.argmax(Wa))
            self.ja_star = self.Na[max_idx]
            self.wa_star = Wa[max_idx]

    def allocation_task(self, a_star, j_star_a_star):
        # phase2.3: Perform task assignment or remove the current task j.
        if a_star == self.id:
            self.Ta.append(self.ja_star)
            self.p.append(self.tasks[self.ja_star].id)
            idx = self.Na.index(self.ja_star)
            self.Na.pop(idx)
            self.Wa.pop(idx)
            self.wa_star = 0.0
        else:
            if j_star_a_star in self.Na:
                if j_star_a_star == self.Na[0]:
                    self.wa_star = 0.0
                idx = self.Na.index(j_star_a_star)
                self.Na.pop(idx)  # Na is one-to-one corresponding with Wa.
                self.Wa.pop(idx)  # Na is one-to-one corresponding with Wa.

    def score_scheme(self, j):
        S_p = 0  # The total reward value for agent A when performing tasks along the path p.
        if len(self.p) > 0:
            distance_j = 0
            distance_j += self.dis_u2t[0][self.p[0]]
            S_p += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[self.p[0]]
            for p_idx in range(len(self.p) - 1):
                distance_j += self.dis_t2t[self.p[p_idx]][self.p[p_idx + 1]]
                S_p += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[self.p[p_idx + 1]]

        # The reward after adding the task.
        p_temp = copy.deepcopy(self.p)
        p_temp.append(j)
        c_temp = 0
        distance_j = 0
        distance_j += self.dis_u2t[0][p_temp[0]]
        c_temp += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[p_temp[0]]
        if len(p_temp) > 1:
            for p_loc in range(len(p_temp) - 1):
                distance_j += self.dis_t2t[p_temp[p_loc]][p_temp[p_loc + 1]]
                c_temp += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[p_temp[p_loc + 1]]

        # Calculate the marginal benefit value.
        w_aj = c_temp - S_p
        return w_aj

    def reward_score(self, p, pos, targets, scl=1):
        c_bar = np.ones(len(targets))
        S_p = 0  # The total reward value for agent A when performing tasks along the path p.
        if len(p) > 0:
            distance_j = 0
            distance_j += l2norm(pos, targets[p[0]].pos_global_frame) / scl
            S_p += (self.Lambda ** (distance_j / self.vel)) * c_bar[p[0]]
            for p_idx in range(len(p) - 1):
                distance_j += l2norm(targets[p[p_idx]].pos_global_frame, targets[p[p_idx + 1]].pos_global_frame) / scl
                S_p += (self.Lambda ** (distance_j / self.vel)) * c_bar[p[p_idx + 1]]
        return S_p

    def sort_Wa_Na(self):
        """
        Function: sort in decreasing order
        """
        Wa = np.array(self.Wa)
        Na = np.array(self.Na)
        self.Wa.sort(reverse=True)  # Sort in descending order.
        self.Na = list(Na[np.argsort(-Wa)])  # Sort the set Na in descending order based on the values of Wa.

    def max_consensus(self):
        a_star = self.receive_info[0][0]
        j_star_a_star = self.receive_info[0][1]
        w_star_a_star = self.receive_info[0][2]
        for i in range(1, len(self.receive_info)):
            if w_star_a_star < self.receive_info[i][2]:
                a_star = self.receive_info[i][0]
                j_star_a_star = self.receive_info[i][1]
                w_star_a_star = self.receive_info[i][2]
        return a_star, j_star_a_star

    # Distances from the robot to each target point
    def dis_UT(self):
        agent_pos = np.array([self.pos])
        UtoT_matrix = spatial.distance.cdist(agent_pos, self.tasks_pos, metric='euclidean')
        return UtoT_matrix

    # Distances between each pair of target points
    def dis_TT(self):
        TtoT_matrix = spatial.distance.cdist(self.tasks_pos, self.tasks_pos, metric='euclidean')
        return TtoT_matrix

    # Distances from each target point to the termination area
    def dis_TE(self):
        row = 1
        column = len(self.tasks_pos)
        min_x, max_x, min_y, max_y = get_boundaries(self.exit_area)
        max_x = max_x - 0.5
        TtoE_matrix = np.zeros((row, column))
        for j in range(column):
            dist = l2norm(self.tasks_pos[j], [max_x, self.tasks_pos[j][1]])
            TtoE_matrix[0][j] = dist

        return TtoE_matrix


if __name__ == "__main__":
    pass

