import copy
import numpy as np
from scipy import spatial
from matamp.tools.utils import get_boundaries, l2norm


class CBBAPolicy(object):
    def __init__(self):
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

        # Local Winning Agent List
        self.z = None
        # Local Winning Bid List
        self.y = None
        # Bundle
        self.b = []
        # Path
        self.p = []
        # Maximum Task Number
        self.L_t = -1
        # Local Clock
        self.time_step = 0
        # Time Stamp List
        self.s = None

        # This part can be modified depend on the problem
        self.c = None  # Initial Score (Euclidean Distance)

        # socre function parameters
        self.Lambda = 0.95
        self.c_bar = None

        # The y, z, and s data of other agents
        self.Y = None

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
        self.z = np.ones(self.task_num, dtype=np.int16) * self.id
        self.y = np.array([0 for _ in range(self.task_num)], dtype=np.float64)
        self.s = {a: self.time_step for a in range(self.agent_num)}
        self.c = np.zeros(self.task_num)  # Initial Score (Euclidean Distance)
        self.c_bar = np.ones(self.task_num)
        self.dis_u2t = self.dis_UT() / scale
        self.dis_t2t = self.dis_TT() / scale

    def tau(self, j):
        # Estimate time agent will take to arrive at task j's location
        # This function can be used in later
        pass

    def send_message(self):
        """
        Return local winning bid list
        [output]
        y: winning bid list (list:task_num)
        z: winning agent list (list:task_num)
        s: Time Stamp List (Dict:{agent_id:update_time})
        """
        return self.y.tolist(), self.z.tolist(), self.s

    def receive_message(self, Y):
        self.Y = Y

    def build_bundle(self):
        """
        Construct bundle and path list with local information
        """
        J = [j for j in range(self.task_num)]

        while len(self.b) < self.L_t:
            # Calculate S_p for constructed path list
            S_p = 0
            if len(self.p) > 0:
                distance_j = 0
                distance_j += self.dis_u2t[0][self.p[0]]
                S_p += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[self.p[0]]
                for p_idx in range(len(self.p) - 1):
                    distance_j += self.dis_t2t[self.p[p_idx]][self.p[p_idx + 1]]
                    S_p += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[self.p[p_idx + 1]]

            # Calculate c_ij for each task j
            best_pos = {}
            for j in J:
                c_list = []
                if j in self.b:  # If already in bundle list
                    self.c[j] = 0  # Minimum Score
                else:
                    for n in range(len(self.p) + 1):
                        p_temp = copy.deepcopy(self.p)
                        p_temp.insert(n, j)
                        c_temp = 0
                        distance_j = 0
                        distance_j += self.dis_u2t[0][p_temp[0]]
                        c_temp += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[p_temp[0]]
                        if len(p_temp) > 1:
                            for p_loc in range(len(p_temp) - 1):
                                distance_j += self.dis_t2t[p_temp[p_loc]][p_temp[p_loc + 1]]
                                c_temp += (self.Lambda ** (distance_j / self.vel)) * self.c_bar[p_temp[p_loc + 1]]

                        c_jn = c_temp - S_p
                        c_list.append(c_jn)

                    max_idx = int(np.argmax(c_list))
                    c_j = c_list[max_idx]
                    self.c[j] = c_j
                    best_pos[j] = max_idx

            h = (self.c > self.y)
            if sum(h) == 0:  # No valid task
                break
            self.c[~h] = 0
            J_i = int(np.argmax(self.c))
            n_J = best_pos[J_i]

            self.b.append(J_i)
            self.p.insert(n_J, J_i)

            self.y[J_i] = self.c[J_i]
            self.z[J_i] = self.id

    def update_task(self):
        """
        [input]
        Y: winning bid lists from neighbors (dict:{neighbor_id:(winning bid_list, winning agent list, time stamp list)})
        time: for simulation,
        """

        old_p = copy.deepcopy(self.p)

        id_list = list(self.Y.keys())
        id_list.insert(0, self.id)

        # Update time list
        for id in list(self.s.keys()):
            if id in id_list:
                self.s[id] = self.time_step
            else:
                s_list = []
                for neighbor_id in id_list[1:]:
                    s_list.append(self.Y[neighbor_id][2][id])
                if len(s_list) > 0:
                    self.s[id] = max(s_list)

        # Update Process
        for j in range(self.task_num):
            for k in id_list[1:]:
                y_k = self.Y[k][0]
                z_k = self.Y[k][1]
                s_k = self.Y[k][2]

                z_ij = self.z[j]
                z_kj = z_k[j]
                y_kj = y_k[j]

                i = self.id
                y_ij = self.y[j]

                # Rule Based Update
                # Rule 1~4
                if z_kj == k:
                    # Rule 1
                    if z_ij == self.id:
                        if y_kj > y_ij:
                            self.__update(j, y_kj, z_kj)
                        elif abs(y_kj - y_ij) < np.finfo(float).eps:  # Tie Breaker
                            if k < self.id:
                                self.__update(j, y_kj, z_kj)
                        else:
                            self.__leave()
                    # Rule 2
                    elif z_ij == k:
                        self.__update(j, y_kj, z_kj)
                    # Rule 3
                    elif z_ij != -1:
                        m = z_ij
                        if (s_k[m] > self.s[m]) or (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif abs(y_kj - y_ij) < np.finfo(float).eps:  # Tie Breaker
                            if k < self.id:
                                self.__update(j, y_kj, z_kj)
                    # Rule 4
                    elif z_ij == -1:
                        self.__update(j, y_kj, z_kj)
                    else:
                        raise Exception("Error while updating")
                # Rule 5~8
                elif z_kj == i:
                    # Rule 5
                    if z_ij == i:
                        self.__leave()
                    # Rule 6
                    elif z_ij == k:
                        self.__reset(j)
                    # Rule 7
                    elif z_ij != -1:
                        m = z_ij
                        if s_k[m] > self.s[m]:
                            self.__reset(j)
                    # Rule 8
                    elif z_ij == -1:
                        self.__leave()
                    else:
                        raise Exception("Error while updating")
                # Rule 9~13
                elif z_kj != -1:
                    m = z_kj
                    # Rule 9
                    if z_ij == i:
                        if (s_k[m] >= self.s[m]) and (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] >= self.s[m]) and (abs(y_kj - y_ij) < np.finfo(float).eps):  # Tie Breaker
                            if m < self.id:
                                self.__update(j, y_kj, z_kj)
                    # Rule 10
                    elif z_ij == k:
                        if (s_k[m] > self.s[m]):
                            self.__update(j, y_kj, z_kj)
                        else:
                            self.__reset(j)
                    # Rule 11
                    elif z_ij == m:
                        if s_k[m] > self.s[m]:
                            self.__update(j, y_kj, z_kj)
                    # Rule 12
                    elif z_ij != -1:
                        n = z_ij
                        if (s_k[m] > self.s[m]) and (s_k[n] > self.s[n]):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] > self.s[m]) and (y_kj > y_ij):
                            self.__update(j, y_kj, z_kj)
                        elif (s_k[m] > self.s[m]) and (abs(y_kj - y_ij) < np.finfo(float).eps):  # Tie Breaker
                            if m < n:
                                self.__update(j, y_kj, z_kj)
                        elif (s_k[n] > self.s[n]) and (self.s[m] > s_k[m]):
                            self.__update(j, y_kj, z_kj)
                    # Rule 13
                    elif z_ij == -1:
                        if s_k[m] > self.s[m]:
                            self.__update(j, y_kj, z_kj)
                    else:
                        raise Exception("Error while updating")
                # Rule 14~17
                elif z_kj == -1:
                    # Rule 14
                    if z_ij == i:
                        self.__leave()
                    # Rule 15
                    elif z_ij == k:
                        self.__update(j, y_kj, z_kj)
                    # Rule 16
                    elif z_ij != -1:
                        m = z_ij
                        if s_k[m] > self.s[m]:
                            self.__update(j, y_kj, z_kj)
                    # Rule 17
                    elif z_ij == -1:
                        self.__leave()
                    else:
                        raise Exception("Error while updating")
                else:
                    raise Exception("Error while updating")

        n_bar = len(self.b)
        # Get n_bar
        for n in range(len(self.b)):
            b_n = self.b[n]
            if self.z[b_n] != self.id:
                n_bar = n
                break

        b_idx1 = copy.deepcopy(self.b[n_bar + 1:])

        if len(b_idx1) > 0:
            self.y[b_idx1] = 0
            self.z[b_idx1] = -1

        if n_bar < len(self.b):
            del self.b[n_bar:]

        self.p = []
        for task in self.b:
            self.p.append(task)

        self.time_step += 1

        converged = False
        if old_p == self.p:
            converged = True

        return converged

    def __update(self, j, y_kj, z_kj):
        """
        Update values
        """
        self.y[j] = y_kj
        self.z[j] = z_kj

    def __reset(self, j):
        """
        Reset values
        """
        self.y[j] = 0
        self.z[j] = -1  # -1 means "none"

    def __leave(self):
        """
        Do nothing
        """
        pass

    def reward_score(self, p, pos, targets, scl=1):
        c_bar = np.ones(len(targets))
        S_p = 0
        if len(p) > 0:
            distance_j = 0
            distance_j += l2norm(pos, targets[p[0]].pos_global_frame) / scl
            S_p += (self.Lambda ** (distance_j / self.vel)) * c_bar[p[0]]
            for p_idx in range(len(p) - 1):
                distance_j += l2norm(targets[p[p_idx]].pos_global_frame, targets[p[p_idx + 1]].pos_global_frame) / scl
                S_p += (self.Lambda ** (distance_j / self.vel)) * c_bar[p[p_idx + 1]]
        return S_p

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
        TtoE = np.zeros((row, column))
        for i in range(row):
            for j in range(column):
                dist = l2norm(self.tasks_pos[j], [max_x, self.tasks_pos[j][1]])
                TtoE[i][j] = dist

        return TtoE


if __name__ == "__main__":
    pass
