"""
@ D* in 2D worskspace
@ Revised: Gang Xu
@ Date: 2023.5.16
@ Reference: https://github.com/zhm-real/PathPlanning.git
"""

import math
import time
import copy
import heapq
import numpy as np
from matamp.tools.utils import l2norm, seg_cross_circle
import matplotlib.pyplot as plt


class DStar:
    def __init__(self):
        self.id = -1
        self.s_start, self.s_goal = None, None
        self.pos = None
        self.goal = None
        self.end_type = 0
        self.step = int(10)
        self.motions = [(self.step, 0), (self.step, self.step), (0, self.step), (-self.step, self.step),
                        (-self.step, 0), (-self.step, -self.step), (0, -self.step), (self.step, -self.step)]
        self.inflation = 0.0
        self.rob_radius = 1.0
        self.obs = set()
        self.obsts = []
        self.dict_comm = None
        self.intersect_obj = None
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        self.OPEN = []
        self.t = dict()
        self.PARENT = dict()
        self.h = dict()
        self.k = dict()
        self.path = []
        self.path_smooth = []
        self.old_path = []
        self.old_path_smooth = []
        self.visited = set()
        self.count = 0
        self.max_count = 0
        self.pri_time = 0.0
        self.sort_time = 0.0
        self.sort_time2 = 0.0
        self.neighbs_time = 0.0
        self.check_collis_time = 0.0

    def init(self, agent, dict_comm, inflation, pos, pos_goal, intersect_obj):
        self.id = agent.id
        self.pos, self.goal = pos, pos_goal
        self.end_type = pos_goal[2]
        self.inflation = inflation
        self.rob_radius = agent.radius
        self.dict_comm = dict_comm
        self.intersect_obj = intersect_obj
        self.obs = set()
        self.obsts = []
        self.OPEN = []
        self.t = dict()
        self.PARENT = dict()
        self.h = dict()
        self.k = dict()
        self.path = []
        self.path_smooth = []
        self.visited = set()
        self.count = 0
        self.max_count = 0

        self.get_xy_range(agent, pos, pos_goal, scale=1.)
        self.s_start = (int(round(pos[0] / 10, 0) * 10), int(round(pos[1] / 10, 0) * 10))
        self.s_goal = (int(round(pos_goal[0] / 10, 0) * 10), int(round(pos_goal[1] / 10, 0) * 10))
        for obj in agent.neighbors:
            obj = obj[0]
            if obj.is_agent or obj in agent.targets or (obj.is_sl and obj.is_cleared):
                continue
            self.obsts.append(obj)
            self.add_objs_grid(agent, obj)
        targets = self.dict_comm['targets']
        obstacles = self.dict_comm['obstacles']
        obj_obs = sorted(obstacles + targets, key=lambda x: l2norm(self.intersect_obj.pos_global_frame, x.pos[:2]))
        for obj in obj_obs[1:agent.maxNeighbors]:
            if obj.is_agent or obj in agent.targets or (obj.is_sl and obj.is_cleared):
                continue
            if obj not in self.obsts:
                self.obsts.append(obj)
                self.add_objs_grid(agent, obj)

        for i in range(self.min_x, self.max_x+1, self.step):
            for j in range(self.min_y, self.max_y+1, self.step):
                self.t[(i, j)] = 'NEW'
                self.k[(i, j)] = 0.0
                self.h[(i, j)] = float("inf")
                self.PARENT[(i, j)] = None

        self.h[self.s_goal] = 0.0

    def get_xy_range(self, agent, pos, pos_goal, scale=0.5):
        self.min_x = int(min(pos[0], pos_goal[0]) - scale * agent.forwardDist)
        if self.min_x < min(agent.min_x, agent.min_xe):
            self.min_x = int(min(agent.min_x, agent.min_xe))

        self.max_x = int(max(pos[0], pos_goal[0]) + scale * agent.forwardDist)
        if self.max_x > max(agent.max_x, agent.max_xe):
            self.max_x = int(max(agent.max_x, agent.max_xe))

        self.min_y = int(min(pos[1], pos_goal[1]) - scale * agent.forwardDist)
        if self.min_y < min(agent.min_y, agent.min_ye):
            self.min_y = int(min(agent.min_y, agent.min_ye))

        self.max_y = int(max(pos[1], pos_goal[1]) + scale * agent.forwardDist)
        if self.max_y > max(agent.max_y, agent.max_ye):
            self.max_y = int(max(agent.max_y, agent.max_ye))
        self.min_x = int(round(self.min_x / 10, 0) * 10)
        self.max_x = int(round(self.max_x / 10, 0) * 10)
        self.min_y = int(round(self.min_y / 10, 0) * 10)
        self.max_y = int(round(self.max_y / 10, 0) * 10)

    def add_objs_grid(self, agent, obj):
        combinedRadius = agent.radius + obj.radius + 2 * self.inflation
        obj_grid_p = (int(round(obj.pos[0] / 10, 0) * 10), int(round(obj.pos[1] / 10, 0) * 10))
        g_num = int(1.5 * combinedRadius / self.step)
        x1 = obj_grid_p[0] - g_num * self.step
        x2 = obj_grid_p[0] + g_num * self.step
        y1 = obj_grid_p[1] - g_num * self.step
        y2 = obj_grid_p[1] + g_num * self.step
        for i in range(x1, x2 + 1, self.step):
            for j in range(y1, y2 + 1, self.step):
                s = (i, j)
                if l2norm(s, obj.pos) <= combinedRadius and s not in self.obs:
                    self.obs.add(s)

    def run(self, agent, dict_comm, inflation, pos, pos_goal, intersect_obj):
        if self.s_goal is None or (l2norm(self.goal, pos_goal[:2]) > 1e-5):     # Using D* for the first time and switching targets using D*
            t1 = time.time()
            self.init(agent, dict_comm, inflation, pos, pos_goal, intersect_obj)
            t2 = time.time()
            cost_t = t2-t1
            if agent.max_init_cost < cost_t:
                agent.max_init_cost = cost_t
            self.insert(self.s_goal, 0)

            while True:
                self.process_state()
                if self.t[self.s_start] == 'CLOSED':
                    self.count = 0
                    break

            self.path = self.extract_path(self.s_start, self.s_goal, agent)
            self.smooth(agent)
            t3 = time.time()
            print(agent.id, 'init time: ', cost_t, len(self.visited))
            print(agent.id, 'solve time: ', t3 - t1)
            # if agent.id == 9:
            #     self.plot_path(self.path, self.path_smooth)
            return True
        else:
            return False

    def dynamic_call(self, agent, pos, intersect_obj):
        t1 = time.time()
        self.intersect_obj = intersect_obj
        is_add_new_obj = False
        targets = self.dict_comm['targets']
        obstacles = self.dict_comm['obstacles']
        for obj in agent.neighbors:
            obj = obj[0]
            if obj.is_agent or obj in agent.targets or (obj.is_sl and obj.is_cleared):
                continue
            if obj not in self.obsts:
                self.obsts.append(obj)
                is_add_new_obj = True
                self.add_objs_grid(agent, obj)
        obj_obs = sorted(obstacles + targets, key=lambda x: l2norm(self.intersect_obj.pos_global_frame, x.pos[:2]))
        for obj in obj_obs[1:agent.maxNeighbors]:
            if obj.is_agent or obj in agent.targets or (obj.is_sl and obj.is_cleared):
                continue
            if obj not in self.obsts:
                self.obsts.append(obj)
                self.add_objs_grid(agent, obj)

        if is_add_new_obj:
            self.pos = pos
            self.pri_time = 0.0
            self.neighbs_time = 0.0
            self.check_collis_time = 0.0
            self.sort_time, self.sort_time2 = 0.0, 0.0
            self.s_start = (int(round(pos[0] / 10, 0) * 10), int(round(pos[1] / 10, 0) * 10))
            s = self.s_start
            if s in self.obs:
                k = 1
                while s in self.obs:
                    motions = k * np.array(self.motions)
                    for u in motions:
                        s_next = tuple([s[i] + u[i] for i in range(2)])
                        if s_next not in self.obs:
                            self.s_start = s_next
                            s = self.s_start
                            print(agent.id, s, pos)
                            break
                    k += 1
            self.visited = set()
            n = 0
            while l2norm(s, self.s_goal) > 1e-5:
                print(agent.id, 'dynamic_call', s, self.PARENT[s], self.h[s])
                n += 1
                if self.is_collision(s, self.PARENT[s], is_add_boj=True):
                    self.modify(s)
                    continue
                s = self.PARENT[s]

            self.old_path = self.path
            self.old_path_smooth = self.path_smooth
            self.path = self.extract_path(self.s_start, self.s_goal, agent)
            self.smooth(agent)
            t2 = time.time()
            t_cost = t2 - t1
            if agent.max_path_replan_cost < t_cost:
                agent.max_path_replan_cost = t_cost
            return True
        else:
            return False

    def extract_path(self, s_start, s_end, agent):
        path = [[self.pos[0], self.pos[1], int(3)]]
        s = s_start
        while True:
            s = self.PARENT[s]
            if s is not None and l2norm(s, s_end) > 1e-5:
                if self.s_not_in_tar(agent, s):
                    path.append([s[0], s[1], int(3)])
            else:
                path.append([self.goal[0], self.goal[1], int(self.goal[2])])
                return path

    def s_not_in_tar(self, agent, s):
        for tar in agent.targets:
            if l2norm(tar.pos, s) < tar.radius + agent.radius + 2*self.inflation:
                return False
        return True

    def process_state(self):
        self.count += 1
        s = self.min_state()  # get node in OPEN set with min k value
        self.visited.add(s)

        if s is None:
            print(self.id, len(self.OPEN), self.s_start, self.s_goal, self.pos, self.goal)
            self.plot_visited()
            return -1  # OPEN set is empty

        k_old = self.get_k_min()  # record the min k value of this iteration (min path cost)
        self.delete(s)  # change state s from OPEN to CLOSED

        # k_min < h[s] --> s: RAISE state (increased cost)
        if k_old < self.h[s]:
            for s_n in self.get_neighbor(s):
                if self.h[s_n] <= k_old and \
                        self.h[s] > self.h[s_n] + self.cost(s_n, s):

                    # update h_value and choose parent
                    self.PARENT[s] = s_n
                    self.h[s] = self.h[s_n] + self.cost(s_n, s)

        # s: k_min >= h[s] -- > s: LOWER state (cost reductions)
        if k_old == self.h[s]:
            for s_n in self.get_neighbor(s):
                if self.t[s_n] == 'NEW' or \
                        (self.PARENT[s_n] == s and self.h[s_n] != self.h[s] + self.cost(s, s_n)) or \
                        (self.PARENT[s_n] != s and self.h[s_n] > self.h[s] + self.cost(s, s_n)):

                    # Condition:
                    # 1) t[s_n] == 'NEW': not visited
                    # 2) s_n's parent: cost reduction
                    # 3) s_n find a better parent
                    self.PARENT[s_n] = s
                    self.insert(s_n, self.h[s] + self.cost(s, s_n))
        else:
            for s_n in self.get_neighbor(s):
                if self.t[s_n] == 'NEW' or \
                        (self.PARENT[s_n] == s and self.h[s_n] != self.h[s] + self.cost(s, s_n)):

                    # Condition:
                    # 1) t[s_n] == 'NEW': not visited
                    # 2) s_n's parent: cost reduction
                    self.PARENT[s_n] = s
                    self.insert(s_n, self.h[s] + self.cost(s, s_n))
                else:
                    if self.PARENT[s_n] != s and \
                            self.h[s_n] > self.h[s] + self.cost(s, s_n):

                        # Condition: LOWER happened in OPEN set (s), s should be explored again
                        self.insert(s, self.h[s])
                    else:
                        if self.PARENT[s_n] != s and \
                                self.h[s] > self.h[s_n] + self.cost(s_n, s) and \
                                self.t[s_n] == 'CLOSED' and \
                                self.h[s_n] > k_old:

                            # Condition: LOWER happened in CLOSED set (s_n), s_n should be explored again
                            self.insert(s_n, self.h[s_n])

        return self.get_k_min()

    def min_state(self):
        """
        choose the node with the minimum k value in OPEN set.
        :return: state
        """
        if not self.OPEN:
            return None, None

        _, min_s = self.OPEN[0]
        return min_s

    def get_k_min(self):
        """
        calc the min k value for nodes in OPEN set.
        :return: k value
        """
        if not self.OPEN:
            return -1

        k_min = self.OPEN[0][0]
        return k_min

    def insert(self, s, h_new):
        """
        insert node into OPEN set.
        :param s: node
        :param h_new: new or better cost to come value
        """

        if self.t[s] == 'NEW':
            self.k[s] = h_new
        elif self.t[s] == 'OPEN':
            self.k[s] = min(self.k[s], h_new)
        elif self.t[s] == 'CLOSED':
            self.k[s] = min(self.h[s], h_new)

        self.h[s] = h_new
        self.t[s] = 'OPEN'
        heapq.heappush(self.OPEN, (self.k[s], s))

    def delete(self, s):
        """
        delete: move state s from OPEN set to CLOSED set.
        :param s: state should be deleted
        """

        if self.t[s] == 'OPEN':
            self.t[s] = 'CLOSED'
        heapq.heappop(self.OPEN)

    def modify(self, s):
        """
        start processing from state s.
        :param s: is a node whose status is RAISE or LOWER.
        """

        self.modify_cost(s)

        while True:
            k_min = self.process_state()
            if k_min >= self.h[s]:
                break

    def modify_cost(self, s):

        if self.t[s] == 'CLOSED':
            self.insert(s, self.h[self.PARENT[s]] + self.cost(s, self.PARENT[s], is_add_boj=True))

    def get_neighbor(self, s):
        nei_list = set()
        for u in self.motions:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            s_next_in_range = self.min_x < s_next[0] < self.max_x and self.min_y < s_next[1] < self.max_y
            if s_next_in_range and s_next not in self.obs:
                nei_list.add(s_next)
        return nei_list

    def s_next_in_forbid_area(self, s_next, obst):
        if self.obs:
            agent_rad = self.rob_radius + self.inflation
            obj_rad = obst.radius + self.inflation
            combinedRadius = agent_rad + obj_rad
            if l2norm(s_next, obst.pos) < combinedRadius:
                return True
            else:
                return False
        else:
            return False

    def cost(self, s_start, s_goal, is_add_boj=False):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :param is_add_boj: again planning
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        if self.is_collision(s_start, s_goal, is_add_boj=is_add_boj):
            return float("inf")

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end, is_add_boj=False):
        seg_cross_obs = False
        is_add_boj = False
        if is_add_boj:
            for obst in self.obsts:
                if obst.is_sl and obst.is_cleared:
                    continue
                agent_rad = self.rob_radius + self.inflation
                obj_rad = obst.radius + self.inflation
                combinedRadius = agent_rad + obj_rad
                seg_cross_obs = seg_cross_circle(s_start, s_end, obst.pos_global_frame, combinedRadius)
                if seg_cross_obs:
                    break
        else:
            if s_start in self.obs or s_end in self.obs:
                seg_cross_obs = True

            if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
                if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                    s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                    s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                else:
                    s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                    s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

                if s1 in self.obs or s2 in self.obs:
                    seg_cross_obs = True

        return seg_cross_obs

    def isAvoDanger(self, agent, x1, y1, x2, y2):
        scale = 2
        agent_rad = agent.radius + self.inflation*scale
        for i in range(len(self.obsts)):
            obj_rad = self.obsts[i].radius + self.inflation*scale
            combinedRadius = agent_rad + obj_rad
            a = (x2 - x1) * (self.obsts[i].pos[0] - x1) + (y2 - y1) * (self.obsts[i].pos[1] - y1)
            b = max((x2 - x1) ** 2 + (y2 - y1) ** 2, 1e-5)
            u = a / b
            x = x1 + u * (x2 - x1)
            y = y1 + u * (y2 - y1)

            dis = math.sqrt((x - self.obsts[i].pos[0]) ** 2 + (y - self.obsts[i].pos[1]) ** 2)
            dis1 = math.sqrt((x1 - self.obsts[i].pos[0]) ** 2 + (y1 - self.obsts[i].pos[1]) ** 2)
            dis2 = math.sqrt((x2 - self.obsts[i].pos[0]) ** 2 + (y2 - self.obsts[i].pos[1]) ** 2)

            if dis < combinedRadius:
                if dis1 < combinedRadius or dis2 < combinedRadius:
                    return False
                else:
                    if (x1 <= x <= x2) or (x2 <= x <= x1) and (y1 <= y <= y2) or (y2 <= y <= y1):
                        return False
        return True

    def smooth(self, agent):
        self.path_smooth = []
        route = self.path[:]
        self.path_smooth.append([self.pos[0], self.pos[1], route[0][2]])
        while True:
            # print('smooth')
            n = len(route)
            if n == 1:
                break
            is_refine = False
            for i in range(n - 1, 0, -1):
                if self.isAvoDanger(agent, route[0][0], route[0][1], route[i][0], route[i][1]):
                    is_refine = True
                    self.path_smooth.append(route[i])
                    for j in range(i):
                        route.pop(0)
                    break
            if not is_refine:
                # print(route[0], self.path_smooth)
                p = route.pop(0)
                if p not in self.path_smooth:
                    self.path_smooth.append(p)

    def plot_path(self, path, path_smooth):
        fig = plt.figure(0)
        fig_size = (10, 8)
        fig.set_size_inches(fig_size[0], fig_size[1])
        ax = fig.add_subplot(1, 1, 1)
        num_p = 0
        for key in self.PARENT:
            if self.PARENT[key] is not None:
                num_p += 1
        for x in self.obs:
            plt.plot(x[0], x[1], marker='s', color='black')

        px = [x[0] for x in path]
        py = [x[1] for x in path]
        px1 = [x[0] for x in path_smooth]
        py1 = [x[1] for x in path_smooth]
        plt.plot(px, py, linewidth=2, color='blue')
        plt.plot(px1, py1, linewidth=2, color='red')
        if self.old_path:
            px = [x[0] for x in self.old_path]
            py = [x[1] for x in self.old_path]
            px1 = [x[0] for x in self.old_path_smooth]
            py1 = [x[1] for x in self.old_path_smooth]
            plt.plot(px, py, linewidth=2, color='grey')
            plt.plot(px1, py1, linewidth=2, color='orange')
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "bs")
        plt.plot(self.pos[0], self.pos[1], marker='o', color='red', zorder=3)
        plt.plot(self.goal[0], self.goal[1], marker='o', color='red', zorder=3)
        plt.axis("equal")
        plt.show()

    def plot_visited(self):
        fig = plt.figure(0)
        fig_size = (10, 8)
        fig.set_size_inches(fig_size[0], fig_size[1])
        plt.plot(self.s_start[0], self.s_start[1], marker='s', color='green', zorder=3)
        plt.plot(self.s_goal[0], self.s_goal[1], marker='s', color='red', zorder=3)
        for x in self.visited:
            if x is not None:
                plt.plot(x[0], x[1], marker='s', color='green')
        for x in self.obs:
            plt.plot(x[0], x[1], marker='s', color='black')
        plt.axis("equal")
        plt.show()


def main():
    s_start = (5, 5)
    s_goal = (45, 25)
    # dstar = DStar(s_start, s_goal)
    # dstar.run(s_start, s_goal)


if __name__ == '__main__':
    main()
