"""
@ ORCA-Dubins policy in 2D worskspace
@ Author: Gang Xu
@ Date: 2023.7.10
@ Function: A multi-robot motion planning method for continue target visit
"""

import numpy as np
from math import sqrt, atan2, pi, cos, sin, acos, inf

from matamp.configs import config
from matamp.planner import dubinsmaneuver2d
from matamp.tools.utils import absSq, l2norm, norm, sqr, det, mod2pi, pi_2_pi, unit_normal_vector, point_in_circle
from matamp.tools.utils import normalize, angle_2_vectors, seg_cross_circle, seg_is_intersec, point_line_dist

eps = config.eps


class Line(object):
    def __init__(self):
        self.direction = np.array([0.0, 0.0])   # The direction of the directed line.
        self.point = np.array([0.0, 0.0])       # A point on the directed line.


class ORCAPolicy(object):
    def __init__(self):
        self.type = "internal"
        self.now_goal = None
        self.epsilon = 1e-5
        self.inflation = 1.0
        self.orcaLines = []
        self.dist_update_now_goal = -1
        self.newVelocity = np.array([0.0, 0.0])

    def find_next_action(self, dict_comm, agent, kdTree):
        if self.dist_update_now_goal < 0:
            self.dist_update_now_goal = agent.near_goal_threshold * 1.0
        dis = 0.0 if self.now_goal is None else l2norm(agent.pos_global_frame, self.now_goal[:2])
        computeNeighbors(agent, kdTree)
        self.get_trajectory(agent, dict_comm)
        self.orcaLines.clear()
        self.newVelocity = np.array([0.0, 0.0])
        v_pref = compute_v_pref(agent, self.now_goal)
        agent_rad = agent.radius + 2 * self.inflation

        # Create polygonal obstacle ORCA lines.
        invTimeHorizonObs = 1.0 / agent.timeHorizonObst
        for obj in agent.poly_obs_neighbors:
            vertice1 = obj[0]
            vertice2 = vertice1.next_

            relative_pos1 = vertice1.point_ - agent.pos_global_frame
            relative_pos2 = vertice2.point_ - agent.pos_global_frame

            # Check if velocity obstacle of obstacle is already taken care of by previously
            # constructed obstacle ORCA lines.
            alreadyCovered = False

            for j in range(len(self.orcaLines)):
                det1 = det(invTimeHorizonObs * relative_pos1 - self.orcaLines[j].point, self.orcaLines[j].direction)
                det2 = det(invTimeHorizonObs * relative_pos2 - self.orcaLines[j].point, self.orcaLines[j].direction)
                if (det1 - invTimeHorizonObs * agent_rad >= -self.epsilon) and (
                        det2 - invTimeHorizonObs * agent_rad >= -self.epsilon):
                    alreadyCovered = True
                    break

            if alreadyCovered:
                continue

            # Not yet covered. Check for collisions.
            distSq1 = absSq(relative_pos1)
            distSq2 = absSq(relative_pos2)

            radiusSq = sqr(agent_rad)

            obstacleVector = vertice2.point_ - vertice1.point_
            s = (-np.dot(relative_pos1, obstacleVector)) / absSq(obstacleVector)
            distSqLine = absSq(-relative_pos1 - s * obstacleVector)

            line = Line()

            if s < 0.0 and distSq1 <= radiusSq:
                # Collision with left vertex. Ignore if non-convex.
                if vertice1.convex_:
                    line.point = np.array([0.0, 0.0])
                    line.direction = normalize([-relative_pos1[1], relative_pos1[0]])
                    self.orcaLines.append(line)
                continue
            elif s > 1.0 and distSq2 <= radiusSq:
                # Collision with right vertex. Ignore if non-convex or if it will be taken care of
                # by neighboring obstacle.
                if vertice2.convex_ and det(relative_pos2, vertice2.direction_) >= 0.0:
                    line.point = np.array([0.0, 0.0])
                    line.direction = normalize([-relative_pos2[1], relative_pos2[0]])
                    self.orcaLines.append(line)
                continue
            elif 0.0 <= s < 1.0 and distSqLine <= radiusSq:
                # Collision with obstacle segment.
                line.point = np.array([0.0, 0.0])
                line.direction = -vertice1.direction_
                self.orcaLines.append(line)
                continue

            # No collision. Compute legs. When obliquely viewed, both legs can come from a single vertex.
            # Legs extend cut-off line when non-convex vertex.
            # leftLegDirection = None
            # rightLegDirection = None

            if s < 0.0 and distSqLine <= radiusSq:
                # Obstacle viewed obliquely so that left vertex defines velocity obstacle.
                if not vertice1.convex_:
                    # Ignore obstacle.
                    continue

                vertice2 = vertice1

                leg1 = sqrt(distSq1 - radiusSq)
                leftLegDirection = np.array([relative_pos1[0] * leg1 - relative_pos1[1] * agent_rad,
                                             relative_pos1[0] * agent_rad + relative_pos1[1] * leg1]) / distSq1
                rightLegDirection = np.array([relative_pos1[0] * leg1 + relative_pos1[1] * agent_rad,
                                              -relative_pos1[0] * agent_rad + relative_pos1[1] * leg1]) / distSq1
            elif s > 1.0 and distSqLine <= radiusSq:
                # Obstacle viewed obliquely so that right vertex defines velocity obstacle.
                if not vertice2.convex_:
                    # Ignore obstacle.
                    continue

                vertice1 = vertice2

                leg2 = sqrt(distSq2 - radiusSq)
                leftLegDirection = np.array([relative_pos2[0] * leg2 - relative_pos2[1] * agent_rad,
                                             relative_pos2[0] * agent_rad + relative_pos2[1] * leg2]) / distSq2
                rightLegDirection = np.array([relative_pos2[0] * leg2 + relative_pos2[1] * agent_rad,
                                              -relative_pos2[0] * agent_rad + relative_pos2[1] * leg2]) / distSq2
            else:
                # Usual situation.
                if vertice1.convex_:
                    leg1 = sqrt(distSq1 - radiusSq)
                    leftLegDirection = np.array([relative_pos1[0] * leg1 - relative_pos1[1] * agent_rad,
                                                 relative_pos1[0] * agent_rad + relative_pos1[1] * leg1]) / distSq1
                else:
                    # Left vertex non-convex left leg extends cut-off line.
                    leftLegDirection = -vertice1.direction_

                if vertice2.convex_:
                    leg2 = sqrt(distSq2 - radiusSq)
                    rightLegDirection = np.array([relative_pos2[0] * leg2 + relative_pos2[1] * agent_rad,
                                                  -relative_pos2[0] * agent_rad + relative_pos2[
                                                      1] * leg2]) / distSq2
                else:
                    # Right vertex non-convex right leg extends cut-off line.
                    rightLegDirection = vertice1.direction_

            # Legs can never point into neighboring edge when convex vertex, take cutoff-line of neighboring
            # edge instead. If velocity projected on "foreign" leg, no constraint is added.

            leftNeighbor = vertice1.previous_

            isLeftLegForeign = False
            isRightLegForeign = False

            if vertice1.convex_ and det(leftLegDirection, -leftNeighbor.direction_) >= 0.0:
                # Left leg points into obstacle.
                leftLegDirection = -leftNeighbor.direction_
                isLeftLegForeign = True

            if vertice2.convex_ and det(rightLegDirection, vertice2.direction_) <= 0.0:
                # Right leg points into obstacle.
                rightLegDirection = vertice2.direction_
                isRightLegForeign = True

            # Compute cut-off centers.
            leftCutOff = invTimeHorizonObs * (vertice1.point_ - agent.pos_global_frame)
            rightCutOff = invTimeHorizonObs * (vertice2.point_ - agent.pos_global_frame)
            cutOffVector = rightCutOff - leftCutOff

            # Project current velocity on velocity obstacle.

            # Check if current velocity is projected on cutoff circles.
            vel_dot_left = np.dot(agent.vel_global_frame - leftCutOff, cutOffVector)
            t = 0.5 if vertice1 == vertice2 else vel_dot_left / absSq(cutOffVector)
            tLeft = np.dot(agent.vel_global_frame - leftCutOff, leftLegDirection)
            tRight = np.dot(agent.vel_global_frame - rightCutOff, rightLegDirection)

            if (t < 0.0 and tLeft < 0.0) or (vertice1 == vertice2 and tLeft < 0.0 and tRight < 0.0):
                # Project on left cut-off circle.
                unitW = normalize(agent.vel_global_frame - leftCutOff)
                line.direction = np.array([unitW[1], -unitW[0]])
                line.point = leftCutOff + agent_rad * invTimeHorizonObs * unitW
                self.orcaLines.append(line)
                continue

            elif t > 1.0 and tRight < 0.0:
                # Project on right cut-off circle.
                unitW = normalize(agent.vel_global_frame - rightCutOff)
                line.direction = np.array([unitW[1], -unitW[0]])
                line.point = rightCutOff + agent_rad * invTimeHorizonObs * unitW
                self.orcaLines.append(line)
                continue

            # Project on left leg, right leg, or cut-off line, whichever is closest to velocity.
            distSqCutoff = inf if t < 0.0 or t > 1.0 or vertice1 == vertice2 else absSq(
                agent.vel_global_frame - (leftCutOff + t * cutOffVector))
            distSqLeft = inf if tLeft < 0.0 else absSq(
                agent.vel_global_frame - (leftCutOff + tLeft * leftLegDirection))
            distSqRight = inf if tRight < 0.0 else absSq(
                agent.vel_global_frame - (rightCutOff + tRight * rightLegDirection))

            if distSqCutoff <= distSqLeft and distSqCutoff <= distSqRight:
                # Project on cut-off line.
                line.direction = -vertice1.direction_
                line.point = leftCutOff + agent_rad * invTimeHorizonObs * np.array([-line.direction[1],
                                                                                    line.direction[0]])
                self.orcaLines.append(line)
                continue

            if distSqLeft <= distSqRight:
                # Project on left leg.
                if isLeftLegForeign:
                    continue

                line.direction = leftLegDirection
                line.point = leftCutOff + agent_rad * invTimeHorizonObs * np.array([-line.direction[1],
                                                                                    line.direction[0]])
                self.orcaLines.append(line)

                continue

            # Project on right leg.
            if isRightLegForeign:
                continue

            line.direction = -rightLegDirection
            line.point = rightCutOff + agent_rad * invTimeHorizonObs * np.array([-line.direction[1],
                                                                                 line.direction[0]])
            self.orcaLines.append(line)

        numObstLines = len(self.orcaLines)

        # Create agent or circle obstacle and target ORCA lines.
        invTimeHorizon = 1.0 / agent.timeHorizon
        for obj in agent.neighbors:
            obj = obj[0]
            if obj in agent.targets or (obj.is_sl and obj.is_cleared) or obj.is_poly:
                continue
            relativePosition = obj.pos_global_frame - agent.pos_global_frame
            relativeVelocity = agent.vel_global_frame - obj.vel_global_frame
            distSq = absSq(relativePosition)
            obj_rad = obj.radius + 2 * self.inflation
            combinedRadius = agent_rad + obj_rad
            combinedRadiusSq = sqr(combinedRadius)

            line = Line()

            if distSq > combinedRadiusSq:
                # No collision.
                w = relativeVelocity - invTimeHorizon * relativePosition

                # Vector from cutoff center to relative velocity.
                wLengthSq = absSq(w)
                dotProduct1 = np.dot(w, relativePosition)

                if dotProduct1 < 0.0 and sqr(dotProduct1) > combinedRadiusSq * wLengthSq:
                    # Project on cut-off circle.
                    wLength = sqrt(wLengthSq)
                    unitW = w / wLength

                    line.direction = np.array([unitW[1], -unitW[0]])
                    u = (combinedRadius * invTimeHorizon - wLength) * unitW
                else:
                    # Project on legs.
                    leg = sqrt(distSq - combinedRadiusSq)

                    if det(relativePosition, w) > 0.0:
                        # Project on left leg.
                        x = relativePosition[0] * leg - relativePosition[1] * combinedRadius
                        y = relativePosition[0] * combinedRadius + relativePosition[1] * leg
                        line.direction = np.array([x, y]) / distSq
                    else:
                        # Project on right leg.
                        x = relativePosition[0] * leg + relativePosition[1] * combinedRadius
                        y = -relativePosition[0] * combinedRadius + relativePosition[1] * leg
                        line.direction = -np.array([x, y]) / distSq

                    dotProduct2 = np.dot(relativeVelocity, line.direction)
                    u = dotProduct2 * line.direction - relativeVelocity
            else:
                # Collision. Project on cut-off circle of time timeStep.
                invTimeStep = 1.0 / agent.timeStep

                # Vector from cutoff center to relative velocity.
                w = relativeVelocity - invTimeStep * relativePosition

                wLength = norm(w)
                unitW = w / wLength

                line.direction = np.array([unitW[1], -unitW[0]])
                u = (combinedRadius * invTimeStep - wLength) * unitW

            line.point = agent.vel_global_frame + 0.5 * u
            self.orcaLines.append(line)

        lineFail, self.newVelocity = self.linear_program2(self.orcaLines, agent.maxSpeed, v_pref, False,
                                                          self.newVelocity)

        if lineFail < len(self.orcaLines):
            self.newVelocity = self.linear_program3(self.orcaLines, numObstLines,
                                                    lineFail, agent.maxSpeed, self.newVelocity)

        vA_post = np.array([self.newVelocity[0], self.newVelocity[1]])
        action = to_unicycle(vA_post, agent)
        agent.vel_global_unicycle[0] = round(1.0 * action[0], 5)
        agent.vel_global_unicycle[1] = round(0.2 * 0.5 * action[1] / agent.dt_nominal, 5)
        dist = l2norm(agent.pos_global_frame, agent.goal_global_frame)

        info = 'agent' + str(agent.id) + ':' + str(action) + 'Distance2Goal: ' + str(dist)
        print(info, len(agent.neighbors))
        agent.history_speed.append(action[0])
        agent.current_speed = action[0]

        if action[0] < 1e-2 or np.linalg.norm(agent.vel_global_frame) < 1e-2:
            agent.stuck_num += 1
        else:
            agent.stuck_num = 0
        if agent.stuck_num >= 300:
            agent.is_out_of_max_time = True
            agent.stuck_num = 0
        if dis <= self.dist_update_now_goal:
            agent.path_speed.append(action[0])

        return action

    def circle_is_interect(self, agent, inflation, o):
        o_inersect = False
        agent_rad = agent.radius + inflation  # The joint radius here is consistent with the combined radius of adding avoidance points.
        radius_min = agent.turning_radius + self.inflation + agent_rad  # The error of the Dubins path, with a turning radius of 19, is approximately 1.
        for obj in agent.neighbors:
            obj = obj[0]
            if obj.is_agent or obj in agent.targets or (obj.is_sl and obj.is_cleared):
                continue
            obj_rad = obj.radius + inflation
            combinedRadius = radius_min + obj_rad
            if not o_inersect and l2norm(o, obj.pos) <= combinedRadius:
                o_inersect = True
        return o_inersect

    def ctrl_turning_direction(self, agent, p_strat, p_end, dir_start, dir_end, inflation):
        nsLeft, nsRight = unit_normal_vector(dir_start)
        neLeft, neRight = unit_normal_vector(dir_end)

        osL = p_strat + agent.turning_radius * nsLeft       # The center of the left turn.
        osR = p_strat + agent.turning_radius * nsRight      # The center of the right turn.

        oeL = p_end + agent.turning_radius * neLeft
        oeR = p_end + agent.turning_radius * neRight
        o3L = compute_ccc_circle(osR, oeR, agent)      # The center of the second turn in the RLR mode.
        o3R = compute_ccc_circle(osL, oeL, agent)      # The center of the second turn in the LRL mode.

        d = l2norm(p_strat, p_end)
        rmin2 = 2 * agent.turning_radius
        LSL_tangent_p = osoe_tangent_point(osL, oeL, agent.turning_radius, 'LSL')
        LSR_tangent_p = osoe_tangent_point(osL, oeR, agent.turning_radius, 'LSR')
        RSR_tangent_p = osoe_tangent_point(osR, oeR, agent.turning_radius, 'RSR')
        RSL_tangent_p = osoe_tangent_point(osR, oeL, agent.turning_radius, 'RSL')

        LSL_tang_p_in = point_in_obj_area(agent, inflation, LSL_tangent_p)
        RSR_tang_p_in = point_in_obj_area(agent, inflation, RSR_tangent_p)
        if d < rmin2:
            LSR_tang_p_in = False
            RSL_tang_p_in = False
        else:
            LSR_tang_p_in = point_in_obj_area(agent, inflation, LSR_tangent_p)
            RSL_tang_p_in = point_in_obj_area(agent, inflation, RSL_tangent_p)

        osL_inersect = LSL_tang_p_in or LSR_tang_p_in
        osR_inersect = RSR_tang_p_in or RSL_tang_p_in

        if not osL_inersect and osR_inersect:
            control_direction = 'L'
            if len(o3R) > 0 and self.circle_is_interect(agent, inflation, o3R):  # Determine if the second circle in LRL mode collides.
                control_direction = 'LS'
        elif osL_inersect and not osR_inersect:
            control_direction = 'R'
            if len(o3L) > 0 and self.circle_is_interect(agent, inflation, o3L):  # Determine if the second circle in RLR mode collides.
                control_direction = 'RS'
        elif not osL_inersect and not osR_inersect:
            control_direction = '*'
            o3L_inersect, o3R_inersect = False, False
            if len(o3R) > 0 and self.circle_is_interect(agent, inflation, o3R):  # Determine if the second circle in LRL mode collides.
                o3R_inersect = True
            if len(o3L) > 0 and self.circle_is_interect(agent, inflation, o3L):  # Determine if the second circle in RLR mode collides.
                o3L_inersect = True
            if not o3L_inersect and o3R_inersect:       # Remove the RLR mode.
                control_direction = '*-L'
            elif o3L_inersect and not o3R_inersect:     # Remove the LRL mode.
                control_direction = '*-R'
            elif o3L_inersect and o3R_inersect:         # Remove the RLR and LRL mode.
                control_direction = 'S'
        else:                                           # Both left and right turns result in collisions, so turning is not permitted.
            control_direction = ''
        return control_direction

    def is_intersect_obj(self, agent, inflation):
        pos, p_goal = agent.pos_global_frame, np.array(self.now_goal[:2])
        constraint3 = True
        for obj in agent.neighbors:
            obj = obj[0]
            if obj.is_agent or obj in agent.targets or (obj.is_sl and obj.is_cleared):
                continue
            obj_rad = obj.radius + inflation
            combinedRadius = agent.radius + obj_rad
            seg_cross_obs = seg_cross_circle(pos, p_goal, obj.pos_global_frame, combinedRadius)
            if seg_cross_obs:
                constraint3 = False
                break
        return constraint3

    def is_use_dubins(self, agent, inflation, dir_start, dir_end):
        angle = angle_2_vectors(dir_start, dir_end)
        constraint1 = angle > agent.maxAngular
        constraint2 = l2norm(agent.pos_global_frame, self.now_goal[:2]) > 2 * (agent.turning_radius + inflation)
        constraint3 = self.is_intersect_obj(agent, inflation)
        return constraint1 and constraint2 and constraint3

    def for_smooth_turning(self, agent, inflation):
        """Optimize the trajectory points using Dubins curves to address the issue of sharp angles."""
        c1 = l2norm(agent.pos_global_frame, agent.goal_global_frame) > agent.expansion
        c2 = l2norm(agent.pos_global_frame, agent.initial_pos[:2]) > agent.startPointdis
        if c1 and c2:
            pos = agent.pos_global_frame
            p_goal = np.array(self.now_goal[:2])  # Current trajectory points
            dir_start = normalize(agent.vel_global_frame)
            dir_end = normalize(p_goal - np.array(agent.path_node[-2][:2]))
            constraints = self.is_use_dubins(agent, inflation, dir_start, dir_end)

            if constraints:  # Exceeding the maximum angular velocity (maxAngular), use Dubins curves.
                ctrl_dir = self.ctrl_turning_direction(agent, pos, p_goal, dir_start, dir_end, inflation)
                if len(ctrl_dir) > 0:
                    yaw_i = atan2(dir_start[1], dir_start[0])
                    yaw_f = atan2(dir_end[1], dir_end[0])
                    test_qi = [pos[0], pos[1], yaw_i]           # start_x, start_y, start_yaw
                    test_qf = [p_goal[0], p_goal[1], yaw_f]     # end_x, end_y, end_yaw
                    rmin = agent.turning_radius
                    sampling_size = agent.sampling_size
                    dubins = dubinsmaneuver2d.dubins_path_planning(test_qi, test_qf, rmin, ctrl_dir, sampling_size)
                    path = dubins.path
                    path_m = dubins.path_m

                    path.pop(0)
                    path_m.pop(0)
                    path.reverse()
                    path_m.reverse()

                    new_path = []
                    is_valid_planner = True
                    for i in range(len(path)):
                        pathi_is_in_area = point_in_obj_area(agent, inflation, path[i])
                        if pathi_is_in_area and path_m[i] != 'S':
                            is_valid_planner = False
                            break
                        elif not pathi_is_in_area:
                            new_path.append(path[i])

                    if is_valid_planner:
                        for tar in agent.targets:  # After re-adding the target position, remove the extended trajectory points.
                            if l2norm(self.now_goal[:2], tar.pos_global_frame) < 1e-5 and not tar.is_cleared:
                                if agent.path[-1][2] != 0:
                                    agent.path.pop()
                                break
                        for i in range(len(new_path)):
                            if i > 0:
                                agent.path.append(np.array([new_path[i][0], new_path[i][1], 2]))
                            else:
                                agent.path.append(np.array([new_path[i][0], new_path[i][1], self.now_goal[2]]))

                        if len(new_path) > 0:
                            agent.path_node.pop()  # The old "now_goal" needs to be re-added to the path, and it also needs to be removed from the path_node.
                            self.now_goal = np.array(agent.path.pop(), dtype='float64')
                            agent.path_node.append(np.array([self.now_goal[0], self.now_goal[1], self.now_goal[2]]))

    def get_trajectory(self, agent, dict_comm):
        for tar in agent.targets:
            if not tar.is_cleared and l2norm(agent.pos_global_frame, tar.pos) <= self.dist_update_now_goal:
                tar.is_cleared = True
                agent.cleared_num += 1
                break
        pass_dist = max(0.001, agent.expansion)
        inflation = 2.5 * self.inflation
        if agent.path:
            if self.now_goal is None:  # first
                self.now_goal = np.array(agent.path.pop(), dtype='float64')
                agent.path_node.append(np.array([self.now_goal[0], self.now_goal[1], self.now_goal[2]]))

            self.for_smooth_turning(agent, inflation)
            dis = l2norm(agent.pos_global_frame, self.now_goal[:2])
            condition1 = dis <= self.dist_update_now_goal
            is_tar_pos = False
            if self.now_goal[2] == 0:
                is_tar_pos = True
            vel = agent.vel_global_frame
            posnowg = np.array(self.now_goal[:2]) - agent.pos_global_frame
            condition2 = (not is_tar_pos and np.dot(vel, posnowg) < 0)
            if condition1 or condition2:
                if agent.path:
                    agent.history_pos.append(
                        np.array([agent.pos_global_frame[0], agent.pos_global_frame[1], self.now_goal[2]]))
                    self.now_goal = np.array(agent.path.pop(), dtype='float64')
                    if dis <= self.dist_update_now_goal:
                        agent.path_node.append(np.array([self.now_goal[0], self.now_goal[1], self.now_goal[2]]))
                    for tar in agent.targets:
                        if l2norm(self.now_goal[:2], tar.pos_global_frame) < 1e-5 and not tar.is_cleared:
                            tar_idx = agent.targets.index(tar)
                            if agent.targets.index(tar) < len(agent.targets) - 1:
                                dist = l2norm(tar.pos_global_frame, agent.targets[tar_idx + 1].pos_global_frame)
                                combinedRadius = tar.radius + agent.targets[tar_idx + 1].radius
                                if dist < combinedRadius:
                                    pass_dist = max(0.001, tar.radius / 4)
                            dif = normalize(self.now_goal[:2] - agent.pos_global_frame) * pass_dist
                            pos = self.now_goal[:2] + dif
                            pos = expansion_pos_in_forbid_area(pos, tar, dict_comm, agent, dif, inflation)
                            agent.path.append(np.array([pos[0], pos[1], 2]))  # The extension point can be adjusted according to the actual situation, with type 2.
                            break
                else:
                    self.now_goal = np.array([agent.goal_global_frame[0], agent.goal_global_frame[1], 0])
                    agent.history_pos.append(np.array([agent.pos_global_frame[0], agent.pos_global_frame[1], 0]))
            else:
                agent.history_pos.append(np.array([agent.pos_global_frame[0], agent.pos_global_frame[1], 0]))
            goal_is_in_unsuitable_area(agent)
        else:
            self.now_goal = np.array([agent.goal_global_frame[0], agent.goal_global_frame[1], 0])
            self.for_smooth_turning(agent, inflation)
            agent.history_pos.append(np.array([agent.pos_global_frame[0], agent.pos_global_frame[1], self.now_goal[2]]))
            goal_is_in_unsuitable_area(agent)

    def linear_program1(self, lines, lineNo, radius, optVelocity, directionOpt):
        """
        Solves a one-dimensional linear program on a specified line subject to linear constraints defined by lines and
        a circular constraint.
        Args:
            lines (list): Lines defining the linear constraints.
            lineNo (int): The specified line constraint.
            radius (float): The radius of the circular constraint.
            optVelocity (Vector2): The optimization velocity.
            directionOpt (bool): True if the direction should be optimized.
        Returns:
            bool: True if successful.
            Vector2: A reference to the result of the linear program.
        """
        dotProduct = np.dot(lines[lineNo].point, lines[lineNo].direction)
        discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines[lineNo].point)
        dotProduct = lines[lineNo].point @ lines[lineNo].direction

        if discriminant < 0.0:
            # Max speed circle fully invalidates line lineNo.
            return False, None

        sqrtDiscriminant = sqrt(discriminant)
        tLeft = -dotProduct - sqrtDiscriminant
        tRight = -dotProduct + sqrtDiscriminant

        for i in range(lineNo):
            denominator = det(lines[lineNo].direction, lines[i].direction)
            numerator = det(lines[i].direction, lines[lineNo].point - lines[i].point)

            if abs(denominator) <= self.epsilon:
                # Lines lineNo and i are (almost) parallel.
                if numerator < 0.0:
                    return False, None
                continue

            t = numerator / denominator

            if denominator >= 0.0:
                # Line i bounds line lineNo on the right.
                tRight = min(tRight, t)
            else:
                # Line i bounds line lineNo on the left.
                tLeft = max(tLeft, t)

            if tLeft > tRight:
                return False, None

        if directionOpt:
            # Optimize direction.
            if np.dot(optVelocity, lines[lineNo].direction) > 0.0:
                # Take right extreme.
                result = lines[lineNo].point + tRight * lines[lineNo].direction
            else:
                # Take left extreme.
                result = lines[lineNo].point + tLeft * lines[lineNo].direction
        else:
            # Optimize closest point.
            t = np.dot(lines[lineNo].direction, optVelocity - lines[lineNo].point)

            if t < tLeft:
                result = lines[lineNo].point + tLeft * lines[lineNo].direction
            elif t > tRight:
                result = lines[lineNo].point + tRight * lines[lineNo].direction
            else:
                result = lines[lineNo].point + t * lines[lineNo].direction

        return True, result

    def linear_program2(self, lines, radius, optVelocity, directionOpt, result):
        """
        Solves a two-dimensional linear program subject to linear constraints
        defined by lines and a circular constraint.
        Args:
            lines (list): Lines defining the linear constraints.
            radius (float): The radius of the circular constraint.
            optVelocity (Vector2): The optimization velocity.
            directionOpt (bool): True if the direction should be optimized.
            result (Vector2): A reference to the result of the linear program.
        Returns:
            int: The number of the line it fails on, and the number of lines if successful.
            Vector2: A reference to the result of the linear program.
        """
        if directionOpt:
            # Optimize direction. Note that the optimization velocity is of unit length in this case.
            result = optVelocity * radius
        elif absSq(optVelocity) > sqr(radius):
            # Optimize closest point and outside circle.
            result = normalize(optVelocity) * radius
        else:
            # Optimize closest point and inside circle.
            result = optVelocity

        for i in range(len(lines)):
            if det(lines[i].direction, lines[i].point - result) > 0.0:
                # Result does not satisfy constraint i. Compute new optimal result.
                tempResult = result
                success, result = self.linear_program1(lines, i, radius, optVelocity, directionOpt)
                if not success:
                    result = tempResult
                    return i, result

        return len(lines), result

    def linear_program3(self, lines, numObstLines, beginLine, radius, result):
        """
        Solves a two-dimensional linear program subject to linear constraints
        defined by lines and a circular constraint.
        Args:
            lines (list): Lines defining the linear constraints.
            numObstLines (int): Count of obstacle lines.
            beginLine (int): The line on which the 2-d linear program failed.
            radius (float): The radius of the circular constraint.
            result (Vector2): A reference to the result of the linear program.
        Returns:
            Vector2: A reference to the result of the linear program.
        """
        distance = 0.0

        for i in range(beginLine, len(lines)):
            if det(lines[i].direction, lines[i].point - result) > distance:
                # Result does not satisfy constraint of line i.
                projLines = []

                for ii in range(numObstLines):
                    projLines.append(lines[ii])

                for j in range(numObstLines, i):
                    line = Line()
                    determinant = det(lines[i].direction, lines[j].direction)

                    if abs(determinant) <= self.epsilon:
                        # Line i and line j are parallel.
                        if np.dot(lines[i].direction, lines[j].direction) > 0.0:
                            # Line i and line j point in the same direction.
                            continue
                        else:
                            # Line i and line j point in opposite direction.
                            line.point = 0.5 * (lines[i].point + lines[j].point)
                    else:
                        line.point = lines[i].point + (
                                det(lines[j].direction, lines[i].point - lines[j].point) / determinant) * lines[
                                         i].direction

                    line.direction = normalize(lines[j].direction - lines[i].direction)
                    projLines.append(line)

                tempResult = result
                optVelocity = np.array([-lines[i].direction[1], lines[i].direction[0]])
                lineFail, result = self.linear_program2(projLines, radius, optVelocity, True, result)
                if lineFail < len(projLines):
                    """
                    This should in principle not happen. The result is by definition already in 
                    the feasible region of this linear program. If it fails, it is due to small 
                    floating point error, and the current result is kept.
                    """
                    result = tempResult

                distance = det(lines[i].direction, lines[i].point - result)
        return result


def compute_v_pref(agent, now_goal):
    goal = now_goal[:2]
    n_diff = normalize(goal - agent.pos_global_frame)
    v_pref = agent.maxSpeed * n_diff
    for tar in agent.targets:
        if l2norm(tar.pos_global_frame, agent.pos_global_frame) <= 1.5 * agent.expansion and not tar.is_cleared:
            v_pref = agent.pref_speed * n_diff
            break
    if l2norm(goal, agent.pos_global_frame) < 5 * agent.maxSpeed:
        v_pref = agent.pref_speed * n_diff
    if l2norm(agent.goal_global_frame, agent.pos_global_frame) < agent.near_goal_threshold:
        v_pref = np.zeros_like(v_pref)

    return v_pref


def computeNeighbors(agent, kdTree):
    agent.neighbors.clear()
    agent.poly_obs_neighbors.clear()

    # check obstacle neighbors
    rangeSq = agent.neighborDist ** 2
    if len(agent.neighbors) != agent.maxNeighbors:
        rangeSq = agent.neighborDist ** 2
    kdTree.computeObstacleNeighbors(agent, rangeSq)

    # check polygonal obstacle neighbors
    kdTree.computePolyObstacleNeighbors(agent, rangeSq)

    # check target neighbors
    if len(agent.neighbors) != agent.maxNeighbors:
        rangeSq = agent.neighborDist ** 2
    kdTree.computeTargetNeighbors(agent, rangeSq)

    if agent.is_collision:
        return

    # check other agents
    if len(agent.neighbors) != agent.maxNeighbors:
        rangeSq = agent.neighborDist ** 2
    kdTree.computeAgentNeighbors(agent, rangeSq)


def osoe_tangent_point(os, oe, turning_radius, mode):
    dist_osoe = l2norm(os, oe)
    n_osoe = normalize(oe - os)
    n_RSR, n_LSL = unit_normal_vector(n_osoe)  #
    if mode == 'LSL':
        tangent_point = os + n_LSL * turning_radius  # The cut-off point in the LSL mode
    elif mode == 'LSR' and dist_osoe > 2 * turning_radius:
        co = -2 * turning_radius / dist_osoe
        si = sqrt(1 - co ** 2)
        n_LSR = np.array([co * n_osoe[0] - si * n_osoe[1], si * n_osoe[0] + co * n_osoe[1]])
        tangent_point = os - n_LSR * turning_radius  # The cut-off point in the LSR mode
    elif mode == 'RSR':
        tangent_point = os + n_RSR * turning_radius  # The cut-off point in the RSR mode
    elif mode == 'RSL' and dist_osoe > 2 * turning_radius:
        co = -2 * turning_radius / dist_osoe
        si = sqrt(1 - co ** 2)
        n_RSL = np.array([co * n_osoe[0] + si * n_osoe[1], -si * n_osoe[0] + co * n_osoe[1]])
        tangent_point = os - n_RSL * turning_radius  # The cut-off point in the RSL mode
    else:
        tangent_point = []
    return tangent_point


def point_in_obj_area(agent, inflation, point):
    is_in_area = False
    if len(point) > 0:
        for obj in agent.neighbors:
            obj = obj[0]
            if obj.is_agent or obj in agent.targets or (obj.is_sl and obj.is_cleared):
                continue
            else:
                if l2norm(point, obj.pos_global_frame) <= obj.radius + agent.radius + 2 * inflation:
                    is_in_area = True
                    break
    return is_in_area


def compute_ccc_circle(os, oe, agent):  # The center of the second circle in the Dubins curve CCC mode
    dist_osoe = l2norm(os, oe)
    if dist_osoe < 4 * agent.turning_radius:
        osoe = oe - os  # RLR mode
        beta = atan2(osoe[1], osoe[0])
        cos_theta = l2norm(os, oe) / (4 * agent.turning_radius)
        theta = acos(cos_theta)
        dir_n = np.array([cos(beta - theta), sin(beta - theta)])
        o3 = os + dir_n * 2 * agent.turning_radius
    else:
        o3 = []
    return o3


def expansion_pos_in_forbid_area(pos, tar, dict_comm, agent, dif, inflation):
    targets = dict_comm['targets']
    obstacles = dict_comm['obstacles']
    obj_obs = sorted(obstacles + targets, key=lambda x: l2norm(tar.pos_global_frame, x.pos[:2]))
    for obj in obj_obs[:agent.maxNeighbors]:
        if obj.is_agent:
            continue
        elif obj in agent.targets:
            continue
        elif obj.tid[obj.id] != tar.tid[tar.id]:
            dist = obj.radius + agent.radius + agent.turning_radius + 2 * inflation
            if l2norm(pos, obj.pos_global_frame) < dist:  # The positions overlap.
                n_dif = normalize(dif)
                length_move = agent.radius  # Move the endpoint out of the threat zone.
                l_pg_po = length_move * n_dif
                pos = tar.pos_global_frame + l_pg_po
    return pos


def goal_is_in_unsuitable_area(ag):
    if len(ag.path) == 2:
        d = 3 * ag.radius
        d2 = 6 * ag.radius
        for ob in ag.neighbors:
            ob = ob[0]
            if ob.is_agent and l2norm(ob.goal_global_frame, ag.goal_global_frame) < ag.radius + ob.radius + 4.:
                n_dif = normalize(np.array(ag.path[1][:2]) - ag.pos_global_frame) * np.random.uniform(d, d2)
                ag.goal_global_frame = ag.goal_global_frame - n_dif
                ag.path[0] = [ag.goal_global_frame[0], ag.goal_global_frame[1], 0]


def to_unicycle(vA_post, agent):
    max_delta = agent.maxAngular
    vA_post = np.array(vA_post)
    norm_vA = norm(vA_post)
    yaw_next = mod2pi(atan2(vA_post[1], vA_post[0]))
    yaw_current = mod2pi(agent.heading_global_frame)
    delta_theta = yaw_next - yaw_current
    delta_theta = pi_2_pi(delta_theta)
    action = np.array([norm_vA, delta_theta * agent.dt_nominal])
    if delta_theta < -pi:
        delta_theta = delta_theta + 2 * pi
    if delta_theta > pi:
        delta_theta = delta_theta - 2 * pi
    if delta_theta > max_delta:
        delta_theta = max_delta
        action = np.array([min(norm_vA, agent.pref_speed), delta_theta * agent.dt_nominal])  # constraints
    elif delta_theta < -max_delta:
        delta_theta = -max_delta
        action = np.array([min(norm_vA, agent.pref_speed), delta_theta * agent.dt_nominal])  # constraints

    return action


if __name__ == '__main__':
    pass
