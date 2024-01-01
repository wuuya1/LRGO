"""
@ Revise: Gang Xu
@ Date: 2021.8.13
@ Details: dubins path planner
@ reference: https://github.com/phanikiran1169/RRTstar_3D-Dubins.git
"""

import math
import numpy as np
from math import sqrt, atan2, pi, cos, sin, acos
from matamp.tools.utils import l2norm, unit_normal_vector, normalize
from matamp.configs.config import SAMPLE_SIZE


class DubinsManeuver(object):
    def __init__(self, qi, qf, r_min, sampling_size):
        self.qi = qi
        self.qf = qf
        self.r_min = r_min
        self.t = -1.0
        self.p = -1.0
        self.q = -1.0
        self.px = []
        self.py = []
        self.pyaw = []
        self.path = []
        self.path_x = []
        self.path_y = []
        self.path_m = []
        self.pose = []
        self.mode = None
        self.length = -1.0
        self.sampling_size = sampling_size
        self.planner_mode = 'LRL'


def mod2pi(theta):  # to 0-2*pi
    return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)


def pi_2_pi(angle):  # to -pi-pi
    return (angle + math.pi) % (2 * math.pi) - math.pi


def LSL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d + sa - sb

    mode = ["L", "S", "L"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((cb - ca), tmp0)
    t = mod2pi(-alpha + tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(beta - tmp1)

    return t, p, q, mode


def RSR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d - sa + sb
    mode = ["R", "S", "R"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((ca - cb), tmp0)
    t = mod2pi(alpha - tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(-beta + tmp1)

    return t, p, q, mode


def LSR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(-mod2pi(beta) + tmp2)

    return t, p, q, mode


def RSL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(beta - tmp2)

    return t, p, q, mode


def RLR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["R", "L", "R"]
    tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp_rlr) > 1.0:
        return None, None, None, mode

    p = mod2pi(2 * math.pi - math.acos(tmp_rlr))
    t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
    q = mod2pi(alpha - beta - t + mod2pi(p))
    return t, p, q, mode


def LRL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["L", "R", "L"]
    tmp_lrl = (6. - d * d + 2 * c_ab + 2 * d * (- sa + sb)) / 8.
    if abs(tmp_lrl) > 1:
        return None, None, None, mode
    p = mod2pi(2 * math.pi - math.acos(tmp_lrl))
    t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.)
    q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

    return t, p, q, mode


def dubins_path_planning_from_origin(ex, ey, syaw, eyaw, c, ctrl_dir):
    # nomalize
    dx = ex
    dy = ey
    D = math.sqrt(dx ** 2.0 + dy ** 2.0)
    d = D / c

    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(syaw - theta)
    beta = mod2pi(eyaw - theta)

    if ctrl_dir == 'L':
        planners = [LSL, LSR, LRL]
    elif ctrl_dir == 'LS':
        planners = [LSL, LSR]
    elif ctrl_dir == 'R':
        planners = [RSR, RSL, RLR]
    elif ctrl_dir == 'RS':
        planners = [RSR, RSL]
    elif ctrl_dir == '*-L':
        planners = [LSL, RSR, LSR, RSL, LRL]
    elif ctrl_dir == '*-R':
        planners = [LSL, RSR, LSR, RSL, RLR]
    elif ctrl_dir == 'S':
        planners = [LSL, RSR, LSR, RSL]
    else:
        planners = [LSL, RSR, LSR, RSL, RLR, LRL]

    bcost = float("inf")
    bt, bp, bq, bmode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = c * (abs(t) + abs(p) + abs(q))
        if bcost > cost:
            bt, bp, bq, bmode = t, p, q, mode
            bcost = cost

    px, py, pyaw, pmode = generate_course([bt, bp, bq], bmode, c)

    return px, py, pyaw, pmode, bmode, bcost, bt, bp, bq


def dubins_path_planning(start_pos, end_pos, c, ctrl_dir, sampling_size):
    """
    Dubins path plannner

    input:start_pos, end_pos, c
        start_pos[0]    x position of start point [m]
        start_pos[1]    y position of start point [m]
        start_pos[2]    yaw angle of start point [rad]
        end_pos[0]      x position of end point [m]
        end_pos[1]      y position of end point [m]
        end_pos[2]      yaw angle of end point [rad]
        c               radius [m]

    output: maneuver
        maneuver.t              the first segment curve of dubins
        maneuver.p              the second segment line of dubins
        maneuver.q              the third segment curve of dubins
        maneuver.px             x position sets [m]
        maneuver.py             y position sets [m]
        maneuver.pyaw           heading angle sets [rad]
        maneuver.length         length of dubins
        maneuver.mode           mode of dubins
    """
    maneuver = DubinsManeuver(start_pos, end_pos, c, sampling_size)
    sx, sy = start_pos[0], start_pos[1]
    ex, ey = end_pos[0], end_pos[1]
    syaw, eyaw = start_pos[2], end_pos[2]

    ex = ex - sx
    ey = ey - sy

    lpx, lpy, lpyaw, lpmode, mode, clen, t, p, q = dubins_path_planning_from_origin(ex, ey, syaw, eyaw, c, ctrl_dir)

    px = [math.cos(-syaw) * x + math.sin(-syaw) * y + sx for x, y in zip(lpx, lpy)]
    py = [-math.sin(-syaw) * x + math.cos(-syaw) * y + sy for x, y in zip(lpx, lpy)]
    pyaw = [pi_2_pi(iyaw + syaw) for iyaw in lpyaw]
    maneuver.t, maneuver.p, maneuver.q, maneuver.mode = t, p, q, mode
    maneuver.px, maneuver.py, maneuver.pyaw, maneuver.length = px, py, pyaw, clen
    for index in range(len(maneuver.px)):
        maneuver.pose.append([maneuver.px[index], maneuver.py[index], maneuver.pyaw[index]])
    spath, spath_x, spath_y, spath_m = path_sample(maneuver.pose, maneuver.sampling_size,
                                                   maneuver.qi, maneuver.qf, lpmode)
    maneuver.path, maneuver.path_x, maneuver.path_y, maneuver.path_m = spath, spath_x, spath_y, spath_m
    return maneuver


def path_sample(pose, sampling_size, start, goal, lpmode):
    seg_length = 0.0
    path = [start[:2]]
    path_x = [start[0]]
    path_y = [start[1]]
    path_m = [lpmode[0]]
    num_point = len(pose)
    for k in range(1, num_point):
        dis = l2norm(path[-1], pose[k])
        seg_length += dis
        if lpmode[k] == "S":
            if seg_length >= sampling_size:
                seg_length = 0.0
                if k < num_point - 1 and (lpmode[k - 1] != "S" or lpmode[k + 1] != "S"):
                    path.append(pose[k][:2])
                    path_x.append(pose[k][0])
                    path_y.append(pose[k][1])
                    path_m.append(lpmode[k])
                elif k == num_point - 1:
                    path.append(pose[k][:2])
                    path_x.append(pose[k][0])
                    path_y.append(pose[k][1])
                    path_m.append(lpmode[k])
        else:
            # dis = l2norm(path[-1], pose[k])
            # seg_length += dis
            if seg_length >= sampling_size:
                seg_length = 0.0
                path.append(pose[k][:2])
                path_x.append(pose[k][0])
                path_y.append(pose[k][1])
                path_m.append(lpmode[k])
            elif k == num_point - 1:
                if seg_length < sampling_size:
                    path.pop(-1)
                    path_x.pop(-1)
                    path_y.pop(-1)
                    path_m.pop(-1)
                path.append(goal[:2])
                path_x.append(goal[0])
                path_y.append(goal[1])
                path_m.append(lpmode[k])
    return path, path_x, path_y, path_m


def generate_course(length, mode, c):
    px = [0.0]
    py = [0.0]
    pyaw = [0.0]
    pmode = [-1]
    for m, l in zip(mode, length):
        pd = 0.0
        if m == "S":
            d = np.deg2rad(6.0)
        else:  # turning course
            d = np.deg2rad(6.0)
        while pd < abs(l - d):
            px.append(px[-1] + d * c * math.cos(pyaw[-1]))
            py.append(py[-1] + d * c * math.sin(pyaw[-1]))
            pmode.append(m)
            if m == "L":  # left turn
                pyaw.append(pyaw[-1] + d)
            elif m == "S":  # Straight
                pyaw.append(pyaw[-1])
            elif m == "R":  # right turn
                pyaw.append(pyaw[-1] - d)
            pd += d

        d = l - pd
        px.append(px[-1] + d * c * math.cos(pyaw[-1]))
        py.append(py[-1] + d * c * math.sin(pyaw[-1]))
        pmode.append(m)

        if m == "L":  # left turn
            pyaw.append(pyaw[-1] + d)
        elif m == "S":  # Straight
            pyaw.append(pyaw[-1])
        elif m == "R":  # right turn
            pyaw.append(pyaw[-1] - d)
        pd += d
        pmode[0] = pmode[1]

    return px, py, pyaw, pmode


def get_coordinates(maneuver, offset):
    noffset = offset / maneuver.r_min
    qi = [0., 0., maneuver.qi[2]]

    l1 = maneuver.t
    l2 = maneuver.p
    q1 = get_position_in_segment(l1, qi, maneuver.mode[0])  # Final do segmento 1
    q2 = get_position_in_segment(l2, q1, maneuver.mode[1])  # Final do segmento 2

    if noffset < l1:
        q = get_position_in_segment(noffset, qi, maneuver.mode[0])
    elif noffset < (l1 + l2):
        q = get_position_in_segment(noffset - l1, q1, maneuver.mode[1])
    else:
        q = get_position_in_segment(noffset - l1 - l2, q2, maneuver.mode[2])

    q[0] = q[0] * maneuver.r_min + qi[0]
    q[1] = q[1] * maneuver.r_min + qi[1]
    q[2] = mod2pi(q[2])

    return q


def get_position_in_segment(offset, qi, mode):
    q = [0.0, 0.0, 0.0]
    if mode == 'L':
        q[0] = qi[0] + math.sin(qi[2] + offset) - math.sin(qi[2])
        q[1] = qi[1] - math.cos(qi[2] + offset) + math.cos(qi[2])
        q[2] = qi[2] + offset
    elif mode == 'R':
        q[0] = qi[0] - math.sin(qi[2] - offset) + math.sin(qi[2])
        q[1] = qi[1] + math.cos(qi[2] - offset) - math.cos(qi[2])
        q[2] = qi[2] - offset
    elif mode == 'S':
        q[0] = qi[0] + math.cos(qi[2]) * offset
        q[1] = qi[1] + math.sin(qi[2]) * offset
        q[2] = qi[2]
    return q


def get_sampling_points(maneuver, sampling_size=0.1):
    points = []
    for offset in np.arange(0.0, maneuver.length + sampling_size, sampling_size):
        points.append(get_coordinates(maneuver, offset))
    return points


def compute_ccc_circle(os, oe, turning_radius):
    dist_osoe = l2norm(os, oe)
    if dist_osoe < 4 * turning_radius:
        osoe = oe - os  # RLR mode
        beta = atan2(osoe[1], osoe[0])
        cos_theta = l2norm(os, oe) / (4 * turning_radius)
        theta = acos(cos_theta)
        dir_n = np.array([cos(beta - theta), sin(beta - theta)])
        o3 = os + dir_n * 2 * turning_radius
    else:
        o3 = []
    return o3


def circle_is_interect(obj_pos, o, turning_radius):
    o_inersect = False
    radius_min = turning_radius + 1.0 + 12.5
    for obj in obj_pos:
        obj_rad = 50 + 2.5
        combinedRadius = radius_min + obj_rad
        if not o_inersect and l2norm(o, obj) <= combinedRadius:
            o_inersect = True
    return o_inersect


def osoe_tangent_point(os, oe, turning_radius, mode):
    dist_osoe = l2norm(os, oe)
    n_osoe = normalize(oe - os)
    n_RSR, n_LSL = unit_normal_vector(n_osoe)
    if mode == 'LSL':
        tangent_point = os + n_LSL * turning_radius
    elif mode == 'LSR' and dist_osoe > 2 * turning_radius:
        co = -2 * turning_radius / dist_osoe
        si = sqrt(1 - co ** 2)
        n_LSR = np.array([co * n_osoe[0] - si * n_osoe[1], si * n_osoe[0] + co * n_osoe[1]])
        tangent_point = os - n_LSR * turning_radius
    elif mode == 'RSR':
        tangent_point = os + n_RSR * turning_radius
    elif mode == 'RSL' and dist_osoe > 2 * turning_radius:
        co = -2 * turning_radius / dist_osoe
        si = sqrt(1 - co ** 2)
        n_RSL = np.array([co * n_osoe[0] + si * n_osoe[1], -si * n_osoe[0] + co * n_osoe[1]])
        tangent_point = os - n_RSL * turning_radius
    else:
        tangent_point = []
    return tangent_point


def ctrl_turning_direction(turning_radius, p_strat, p_end, dir_start, dir_end, obj_pos):
    nsLeft, nsRight = unit_normal_vector(dir_start)
    neLeft, neRight = unit_normal_vector(dir_end)

    osL = p_strat + turning_radius * nsLeft
    osR = p_strat + turning_radius * nsRight

    oeL = p_end + turning_radius * neLeft
    oeR = p_end + turning_radius * neRight
    o3L = compute_ccc_circle(osR, oeR, turning_radius)
    o3R = compute_ccc_circle(osL, oeL, turning_radius)

    osL_inersect = circle_is_interect(obj_pos, osL, turning_radius)
    osR_inersect = circle_is_interect(obj_pos, osR, turning_radius)
    if not osL_inersect and osR_inersect:
        control_direction = 'L'
        if len(o3R) > 0 and circle_is_interect(obj_pos, o3R, turning_radius):
            control_direction = 'LS'
    elif osL_inersect and not osR_inersect:
        control_direction = 'R'
        if len(o3L) > 0 and circle_is_interect(obj_pos, o3L, turning_radius):
            control_direction = 'RS'
    elif not osL_inersect and not osR_inersect:
        control_direction = '*'
        o3L_inersect, o3R_inersect = False, False
        if len(o3R) > 0 and circle_is_interect(obj_pos, o3R, turning_radius):
            o3R_inersect = True
        if len(o3L) > 0 and circle_is_interect(obj_pos, o3L, turning_radius):
            o3L_inersect = True
        if not o3L_inersect and o3R_inersect:
            control_direction = '*-L'
        elif o3L_inersect and not o3R_inersect:
            control_direction = '*-R'
        elif o3L_inersect and o3R_inersect:
            control_direction = 'S'
    else:
        control_direction = ''
    return control_direction, [osL, osR, oeL, oeR, o3L, o3R]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matamp.tools.utils import l2norm

    print("Dubins Path Planner Start!!")
    p2 = np.array([0.0, 500.0])
    p3 = np.array([20/np.deg2rad(30)-0.0005, 500.0])
    yaw_i = np.deg2rad(180.0)
    yaw_f = np.deg2rad(0.0)
    heading_s = np.array([cos(yaw_i), sin(yaw_i)])
    heading_e = np.array([cos(yaw_f), sin(yaw_f)])
    test_qi = [[p2[0], p2[1], yaw_i]]  # start_x, start_y, start_yaw
    test_qf = [[p3[0], p3[1], yaw_f]]  # end_x, end_y, end_yaw
    print((yaw_f - yaw_i) * 57.3)

    rmin = 10/np.deg2rad(30)

    nLeft, nRight = unit_normal_vector(heading_s)
    n2Left, n2Right = unit_normal_vector(heading_e)
    circle_centerL = p2 + rmin * nLeft
    circle_centerR = p2 + rmin * nRight
    circle2_centerL = p3 + rmin * n2Left
    circle2_centerR = p3 + rmin * n2Right

    tangent_point = osoe_tangent_point(circle_centerL, circle2_centerL, rmin, 'LSL')
    LSL_tangent_p = osoe_tangent_point(circle_centerL, circle2_centerL, rmin, 'LSL')
    LSR_tangent_p = osoe_tangent_point(circle_centerL, circle2_centerR, rmin, 'LSR')
    RSR_tangent_p = osoe_tangent_point(circle_centerR, circle2_centerR, rmin, 'RSR')
    RSL_tangent_p = osoe_tangent_point(circle_centerR, circle2_centerL, rmin, 'RSL')

    o1o2 = circle2_centerR - circle_centerR
    beta1 = atan2(o1o2[1], o1o2[0])
    dist1 = l2norm(circle_centerR, circle2_centerR)
    r4 = 4*rmin

    cos_theta1 = dist1 / r4
    theta1 = acos(min(cos_theta1, 1))
    dir_n1 = np.array([cos(beta1-theta1), sin(beta1-theta1)])
    circle3_center = circle_centerR + dir_n1 * 2 * rmin

    # ob_pos = np.array([3839.81, 4167.44])

    ctrol_direction, o_points = ctrl_turning_direction(rmin, p2, p3, heading_s, heading_e, [])
    osL, osR, oeL, oeR, o3L, o3R = o_points
    test_maneuver = dubins_path_planning(test_qi[0], test_qf[0], rmin, 'R')
    # test_maneuver.path.pop(0)
    # test_maneuver.path.pop()
    # test_maneuver.path_x.pop(0)
    # test_maneuver.path_x.pop()
    # test_maneuver.path_y.pop(0)
    # test_maneuver.path_y.pop()
    print(test_maneuver.mode, len(test_maneuver.path), test_maneuver.length, test_maneuver.path)

    # -----------------------------------plotting -----------------------------------
    ob_radius = 50 + 2.5
    ag_radius = 10. + 2.5
    fig = plt.figure(0)
    fig_size = (10, 8)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    # plt.plot(ob_pos[0], ob_pos[1], color='red', marker='o')
    # ax.add_patch(plt.Circle((ob_pos[0], ob_pos[1]), radius=ob_radius, fc=[0.8, 0.8, 0.8], ec='black'))
    for i in range(len(test_maneuver.path)):
        if test_maneuver.path_m[i] == 'S':
            ax.add_patch(plt.Circle((test_maneuver.path[i][0], test_maneuver.path[i][1]),
                                    radius=ag_radius, fc='blue', ec='black', alpha=0.5))
            # print(test_maneuver.path[i])
        # else:
        #     ax.add_patch(plt.Circle((test_maneuver.path[i][0], test_maneuver.path[i][1]),
        #                             radius=ag_radius, fc=[0.51, 0.43, 1.0], ec='black', alpha=0.2))
    plt.plot(test_maneuver.path_x, test_maneuver.path_y, color='green')
    plt.scatter(test_maneuver.path_x, test_maneuver.path_y, marker='o', color='red', s=10, zorder=3)
    plt.plot(test_qi[0][0], test_qi[0][1], color='blue', marker='*', markersize=10)
    # ax.add_patch(plt.Circle((test_qi[0][0], test_qi[0][1]), radius=rmin, fc='red', ec='red', alpha=0.2))
    plt.plot(test_qf[0][0], test_qf[0][1], color='orange', marker='*', markersize=10)
    plt.plot(circle_centerL[0], circle_centerL[1], color='blue', marker='o', markersize=5)
    plt.plot(circle_centerR[0], circle_centerR[1], color='blue', marker='o', markersize=5)
    plt.plot(circle2_centerL[0], circle2_centerL[1], color='red', marker='o', markersize=5)
    plt.plot(circle2_centerR[0], circle2_centerR[1], color='red', marker='o', markersize=5)
    # plt.plot(circle3_center[0], circle3_center[1], color='red', marker='o', markersize=5)

    # if len(tangent_point) > 0:
    plt.plot(tangent_point[0], tangent_point[1], color='blue', marker='o', markersize=5)

    ax.add_patch(plt.Circle((circle_centerL[0], circle_centerL[1]), radius=rmin, fc='blue', ec='blue', alpha=0.3))
    ax.add_patch(plt.Circle((circle_centerR[0], circle_centerR[1]), radius=rmin, fc='blue', ec='blue', alpha=0.3))
    ax.add_patch(plt.Circle((circle2_centerL[0], circle2_centerL[1]), radius=rmin, fc='red', ec='red', alpha=0.3))
    ax.add_patch(plt.Circle((circle2_centerR[0], circle2_centerR[1]), radius=rmin, fc='red', ec='red', alpha=0.3))
    # ax.add_patch(plt.Circle((circle3_center[0], circle3_center[1]), radius=rmin, fc='orange', ec='red', alpha=0.6))
    plt.grid(True)
    plt.axis("equal")
    plt.show()
