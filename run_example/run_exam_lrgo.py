import os
import json
import time
import random
import numpy as np
import pandas as pd
from draw.plt2d import plt_visulazation
from matamp.agents.agent import Agent
from matamp.agents.target import Target
from matamp.agents.obstacle import Obstacle
from matamp.envs.mampenv import MACAEnv
from matamp.ta_policies import ta_policy_dict
from matamp.policies.gosPolicy import GOSPolicy
from matamp.configs.config import DT, NEAR_GOAL_THRESHOLD, SAMPLE_SIZE
from matamp.tools.utils import l2norm, normalize, get_boundaries, norm


def get_exit_pos(exit_area, agent_num, task_area, ls=0.3, us=0.2):
    min_xa, max_xa, min_ya, max_ya = get_boundaries(task_area)
    min_x, max_x, min_y, max_y = get_boundaries(exit_area)
    if abs(min_x - min_xa) < abs(min_x - max_xa) and abs(min_y - min_ya) < 0.5:
        in_direction = 'west'
    elif abs(min_x - min_xa) > abs(min_x - max_xa) and abs(min_y - min_ya) < 0.5:
        in_direction = 'east'
    elif abs(min_y - min_ya) > abs(min_y - max_ya) and abs(min_x - min_xa) < 0.5:
        in_direction = 'north'
    elif abs(min_y - min_ya) < abs(min_y - max_ya) and abs(min_x - min_xa) < 0.5:
        in_direction = 'south'
    else:
        in_direction = 'east'
    agent_goal = np.random.uniform(low=(max_x - us, min_y + ls), high=(max_x - us, max_y - ls), size=(agent_num, 2))

    return agent_goal, in_direction


def gen_polygonal_vertices(center, radius, num_vertice, is_random=True):
    regular_polygon = True if num_vertice < 6 else False

    if regular_polygon:
        vertices = []
        s_theta = np.random.uniform(0, 360 / num_vertice) if is_random else 360 / num_vertice / 2
        for theta in [np.deg2rad(i * (360 / num_vertice)+s_theta) for i in range(num_vertice)]:
            vertice = center + radius * np.array([np.cos(theta), np.sin(theta)])
            vertices.append(vertice)
    else:
        points_num = int(2 * np.pi * radius / 30.0)
        all_vertices = []
        for angular in [np.deg2rad(i * (360 / points_num)) for i in range(points_num)]:
            vertice = center + radius * np.array([np.cos(angular), np.sin(angular)])
            all_vertices.append([vertice, angular])
        vertices_theta = random.sample(all_vertices, num_vertice)
        vertices_theta = sorted(vertices_theta, key=lambda x: x[1])
        vertices = []
        for vertice in vertices_theta:
            vertices.append(vertice[0])

    return vertices


def gen_objs_pos_sim(ag_num=100, tar_num=500, ob_num=160):
    scl = 1000
    x_length = 6.0 * scl
    y_length = 5.0 * scl
    x_end = 0.8 * scl
    x_ends = 5.2 * scl
    x_endg = 6.0 * scl
    y_start = 0.0
    grid_rob = 25
    grid_length = 150.0
    ls = 0.1 * scl
    us = 0.25 * scl

    # simulation 1: The endpoint and the starting point are located at opposite ends
    task_area = [[0.0, y_start], [x_length, y_start], [x_length, y_length], [0.0, y_length], [0.0, y_start]]
    exit_area = [[-x_end, y_start], [0.0, y_start], [0.0, y_length], [-x_end, y_length], [-x_end, y_start]]

    # simulation 2: The endpoint and starting point are in the same area
    # task_area = [[-x_end, y_start], [x_length, y_start], [x_length, y_length], [-x_end, y_length], [-x_end, y_start]]
    # exit_area = [[x_ends, y_start], [x_endg, y_start], [x_endg, y_length], [x_ends, y_length], [x_ends, y_start]]

    # Generate preselected positions for the robots.
    min_x, max_x, min_y, max_y = get_boundaries(task_area)
    ys = [round(ls + grid_rob * j, 2) for j in range(int((max_y - ls - min_y) / grid_rob))]
    # Based on the robot diameter, which has a maximum diameter of 20,
    # and with each grid having a length of 25, the random selection
    # can be made within the range of (25-20)/2.
    r = 2.5
    ag_poses = [[round(max_x - 1*ls + np.random.uniform(-r, r), 2), round(y + np.random.uniform(-r, r), 2)] for y in ys]
    # Generate preselected positions for targets or obstacles.
    xs = [ls + grid_length * i for i in range(int((max_x - 1.2 * us - min_x) / grid_length))]
    ys = [ls + grid_length * j for j in range(int((max_y - min_y) / grid_length))]
    obj_poses = []
    # Based on the robot diameter, which has a maximum diameter of 100,
    # and with each grid having a length of 150, a certain margin is left,
    # allowing for a slight overlap between targets and obstacles.
    r = 30
    for x in xs:
        if x > max_x - us:
            continue
        for y in ys:
            obj_poses.append([round(x + np.random.uniform(-r, r), 2), round(y + np.random.uniform(-r, r), 2)])
    objs_pos = random.sample(obj_poses, tar_num + ob_num)
    agent_start = random.sample(ag_poses, ag_num)
    task_pos = objs_pos[:tar_num]
    obstacle_pos = objs_pos[tar_num:]

    # simulation 1: 50 robots and 200 targets, The endpoint and the starting point are located at opposite ends.
    agent_start = np.array([[5901.57, 2301.74], [5898.36, 4322.52], [5899.96, 3525.11], [5899.22, 747.83], [5902.26, 1523.48], [5900.72, 3948.42], [5901.13, 4475.77], [5901.22, 3147.82], [5900.93, 1049.76], [5901.15, 1626.56], [5900.91, 1825.99], [5898.71, 2876.57], [5900.95, 3747.59], [5898.92, 4647.89], [5901.2, 4773.51], [5900.84, 4298.31], [5898.8, 2424.05], [5898.65, 4522.73], [5899.24, 1500.1], [5898.59, 1775.4], [5899.17, 573.4], [5902.01, 4626.78], [5899.86, 3676.98], [5898.58, 2273.79], [5901.35, 4451.24], [5901.68, 1950.89], [5898.92, 2949.07], [5901.89, 4176.93], [5900.79, 199.36], [5899.13, 650.73], [5901.14, 2151.24], [5900.37, 997.84], [5901.78, 3601.22], [5897.63, 4850.58], [5901.86, 2725.45], [5901.9, 2500.89], [5900.3, 899.46], [5899.91, 1124.52], [5901.47, 3873.68], [5899.44, 3502.1], [5898.47, 3724.16], [5900.13, 4951.39], [5898.48, 2448.86], [5897.68, 847.82], [5900.32, 2851.66], [5902.43, 724.35], [5902.12, 3701.27], [5898.62, 2026.04], [5901.1, 1324.9], [5902.3, 2400.33]])
    task_pos = np.array([[3888, 4337], [3400.93, 4234.98], [4669, 3196], [956, 1863], [1074, 1748], [838, 1717],
                         [3427.76, 1439.68], [269.15, 4609.63], [2100.85, 4606.14], [3221.62, 1577.98], [5207.89, 865.19], [247.71, 863.87], [559.72, 1925.91], [667.54, 4928.82], [2473.86, 4449.04], [4721.51, 2681.61], [3112.12, 2536.54], [3273.55, 385.46], [2225.57, 2934.14], [2684.21, 815.9], [667.52, 1115.57], [2958.62, 2515.24], [1180.54, 4610.82], [2215.38, 586.23], [865.92, 60.59], [3118.3, 4738.25], [2517.53, 3265.53], [5655.8, 1028.02], [729.83, 276.66], [370.74, 4864.33], [4903.29, 1705.36], [1162.59, 2214.56], [4909.58, 1870.99], [2379.61, 3954.45], [1185.91, 575.8], [3072.05, 1927.49], [3668.27, 3712.22], [4470.6, 2675.28], [3075.6, 1787.8], [5372.85, 4721.03], [4740.38, 1752.55], [5623.97, 2081.6], [1557.59, 4758.99], [259.58, 1140.83], [2968.68, 1330.95], [2924.14, 4883.24], [3215.38, 3694.1], [2745.65, 3276.01], [3997.26, 1441.26], [671.08, 2463.97], [4015.59, 92.9], [2077.77, 4185.68], [1570.07, 3711.96], [3331.31, 2045.43], [3436.31, 4035.32], [1550.51, 1570.37], [4926.28, 2464.72], [3729.07, 1449.76], [5387.97, 3082.1], [2654.94, 3805.8], [3531.54, 4863.27], [3127.27, 4134.22], [2658.06, 1731.91], [4120.5, 92.59], [4896.52, 4901.12], [4414.93, 2387.38], [4936.1, 3966.28], [1449.16, 3529.1], [3873.84, 2183.75], [1885.31, 1896.44], [3233.86, 4144.38], [1161.24, 4464.67], [5671.08, 3708.93], [4895.06, 4572.11], [4137.55, 3064.25], [1260.93, 661.48], [3697.67, 3983.8], [4747.25, 2519.21], [3682.55, 3571.97], [2661.78, 4357.63], [1002.78, 2658.96], [5489.5, 4154.6], [2695.16, 1431.82], [4881.1, 2221.84], [5215.65, 3388.67], [3660.17, 997.17], [65.05, 4716.21], [5376.44, 4619.46], [5352.52, 4938.22], [1662.72, 4272.73], [4574.45, 1482.87], [2914.59, 3077.79], [4414.68, 1773.26], [125.05, 1580.5], [3809.69, 1781.46], [1717.31, 3548.87], [520.01, 2462.35], [4895.58, 1167.11], [78.72, 2763.81], [1667.08, 4397.31], [3246.87, 4318.86], [1271.35, 4146.97], [4918.54, 2664.93], [4459.57, 400.12], [3249.37, 1788.49], [4897.59, 4323.05], [4584.06, 4432.59], [104.54, 728.86], [5362.27, 3710.1], [4610.3, 2363.85], [4030.0, 1332.97], [1271.8, 2351.27], [1413.64, 854.82], [4485.12, 2799.78], [4307.14, 3288.75], [368.11, 3218.83], [3964.0, 548.31], [1020.48, 1478.8], [3424.75, 510.41], [3239.64, 2954.58], [4579.2, 1575.01], [3562.58, 1148.15], [2496.06, 2231.12], [4563.56, 2817.03], [2359.7, 4158.72], [701.98, 1323.66], [2173.17, 4310.01], [2922.61, 1798.13], [1169.95, 1576.68], [1772.84, 4021.14], [1026.18, 884.97], [5332.17, 1030.62], [4119.89, 4568.4], [4581.59, 3985.77], [3816.93, 4255.98], [3257.24, 1278.81], [3418.84, 1607.09], [3280.77, 4711.84], [1021.41, 2171.18], [4473.16, 1459.19], [856.93, 3249.44], [5468.16, 430.11], [2021.06, 1167.84], [2485.77, 2680.04], [1611.8, 219.14], [4118.05, 3284.92], [2065.42, 2615.75], [3404.24, 3215.44], [2803.4, 1026.47], [3415.37, 2166.81], [3814.88, 261.91], [2364.6, 1907.65], [256.04, 1590.97], [3541.61, 4710.61], [2359.61, 4727.66], [2776.73, 4609.01], [3982.07, 730.88], [3988.85, 3862.58], [4443.22, 3401.95], [302.38, 2704.64], [2201.0, 3560.67], [881.38, 1560.79], [5217.72, 83.82], [3262.59, 553.67], [2477.83, 4923.1], [1614.24, 2923.12], [2331.95, 1624.7], [2826.09, 834.9], [3405.52, 2610.25], [4422.22, 4423.43], [2910.45, 4151.68], [90.87, 4631.85], [5226.21, 1265.28], [3077.62, 2053.52], [5385.3, 2920.02], [1331.21, 2063.19], [5521.23, 3818.54], [2214.14, 3980.03], [879.1, 2792.24], [853.98, 3547.05], [1421.85, 4712.75], [5628.8, 2940.58], [4187.95, 2932.24], [1712.96, 241.12], [5490.0, 3285.13], [836.92, 2014.72], [2672.31, 427.27], [5180.02, 4580.18], [373.23, 849.55], [3665.89, 134.44], [548.0, 2176.69], [394.88, 2208.4], [3252.24, 3997.56], [4585.42, 3721.24], [1480.69, 1277.59], [4619.33, 420.76], [84.76, 3403.49], [1455.5, 3691.22], [5215.03, 3103.89], [4157.05, 319.43], [1713.75, 4610.57], [3698.3, 3402.46], [1266.08, 361.1]])
    obstacle_pos = np.array([[4038, 4356], [3955.64, 4415.83], [3848.09, 4451.03], [3733.09, 4397.91], [3684.68, 4290.56], [3714.86, 4184.99], [3816.81, 4145.44], [3936, 4164], [4042, 4214],
                             [5050.87, 3618.63], [5054, 3508], [5051.84, 3380.93], [5053.12, 3272.99],
                             [5052.6, 3165.06],  [5052.39, 3062.08], [5051.87, 2945.63], [4932.6, 3275.06],
                             [3512.46, 240.77], [1744.34, 365.57], [1482.21, 668.55], [4597.89, 1318.52], [1901.19, 536.62], [4264.97, 272.03], [3404.64, 4735.96], [3514.89, 1038.83], [3976.04, 3534.33], [4563.68, 2530.17], [2910.33, 2178.77], [2360.66, 4630.88], [1724.63, 2655.81], [849.3, 3403.76], [573.53, 2674.02], [82.58, 1030.07], [3545.51, 527.4], [3114.94, 2826.53], [1460.56, 421.79], [1270.76, 1887.69], [4618.2, 2081.17], [717.31, 3817.17], [1135.87, 2313.44], [223.06, 981.72], [1602.97, 2196.34], [3718.71, 395.27], [1220.03, 4908.32], [5215.7, 398.17], [850.0, 1037.88], [3428.56, 124.08], [1951.7, 3872.38], [5461.35, 3967.56], [5229.11, 4898.84], [2966.95, 1893.33], [2641.5, 2059.86], [2321.21, 1131.71], [522.93, 3660.73], [1425.18, 3379.12], [3382.81, 1266.78], [4767.44, 397.43], [371.96, 1741.17], [420.69, 2829.24], [2670.76, 102.28], [3575.96, 68.14], [78.86, 1293.07], [3377.24, 880.83], [2202.51, 3419.82], [3072.36, 2206.37], [2366.72, 4312.89], [4206.76, 3416.55], [4724.13, 4443.07], [5500.84, 2783.64], [824.71, 2370.42], [398.43, 4298.32], [2465.06, 1914.74], [5355.98, 434.82], [1923.43, 3671.84], [2769.23, 4782.2], [1634.34, 1003.21], [2767.21, 4865.72], [3548.19, 3219.07], [5686.79, 1262.43], [3061.37, 2613.3], [1861.56, 2275.08], [4712.04, 1879.63], [2788.88, 2386.11], [4008.08, 2490.56], [5025.14, 2170.67], [4639.27, 516.72], [5472.22, 2031.28], [2530.86, 1765.54], [2171.66, 1181.41], [878.92, 2488.2], [4157.17, 2369.33], [557.13, 660.74], [5066.5, 251.76], [273.52, 1430.94], [545.58, 4638.59], [4121.8, 3715.01], [2625.91, 998.72], [2963.55, 4328.54], [2974.0, 270.48], [4410.8, 108.35], [413.52, 2023.13], [1893.38, 722.14], [4714.18, 114.13], [5639.88, 3572.67], [5052.09, 1786.96], [5036.6, 4595.7], [1478.8, 2631.66], [4118.42, 2533.01], [4784.41, 260.24], [554.62, 2356.13], [5175.43, 701.98], [5352.13, 3426.41], [1424.73, 289.85], [1601.2, 2358.41], [3375.15, 3533.27], [5048.93, 2341.02], [966.61, 4028.23], [1121.32, 2021.55], [5532.91, 1449.86], [3547.59, 2348.7], [2621.52, 664.46], [4616.39, 4155.63], [3075.09, 4907.01], [4171.17, 1171.99], [5321.54, 4021.67], [1395.63, 1588.99], [4893.79, 4465.41], [2622.45, 4574.46], [1123.31, 2512.27], [511.12, 3439.25], [2804.44, 1318.6], [214.19, 4777.73], [699.93, 2631.27], [3393.7, 3880.73], [3704.03, 3106.59], [222.78, 4223.26], [1763.42, 837.4], [899.85, 3074.64], [579.42, 1716.98], [1153.5, 3911.88], [83.44, 275.55], [4584.79, 690.23], [1989.03, 4028.55], [2490.91, 369.79], [2777.09, 74.49], [2919.88, 4630.29], [4781.25, 544.49], [3224.48, 2819.22], [2073.38, 999.83], [3529.36, 3574.56], [1861.98, 2916.97], [4595.96, 4770.52], [470.96, 1221.91], [2509.35, 2954.7], [1920.32, 4413.32], [5264.42, 4349.22], [2294.3, 2512.55], [1252.99, 1093.31], [622.06, 4108.77], [4293.28, 3928.15], [873.63, 687.25], [966.01, 257.98], [3084.75, 3231.15], [2946.13, 1566.33], [3146.13, 4511.61], [2954.46, 544.26], [860.02, 4466.95], [4262.25, 587.19], [228.54, 3874.28], [4423.84, 3070.46], [3715.47, 2637.67], [1926.32, 4829.36], [4780.03, 934.75], [3812.75, 1233.47], [3565.54, 1911.43], [5414.61, 1721.33], [4133.59, 1728.31], [4288.5, 950.09], [3837.29, 4800.63], [2131.08, 223.87], [4929.69, 1439.77], [236.18, 2492.64], [2979.14, 3888.89], [4188.07, 2109.28], [5365.11, 2251.67], [3100.35, 1056.4], [1150.13, 3529.29], [5196.07, 2631.46], [2762.04, 3516.99], [309.62, 474.85], [2614.35, 4086.64],
                             [1410, 4420], [1550, 3980], [1832, 3322], [1265, 3064], [2412, 3735],
                             [2063, 1900], [1713, 1900], [1713, 1550], [2063, 1550]])

    # simulation 2: 50 robots and 200 targets, The endpoint and starting point are in the same area.
    # agent_start = agent_start - np.array([600, 0])
    # task_pos = task_pos - np.array([600, 0])
    # obstacle_pos = obstacle_pos - np.array([600, 0])

    print(list(agent_start))
    print(list(task_pos))
    print(list(obstacle_pos))
    return exit_area, task_area, agent_start, task_pos, obstacle_pos


def generative_agent_para(agent_num):
    agent_type1 = {'radius': 5.0, 'capacity': 8, 'pref_speed': 6.0,
                   'max_speed': 8.0, 'max_angular': np.deg2rad(30), 'min_speed': 3.0}
    agent_type2 = {'radius': 8.0, 'capacity': 10, 'pref_speed': 6.0,
                   'max_speed': 9.0, 'max_angular': np.deg2rad(30), 'min_speed': 4.0}
    agent_type3 = {'radius': 10.0, 'capacity': 12, 'pref_speed': 6.0,
                   'max_speed': 10.0, 'max_angular': np.deg2rad(30), 'min_speed': 5.0}
    agents_type = []
    for i in range(agent_num):
        if i < agent_num / 3:
            agents_type.append(agent_type3)
        elif agent_num / 3 <= i < (2*agent_num / 3):
            agents_type.append(agent_type2)
        else:
            agents_type.append(agent_type1)
    return agents_type


def build_agent_target_sim(ta_name):
    ready_done = True
    target_radius = 50.0
    obstacle_radius = 50.0
    exit_area, task_area, agents_pos, targets_pos, ob_pos = gen_objs_pos_sim(ag_num=100, tar_num=400, ob_num=200)
    robots_num, target_num, ob_num, = len(agents_pos), len(targets_pos), len(ob_pos)
    agents_type = generative_agent_para(robots_num)

    # build agents
    agents = []
    init_vel = [0.0, 0.0]
    expansion = 1.05 * target_radius  # Continue the straight line through the target position.
    agents_goal, in_direction = get_exit_pos(exit_area, robots_num, task_area, ls=50, us=350)
    all_agent_load = 0
    for i in range(robots_num):
        agent_radius = agents_type[i]['radius']
        pref_speed = agents_type[i]['pref_speed']
        maxSpeed = agents_type[i]['max_speed']
        minSpeed = agents_type[i]['min_speed']
        max_angular = agents_type[i]['max_angular']
        turning_radius = maxSpeed / agents_type[i]['max_angular']
        Lt = int(target_num / robots_num + 1) if robots_num > 80 or target_num > 300 else 8
        capacity = agents_type[i]['capacity'] if ta_name != 'cbba' else Lt
        all_agent_load += capacity
        agents.append(Agent(start_pos=agents_pos[i], goal_pos=agents_goal[i], vel=init_vel, radius=agent_radius,
                            pref_speed=pref_speed, maxSpeed=maxSpeed, maxAngular=max_angular, minSpeed=minSpeed,
                            turning_radius=turning_radius, policy=GOSPolicy, taPolicy=ta_policy_dict[ta_name],
                            id=i, tid='robot' + str(i), dt=DT, near_goal_threshold=NEAR_GOAL_THRESHOLD,
                            sampling_size=SAMPLE_SIZE, member_num=robots_num, capacity=capacity,
                            expansion=expansion, task_area=task_area, exit_area=exit_area, in_direction=in_direction))

    # build targets
    targets = []
    tar_num = target_num
    for i in range(tar_num):
        targets.append(Target(pos=targets_pos[i][:2], shape_dict={'shape': 'circle', 'feature': target_radius},
                              id=i, tid=str(i), is_sl=True))

    # build obstacles
    obstacles = []
    s_num = ob_num - 50
    for i in range(s_num):
        obstacles.append(Obstacle(pos=ob_pos[i][:2], shape_dict={'shape': 'circle', 'feature': obstacle_radius},
                                  id=i, tid='obstacle' + str(i)))
    n = 0
    for i in range(s_num, ob_num):
        radius = 3 * obstacle_radius
        if ob_num - 10 < i <= ob_num - 5:     # The fifth-from-last is concave.
            radius = 3 * obstacle_radius
            vertices = gen_polygonal_vertices(ob_pos[i][:2], radius, 3, is_random=False)
            vertices.append(ob_pos[i][:2])
        elif i > ob_num - 5:        # The last four are squares.
            radius = 3 * obstacle_radius
            vertices = gen_polygonal_vertices(ob_pos[i][:2], radius, 4, is_random=False)
        elif i <= s_num+13:
            vertices = gen_polygonal_vertices(ob_pos[i][:2], radius, 3, is_random=False)
        elif s_num+13 < i <= s_num+27:
            vertices = gen_polygonal_vertices(ob_pos[i][:2], radius, 4, is_random=False)
        else:
            vertices = gen_polygonal_vertices(ob_pos[i][:2], radius, 5, is_random=False)
        obstacles.append(Obstacle(pos=ob_pos[i][:2], shape_dict={'shape': 'polygon', 'feature': radius},
                                  id=i, tid='obstacle' + str(i), vertices=vertices, is_poly=True))
    ob_vertices = []
    for k in range(len(obstacles)):
        if obstacles[k].is_poly:
            ob_vertices += obstacles[k].vertices_

    for j in range(len(ob_vertices)):
        ob_vertices[j].id_ = j

    poses = [list(agents_pos), list(targets_pos), list(ob_pos)]
    return agents, targets, obstacles, ob_vertices, ready_done, poses


def get_agent_max_run_dist(agent):
    agent.max_run_dist = 0.0
    for i in range(len(agent.path) - 1):
        agent.max_run_dist += l2norm(agent.path[i], agent.path[i + 1])
    max_run_dist = max(agent.max_run_dist, agent.expansion) * 10
    agent.max_run_dist = max_run_dist


def goal_is_in_unsuatable_area(agents, targets, obstacles, agent):
    env_objects = agents + targets + obstacles
    goal = np.array([agent.goal_global_frame[0], agent.goal_global_frame[1]])
    if agent.is_exit_area:
        for obj in agents:
            if obj.id == agent.id:
                continue
            else:
                if 1e-5 < l2norm(goal, obj.goal_global_frame) < obj.radius + agent.radius + 0.02:  # The positions overlap.
                    pg_po = goal - obj.goal_global_frame
                    n_pg_po = normalize(pg_po)
                    length_move = obj.radius + agent.radius + 0.02 - norm(pg_po)  # Move the endpoint out of the threat zone by the specified distance.
                    l_pg_po = length_move * n_pg_po
                    goal = goal + l_pg_po
                    agent.goal_global_frame = np.array(goal)
    else:
        for obj in env_objects:
            if obj.tid[obj.id] == agent.tid[agent.id]:
                continue
            else:
                if 1e-5 < l2norm(goal, obj.goal_global_frame) < obj.radius + agent.radius + 0.02:  # The positions overlap.
                    pg_po = goal - obj.goal_global_frame
                    n_pg_po = normalize(pg_po)
                    length_move = obj.radius + agent.radius + 0.02 - norm(pg_po)  # Move the endpoint out of the threat zone by the specified distance.
                    l_pg_po = length_move * n_pg_po
                    goal = goal + l_pg_po
                    agent.goal_global_frame = np.array(goal)


def update_goal_pos(agents, targets, obstacles, agent):
    if agent.in_direction == 'west' or agent.in_direction == 'east':
        goal = np.array([agent.goal_global_frame[0], agent.path[1][1]])
        agent.goal_global_frame = np.array([goal[0], goal[1]])
    else:
        goal = np.array([agent.path[1][0], agent.goal_global_frame[1]])
        agent.goal_global_frame = np.array([goal[0], goal[1]])
    goal_is_in_unsuatable_area(agents, targets, obstacles, agent)
    agent.path[0] = [agent.goal_global_frame[0], agent.goal_global_frame[1], 0]


def adjust_agent_goal(ag):
    if len(ag.exit_area) == 0:
        goal = np.array([ag.path[1][0], ag.path[1][1]])
        ag.path[0] = np.array([goal[0], goal[1], 0])
        ag.goal_global_frame = np.array(goal)
        get_agent_max_run_dist(ag)  # Update the maximum distance.
    else:
        get_agent_max_run_dist(ag)  # Update the maximum distance.


def uodate_goal_and_maxdist(agents, targets, obstacles):
    # Adjust the endpoint position and update the maximum distance.
    for agent in agents:
        update_goal_pos(agents, targets, obstacles, agent)
        adjust_agent_goal(agent)


def write_trajs(agents, trajs_save_dir):
    os.makedirs(trajs_save_dir, exist_ok=True)
    writer = pd.ExcelWriter(trajs_save_dir + '/trajs.xlsx')
    for agent in agents:
        agent.history_info.to_excel(writer, sheet_name='agent' + str(agent.id))
    writer.save()


def write_env_cfg(agents, trajs_save_dir, objs, obstacles, assignment_results):
    task_area = agents[0].task_area
    exit_area = agents[0].exit_area
    agents_num = len(agents)
    tar_num = len(objs)
    info_dict_to_visualize = {
        'TotalTaskAssignRewards': round(agents[0].all_agent_rewards, 5),
        'AllTravelDist': 0.0,
        'MaximumTravelDist': 0.0,
        'MaximumTravelTime': 0.0,
        'CoillisionSuccessRate': 0.0,
        'MaxYawRate': 0.0,
        'TaskCompletedRate': 0.0,
        'AverageCost': 0.0,
        'all_compute_time': 0.0,
        'successful_num': 0,
        'cleard_tar_num': 0,
        'all_step_num': 0,
        'solve_ta_cost': round(agents[0].solve_ta_cost, 3),
        'assignment_results': assignment_results,
        'all_agent_info': [],
        'all_target': [],
        'all_obstacle': [],
        'TaskArea': list(task_area),
        'ExitArea': list(exit_area),
    }
    all_step_num = 0
    all_cleared_tar_num = 0
    all_travel_dist = 0.0
    agents_num_of_success = 0
    total_compute_time = 0.0
    agents_max_yaw_rate = []
    agents_travel_dist = []
    agents_travel_time = []

    for agent in agents:
        agent_info_dict = {'id': agent.id, 'gp': agent.group, 'radius': agent.radius,
                           'goal_pos': agent.goal_global_frame.tolist()}
        info_dict_to_visualize['all_agent_info'].append(agent_info_dict)
        if not agent.is_collision and not agent.is_out_of_max_time and agent.is_in_ending_area():
            agents_num_of_success += 1
            total_compute_time += agent.solve_time_cost
            all_step_num += agent.step_num
            all_cleared_tar_num += agent.cleared_num
            all_travel_dist += agent.travel_dist
            agents_max_yaw_rate.append(agent.max_yaw_rate)
            agents_travel_dist.append(agent.travel_dist)
            agents_travel_time.append(agent.travel_time)

    MaximumTravelDist = round(np.float(max(agents_travel_dist)), 2)
    MaximumTravelTime = round(np.float(max(agents_travel_time)), 2)
    CoillisionSuccessRate = round(100 * agents_num_of_success / agents_num, 2)
    MaxYawRate = round(np.float(max(agents_max_yaw_rate)) / DT, 3)
    TaskCompletedRate = round(100 * all_cleared_tar_num / tar_num, 2)
    AverageCost = round(1000 * total_compute_time / max(all_step_num, 1), 3)

    info_dict_to_visualize['all_compute_time'] = round(total_compute_time, 3)
    info_dict_to_visualize['successful_num'] = agents_num_of_success
    info_dict_to_visualize['cleard_tar_num'] = all_cleared_tar_num
    info_dict_to_visualize['all_step_num'] = all_step_num
    info_dict_to_visualize['AllTravelDist'] = all_travel_dist
    info_dict_to_visualize['MaximumTravelDist'] = MaximumTravelDist
    info_dict_to_visualize['MaximumTravelTime'] = MaximumTravelTime
    info_dict_to_visualize['CoillisionSuccessRate'] = CoillisionSuccessRate
    info_dict_to_visualize['MaxYawRate'] = MaxYawRate
    info_dict_to_visualize['TaskCompletedRate'] = TaskCompletedRate
    info_dict_to_visualize['AverageCost'] = AverageCost

    for target in objs:
        obstacle_info_dict = {'id': target.id, 'position': list(target.pos), 'shape': target.shape, 'feature': target.feature}
        info_dict_to_visualize['all_target'].append(obstacle_info_dict)
    for obj in obstacles:
        obstacle_info_dict = {'position': list(obj.pos), 'shape': obj.shape, 'feature': obj.feature,
                              'vertices': obj.vertices_pos}
        info_dict_to_visualize['all_obstacle'].append(obstacle_info_dict)

    info_str = json.dumps(info_dict_to_visualize, indent=4)
    with open(trajs_save_dir + '/env_cfg.json', 'w') as json_file:
        json_file.write(info_str)
    json_file.close()


def plt_path_node(agents, objs, obstacles):
    task_area = agents[0].task_area
    exit_area = agents[0].exit_area
    paths_smooth = []
    for a in agents:
        path_smooth = a.path_node
        paths_smooth.append(path_smooth)
    plt_visulazation(paths_smooth, agents, objs, obstacles, task_area, exit_area)


def plt_final_trajs(agents, objs, obstacles):
    taskArea = agents[0].task_area
    exitArea = agents[0].exit_area
    paths_smooth = []
    for a in agents:
        path_smooth = a.history_pos
        paths_smooth.append(path_smooth)
    plt_visulazation(paths_smooth, agents, objs, obstacles, taskArea, exitArea)


def run_temp(total_time=900):
    ta_name = 'lrca'
    # Build robots, targets and obstacles.
    robots, targets, obstacles, polyobs_verts, ready_done, objs_pos = build_agent_target_sim(ta_name=ta_name)
    # # scenario information
    if ready_done:
        st = time.time()
        env = MACAEnv()
        env.set_agents(robots, targets=targets, obstacles=obstacles, obs_verts=polyobs_verts, ta_name=ta_name)
        uodate_goal_and_maxdist(robots, targets, obstacles)

        step = 0
        while step * DT < total_time:
            actions = {}
            which_agents_done = env.step(actions)

            # Is arrived.
            if which_agents_done:
                print("All agents finished!" + str(step))
                break
            step += 1
            print("step is ", step)

        et = time.time()
        print(et - st)
        for obj_type in objs_pos:
            print(obj_type)

        assignment_results = {}
        max_path_plan_cost = []
        for rob in robots:
            rob_assign = []
            for tar in rob.targets:
                rob_assign.append(tar.id)
            assignment_results[rob.id] = rob_assign
            max_path_plan_cost.append(rob.max_path_plan_cost)
        print(robots[0].assignment_results)
        print(assignment_results)
        print('max_path_plan_cost: ', max_path_plan_cost)
        # plt_path_node(robots, targets, obstacles)
        plt_final_trajs(robots, targets, obstacles)  # Visualize the trajectory points.

        # scenario information
        trajs_save_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/../draw/lrgo/log/'
        write_env_cfg(robots, trajs_save_dir, targets, obstacles, assignment_results)

        # write trajectories
        write_trajs(robots, trajs_save_dir)


if __name__ == "__main__":
    run_temp(total_time=2400)
