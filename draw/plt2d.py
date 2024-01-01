import copy
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from math import sin, cos, atan2, sqrt, pow

from draw.vis_util import get_2d_car_model, get_2d_uav_model
from matamp.tools.utils import get_boundaries, l2norm
from matamp.configs.config import NEAR_GOAL_THRESHOLD, NEAR_GOAL_THRESHOLD1

simple_plot = True


# img = plt.imread('beijing.jpg', 100)


def convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame):
    alpha = heading_global_frame
    for point in agent_model:
        x = point[0]
        y = point[1]
        ll = sqrt(pow(x, 2) + pow(y, 2))
        alpha_model = atan2(y, x)
        alpha_ = alpha + alpha_model - np.pi / 2
        point[0] = ll * cos(alpha_) + pos_global_frame[0]
        point[1] = ll * sin(alpha_) + pos_global_frame[1]


def draw_agent_2d(ax, pos_global_frame, heading_global_frame, my_agent_model, color='grey', alpha=0.9):
    agent_model = my_agent_model
    convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame)

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    path = Path(agent_model, codes)
    col = [0.8, 0.8, 0.8]
    patch = patches.PathPatch(path, fc=color, ec=color, lw=1.0, alpha=0.9)

    ax.add_patch(patch)


def draw_polygon_2d(ax, vertices, color='blue', alpha=0.9):
    vertices.append(vertices[0])
    codes = []
    for i in range(len(vertices)):
        if i == 0:
            codes.append(Path.MOVETO)
        elif i == len(vertices) - 1:
            codes.append(Path.CLOSEPOLY)
        else:
            codes.append(Path.LINETO)

    path = Path(vertices, codes)
    col = [0.8, 0.8, 0.8]
    patch = patches.PathPatch(path, fc='grey', ec='grey', lw=1.0, alpha=alpha)

    ax.add_patch(patch)


def draw_polygon_2d_ins(ax, axins, vertices, color='blue', alpha=0.9):
    vertices.append(vertices[0])
    codes = []
    for i in range(len(vertices)):
        if i == 0:
            codes.append(Path.MOVETO)
        elif i == len(vertices) - 1:
            codes.append(Path.CLOSEPOLY)
        else:
            codes.append(Path.LINETO)

    path = Path(vertices, codes)
    col = [0.8, 0.8, 0.8]

    patch = patches.PathPatch(path, fc='grey', ec='grey', lw=1.0, alpha=alpha)
    ax.add_patch(patch)

    patch1 = patches.PathPatch(path, fc='grey', ec='grey', lw=1.0, alpha=alpha)
    axins.add_patch(patch1)


def draw_traj_2d(ax, tar_and_obs_info, agents_info, agents_traj_list, step_num_list, current_step,
                 task_area, exit_area, plot_target, target_color):
    plt_colors = get_colors()
    for idx, agent_traj in enumerate(agents_traj_list):
        agent_id = agents_info[idx]['id']
        agent_gp = agents_info[idx]['gp']
        agent_rd = agents_info[idx]['radius']
        agent_goal = agents_info[idx]['goal_pos']
        group = agent_gp
        color_ind = agent_id % len(plt_colors)
        # p_color = p_colors[color_ind]
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num - 1:
            plot_step = ag_step_num - 1
        else:
            plot_step = current_step

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        alpha = agent_traj['alpha']
        spd = agent_traj['speed']

        # Draw the end area.
        if len(exit_area) > 0:
            min_xe, max_xe, min_ye, max_ye = get_boundaries(exit_area)
            width = max_xe - min_xe
            height = max_ye - min_ye
            # rect = patches.Rectangle((min_xe, min_ye), width, height,
            #                          linewidth=2, edgecolor='green', facecolor='none', zorder=3, alpha=1)
            # ax.add_patch(rect)
            ax.add_patch(plt.Rectangle((min_xe, min_ye), width, height, linewidth=2, edgecolor='green',
                                       facecolor='none', zorder=3, alpha=1))
            # axins.add_patch(plt.Rectangle((min_xe, min_ye), width, height, linewidth=2, edgecolor='green',
            #                               facecolor='none', zorder=3, alpha=1))

        # Draw the task area.
        min_xt, max_xt, min_yt, max_yt = get_boundaries(task_area)
        # max_xt = 5800
        width_t = max_xt - min_xt
        height_t = max_yt - min_yt
        # rect = patches.Rectangle((min_xt, min_yt), width_t, height_t,
        #                          linewidth=2, edgecolor='blue', facecolor='none', alpha=1)
        # ax.add_patch(rect)
        ax.add_patch(plt.Rectangle((min_xt, min_yt), width_t, height_t, linewidth=2, edgecolor='blue',
                                   facecolor='none', alpha=1))
        # axins.add_patch(plt.Rectangle((min_xt, min_yt), width_t, height_t, linewidth=2, edgecolor='blue',
        #                               facecolor='none', alpha=1))

        # # Draw the robot's starting position.
        ax.add_patch(plt.Circle((pos_x[0], pos_y[0]), radius=agent_rd, fc='none', ec=plt_color, linewidth=1, alpha=1))
        # axins.add_patch(
        #     plt.Circle((pos_x[0], pos_y[0]), radius=agent_rd, fc='none', ec=plt_color, linewidth=1, alpha=1))

        # Draw a solid line.
        plt.plot(pos_x[:plot_step], pos_y[:plot_step], linewidth=0.6, color=plt_color, zorder=3)

        # Draw the path passing through the cleared target locations.
        for i in range(len(tar_and_obs_info['targets_info'])):
            tar_id = tar_and_obs_info['targets_info'][i]['id']
            pos = tar_and_obs_info['targets_info'][i]['position']
            shape = tar_and_obs_info['targets_info'][i]['shape']
            if shape == 'circle':
                ob_rd = tar_and_obs_info['targets_info'][i]['feature']
            else:
                ob_rd1 = tar_and_obs_info['targets_info'][i]['feature']
                ob_rd = np.sqrt(ob_rd1[0] ** 2 + ob_rd1[1] ** 2) / 2

            # large-scale use NEAR_GOAL_THRESHOLD, real experiment use NEAR_GOAL_THRESHOLD1
            if l2norm([pos_x[plot_step], pos_y[plot_step]], pos) < NEAR_GOAL_THRESHOLD1 or plot_target[i]:
                plot_target[i] = True
                plt.plot(pos[0], pos[1], color=target_color[tar_id], marker='*', markersize=6, alpha=1)
                ax.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='none',
                                        ec='green', alpha=0.05, linestyle='--', linewidth=0.3))
                # axins.plot(pos[0], pos[1], color=target_color[tar_id], marker='*', markersize=3, alpha=1)
                # axins.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='none',
                #                            ec='green', alpha=0.05, linestyle='--', linewidth=0.3))

        # Draw an arrow.
        ax.arrow(pos_x[plot_step], pos_y[plot_step], 3 * agent_rd * cos(alpha[plot_step]),
                 3 * agent_rd * sin(alpha[plot_step]),
                 fc=plt_color, ec=plt_color, head_width=2 * agent_rd, head_length=2 * agent_rd)

        if simple_plot:
            ax.add_patch(plt.Circle((pos_x[plot_step], pos_y[plot_step]), radius=agent_rd, fc=plt_color, ec=plt_color))
        else:
            if group == 0:
                my_model = get_2d_car_model(size=agent_rd)
            else:
                my_model = get_2d_uav_model(size=agent_rd)
            pos = [pos_x[plot_step], pos_y[plot_step]]
            heading = alpha[plot_step]
            draw_agent_2d(ax, pos, heading, my_model)

    for i in range(len(tar_and_obs_info['targets_info'])):
        if tar_and_obs_info['targets_info'][i]['shape'] == 'circle':
            if plot_target[i]:
                pos = tar_and_obs_info['targets_info'][i]['position']
                ob_rd = tar_and_obs_info['targets_info'][i]['feature']
            else:
                tar_id = tar_and_obs_info['targets_info'][i]['id']
                pos = tar_and_obs_info['targets_info'][i]['position']
                ob_rd = tar_and_obs_info['targets_info'][i]['feature']
                ax.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='red', ec='red', alpha=0.5))
                plt.plot(pos[0], pos[1], color=target_color[tar_id], marker='*', markersize=6, alpha=1, zorder=3)

    for i in range(len(tar_and_obs_info['obstacles_info'])):
        pos = tar_and_obs_info['obstacles_info'][i]['position']
        shape = tar_and_obs_info['obstacles_info'][i]['shape']
        if shape == 'circle':
            ob_rd = tar_and_obs_info['obstacles_info'][i]['feature']
            ax.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='grey', ec='grey', alpha=0.9))
        elif shape == 'polygon':
            vertices = tar_and_obs_info['obstacles_info'][i]['vertices']
            draw_polygon_2d(ax, vertices, color='grey', alpha=0.9)
            # draw_polygon_2d_ins(ax, axins, vertices, color='grey', alpha=0.9)
        elif shape == 'rect':
            heading = 0.0
            rd = tar_and_obs_info['obstacles_info'][i]['feature']
            agent_rd = rd[0] / 2
            my_model = get_2d_car_model(size=agent_rd)
            draw_agent_2d(ax, pos, heading, my_model, color='grey', alpha=0.9)


def plot_save_one_pic(tar_and_obs_info, agents_info, agents_traj_list, step_num_list, filename,
                      current_step, task_area, exit_area, plot_target, target_color):
    fig = plt.figure(0)
    fig_size = (10, 7)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('equal')
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    draw_traj_2d(ax, tar_and_obs_info, agents_info, agents_traj_list,
                 step_num_list, current_step, task_area, exit_area, plot_target, target_color)

    fig.savefig(filename)
    # if current_step == 0: plt.show()
    # if current_step == 1500: plt.show()
    if current_step == 8340: plt.show()
    # if current_step == 8342: plt.show()
    # if current_step == 15494: plt.show()
    # if current_step == 15626: plt.show()
    # fig.savefig(filename)
    # plt.show()
    plt.close()


def plot_episode(tar_and_obs_info, agents_info, traj_list, step_num_list, plot_save_dir, base_fig_name, last_fig_name,
                 task_area, exit_area, tasks_scheme, show=False):
    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    print('num_agents:', num_agents, 'total_step:', total_step)
    assignment_results = tasks_scheme
    plot_target = [False for _ in range(len(tar_and_obs_info['targets_info']))]
    target_color = ['red' for _ in range(len(tar_and_obs_info['targets_info']))]
    plt_colors = get_colors()
    for agent_id in assignment_results:
        color_ind = int(agent_id) % len(plt_colors)
        plt_color = plt_colors[color_ind]
        for tar_id in assignment_results[agent_id]:
            target_color[tar_id] = plt_color

    while current_step < total_step:
        fig_name = base_fig_name + "_{:05}".format(current_step) + '.png'
        filename = plot_save_dir + fig_name
        plot_save_one_pic(tar_and_obs_info, agents_info, traj_list, step_num_list, filename,
                          current_step, task_area, exit_area, plot_target, target_color)
        print(filename)
        current_step += 5
    filename = plot_save_dir + last_fig_name
    plot_save_one_pic(tar_and_obs_info, agents_info, traj_list, step_num_list, filename,
                      total_step, task_area, exit_area, plot_target, target_color)


def get_cmap(N):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def get_colors():
    py_colors = np.array(
        [
            [255, 0, 0], [255, 165, 0], [0, 0, 255], [0, 255, 0], [160, 32, 240], [152, 251, 152], [255, 69, 0],
            [255, 99, 71], [132, 112, 255], [0, 255, 255], [255, 69, 0], [148, 0, 211], [255, 192, 203],
            [255, 127, 0], [0, 191, 255], [255, 0, 255],
        ]
    )
    return py_colors / 255


def draw_optied_traj_2d(ax, obstacles, agents, agents_traj_list, task_area, exit_area):
    plt_colors = get_colors()
    for idx, agent_traj in enumerate(agents_traj_list):
        agent_goal = agents[idx].goal_global_frame
        agent_r = agents[idx].radius
        ag_init_pos = agents[idx].initial_pos[:2]
        color_ind = idx % len(plt_colors)
        plt_color = plt_colors[color_ind]

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        spd = agent_traj['spd']
        heading = agents[idx].heading_global_frame
        ag_pos = agents[idx].pos_global_frame

        min_xt, max_xt, min_yt, max_yt = get_boundaries(task_area)
        width_t = max_xt - min_xt
        height_t = max_yt - min_yt
        rect = patches.Rectangle((min_xt, min_yt), width_t, height_t,
                                 linewidth=2, edgecolor='blue', facecolor='none', alpha=1)
        ax.add_patch(rect)

        if len(exit_area) > 0:
            min_xe, max_xe, min_ye, max_ye = get_boundaries(exit_area)
            width = max_xe - min_xe
            height = max_ye - min_ye
            rect = patches.Rectangle((min_xe, min_ye), width, height,
                                     linewidth=2, edgecolor='green', facecolor='none', alpha=1, zorder=3)
            ax.add_patch(rect)

        plt.plot(agent_goal[0], agent_goal[1], color='green', marker='*', markersize=5)

        plt.text(pos_x[-1] - 8, pos_y[-1] + 8, str(agents[idx].id), fontsize=8, color=plt_color)

        ax.add_patch(plt.Circle((ag_init_pos[0], ag_init_pos[1]), radius=agent_r, fc='none', ec=plt_color, alpha=1))
        ax.add_patch(plt.Circle((pos_x[-1], pos_y[-1]), radius=agent_r, fc=plt_color, ec=plt_color, alpha=1))
        plt.text(pos_x[-1] - 0.25, pos_y[-1] + 0.2, 'agent' + str(agents[idx].id), color=plt_color)

        plt.arrow(ag_pos[0], ag_pos[1], 0.35*cos(heading), 0.35*sin(heading),
                  fc=plt_color, ec=plt_color, head_width=agent_r, head_length=agent_r)

        plt.plot(pos_x[:], pos_y[:], color=plt_color, linewidth=0.5, alpha=1)
        if len(pos_x) < 200:
            ax.scatter(pos_x[:], pos_y[:], marker='o', color=plt_color, s=12, zorder=3)

    for i in range(len(obstacles)):
        pos = obstacles[i].pos_global_frame
        rd = obstacles[i].radius
        if not obstacles[i].is_sl:
            if obstacles[i].shape == 'circle':
                ax.add_patch(plt.Circle((pos[0], pos[1]), radius=rd, fc='grey', ec='grey', alpha=0.7))
                # plt.plot(pos[0], pos[1], color='black', marker='o', markersize=2)
            elif obstacles[i].shape == 'polygon':
                vertices = copy.deepcopy(obstacles[i].vertices_pos)
                draw_polygon_2d(ax, vertices, color='grey')
                # ax.add_patch(plt.Circle((pos[0], pos[1]), radius=rd, fc='red', ec='red', alpha=0.5, linestyle='dotted'))
            else:
                # ax.add_patch(plt.Circle((pos[0], pos[1]), radius=rd, fc='grey', ec='grey', alpha=0.7))
                heading = 0.0
                agent_rd = obstacles[i].width/2
                my_model = get_2d_car_model(size=agent_rd)
                draw_agent_2d(ax, pos, heading, my_model)
        else:  # target
            ax.add_patch(plt.Circle((pos[0], pos[1]), radius=rd, fc='red', ec='red', linewidth=1, alpha=0.6))


def plt_visulazation(wrts_path_smooth, wrts, targets, anti_targets, task_area, exit_area):
    trajs = []
    for wrt_path in wrts_path_smooth:
        traj = {'pos_x': [], 'pos_y': [], 'is_sl': [], 'spd': []}
        for ii in range(len(wrt_path)):
            traj['pos_x'].append(wrt_path[ii][0])
            traj['pos_y'].append(wrt_path[ii][1])
            traj['is_sl'].append(wrt_path[ii][2])
            a = len(wrt_path[ii])
            if len(wrt_path[ii]) > 3:
                traj['spd'].append(wrt_path[ii][3])
        trajs.append(traj)

    fig = plt.figure(0)
    fig_size = (10 * 1, 8 * 1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='X',
           ylabel='Y',
           )
    ax.axis('equal')
    draw_optied_traj_2d(ax, targets + anti_targets, wrts, trajs, task_area, exit_area)
    # plt.axis('off')
    plt.show()
