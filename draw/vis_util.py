import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2, sqrt, pow

plt_colors = [[0.8500, 0.3250, 0.0980], [0.0, 0.4470, 0.7410], [0.4660, 0.8740, 0.1880],
              [0.4940, 0.1840, 0.5560],
              [0.9290, 0.6940, 0.1250], [0.3010, 0.7450, 0.9330], [0.6350, 0.0780, 0.1840]]


# plt_colors = ['red', 'green', 'blue', 'orange', 'yellow', 'purple']


def get_2d_car_model(size):
    verts = [
        [-1. * size, 1. * size],  # Coordinates of the bottom-left corner of the rectangle (left,bottom)
        [-1. * size, -1. * size],  # Coordinates of the top-left corner of the rectangle (left,top)
        [1. * size, -1. * size],  # Coordinates of the top-right corner of the rectangle (right,top)
        [1. * size, 1. * size],  # Coordinates of the bottom-right corner of the rectangle (right, bottom)
        [-1. * size, 1. * size],  # Close to the starting point
    ]
    return verts


def get_2d_uav_model0(size):
    # size /= 2
    verts = [
        [0., 1. * size],  # Coordinates of the bottom-left corner of the rectangle (left, bottom)
        [-0.5 * size, -0.8 * size],  # Coordinates of the top-left corner of the rectangle (left, top)
        [0, -0.2 * size],  # Coordinates of the top-right corner of the rectangle (right, top)
        [0.5 * size, -0.8 * size],  # Coordinates of the bottom-right corner of the rectangle (right, bottom)
        [0., 1. * size],  # Close to the starting point
    ]
    return verts


def get_2d_uav_model(size):
    # size /= 2
    verts = [
        [0., 1. * size],  # Coordinates of the bottom-left corner of the rectangle (left, bottom)
        [-1. * size, -1. * size],  # Coordinates of the top-left corner of the rectangle (left, top)
        [0., -0.5 * size],  # Coordinates of the top-right corner of the rectangle (right, top)
        [1. * size, -1. * size],  # Coordinates of the bottom-right corner of the rectangle (right, bottom)
        [0., 1. * size],  # Close to the starting point
    ]
    return verts


def rgba2rgb(rgba):
    # rgba is a list of 4 color elements btwn [0.0, 1.0]
    # or a 2d np array (num_colors, 4)
    # returns a list of rgb values between [0.0, 1.0] accounting for alpha and background color [1, 1, 1] == WHITE
    if isinstance(rgba, list):
        alpha = rgba[3]
        r = max(min((1 - alpha) * 1.0 + alpha * rgba[0], 1.0), 0.0)
        g = max(min((1 - alpha) * 1.0 + alpha * rgba[1], 1.0), 0.0)
        b = max(min((1 - alpha) * 1.0 + alpha * rgba[2], 1.0), 0.0)
        return [r, g, b]
    elif rgba.ndim == 2:
        alphas = rgba[:, 3]
        r = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 0], 0, 1)
        g = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 1], 0, 1)
        b = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 2], 0, 1)
        return np.vstack([r, g, b]).T


def draw_env(ax, buildings_obj_list, keypoints_obj_list, connection_matrix):
    pass


def draw(origin_pos, objects_list, buildings_obj_list, keypoints_obj_list, connection_matrix):
    fig = plt.figure()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim([-500.0, 3500.0])
    plt.ylim([-1000.0, 1000.0])
    ax = fig.add_subplot(1, 1, 1)

    ax.set_aspect('equal')

    draw_objects(ax, objects_list)
    draw_buildings(ax, origin_pos, buildings_obj_list)
    draw_roads(ax, keypoints_obj_list, connection_matrix)
    plt.show()


def draw_buildings(ax, origin_pos, buildings_obj):
    for building_obj in buildings_obj:
        id = building_obj.id
        pos = building_obj.pos
        draw_rectangle(ax, origin_pos, pos)


def draw_roads(ax, keypoints_obj_list, connection_matrix):
    for index_i, info in enumerate(connection_matrix):
        self_pos = keypoints_obj_list[index_i].pos
        for index_j, distance in enumerate(info):
            if distance > 0:
                target_pos = keypoints_obj_list[index_j].pos
                x = [self_pos[0], target_pos[0]]
                y = [self_pos[1], target_pos[1]]
                plt.plot(x, y, color='r')
                plt.scatter(x, y, color='b')


def draw_objects(ax, objects_obj):
    for object_obj in objects_obj:
        pass


def draw_rectangle(ax, origin_pos, pos):
    pos_x = pos[0]
    pos_y = pos[1]
    ax.add_patch(
        plt.Rectangle(
            (pos_x - 5, pos_y - 5),
            10,
            10,
            color='maroon',
            alpha=0.5
        ))


def convert_to_actual_model_3d(agent_model, pos_global_frame, heading_global_frame):
    alpha = heading_global_frame[0]
    beta = heading_global_frame[1]
    gamma = heading_global_frame[2]
    for point in agent_model:
        x = point[0]
        y = point[1]
        z = point[2]
        # Perform pitch calculation
        r = sqrt(pow(y, 2) + pow(z, 2))
        beta_model = atan2(z, y)
        beta_ = beta + beta_model
        y = r * cos(beta_)
        z = r * sin(beta_)
        # Perform roll calculation
        h = sqrt(pow(x, 2) + pow(z, 2))
        gama_model = atan2(z, x)
        gamma_ = gamma + gama_model
        x = h * cos(gamma_)
        z = h * sin(gamma_)
        # Perform heading calculation
        l = sqrt(pow(x, 2) + pow(y, 2))
        alpha_model = atan2(y, x)
        alpha_ = alpha + alpha_model - np.pi / 2
        point[0] = l * cos(alpha_) + pos_global_frame[0]
        point[1] = l * sin(alpha_) + pos_global_frame[1]
        point[2] = z + pos_global_frame[2]
