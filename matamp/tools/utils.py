import time
import random
import numpy as np
from math import sqrt, atan2, acos, asin
from pyproj import Transformer

trans2utm = Transformer.from_crs("EPSG:4326", "EPSG:3857")
trans2lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326")
eps = 1e5


# Converting from latitude and longitude to UTM coordinate system,
# with units in meters, where the x-axis points east and the y-axis points north.
def lonlat2grid(lonlat_pos):
    x, y = trans2utm.transform(lonlat_pos[1], lonlat_pos[0])
    return [round(x, 3), round(y, 3)]


def grid2lonlat(utm_pos):
    lat, lon = trans2lonlat.transform(utm_pos[0], utm_pos[1])
    return [round(lon, 8), round(lat, 8)]


# The four boundary lines of the rectangular area.
def get_boundaries(area):
    min_x = 999999999999.
    max_x = -999999999999.
    min_y = 999999999999.
    max_y = -999999999999.
    for k in range(len(area) - 1):
        min_x = min(min_x, area[k][0])
        min_y = min(min_y, area[k][1])
        max_x = max(max_x, area[k][0])
        max_y = max(max_y, area[k][1])
    return min_x, max_x, min_y, max_y


def takeSecond(elem):
    return elem[1]


def pedal(p1, p2, p3):
    """
    Calculate the pedal coordinates of the line connecting p1 and p2 at the point p3
    Line 1: Pedal coordinates and line connecting p3
    Line 2: Line connecting p1 and p2
    The two lines are perpendicular and their intersection is the pedal point
    :param p1: (x1, y1)
    :param p2: (x2, y2)
    :param p3: (x3, y3)
    :return: Pedal coordinates (x, y)
    """
    if p2[0] != p1[0]:
        # ########## Calculate the linear equation's k, b based on points x1 and x2
        k, b = np.linalg.solve([[p1[0], 1], [p2[0], 1]], [p1[1], p2[1]])  # Obtain k and b
        # #######Principle: The scalar product of perpendicular vectors is 0
        x = np.divide(((p2[0] - p1[0]) * p3[0] + (p2[1] - p1[1]) * p3[1] - b * (p2[1] - p1[1])),
                      (p2[0] - p1[0] + k * (p2[1] - p1[1])))
        y = k * x + b

    else:  # When the line connecting points p1 and p2 is perpendicular to the x-axis
        x = p1[0]
        y = p3[1]

    return np.array([x, y])


# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def left_of(a, b, c):
    return det(a - c, b - a)        #


def l2norm(p1, p2):
    """ Compute Euclidean distance in 2D domains"""
    return round(sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 5)


def l2normsq(x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2


def sqr(a):
    return a ** 2


def unit_normal_vector(p1p2):
    """ Compute the unit normal vector of  vector p1p2"""
    nRight = normalize(np.array([p1p2[1], -p1p2[0]]))
    nLeft = -nRight
    return nLeft, nRight


def linear_equation(p1, p2):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p1[0] * (p1[1] - p2[1]) + p1[1] * (p2[0] - p1[0])
    return a, b, c


def norm(vec):
    return round(np.linalg.norm(vec), 5)


def normalize(vec):
    if np.linalg.norm(vec) > 0:
        return vec / np.linalg.norm(vec)
    else:
        return np.array([0.0, 0.0])


def absSq(vec):
    return np.dot(vec, vec)


def det(p, q):
    return p[0] * q[1] - p[1] * q[0]


def pi_2_pi(angle):  # to -pi-pi
    return round((angle + np.pi) % (2 * np.pi) - np.pi, 5)


def mod2pi(theta):  # to 0-2*pi
    return round(theta - 2.0 * np.pi * np.floor(theta / 2.0 / np.pi), 5)


def is_parallel(vec1, vec2):
    """ Determine whether two vectors are parallel """
    assert vec1.shape == vec2.shape, r'The parameter "shape" must be the same.'
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    vec1_normalized = vec1 / norm_vec1
    vec2_normalized = vec2 / norm_vec2
    if norm_vec1 <= 1e-3 or norm_vec2 <= 1e-3:
        return True
    elif 1.0 - abs(np.dot(vec1_normalized, vec2_normalized)) < 1e-3:
        return True
    else:
        return False


def angle_2_vectors(v1, v2):
    v1v2_norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if v1v2_norm == 0.0:
        v1v2_norm = 1e-5
    cosdv = np.dot(v1, v2) / v1v2_norm
    if cosdv > 1.0:
        cosdv = 1.0
    elif cosdv < -1.0:
        cosdv = -1.0
    else:
        cosdv = cosdv
    angle = acos(cosdv)
    return angle


# Check if a point is inside a circle.
def point_in_circle(pos, obj_pos, combinedRadius):
    if l2norm(pos, obj_pos) < combinedRadius:
        return True
    return False


# Determine if a line segment intersects with a circle.
def seg_cross_circle(p_1, p_2, obj_pos, combinedRadius):
    p1_in_circle = point_in_circle(p_1, obj_pos, combinedRadius)
    p2_in_circle = point_in_circle(p_2, obj_pos, combinedRadius)
    if p1_in_circle and p2_in_circle:
        return False
    elif (p1_in_circle and not p2_in_circle) or (not p1_in_circle and p2_in_circle):
        return True
    if p_1[0] == p_2[0]:
        a, b, c = 1, 0, -p_1[0]
    elif p_1[1] == p_2[1]:
        a, b, c = 0, 1, -p_1[1]
    else:
        a = p_1[1] - p_2[1]
        b = p_2[0] - p_1[0]
        c = p_1[0] * p_2[1] - p_1[1] * p_2[0]
    dist_1 = (a * obj_pos[0] + b * obj_pos[1] + c)**2
    dist_2 = (a * a + b * b) * combinedRadius * combinedRadius
    if dist_1 > dist_2:     # The distance from the point to the line is greater than the radius "r," indicating no intersection.
        return False
    angle_1 = (obj_pos[0] - p_1[0]) * (p_2[0] - p_1[0]) + (obj_pos[1] - p_1[1]) * (p_2[1] - p_1[1])
    angle_2 = (obj_pos[0] - p_2[0]) * (p_1[0] - p_2[0]) + (obj_pos[1] - p_2[1]) * (p_1[1] - p_2[1])
    if angle_1 > 0 and angle_2 > 0:
        return True
    return False


def cross(p1, p2, p3):  #
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def seg_is_intersec(p1, p2, p3, p4):
    """Determine whether the line segment p1p2 intersects with the line segment p3p4
    """
    # Quick rejection, the rectangle formed by l1 and l2's diagonals
    # must intersect, otherwise the two line segments do not intersect
    if (max(p1[0], p2[0]) >= min(p3[0], p4[0])              # The rightmost point of rectangle 1 is greater than the leftmost point of rectangle 2
            and max(p3[0], p4[0]) >= min(p1[0], p2[0])      # The rightmost point of rectangle 2 is greater than the leftmost point of rectangle 1
            and max(p1[1], p2[1]) >= min(p3[1], p4[1])      # The highest point of rectangle 1 is greater than the lowest point of rectangle 2
            and max(p3[1], p4[1]) >= min(p1[1], p2[1])):    # The highest point of rectangle 2 is greater than the lowest point of rectangle 1

        # If the quick rejection test passes, perform the crossing experiment
        if (cross(p1, p2, p3) * cross(p1, p2, p4) <= 0
                and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0):
            D = 1
        else:
            D = 0
    else:
        D = 0
    return bool(D)


def point_line_dist(p, pos, forward_pos):
    a, b, c = linear_equation(pos, forward_pos)
    return abs(a * p[0] + b * p[1] + c) / np.sqrt(a ** 2 + b ** 2)


def dist_sq_point_line_segment(vector1, vector2, vector3):
    """
    Computes the squared distance from a line segment with the specified endpoints to a specified point.

    Args:
        vector1 (Vector2): The first endpoint of the line segment.
        vector2 (Vector2): The second endpoint of the line segment.
        vector3 (Vector2): The point to which the squared distance is to be calculated.

    Returns:
        float: The squared distance from the line segment to the point.
    """
    r = np.dot((vector3 - vector1), (vector2 - vector1)) / absSq(vector2 - vector1)

    if r < 0.0:
        return absSq(vector3 - vector1)

    if r > 1.0:
        return absSq(vector3 - vector2)

    return absSq(vector3 - (vector1 + r * (vector2 - vector1)))


def scal(data, sec_dis):
    """
    Args:
        data: The set of points describing the polygon arranged in counterclockwise order
        sec_dis: Scaling distance

    Returns:
        The set of points describing the scaled polygon
    """
    num = len(data)
    scal_data = []
    for k in range(num):
        x1 = data[k % num][0] - data[(k - 1) % num][0]
        y1 = data[k % num][1] - data[(k - 1) % num][1]
        x2 = data[(k + 1) % num][0] - data[k % num][0]
        y2 = data[(k + 1) % num][1] - data[k % num][1]

        d_A = (x1 ** 2 + y1 ** 2) ** 0.5
        d_B = (x2 ** 2 + y2 ** 2) ** 0.5

        Vec_Cross = (x1 * y2) - (x2 * y1)

        sin_theta = Vec_Cross / (d_A * d_B)

        dv = sec_dis / sin_theta

        v1_x = (dv / d_A) * x1
        v1_y = (dv / d_A) * y1

        v2_x = (dv / d_B) * x2
        v2_y = (dv / d_B) * y2

        PQ_x = v1_x - v2_x
        PQ_y = v1_y - v2_y

        Q_x = data[k % num][0] + PQ_x
        Q_y = data[k % num][1] + PQ_y
        scal_data.append([Q_x, Q_y])
    return scal_data


def gen_polygonal_vertices(center, radius, num_vertice):
    regular_polygon = True if num_vertice < 6 else False  # If the number of sides of the polygon is less than 6, then it is considered a regular polygon; otherwise, it is treated as an irregular polygon.
    if regular_polygon:
        vertices = []
        s_theta = 360 / num_vertice / 2
        for theta in [np.deg2rad(i * (360 / num_vertice)+s_theta) for i in range(num_vertice)]:
            vertice = center + radius * np.array([np.cos(theta), np.sin(theta)])
            vertices.append(vertice)
    else:
        points_num = int(2 * np.pi * radius / 20.0)
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as patches

    fig = plt.figure(0)
    fig_size = (10, 8)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)

    center = np.array([10.0, -10.0])
    radius = 50.0
    num_vertice = 3
    vertices = gen_polygonal_vertices(center, radius, num_vertice)
    vertices.append(vertices[0])

    print(len(vertices), vertices)
    ax.add_patch(plt.Circle((center[0], center[1]), radius=radius, fc='blue', ec='blue', alpha=0.3))
    for i in range(num_vertice):
        plt.plot(vertices[i][0], vertices[i][1], color='orange', marker='o', markersize=10)
    plt.plot(vertices[0][0], vertices[0][1], color='red', marker='o', markersize=10)
    plt.plot(vertices[1][0], vertices[1][1], color='red', marker='s', markersize=10)

    codes = []
    for i in range(len(vertices)):
        if i == 0:
            codes.append(Path.MOVETO)
        elif i == len(vertices) - 1:
            codes.append(Path.CLOSEPOLY)
        else:
            codes.append(Path.LINETO)

    path = Path(vertices, codes)
    # Step 2: Create a patch. The path is still implemented through the patch, but is now referred to as a "path patch."
    col = [0.8, 0.8, 0.8]
    patch = patches.PathPatch(path, fc=col, ec='black', lw=1.5)

    ax.add_patch(patch)

    plt.axis("equal")
    plt.show()



