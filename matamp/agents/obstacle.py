import numpy as np
from matamp.tools.utils import normalize, left_of


class Vertice(object):
    """
    Defines vertice of polygonal static obstacles in the simulation.
    """

    def __init__(self):
        self.next_ = None
        self.previous_ = None
        self.direction_ = None
        self.point_ = None
        self.id_ = -1
        self.convex_ = False


class Obstacle(object):
    def __init__(self, pos, shape_dict, id, tid, vertices=None, is_poly=False):
        self.shape = shape_dict['shape']
        self.radius = shape_dict['feature']
        self.feature = shape_dict['feature']
        if self.shape == 'rect':
            self.width, self.heigh = shape_dict['feature']
            self.radius = np.sqrt(self.width ** 2 + self.heigh ** 2) / 2
        self.pos_global_frame = np.array(pos[:2], dtype='float64')
        self.goal_global_frame = np.array(pos[:2], dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0])
        self.pos = pos[:2]
        self.id = id
        self.tid = {self.id: tid}
        self.t = 0.0
        self.step_num = 0
        self.is_at_goal = True
        self.is_agent = False
        self.is_obstacle = True
        self.is_sl = False
        self.was_in_collision_already = False
        self.is_collision = False

        self.x = pos[0]
        self.y = pos[1]
        self.r = self.radius

        self.is_poly = is_poly
        self.vertices_pos = []
        self.vertices_ = []
        if is_poly and vertices is not None:
            self.connect_vertices(vertices)
            for pos in vertices:
                self.vertices_pos.append(list(pos))

    def connect_vertices(self, vertices):
        """
        build a new polygonal obstacle to the simulation.

        Args:
            vertices (list): List of the vertices of the polygonal obstacle in counterclockwise order.

        Returns:
            int: The number of the first vertex of the obstacle, or -1 when the number of vertices is less than two.

        Remarks:
            To add a "negative" obstacle, e.g. a bounding polygon around the environment, the vertices should be
            listed in clockwise order.
        """
        if len(vertices) < 2:
            raise Exception('Must have at least 2 vertices.')

        # obstacleNo = len(self.obstacles_)

        for i in range(len(vertices)):
            vertice = Vertice()
            vertice.point_ = vertices[i]

            if i != 0:
                vertice.previous_ = self.vertices_[len(self.vertices_) - 1]
                vertice.previous_.next_ = vertice

            if i == len(vertices) - 1:
                vertice.next_ = self.vertices_[0]
                vertice.next_.previous_ = vertice

            vertice.direction_ = normalize(vertices[0 if i == len(vertices) - 1 else i + 1] - vertices[i])

            if len(vertices) == 2:
                vertice.convex_ = True
            else:
                vertice.convex_ = left_of(
                    vertices[len(vertices) - 1 if i == 0 else i - 1],
                    vertices[i],
                    vertices[0 if i == len(vertices) - 1 else i + 1]) >= 0.0

            vertice.id_ = i
            self.vertices_.append(vertice)

