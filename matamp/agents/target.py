import numpy as np


class Target(object):
    def __init__(self, pos, shape_dict, id, tid, is_sl=True):
        self.shape = shape = shape_dict['shape']
        self.feature = feature = shape_dict['feature']
        if shape == 'rect':
            self.width, self.heigh, self.rect_pos = shape_dict['feature']
            self.radius = np.sqrt(self.width ** 2 + self.heigh ** 2) / 2
        elif shape == 'circle':
            self.radius = shape_dict['feature']
        else:
            raise NotImplementedError
        self.pos_global_frame = np.array(pos[:2], dtype='float64')
        self.goal_global_frame = np.array(pos[:2], dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0])
        self.pos = pos[:2]
        self.id = id
        self.tid = {self.id: tid}
        self.t = 0.0
        self.step_num = 0
        self.is_close = False
        self.is_at_goal = True
        self.is_agent = False
        self.is_obstacle = False
        self.was_in_collision_already = False
        self.is_collision = False
        self.is_poly = False
        self.is_expansion = False

        self.is_sl = is_sl
        self.is_cleared = False
        self.x = pos[0]
        self.y = pos[1]
        self.r = shape_dict['feature']
