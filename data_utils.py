import torch


class SubTrajectory(object):
    def __init__(self, size):
        self.obs = None
        self.pos = None
        self.ori = None
        self.action = None
        self.new_obs = None
        self.new_pos = None
        self.new_ori = None
        self.belief = None
        self.maxSize = size
        self.len = 0



class TrajectoryBuffer(object):
    def __init__(self, size, trajectory_len, obs_dim, pos_dim, ori_dim, action_dim=4):
        self.obs = torch.zeros(size=(size, trajectory_len, *obs_dim), dtype=torch.float32)
        self.pos = torch.zeros(size=(size, trajectory_len, *pos_dim), dtype=torch.float32)
        self.ori = torch.zeros(size=(size, trajectory_len, *ori_dim), dtype=torch.float32)
        self.action = torch.zeros((size, trajectory_len, *action_dim), dtype=torch.float32)
        self.new_obs = torch.zeros((size, trajectory_len, *obs_dim), dtype=torch.float32)
        self.new_pos = torch.zeros((size, trajectory_len, *pos_dim), dtype=torch.float32)
        self.new_ori = torch.zeros((size, trajectory_len, *ori_dim), dtype=torch.float32)
        self.maxSize = size
        self.len = 0

    def add_trajectory(self, trajectory):
        self.obs[self.len,:] = trajectory.obs
        self.pos[self.len, :] = trajectory.orig_position
        self.ori[self.len, :] = trajectory.orig_orientation
        self.new_obs[self.len, :] = trajectory.new_obs
        self.new_pos[self.len, :] = trajectory.position_list
        self.new_ori[self.len, :] = trajectory.orientation_list
        self.action[self.len, :] = trajectory.action
        self.len += 1

    def clear(self):
        self.len = 0