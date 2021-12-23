import torch
import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    def __init__(self, rows, cols, init_position=None):
        self.num_rows = rows
        self.num_cols = cols
        if init_position is None:
            initial_row = np.random.randint(0, rows)
            initial_col = np.random.randint(0, cols)
        else:
            initial_row = init_position[0]
            initial_col = init_position[1]

        self.position = (initial_row, initial_col)
        self.orientation = np.random.choice([0, 90, 180, 270])

    def reset(self):
        initial_row = np.random.randint(0, self.num_rows)
        initial_col = np.random.randint(0, self.num_cols)
        self.position = (initial_row, initial_col)
        self.orientation = np.random.choice([0, 90, 180, 270])

    def render(self):
        fig, ax = plt.subplots()
        ax.scatter(self.position[0], self.position[1], marker=(3, 0, self.orientation), color='black')
        ax.set_xticks(np.arange(self.num_rows + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.num_cols + 1) - 0.5, minor=True)
        ax.grid(which="minor")
        ax.tick_params(which="minor", size=0)
        plt.show()

    def get_observation(self):
        grid = np.ones((self.num_cols+4, self.num_rows+4))
        grid[2:-2,2:-2] = np.zeros((self.num_cols, self.num_rows))
        obs = grid[self.position[1]:2 + self.position[1] + 3, self.position[0] :self.position[0] + 2 + 3][::-1, :][np.newaxis, : , :]
        return obs, self.position, self.orientation

    def go_forward(self):
        if self.orientation == 0:
            self.position = (self.position[0], np.minimum(self.num_rows-1, self.position[1] + 1))
        if self.orientation == 90:
            self.position = (np.maximum(self.position[0]-1,0), self.position[1])
        if self.orientation == 180:
            self.position = (self.position[0], np.maximum(0,self.position[1]-1))
        if self.orientation == 270:
            self.position = (np.minimum(self.num_rows-1,self.position[0]+1), self.position[1])

    def go_backward(self):
        self.rotate_left(2)
        self.go_forward()
        self.rotate_left(2)

    def rotate_right(self, num_rotations=1):
        for j in range(num_rotations):
            self.orientation -= 90
        self.orientation = self.orientation % 360

    def rotate_left(self, num_rotations=1):
        for j in range(num_rotations):
            self.orientation += 90
        self.orientation = self.orientation % 360

    def step(self, action):
        if action == 0:
            self.go_forward()
        if action == 1:
            self.go_backward()
        if action == 2:
            self.rotate_left()
        if action == 3:
            self.rotate_right()

class GridWorldAgent(object):
    def __init__(self, env):
        self.env = env
        self.num_action_performed =0
        self.one_hot_action = np.zeros((1,4), dtype=np.float)

    def step(self):
        action = np.random.choice([0,1,2,3], p=[0.6, 0.0, 0.2, 0.2])
        one_hot_action = np.zeros((1, 4), dtype=np.float)
        one_hot_action[0, action] = 1.0
        return one_hot_action



