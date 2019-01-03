import random

import numpy as np

from psokeras.optimizer import BIG_SCORE


class Particle:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.init_weights = model.get_weights()
        self.velocities = [None] * len(self.init_weights)
        self.length = len(self.init_weights)
        for i, layer in enumerate(self.init_weights):
            self.velocities[i] = np.random.rand(*layer.shape) / 5 - 0.10
            # self.velocities[i] = np.zeros(layer.shape)

        self.best_weights = None
        self.best_score = BIG_SCORE

    def get_score(self, x, y, update=True):
        local_score = self.model.evaluate(x, y, verbose=0)
        if local_score < self.best_score and update:
            self.best_score = local_score
            self.best_weights = self.model.get_weights()

        return local_score

    def _update_velocities(self, global_best_weights, depth):
        new_velocities = [None] * len(self.init_weights)
        weights = self.model.get_weights()
        local_rand, global_rand = random.random(), random.random()

        for i, layer in enumerate(weights):
            if i >= depth:
                new_velocities[i] = self.velocities[i]
                continue
            new_v = self.params['acc'] * self.velocities[i]
            new_v = new_v + self.params['local_acc'] * local_rand * (self.best_weights[i] - layer)
            new_v = new_v + self.params['global_acc'] * global_rand * (global_best_weights[i] - layer)
            new_velocities[i] = new_v

        self.velocities = new_velocities

    def _update_weights(self, depth):
        old_weights = self.model.get_weights()
        new_weights = [None] * len(old_weights)
        for i, layer in enumerate(old_weights):
            if i>= depth:
                new_weights[i] = layer
                continue
            new_w = layer + self.velocities[i]
            new_weights[i] = new_w

        self.model.set_weights(new_weights)

    def step(self, x, y, global_best_weights,depth=None):
        if depth is None:
            depth = self.length
        self._update_velocities(global_best_weights, depth)
        self._update_weights(depth)
        return self.get_score(x, y)

    def get_best_weights(self):
        return self.best_weights
