# -*- coding: utf-8 -*-
"""PSOkeras - Particle Swarm Optimizer for Keras models

This module implements a particle swarm optimizer for training the weights of Keras models.  The

"""


from .version import __version__
from .optimizer import Optimizer

__all__ = [
    'Optimizer',
]
