import numpy as np
import matplotlib.pyplot as plt
from csci3202.classifiers.cnn import *
from csci3202.data_utils import get_CIFAR10_data
from csci3202.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from csci3202.layers import *
from csci3202.fast_layers import *
from csci3202.solver import Solver

data = get_CIFAR10_data()
model = ThreeLayerConvNet(filter_size=5, weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()