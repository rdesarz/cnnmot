import numpy as np


def dynamic_model(delta_t):
    return np.array([[delta_t, 0],
                     [0,       delta_t]])


def prediction_step(state_vector, state_transition_model, delta_t):
    return state_transition_model(delta_t).dot(state_vector)
