import numpy as np


m1 = 0.3
m2 = 0.2
g = 9.8
L = 0.5
l = L/2             # CoM of uniform rod                 
I = (1./12) * m2 * (L**2)
b = 1           # dampending coefficient

state_space = np.array([0., 0., 0., 0.])  # state derivatives


def state_space_dynamics(t, state, u):
    # nonlinear system dynamics
    S = np.sin(state[1])
    C = np.cos(state[1])
    denominator = (m2*l*C)**2 - (m1+m2) * (m2*(l**2) + I)
    state_space[0] = state[2]
    state_space[1] = state[3]
    state_space[2] = (1/denominator)*(-(m2**2)*g*(l**2)*C*S - m2*l*S*(m2*(l**2)+I)*(state[3]**2) + b*(m2*(l**2) + I)*state[2]) - ((m2*(l**2)+I)/denominator)*u
    state_space[3] = (1/denominator)*((m1+m2)*m2*g*l*S + (m2**2)*(l**2)*S*C*(state[3]**2) - b*m2*l*C*state[2]) + ((m2*l*C)/denominator)*u

    return state_space