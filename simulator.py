import time

import matplotlib.animation as animation
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class Simulator:
    def __init__(self, model, solver, sim_args):
        self.model = model
        self.solver = solver
        self.initial = sim_args[0]
        self.final = sim_args[1]
        self.dt = sim_args[2]
        self.total_frames = sim_args[3]

        # Set plot and axis
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111, aspect = 'equal', xlim = (-5, 5), ylim = (-1, 1))
        self.axis.grid()

        # Draw initial pendulum and cart
        self.origin = [0.0, 0.0]
        self.pendulumArm = lines.Line2D(self.origin, self.origin, color='r')
        self.cart = patches.Rectangle(self.origin, 0.5, 0.15, color='b')

    def _animate(self, i):
        xPos = self.animation_frame_states.y[0][i]
        theta = self.animation_frame_states.y[1][i]
        x = [self.origin[0] + xPos, self.origin[0] + xPos + self.model.L * np.sin(theta)]
        y = [self.origin[1], self.origin[1] - self.model.L * np.cos(theta)]
        self.pendulumArm.set_xdata(x)
        self.pendulumArm.set_ydata(y)
        cartPos = [self.origin[0] + xPos - self.cart.get_width()/2, self.origin[1] - self.cart.get_height()]
        self.cart.set_xy(cartPos)

        return self.pendulumArm, self.cart

    def _animation_init(self):
        self.axis.add_patch(self.cart)
        self.axis.add_line(self.pendulumArm)

        return self.pendulumArm, self.cart
    
    def run_sim(self):
        # Calculate every state frames
        end_time = self.total_frames * self.dt
        time_steps = np.linspace(0.0, end_time, self.total_frames)
        self.animation_frame_states = solve_ivp(
            lambda t, state: self.model.state_space_dynamics(t, state, self.solver.calculate_input(state, self.final)),
            t_span = [0.0, end_time],
            y0 = self.initial,
            t_eval = time_steps,
            rtol = 1e-8,
        )

        tick = time.time()
        self._animate(0)
        tock = time.time()
        interval = 1000 * self.dt - (tick - tock)

        _ = animation.FuncAnimation(
            self.fig,
            self._animate,
            init_func = self._animation_init,
            frames = self.total_frames,
            interval = interval,
            blit = True
        )

        plt.show()
