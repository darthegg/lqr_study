from time import time

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.integrate import solve_ivp

from dynamics_model import calculate_dynamics


class Simulator:
    def __init__(self, solver, initial, final, dt, total_frames):
        self.solver = solver
        self.final = final
        self.dt = dt
        self.total_frames = total_frames

        # Set plot and axis
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(
            111,
            aspect = 'equal',
            xlim = (-5, 5),
            ylim = (-1, 1),
            title = "Inverted Pendulum Simulation",
        )
        self.axis.grid()

        # Draw initial pendulum and cart
        self.origin = [0.0, 0.0]
        self.pendulumArm = lines.Line2D(self.origin, self.origin, color='r') 
        self.cart = patches.Rectangle(self.origin, 0.5, 0.15, color='b')
        
        # Initialize and calculate every frames
        end_time = total_frames * dt
        time_steps = np.linspace(0.0, end_time, total_frames)
        self.animation_frame_states = solve_ivp(
            calculate_dynamics,
            t_span = [0.0, end_time],
            y0 = initial,
            t_eval = time_steps,
            rtol=1e-8,
        )

    def _animate(self, i):
        xPos = self.animation_frame_states.y[0][i] 
        theta = self.animation_frame_states.y[1][i]
        x = [self.origin[0] + xPos, self.origin[0] + xPos + self.solver.L * np.sin(theta)]
        y = [self.origin[1], self.origin[1] - self.solver.L * np.cos(theta)]
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
        t0 = time()
        self._animate(0)
        t1 = time()
        interval = 1000 * self.dt - (t1 - t0)

        _ = animation.FuncAnimation(self.fig, self._animate, init_func = self._animation_init, frames = self.total_frames, interval = interval, blit = True)

        plt.show()
