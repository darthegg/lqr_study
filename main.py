from time import time

import numpy as np

import lqr_solver
from simulator import Simulator


initial=np.array([-1, np.pi+0.5, 0, 0])      
final=np.array([2, np.pi, 0, 0])
dt = 0.02
total_frames = 200

solver = lqr_solver
sim = Simulator(solver, initial, final, dt, total_frames)

sim.run_sim()
