import numpy as np

import dynamics_model
import lqr_solver
from simulator import Simulator


initial=np.array([-1, np.pi+0.5, 0, 0])      
final=np.array([2, np.pi, 0, 0])
dt = 0.02
total_frames = 200

sim_args = (initial, final, dt, total_frames)

solver = lqr_solver
model = dynamics_model
sim = Simulator(model, solver, sim_args)

sim.run_sim()
