from models.Neurons import LIF, ELIF, AddaptiveELIF
from simulation.Monitors import *
from simulation.Simulate import Simulate
import numpy as np


firing_patterns = []
currents = list(np.arange(0, 20, 0.1))
for i in currents:
    lif = LIF()
    simulate = Simulate(lif, 0.001)
    potential_list, current_list, time_list = simulate.run(2, lambda x: i)
    firing_patterns.append(lif.spike_times)
    # plot_firing_pattern(potential_list, current_list,
    #                     time_list, lif.u_rest, lif.threshold)
plot_f_i_curve(firing_patterns, 2, currents)
