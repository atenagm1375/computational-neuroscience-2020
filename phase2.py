from models.Neurons import LIF
from models.Synapses import Synapse
from models.Connections import Connection
from models.Populations import Population
from simulation.Simulate import Simulate
from simulation.Monitors import raster_plot

import numpy as np

neuron = LIF()
syn = Synapse(10)
pop = Population(800, neuron)
pop.add(200, LIF(is_inh=True, r=10))
conn = Connection(pop, pop, syn)
conn.apply("fixed_pre", p=0.1)
sim = Simulate({"populations": [pop], "connections": [conn]})
sim.run(20, lambda x: np.random.uniform(7, 13) if 2 < x < 18 else 0)
colors = ['g'] * 800 + ['r'] * 200
raster_plot(pop.spikes_per_neuron, colors=colors)
