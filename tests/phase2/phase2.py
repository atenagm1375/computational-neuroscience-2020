from models.Neurons import *
from models.Synapses import Synapse
from models.Connections import Connection
from models.Populations import Population
from simulation.Simulate import Simulate
from simulation.Monitors import raster_plot

import numpy as np


def q1_fixed_pre():
    duration = 30
    neuron_params = {
        'r': 3,
        'tau': 10,
        'threshold': -55,
        'is_inh': True,
        'current': lambda x: np.random.uniform(5, 10) if 5 < x < duration - 5 else 0
    }
    pop = Population(1000, LIF, exc_ratio=0.8, **neuron_params)
    conn = Connection(pop, pop).apply("fixed_pre", p=0.005, mu=1, sigma=0.01)
    sim = Simulate(populations=[pop], connections=[conn], time_step=0.1)
    sim.run(duration)
    print(pop.spikes_per_neuron.shape)
    raster_plot(pop.spikes_per_neuron)


def q1_full():
    duration = 30
    neuron_params = {
        'r': 3,
        'tau': 10,
        'threshold': -55,
        'is_inh': True,
        'current': lambda x: np.random.uniform(5, 10) if 5 < x < duration - 5 else 0
    }
    pop = Population(1000, LIF, exc_ratio=0.8, **neuron_params)
    conn = Connection(pop, pop).apply("full")
    sim = Simulate(populations=[pop], connections=[conn], time_step=0.1)
    sim.run(duration)
    print(pop.spikes_per_neuron.shape)
    raster_plot(pop.spikes_per_neuron)


def q2_fixed_pre():
    pass


def q2_full():
    pass
