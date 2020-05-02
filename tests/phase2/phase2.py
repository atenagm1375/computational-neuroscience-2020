from models.Neurons import *
from models.Synapses import Synapse
from models.Connections import Connection
from models.Populations import Population
from simulation.Simulate import Simulate
from simulation.Monitors import raster_plot, decision_plot

import numpy as np


def current_generator(duration, dt, ranges):
    current_list = []
    for i in np.arange(0, duration, dt):
        if i < 5:
            current_list.append(0)
        elif i < duration / 2:
            current_list.append(np.random.uniform(*ranges[0]))
        elif i < duration - 5:
            current_list.append(np.random.uniform(*ranges[1]))
        else:
            current_list.append(0)
    return current_list


def aggregate_population_spikes(populations):
    spikes = []
    i = 0
    for pop in populations:
        if pop.spikes_per_neuron.shape[0] > 0:
            pop.spikes_per_neuron[:, 0] += i
            if len(spikes) == 0:
                spikes = pop.spikes_per_neuron
            else:
                spikes = np.concatenate(
                    (spikes, pop.spikes_per_neuron), axis=0)
            i += pop.size
    return np.array(spikes)


def q1_fixed_pre():
    duration = 60
    dt = 0.1
    neuron_params = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_generator(duration, dt, [(4, 5), (4, 5)])
    }
    pop = Population(1000, LIF, exc_ratio=0.8, **neuron_params)
    conn = Connection(pop, pop).apply("fixed_pre", p=0.005, mu=0.6, sigma=0.1)
    sim = Simulate(populations=[pop], connections=[conn], time_step=dt)
    sim.run(duration)
    print(pop.spikes_per_neuron.shape)
    colors = ['inh' if pop.neurons[int(p[0])].is_inh
              else 'exc' for p in pop.spikes_per_neuron]
    raster_plot(pop.spikes_per_neuron, colors=colors)


def q1_full():
    duration = 50
    dt = 0.1
    neuron_params = {
        'r': 10,
        'tau': 5,
        'threshold': -60,
        'current': current_generator(duration, dt, [(4, 5), (4, 5)])
    }
    pop = Population(1000, LIF, exc_ratio=0.8, **neuron_params)
    conn = Connection(pop, pop).apply("full")
    sim = Simulate(populations=[pop], connections=[conn], time_step=dt)
    sim.run(duration)
    print(pop.spikes_per_neuron.shape)
    colors = ['inh' if pop.neurons[int(p[0])].is_inh
              else 'exc' for p in pop.spikes_per_neuron]
    raster_plot(pop.spikes_per_neuron, colors=colors)


def q2_fixed_pre_fixed_pre():
    duration = 60
    dt = 0.1
    current_list1 = current_generator(duration, dt, [(4, 5), (1, 2)])
    current_list2 = current_generator(duration, dt, [(2, 3), (4, 5)])
    current_list3 = current_generator(duration, dt, [(0, 1), (0, 1)])
    neuron_params_exc1 = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_list1
    }
    neuron_params_exc2 = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_list2
    }
    neuron_params_inh = {
        'r': 10,
        'tau': 5,
        'threshold': -65,
        'is_inh': True,
        'current': current_list3
    }
    pop1 = Population(400, LIF, exc_ratio=1, **neuron_params_exc1)
    pop2 = Population(200, LIF, exc_ratio=0, **neuron_params_inh)
    pop3 = Population(400, LIF, exc_ratio=1, **neuron_params_exc2)

    conn1 = Connection(pop1, pop1).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn2 = Connection(pop2, pop2).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn3 = Connection(pop3, pop3).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn4 = Connection(pop1, pop2).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn5 = Connection(pop2, pop3).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn6 = Connection(pop2, pop1).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn7 = Connection(pop3, pop2).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn8 = Connection(pop3, pop1).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn9 = Connection(pop1, pop3).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)

    sim = Simulate(populations=[pop1, pop2, pop3], connections=[
                   conn1, conn2, conn3, conn4, conn5, conn6, conn7, conn8, conn9],
                   time_step=dt)
    sim.run(duration)
    print(pop1.spikes_per_neuron.shape)
    print(pop2.spikes_per_neuron.shape)
    print(pop3.spikes_per_neuron.shape)
    spikes = aggregate_population_spikes([pop1, pop2, pop3])
    print(spikes.shape)
    # raster_plot(spikes)
    decision_plot(pop1.activity, pop3.activity)


def q2_full_fixed_pre():
    duration = 60
    dt = 0.1
    current_list1 = current_generator(duration, dt, [(40, 50), (1, 2)])
    current_list2 = current_generator(duration, dt, [(1, 2), (40, 50)])
    current_list3 = current_generator(duration, dt, [(1, 1), (1, 1)])
    neuron_params_exc1 = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_list1
    }
    neuron_params_exc2 = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_list2
    }
    neuron_params_inh = {
        'r': 8,
        'tau': 5,
        'threshold': -65,
        'is_inh': True,
        'current': current_list3
    }
    pop1 = Population(400, LIF, exc_ratio=1, **neuron_params_exc1)
    pop2 = Population(200, LIF, exc_ratio=0, **neuron_params_inh)
    pop3 = Population(400, LIF, exc_ratio=1, **neuron_params_exc2)

    conn1 = Connection(pop1, pop1).apply("full")
    conn2 = Connection(pop2, pop2).apply("full")
    conn3 = Connection(pop3, pop3).apply("full")
    conn4 = Connection(pop1, pop2).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn5 = Connection(pop2, pop3).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn6 = Connection(pop2, pop1).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn7 = Connection(pop3, pop2).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn8 = Connection(pop3, pop1).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn9 = Connection(pop1, pop3).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)

    sim = Simulate(populations=[pop1, pop2, pop3], connections=[
                   conn1, conn2, conn3, conn4, conn5, conn6, conn7, conn8, conn9],
                   time_step=dt)
    sim.run(duration)
    print(pop1.spikes_per_neuron.shape)
    print(pop2.spikes_per_neuron.shape)
    print(pop3.spikes_per_neuron.shape)
    spikes = aggregate_population_spikes([pop1, pop2, pop3])
    print(spikes.shape)
    # raster_plot(spikes)
    decision_plot(pop1.activity, pop3.activity)


def q2_full_full():
    duration = 60
    dt = 0.1
    current_list1 = current_generator(duration, dt, [(20, 30), (1, 2)])
    current_list2 = current_generator(duration, dt, [(1, 2), (30, 40)])
    current_list3 = current_generator(duration, dt, [(1, 1), (1, 1)])
    neuron_params_exc1 = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_list1
    }
    neuron_params_exc2 = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_list2
    }
    neuron_params_inh = {
        'r': 8,
        'tau': 5,
        'threshold': -65,
        'is_inh': True,
        'current': current_list3
    }
    pop1 = Population(400, LIF, exc_ratio=1, **neuron_params_exc1)
    pop2 = Population(200, LIF, exc_ratio=0, **neuron_params_inh)
    pop3 = Population(400, LIF, exc_ratio=1, **neuron_params_exc2)

    conn1 = Connection(pop1, pop1).apply("full")
    conn2 = Connection(pop2, pop2).apply("full")
    conn3 = Connection(pop3, pop3).apply("full")
    conn4 = Connection(pop1, pop2).apply("full")
    conn5 = Connection(pop2, pop3).apply("full")
    conn6 = Connection(pop2, pop1).apply("full")
    conn7 = Connection(pop3, pop2).apply("full")
    conn8 = Connection(pop3, pop1).apply("full")
    conn9 = Connection(pop1, pop3).apply("full")

    sim = Simulate(populations=[pop1, pop2, pop3], connections=[
                   conn1, conn2, conn3, conn4, conn5, conn6, conn7, conn8, conn9],
                   time_step=dt)
    sim.run(duration)
    print(pop1.spikes_per_neuron.shape)
    print(pop2.spikes_per_neuron.shape)
    print(pop3.spikes_per_neuron.shape)
    spikes = aggregate_population_spikes([pop1, pop2, pop3])
    print(spikes.shape)
    decision_plot(pop1.activity, pop3.activity)


def q2_fixed_pre_full():
    duration = 60
    dt = 0.1
    current_list1 = current_generator(duration, dt, [(10, 20), (2, 3)])
    current_list2 = current_generator(duration, dt, [(1, 2), (70, 80)])
    current_list3 = current_generator(duration, dt, [(1, 1), (1, 1)])
    neuron_params_exc1 = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_list1
    }
    neuron_params_exc2 = {
        'r': 5,
        'tau': 10,
        'threshold': -60,
        'current': current_list2
    }
    neuron_params_inh = {
        'r': 8,
        'tau': 5,
        'threshold': -65,
        'is_inh': True,
        'current': current_list3
    }
    pop1 = Population(400, LIF, exc_ratio=1, **neuron_params_exc1)
    pop2 = Population(200, LIF, exc_ratio=0, **neuron_params_inh)
    pop3 = Population(400, LIF, exc_ratio=1, **neuron_params_exc2)

    conn1 = Connection(pop1, pop1).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn2 = Connection(pop2, pop2).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn3 = Connection(pop3, pop3).apply(
        "fixed_pre", p=0.01, mu=0.6, sigma=0.1)
    conn4 = Connection(pop1, pop2).apply("full")
    conn5 = Connection(pop2, pop3).apply("full")
    conn6 = Connection(pop2, pop1).apply("full")
    conn7 = Connection(pop3, pop2).apply("full")
    conn8 = Connection(pop3, pop1).apply("full")
    conn9 = Connection(pop1, pop3).apply("full")

    sim = Simulate(populations=[pop1, pop2, pop3], connections=[
                   conn1, conn2, conn3, conn4, conn5, conn6, conn7, conn8, conn9],
                   time_step=dt)
    sim.run(duration)
    print(pop1.spikes_per_neuron.shape)
    print(pop2.spikes_per_neuron.shape)
    print(pop3.spikes_per_neuron.shape)
    spikes = aggregate_population_spikes([pop1, pop2, pop3])
    print(spikes.shape)
    decision_plot(pop1.activity, pop3.activity)
