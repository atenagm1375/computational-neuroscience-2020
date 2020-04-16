import pickle
import numpy as np


q1_list = [
    {
        'time_window': 10,
        'dt': 0.001,
        'current_values': np.arange(0, 50, 5),
        'neuron_params': {
            'tau': 10,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.001,
        'current_values': np.arange(0, 50, 5),
        'neuron_params': {
            'tau': 5,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.001,
        'current_values': np.arange(0, 50, 5),
        'neuron_params': {
            'tau': 20,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.001,
        'current_values': np.arange(0, 50, 5),
        'neuron_params': {
            'tau': 10,
            'r': 5,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.001,
        'current_values': np.arange(0, 50, 5),
        'neuron_params': {
            'tau': 20,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    }
]
with open('./q1.data', 'wb') as file:
    pickle.dump(q1_list, file)


q2_list = [
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'neuron_params': {
            'tau': 10,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'neuron_params': {
            'tau': 20,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'neuron_params': {
            'tau': 10,
            'r': 5,
            'u_rest': -65,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'neuron_params': {
            'tau': 5,
            'r': 10,
            'u_rest': -70,
            'threshold': -55
        }
    },
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'neuron_params': {
            'tau': 15,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    }
]
with open('./q2.data', 'wb') as file:
    pickle.dump(q2_list, file)


q3_list = [
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 10,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 20,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 10,
            'r': 5,
            'u_rest': -65,
            'threshold': -50
        }
    },
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 5,
            'r': 10,
            'u_rest': -70,
            'threshold': -55
        }
    },
    {
        'time_window': 10,
        'dt': 0.01,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 15,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    }
]
with open('./q3.data', 'wb') as file:
    pickle.dump(q3_list, file)


q4_list = [
    {
        'time_window': 5,
        'dt': 0.001,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 10,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 5,
        'dt': 0.001,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 20,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    },
    {
        'time_window': 5,
        'dt': 0.001,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 10,
            'r': 5,
            'u_rest': -65,
            'threshold': -50
        }
    },
    {
        'time_window': 5,
        'dt': 0.001,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 5,
            'r': 10,
            'u_rest': -70,
            'threshold': -55
        }
    },
    {
        'time_window': 5,
        'dt': 0.001,
        'current_range': (10, 100),
        'current_values': np.arange(10, 100, 10),
        'neuron_params': {
            'tau': 15,
            'r': 10,
            'u_rest': -70,
            'threshold': -50
        }
    }
]
with open('./q4.data', 'wb') as file:
    pickle.dump(q4_list, file)
