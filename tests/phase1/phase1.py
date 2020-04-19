import pickle
import numpy as np

import sys

folder = ''

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]

q1_list = [
    {
        'time_window': 100,
        'dt': 1,
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 1,
            'u_rest': -65,
            'threshold': -55
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 2,
            'u_rest': -65,
            'threshold': -55
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 12,
            'r': 1,
            'u_rest': -65,
            'threshold': -55
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 8,
            'r': 1,
            'u_rest': -65,
            'threshold': -55
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 2,
            'u_rest': -70,
            'threshold': -50
        }
    }
]
with open(f'./{folder}/q1.data', 'wb') as file:
    pickle.dump(q1_list, file)


q2_list = [
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'neuron_params': {
            'tau': 10,
            'r': 1,
            'u_rest': -65,
            'threshold': -55
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'neuron_params': {
            'tau': 10,
            'r': 2,
            'u_rest': -65,
            'threshold': -55
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'neuron_params': {
            'tau': 12,
            'r': 1,
            'u_rest': -65,
            'threshold': -55
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'neuron_params': {
            'tau': 8,
            'r': 1,
            'u_rest': -65,
            'threshold': -55
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'neuron_params': {
            'tau': 10,
            'r': 1,
            'u_rest': -70,
            'threshold': -50
        }
    }
]
with open(f'./{folder}/q2.data', 'wb') as file:
    pickle.dump(q2_list, file)


q3_list = [
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 1,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 1,
            'theta_rh': -58
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 1,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 0.2,
            'theta_rh': -58
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 1,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 2,
            'theta_rh': -58
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 1,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 1,
            'theta_rh': -60
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 1,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 1,
            'theta_rh': -55
        }
    }
]
with open(f'./{folder}/q3.data', 'wb') as file:
    pickle.dump(q3_list, file)


q4_list = [
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 2,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 1,
            'theta_rh': -58,
            'tau_w': 20,
            'a': 1,
            'b': 1
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 2,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 1,
            'theta_rh': -58,
            'tau_w': 5,
            'a': 1,
            'b': 1
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 2,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 1,
            'theta_rh': -58,
            'tau_w': 10,
            'a': 1,
            'b': 1
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 2,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 1,
            'theta_rh': -58,
            'tau_w': 20,
            'a': 1,
            'b': 5
        }
    },
    {
        'time_window': 100,
        'dt': 1,
        'current_range': (7, 13),
        'current_values': np.arange(2, 21, 2),
        'neuron_params': {
            'tau': 10,
            'r': 2,
            'u_rest': -70,
            'threshold': -55,
            'delta_t': 1,
            'theta_rh': -58,
            'tau_w': 20,
            'a': 5,
            'b': 1
        }
    }
]
with open(f'./{folder}/q4.data', 'wb') as file:
    pickle.dump(q4_list, file)
