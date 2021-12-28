import numpy as np

def compounded_returns(values):
    if type(values).__module__ != np.__name__:
        raise Exception(f'Needs a numpy array given {type(values)}')
    prev_values = values[:-1]
    curr_values = values[1:]
    returns = [(np.log(curr / prev) * 100) for curr, prev in zip(curr_values, prev_values)]
    returns.insert(0, 1)
    return returns