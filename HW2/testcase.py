# randomly generate test data
import numpy as np

# sample with mean change
def sampleMean():
    have_change = np.random.choice([True, False])
    val = np.random.randint(0, 100, size=2)
    var = np.random.randint(0, 10, size=1)

    if have_change:
        position_change = np.random.randint(51, 1000)
        data1 = np.random.normal(loc=val[0], scale=var, size=position_change)
        data2 = np.random.normal(loc=val[1], scale=var, size=1000 - position_change)
        data = np.append(data1, data2)
    else:
        position_change = -1
        data = np.random.normal(loc=val[0], scale=var, size=1000)

    return data, position_change

# sample with variance change
def sampleVariance():
    have_change = np.random.choice([True, False])
    val = np.random.randint(0, 100, size=1)
    var = np.random.randint(0, 10, size=2)

    if have_change:
        position_change = np.random.randint(51, 1000)
        data1 = np.random.normal(loc=val, scale=var[0], size=position_change)
        data2 = np.random.normal(loc=val, scale=var[1], size=1000 - position_change)
        data = np.append(data1, data2)
    else:
        position_change = -1
        data = np.random.normal(loc=val, scale=var[0], size=1000)

    return data, position_change

# sample with change in probability of binary
def sampleBinary():
    have_change = np.random.choice([True, False])
    p = np.random.uniform(0, 1, size=2)
    choices = ['a', 'b']

    if have_change:
        position_change = np.random.randint(51, 1000)
        data1 = np.random.choice(choices, position_change, p=[p[0], 1 - p[0]])
        data2 = np.random.choice(choices, 1000 - position_change, p=[p[1], 1 - p[1]])
        data = np.append(data1, data2)
    else:
        position_change = -1
        data = np.random.choice(choices, 1000, p=[p[0], 1 - p[0]])

    return data, position_change
