import numpy as np

def one_side_exponential_smoothing(x_before, y_before, decay_steps=1.):
    """
    This is the one side (regular) exponential moving average, which 
    evenly resamples points based on x-aixs and averages y values with weighting factor
    decreasing exponentially.

    Arguments:
    x_before: array     - x values. Required to be in accending order.
    
    y_before: array     - y values. Required to have same size of x_before.
    
    decay_steps: float  - the number of previous steps trusted. Used to calculate the decay factor.

    Return:
    x_after: array      - x values after resampling.
    
    y_after: array      - y values after smoothing.
    
    y_count: array      - decay values for each steps. 

    """
    
    if x_before is None:
        x_before = np.arange(len(y_before))

    assert len(x_before) == len(y_before), 'length of x_before and y_before is not equal !'
    assert all(x_before[i] <= x_before[i+1] for i in range(len(x_before)-1)),' x_before needs to be sorted in accending order'
    # Resampling
    size = len(x_before)
    x_after = np.linspace(x_before[0],x_before[-1],size)
    y_after = np.zeros(size,dtype =float)
    y_count = np.zeros(size,dtype =float)

   
    alpha = np.exp(-1./decay_steps)  # Weighting factor for data of previous steps
    x_before_length = x_before[-1] - x_before[0]
    x_before_index = 0
    decay_period = x_before_length/(size-1)*decay_steps
    
    for i in range(len(x_after)):
        # Compute current EMA value based on the value of previous time step
        if(i!=0):
            y_after[i] = alpha * y_after[i-1]
            y_count[i] = alpha * y_count[i-1]

        # Compute current EMA value by adding weighted average of old points covered by the new point
        while x_before_index < size:
            if x_after[i] >= x_before[x_before_index]:
                difference = x_after[i] - x_before[x_before_index]
                beta = np.exp(-(difference/decay_period))   # Weighting factor for y value of each old points
                y_after[i] += y_before[x_before_index] * beta
                y_count[i] += beta
                x_before_index += 1
            else:
                break

    y_after = y_after/y_count
    return x_after, y_after, y_count


def exponential_smoothing(x_before, y_before, decay_steps=1.0):
    """
    This is the two side exponential moving average, which 
    performs regular exponential moving average twice from different sides 
    and then combines results together.

    Arguments:
    x_before: array     - x values. Required to be in accending order.
    
    y_before: array     - y values. Required to have same shape of x_before.
    
    decay_steps: float  - the number of previous steps trusted. Used to calculate the decay factor.

    Return:
    x_after: array      - x values after resampling.
    
    y_after: array      - y values after smoothing.
    
    y_count: array      - decay values for each points. 

    """

    x_after1, y_after1, y_count1 = one_side_exponential_smoothing(x_before, y_before, decay_steps)
    x_after2, y_after2, y_count2 = one_side_exponential_smoothing(-x_before[::-1],y_before[::-1],decay_steps)

    y_after2 = y_after2[::-1]
    y_count2 = y_count2[::-1]

    y_after = (y_after1*y_count1+y_after2*y_count2)/(y_count1+y_count2)

    return x_after1, y_after, y_count1+y_count2


smooth = exponential_smoothing

