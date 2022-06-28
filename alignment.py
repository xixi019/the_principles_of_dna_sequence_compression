import numpy as np
import random


def align(x, y, mode='global', match=1, mismatch=-1, gap=-1):
    # Check whether the mode is valid
    valid_modes = ['global', 'local']
    if mode not in valid_modes:
        raise RuntimeError('invalid mode: {}'.format(mode))

    # Store the two sequence lengths
    x_len = len(x)
    y_len = len(y)

    # Start filling the scoring matrix.
    # - Global aligment: Given there is no 'top' or 'top-left' cells for the
    #   first row (first column) only the existing cell to the left (top) can
    #   be used to calculate the score of each cell. Hence, as this represents
    #   a gap, the gap score is added for each shift to the right (bottom).
    # - Local alignment: The first row and column are set to zero.
    F = np.zeros(shape=((x_len + 1), (y_len + 1)))
    x_stop = x_len * gap
    y_stop = y_len * gap
    if mode == 'local':
        x_stop = 0
        y_stop = 0
    F[:, 0] = np.linspace(start=0, stop=x_stop, num=(x_len + 1))
    F[0, :] = np.linspace(start=0, stop=y_stop, num=(y_len + 1))

    # Pointers to trace through an optimal aligment. The cells which give the
    # highest candidate scores must also be recorded. Therefore we store
    # pointers to trace through an optimal alignment. We use 0 for top-left,
    # 1 for left, and 2 for top.
    P = np.zeros(shape=((x_len + 1), (y_len + 1)))
    P[:, 0] = 2
    P[0, :] = 1

    # Fill in the rest of the scoring matrix and construct the pointer matrix.
    # For each cell, we store the pointer of the first maximum that we find.
    # Ideally, all alternative paths should be traced.
    for i in range(x_len):
        for j in range(y_len):
            trace = np.zeros(shape=3)
            if x[i] == y[j]:
                trace[0] = F[i, j] + match
            else:
                trace[0] = F[i, j] + mismatch
            trace[1] = F[i, (j+1)] + gap
            trace[2] = F[(i+1), j] + gap
            trace_max = np.max(trace)
            F[(i+1), (j+1)] = np.max(trace)
            if mode == 'local' and trace_max < 0:
                F[(i+1), (j+1)] = 0
            else:
                P[(i+1), (j+1)] = np.argmax(trace)

    # Set the traceback start
    i = x_len
    j = y_len
    if mode == 'local':
        coords = np.argwhere(F == np.max(F))
        traceback_start = coords[0]
        if len(coords) > 1:
            traceback_start = random.choice(seq=coords)
            print('alignment.align: multiple ({}) possible traceback starts, choosing {}'.format(len(coords), traceback_start))
        i = traceback_start[0]
        j = traceback_start[1]

    # Traceback
    x_aligned = []
    y_aligned = []
    path_x = []
    path_y = []
    while (mode == 'global' and (i > 0 or j > 0)) or (mode == 'local' and F[i, j] > 0):
        path_x.append(i)
        path_y.append(j)
        if P[i, j] == 0:  # top-left
            x_aligned.append(x[i-1])
            y_aligned.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i, j] == 1:  # left
            x_aligned.append(x[i-1])
            y_aligned.append('-')
            i -= 1
        elif P[i, j] == 2:  # top
            x_aligned.append('-')
            y_aligned.append(y[j-1])
            j -= 1

    # Reverse the strings
    x_aligned = ''.join(x_aligned)[::-1]
    y_aligned = ''.join(y_aligned)[::-1]

    # Reverse the paths and let them start with 0
    path_x = path_x[::-1]
    path_y = path_y[::-1]
    path_x = [(i - 1) for i in path_x]
    path_y = [(j - 1) for j in path_y]

    # Remove the first row and column from F
    F = F[:, 1:]
    F = F[1:, :]

    return x_aligned, y_aligned, path_x, path_y, F
