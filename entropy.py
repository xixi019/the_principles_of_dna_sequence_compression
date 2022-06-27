import math
from collections import Counter

def calculate(data):
    base = 2

    if len(data) <= 1:
        return 0

    counts = Counter()
    for d in data:
        counts[d] += 1

    eta = 0
    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            eta -= p * math.log(p, base)

    return eta
