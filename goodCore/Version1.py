import random


"""
performance issues , too slow
"""
def get_samples(corpus, C, size_samples=100):
    candidates = list(set(range(len(corpus))) - set(C))
    if len(candidates) <= size_samples:
        return candidates
    return random.sample(candidates, size_samples)

def distance(a,b):
    return np.linalg.norm(a-b)  
      ## tooooooooo slow
def compute_error(corpus, C):
    """
    we are mesuring the distnace betwene the corset points and the rest of the dataset , 
    we want it nice disparcity
    want it to be small 
    """
    error = 0
    for x in corpus:
        d = min(distance(x, corpus[c]) for c in C)
        error += d
    return error
def compute_utility(corpus, C, t):

    if len(C) == 0:
        return float("inf")

    error_before = compute_error(corpus, C)
    error_after = compute_error(corpus, C + [t])

    return error_before - error_after
def greedy_coreset(corpus, K=50, sample_size=100):
    C = []
    while len(C) < K:
        print(len(c))
        samples = get_samples(corpus, C, sample_size)
        best_t = None
        best_utility = -float("inf")
        for t in samples:
            if t in C:
                continue

            u = compute_utility(corpus, C, t)

            if u > best_utility:
                best_utility = u
                best_t = t

        if best_t is not None:
            C.append(best_t)

    return C
  

