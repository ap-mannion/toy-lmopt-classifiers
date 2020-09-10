import numpy as np
from collections import defaultdict


def gaussian(a, b, variance, sigma=None):
    """
    Computes the Gaussian similarity measure between the given arrays a and b, with
    the sd specified by sigma or variance specified directly.
    """
    ssq = variance if sigma is None else sigma**2
    return np.exp(-np.sqrt(np.linalg.norm(a-b)**2/(2*ssq)))


def spectrum(a, b, k, sim_fn=None):
    """
    Computes the spectrum kernel, which is a similarity measure of two sequences
    based on the number of occurrences of subsequences of a given length in each one,
    without explicitly mapping them to the feature space.

    Parameters
    ----------
    a, b: numpy.ndarray or list
    k: int
        Length of subsequences
    sim_fn: callable
        Function to use to calculate the similarity measure between the calculated frequency
        vectors. The function must take either dictionaries representing sparse vectors or numpy
        arrays as inputs; if this argument is not specified, the inner
        product (dot product) is used.
    """
    assert(len(a) == len(b))
    count_a, count_b = defaultdict(int), defaultdict(int)
    for p in range(len(a)-k+1):
        count_a[a[p:p+k]] += 1
        count_b[b[p:p+k]] += 1

    if sim_fn is None:
        res = 0
        for subseq, freq in count_a.items():
            if subseq in count_b:
                res += freq*count_b[subseq]
    else:
        try:
            res = sim_fn(count_a, count_b)
        except:
            # make numpy arrays from the sparse dictionary representation (this is not recommended)
            a_list, b_list = [], []
            for subseq in np.unique(np.array(list(count_a.keys())+list(count_b.keys()))):
                if subseq in count_a and subseq in count_b:
                    a_list.append(count_a[subseq])
                    b_list.append(count_b[subseq])
                elif subseq in count_a and subseq not in count_b:
                    a_list.append(count_a[subseq])
                    b_list.append(0)
                else:
                    a_list.append(0)
                    b_list.append(count_b[subseq])
            res = sim_fn(np.array(a_list), np.array(b_list))

    return res                
                    
                

