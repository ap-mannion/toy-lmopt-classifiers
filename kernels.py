import numpy as np
from collections import defaultdict
from warnings import warn


linear = lambda a, b: np.dot(a, b)


def gaussian(a, b, variance, sigma=None):
    """
    Computes the Gaussian similarity measure between the given arrays a and b, with
    the sd specified by sigma or variance specified directly.
    """
    ssq = variance if sigma is None else sigma**2
    
    return np.exp(-1*np.linalg.norm(a-b)**2/(2*ssq))


polynomial = lambda a, b, deg=3, intercept=0.0: (np.dot(a, b)+intercept)**deg


def spectrum(a, b, k, sim_fn=None, **sf_params):
    """
    Computes the spectrum kernel, which is a similarity measure of two sequences
    based on the number of occurrences of subausequences of a given length in each one,
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
            res = sim_fn(count_a, count_b, **sf_params)
        except:
            try:
                # see if the function will take compressed sparse row matrices
                from scipy.sparse import coo_matrix, csr_matrix

                subsequences = np.unique(np.array(list(count_a.keys())+list(count_b.keys())))
                coords = {}
                coords.update((seq, i) for i, seq in enumerate(subsequences))
                
                def _tosparse(fdict):
                    rows, cols, freqs = [], [], []
                    for k, v in fdict.items():
                        rows.append(1)
                        cols.append(coords[k])
                        freqs.append(v)

                    return csr_matrix(coo_matrix((freqs, (rows, cols))))

                res = sim_fn(_tosparse(count_a), _tosparse(count_b), **sf_params)

            except:
                # try with numpy arrays
                warn(f"""Provided similarity function {sim_fn.__name__} doesn't support sparse vector representations - trying with dense numpy arrays.
This can be very slow for long sequences with high-dimensional frequency spaces""", RuntimeWarning)
                try:
                    def _tonp(fdict):
                        l = []
                        for s, i in coords.items():
                            try:
                                l.append(fdict[s])
                            except KeyError:
                                l.append(0)

                        return np.array(l)

                    res = sim_fn(_tonp(count_a), _tonp(count_b), **sf_params)
                except:
                    raise TypeError("""Similarity function must take dictionary representations of sparse matrices, scipy.sparse.csr_matrix objects
or numpy arrays""")

    return res             
                    
                

