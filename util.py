import numpy as np
from warnings import warn


def check_input_dims(X, y, w):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"""Number of samples ({X.shape[0]}) doesn't match the length of
the target data vector ({y.shape[0]})""")
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"""Dimension of X ({X.shape[1]}) doesn't match the dimension of w
({w.shape[0]})""")


smoothness = lambda X, l2: 0.25*max(np.linalg.norm(X, 2, axis=1))**2+l2


def logistic_stable(x):
    """
    Implementation of the logistic function
        f(x) = 1 / (1 + exp(-x))
    that avoids numerical overflow in the exponential

    ref. http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    try:
        d = len(x)
    except TypeError:
        return 1./(1.+np.exp(-x)) if x > 0. else np.exp(x)/(1.+np.exp(x))

    ret = np.empty(d, dtype=np.float)
    pos_idx = x > 0.

    ret[pos_idx] = 1./(1.+np.exp(-x[pos_idx]))
    e_tmp = np.exp(x[~pos_idx])
    ret[~pos_idx] = e_tmp/(1.+e_tmp)

    return ret


def prox(w, c):
    """
    Proximal gradient operator
    """
    ret = []
    for weight in w:
        if weight < -c:
            ret.append(weight+c)
        elif abs(weight) <= c:
            ret.append(0)
        else:
            ret.append(weight-c)

    return np.array(ret)


def bfgs_hessapprox_update(H, s, t):
    """
    Hessian approximation update used in the BFGS method and the first few iterations of the
    limited-memory BFGS implementations.
    """
    d = np.dot(t, s)
    if d == 0.:
        # pass empty object back to the gradient method scope to trigger a TypeError in the update
        res = None
    else:
        try:
            res = (np.dot(np.outer(s, t), H)+np.dot(H, np.outer(t, s)))/d+\
                np.outer(s, s)*(1+np.dot(t, np.dot(H, t)))/d**2
        except RuntimeWarning:
            warn('Invalid value(s) encountered in BFGS update, reverting to newly initialised Hessian estimation', RuntimeWarning)
            res = np.eye(H.shape)

    return res


bfgs_stopcondition = lambda current, previous, epsilon: np.abs(current-previous) <= epsilon*np.max((previous, current, 1.))


def lbfgs_recursion(iter_, memory_size, grad, dim, disp, grad_disp):
    A = []
    for i in reversed(range(iter_-memory_size, iter_)):
        wdisp, gdisp = disp[i%memory_size], grad_disp[i%memory_size]
        a = grad*np.dot(wdisp, gdisp)/np.dot(gdisp, gdisp)
        grad -= a*gdisp
        A.append(a)
    s, t = disp[memory_size-1], grad_disp[memory_size-1]
    z = np.dot(np.eye(dim)*np.dot(s, t)/np.dot(t, t), grad)
    for i in range(iter_-memory_size, iter_):
        z += np.dot(wdisp, A[i%memory_size]-z*gdisp/np.dot(gdisp, wdisp))

    return z


def gram(X, kernel_fn, **kernelparams):
    """
    Computes the Gram matrix for a given data matrix and kernel function
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i,j] = kernel_fn(X[i], X[j], **kernelparams)
            if i != j:
                K[j,i] = K[i,j]

    return K