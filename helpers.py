import numpy as np


def check_input_dims(X, y, w):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"""Number of samples in X ({X.shape[0]}) doesn't match the number
of samples in y ({y.shape[0]})""")
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"""Dimension of X ({X.shape[1]}) doesn't match the dimension of w
({w.shape[0]})""")


def linesearch(obj_fn, X, y, w, stepsize, alpha, beta, grad, hess=None, gh=None):
    # Wolfe line search
    if hess is None:
        hess = obj_fn.hess(X, y, w)
    gH_product = np.linalg.solve(hess, grad) if gh is None else gh
    while obj_fn(X, y, w+stepsize*gH_product) > obj_fn(X, y, w)+alpha*stepsize*np.dot(grad, gH_product):
        stepsize *= beta

    return stepsize, gH_product


def bfgs_hessapprox_update(H, s, t):
    """
    Hessian approximation update used in the BFGS method and the first few iterations of the
    limited-memory BFGS implementations.
    """
    d = np.dot(t, s)
    if d == 0.0:
        # pass empty object back to the gradient method scope to trigger a TypeError in the update
        res = None
    else:
        try:
            res = (np.dot(np.outer(s, t), H)+np.dot(H, np.outer(t, s)))/d+\
                np.outer(s, s)*(1+np.dot(t, np.dot(H, t)))/d**2
        except RuntimeWarning:
            print("""*solvers msg*: Invalid value(s) encountered in BFGS update, returning newly initialised Hessian estimation""")
            res = np.eye(H.shape)

    return res


bfgs_stopcondition = lambda current, previous, epsilon: np.abs(current-previous) <= epsilon*np.max((previous, current, 1.0))


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