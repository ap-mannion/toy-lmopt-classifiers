import numpy as np


def check_input_dims(X, y, w):
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"""Number of samples in X ({X.shape[0]}) doesn't match the number
of samples in y ({y.shape[0]})""")
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"""Dimension of X ({X.shape[1]}) doesn't match the dimension of w
({w.shape[0]})""")


def GD(X, y, w_init, obj_fn, max_iter, smoothness):
    """
    Basic gradient descent implementation.

    Parameters
    ----------
    X: numpy.ndarray
        Data matrix
    y: numpy.ndarray
        Classification labels
    w_init: numpy.ndarray
        Initialisation of the solver
    obj_fn: function
        Loss function object from the linear_model script
    max_iter: int
        Number of iterations (i.e. number of descent steps), in this case, one iteration is one
        epoch i.e, one pass through the data.
    smoothness: float
        Smoothness constant of the objective function

    Returns
    -------
    weights: numpy.ndarray
        final iterate of the solver
    wtab: numpy.ndarray
        table of all the iterates
    """
    check_input_dims(X, y, w_init)
    stepsize = 1.0/smoothness
    w = w_init
    wtab = np.copy(w)

    k = 0 # iterates
    while k <= max_iter:
        w -= stepsize*obj_fn.grad(X, y, w)
        wtab = np.vstack((wtab, w))
        k += 1

    return w, wtab


def SGD(X, y, w_init, obj_fn, max_iter, smoothness, force_complete_pass=False):
    """
    Stochastic Gradient Descent
    The boolean argument `force_complete_pass` can be set to true to make the algorithm use each
    data point exactly once in each pass over the data: max_iter should normally be set to some
    integer multiple of the number of samples (note that for this algorithm `max_iter` controls
    the number of data points iterated over, rather than the number of passes over the dataset as
    in the vanilla gradient descent implementation).
    """
    check_input_dims(X, y, w_init)
    w = w_init
    wtab = np.copy(w)
    n = len(y)

    k = 0
    if force_complete_pass:
        indices = np.arange(n)
        np.random.shuffle(indices)
    while k <= max_iter:
        stepsize = 1.0/(smoothness*(int(k/n+1)**0.6))
        if not force_complete_pass:
            i = np.random.randint(n)
        else:
            i = indices[k%n]
        w -= stepsize*obj_fn.grad(X, y, w, i)
        if k%n == 0: # full pass over n data samples
            wtab = np.vstack((wtab, w))
            if force_complete_pass:
                np.random.shuffle(indices)
        k += 1

    return w, wtab


def SAGA(X, y, w_init, obj_fn, max_iter, smoothness, strong_convexity):
    """
    Implementation of the SAGA gradient method introduced in the 2007 paper 'SAGA: A Fast
    Incremental Gradient Method with Support for Non-Strongly Convex Composite Objectives'
    (arXiv:1407.0202)
    """
    check_input_dims(X, y, w_init)
    w = w_init
    wtab = np.copy(w)
    n = len(y)
    stepsize = 1/(2*strong_convexity*n+smoothness)
    gtab = np.vstack(tuple(obj_fn.grad(X, y, w, i) for i in range(n)))

    k = 0
    while k <= max_iter:
        j = np.random.randint(n)
        w -= stepsize*(obj_fn.grad(X, y, w, j)-gtab[j]+np.sum(gtab)/n)
        if k%n == 0: # add to iteration storage every n iterations
            wtab = np.vstack((wtab, w))
        k += 1

    return w, wtab


def SVRG(X, y, w_init, obj_fn, max_iter, smoothness, strong_convexity):
    """
    Stochastic Variance-Reduced Gradient method, from the paper 'Accelerating Stochastic
    Gradient Descent using Predictive Variance Reduction', published in NIPS 2013
    """
    check_input_dims(X, y, w_init)
    stepsize = 0.1/smoothness
    M = int(1.1*smoothness/strong_convexity)
    wtab = np.copy(w_init)

    k = 0
    while k <= max_iter:
        v0 = wtab[k]
        v = np.copy(v0)
        grad_tmp = obj_fn.grad(X, y, v)
        for _ in range(M):
            i = np.random.randint(len(y))
            y -= stepsize*obj_fn.grad(X, y, v, i)-obj_fn.grad(X, y, v0, i)+grad_tmp
        w = v
        wtab = np.vstack((wtab, w))
        k += 1

    return w, wtab


def NM(X, y, w_init, obj_fn, max_iter, smoothness, stopping_eps=1e-5, wolfe_alpha=0.1, wolfe_beta=0.9):
    """
    Newton's method for logistic regression. The args `X`, `y`, `w_init`, `obj_fn`, `max_iter`, and
    `smoothness` have the same definitions as for gradient descent. If either of the line search 
    parameters are set to `None`, the algorithm will just use a constant step size

    Parameters
    ----------
    stopping_eps:  float
        Precision to be used in the stopping rule
    wolfe_alpha: float
        Constant used to check the Armijo condition in the Wolfe line search for the optimal step
        size
    wolfe_beta: float
        Constant used to scale the step size in the Wolfe search
    """
    check_input_dims(X, y, w_init)
    w = w_init
    wtab = np.copy(w)
    stepsize = 1.0/smoothness

    k = 0
    while k <= max_iter:
        g = obj_fn.grad(X, y, w)
        v = np.linalg.solve(obj_fn.hess(X, y, w), g)
        if wolfe_alpha is not None and wolfe_beta is not None:
            # Wolfe line search
            while obj_fn(X, y, w+stepsize*v) > obj_fn(X, y, w)+wolfe_alpha*stepsize*np.dot(g, v):
                stepsize *= wolfe_beta
        w -= stepsize*v
        wtab = np.vstack((wtab, w))
        if 0.5*np.dot(g, v) <= stopping_eps:
            break
        k += 1

    return w, wtab


def BFGS(X, y, w_init, obj_fn, max_iter, smoothness):
    # ISSUE: RuntimeWarning about overflows in exp and division by zero, weights become nans
    """
    Implementation of the Broyden-Fletcher-Goldfarb-Shanno algorithm, a type of quasi-Newton method.
    """
    check_input_dims(X, y, w_init)
    w = w_init
    wtab = np.copy(w)
    H = np.eye(w.size) # initialise Hessian approximation
    stepsize = 1.0/smoothness

    k = 0
    while k <= max_iter:
        w_prev = np.copy(w)
        g = obj_fn.grad(X, y, w)

        # Newton-method style weight update
        w -= stepsize*np.dot(H, g)

        # update Hessian approximation
        s = w-w_prev
        t = obj_fn.grad(X, y, w)-g
        d = np.dot(t, s)
        H -= (np.dot(np.outer(s, t), H)+np.dot(H, np.outer(t, s)))/d+\
            np.outer(s, s)*(1+np.dot(t, np.dot(H, t)))/d**2

        wtab = np.vstack((wtab, w))
        k += 1

    return w, wtab


def LBFGS(X, y, w_init, obj_fn, max_iter, smoothness, memory_size=10):
    """
    This is a variant of the BFGS algorithm adapted for storage efficiency - it uses an implicit 
    representation of the update term for the Hessian approximation that gives it a linear memory
    requirement. It keeps a record of previous updates of the gradient and directly approximates
    the Hessian-gradient product instead of storing separate versions of both
    """
    check_input_dims(X, y, w_init)
    w = w_init
    wtab = np.copy(w)
    stepsize = 1.0/smoothness
    p = w.size
    H = np.eye(p)
    z = obj_fn.grad(X, y, w) # initialisation of grad-Hess product
    disp, grad_disp = {}, {}

    k = 0
    while k <= max_iter:
        w_prev = np.copy(w)
        g = obj_fn.grad(X, y, w)
        w -= stepsize*z
        disp[k%memory_size] = w-w_prev
        grad_disp[k%memory_size] = obj_fn(X, y, w)-g
        if k < memory_size:
            s, t = disp[k], grad_disp[k]
            d = np.dot(s, t)
            if d == 0:
                return w_init, y
            H -= (np.dot(np.outer(s, t), H)+np.dot(H, np.outer(t, s)))/d+\
                np.outer(s, s)*(1+np.dot(t, np.dot(H, t)))/d**2
            z = np.dot(H, g)
        else:
            A = []
            db_iter1 = 0
            for i in reversed(range(k-memory_size, k)):
                db_iter1 += 1
                wdisp, gdisp = disp[i%memory_size], grad_disp[i%memory_size]
                a = g*np.dot(wdisp, gdisp)/np.dot(gdisp, gdisp)
                g -= a*gdisp
                A.append(a)
            print(f'loop 1 iters: {db_iter1}')
            print(f'size of A: {len(A)}')
            s, t = disp[memory_size-1], grad_disp[memory_size-1]
            z = np.dot(np.eye(p)*np.dot(s, t)/np.dot(t, t), g)
            db_iter2 = 0
            for i in range(k-memory_size, k):
                db_iter2 += 1
                z += np.dot(wdisp, A[i%memory_size]-z*gdisp/np.dot(gdisp, wdisp))
            print(f'loop 2 iters: {db_iter2}')
        wtab = np.vstack((wtab, w))
        k += 1

    return w, wtab


def SLBFGS(X, y, obj_fn, max_iter, smoothness, n_updates, memory_size, n_curve_updates, batch_size_grad, batch_size_hess):
    # TODO
    pass


