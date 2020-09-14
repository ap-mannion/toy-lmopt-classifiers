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


def _linesearch(obj_fn, X, y, w, stepsize, alpha, beta, grad, hess=None, gh=None):
    # Wolfe line search
    if hess is None:
        hess = obj_fn.hess(X, y, w)
    gH_product = np.linalg.solve(hess, grad) if gh is None else gh
    while obj_fn(X, y, w+stepsize*gH_product) > obj_fn(X, y, w)+alpha*stepsize*np.dot(grad, gH_product):
        stepsize *= beta

    return stepsize, gH_product


def NM(X, y, w_init, obj_fn, max_iter, smoothness, stopping_eps=1e-5, ls_alpha=0.1, ls_beta=0.9):
    """
    Newton's method for logistic regression. The args `X`, `y`, `w_init`, `obj_fn`, `max_iter`, and
    `smoothness` have the same definitions as for gradient descent. If either of the line search 
    parameters are set to `None`, the algorithm will just use a constant step size

    Parameters
    ----------
    stopping_eps:  float
        Precision to be used in the stopping rule
    ls_alpha: float
        Constant used to check the Armijo condition in the Wolfe line search for the optimal step
        size
    ls_beta: float
        Constant used to scale the step size in the Wolfe search
    """
    check_input_dims(X, y, w_init)
    w = w_init
    wtab = np.copy(w)
    stepsize = 1.0/smoothness

    k = 0
    while k <= max_iter:
        g = obj_fn.grad(X, y, w)
        if ls_alpha is not None and ls_beta is not None:
            stepsize, ghprod = _linesearch(obj_fn, X, y, w, stepsize, ls_alpha, ls_beta, g)
        w -= stepsize*ghprod
        wtab = np.vstack((wtab, w))
        if 0.5*np.dot(g, ghprod) <= stopping_eps:
            break
        k += 1

    return w, wtab


def _bfgs_hessapprox_update(H, s, t):
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
            print("""*solvers msg*: Invalid value(s) encountered in BFGS update, returning newly
initialised Hessian estimation""")
            res = np.eye(H.shape)

    return res


_bfgs_stopcondition = lambda current, previous, epsilon: np.abs(current-previous) <= epsilon*np.max((previous, current, 1.0))


def BFGS(X, y, w_init, obj_fn, max_iter, smoothness, stopping_eps=1e-5, ls_alpha=0.1, ls_beta=0.9):
    # ISSUE: RuntimeWarning about overflows in exp and division by zero, weights become nans
    """
    Implementation of the Broyden-Fletcher-Goldfarb-Shanno algorithm, a type of quasi-Newton method,
    with the stopping rule based on the absolute change in the value of the objective function
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
        if ls_alpha is not None and ls_beta is not None:
            stepsize, _ = _linesearch(obj_fn, X, y, w, stepsize, ls_alpha, ls_beta, g, H)

        # Newton-method style weight update
        w -= stepsize*np.dot(H, g)
        wtab = np.vstack((wtab, w))

        # update Hessian approximation
        try:
            H -= _bfgs_hessapprox_update(H, w-w_prev, obj_fn.grad(X, y, w)-g)
        except TypeError:
            print("""BFGS optimiser: Zero-displacement curvature caught at iteration {k} Hessian update:
stopping descent""")
            break
        
        # precision check for stopping condition
        current_objval, prev_objval = obj_fn(X, y, w), obj_fn(X, y, wtab[k-1])
        if _bfgs_stopcondition(current_objval, prev_objval, stopping_eps):
            print(f"BFGS optimiser: stopping condition reached at iteration {k}\n")
            break
        k += 1

    return w, wtab


def _lbfgs_recursion(iter_, memory_size, grad, dim, disp, grad_disp):
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


def LBFGS(X, y, w_init, obj_fn, max_iter, smoothness, memory_size=10, stopping_eps=1e-5, ls_alpha=0.1, ls_beta=0.9):
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
        if ls_alpha is not None and ls_beta is not None:
            stepsize, _ = _linesearch(obj_fn, X, y, w, stepsize, ls_alpha, ls_beta, g, H)
        w -= stepsize*z
        disp[k%memory_size] = w-w_prev
        grad_disp[k%memory_size] = obj_fn(X, y, w)-g
        if k < memory_size:
            try:
                H -= _bfgs_hessapprox_update(H, disp[k], grad_disp[k])
            except TypeError:
                print("""L-BFGS optimiser: Zero-displacement curvature caught at iteration {k} Hessian update:
stopping descent""")
            break
            z = np.dot(H, g)
        else:
            z = _lbfgs_recursion(k, memory_size, g, p, disp, grad_disp)
        wtab = np.vstack((wtab, w))
        current_objval, prev_objval = obj_fn(X, y, w), obj_fn(X, y, wtab[k-1])
        if _bfgs_stopcondition(current_objval, prev_objval, stopping_eps):
            print(f"L-BFGS optimiser: stopping condition reached at iteration {k}\n")
            break
        k += 1

    return w, wtab


def SLBFGS(X, y, w_init, obj_fn, max_iter, smoothness, n_updates, memory_size, n_curve_updates, batch_size_grad, batch_size_hess, stopping_eps=1e-5, ls_alpha=0.1, ls_beta=0.9):
    """
    Stochastic BFGS algorithm implementation with limited memory constraint, as introduced by
    Moritz et al in the AISTAT 2016 paper 'A Linearly-Convergent Stochastic L-BFGS Algorithm'
    (arXiv:1508.02087v2).

    Parameters
    ----------
    n_updates: int
        Number of stochastic updates
    memory_size: int
        Number of gradient updates to hold in memory at a time
    n_curve_updates: int
        Number of stochastic updates per update of the curvature
    batch_size_grad: int
        Batch size for gradient subsampling
    batch_size_hess: int
        Batch size for Hessian subsampling
    """
    w = w_init
    wtab = np.copy(w)
    stepsize = 1.0/smoothness
    n, p = y.size, w_init.size
    H = np.eye(p)
    x = w
    disp, grad_disp = {}, {}

    # iterators: one for the target gradient updates and one for the Hessian curvature estimation updates
    k, r = 0, 0
    while k <= max_iter:
        g = obj_fn.grad(X, y, w)
        v = w
        vtab = np.copy(v)

        for i in range(n_updates):
            batch = np.random.randint(n, size=batch_size_grad).tolist()
            mu1, mu2 = obj_fn.grad(X, y, v, batch), obj_fn.grad(X, y, w, batch)

            v -= stepsize*np.dot(H, mu1-mu2+g)
            vtab = np.vstack((vtab, v))

            if i%n_curve_updates == 0:
                x_prev = np.copy(x)
                x = np.mean(vtab, 0)
                disp[r] = x-x_prev

                # update gradient-Hessian product approximation
                grad_disp[r] = disp[r]*np.mean(
                    np.array(
                        [obj_fn.hess(X, y, x, j) for j in np.random.randint(n, size=batch_size_hess)]
                    )
                )

                if r < memory_size: # first `filling` of the memory with product estimations
                    try:
                        H -= _bfgs_hessapprox_update(H, disp[r], grad_disp[r])
                    except TypeError:
                        print("""BFGS optimiser: Zero-displacement curvature caught at iteration {k} Hessian update:
stopping descent""")
                        break
                else:
                    H = _lbfgs_recursion(r, memory_size, obj_fn.grad(X, y, x), p, disp, grad_disp)
                r += 1
        w = vtab[np.random.randint(n_updates)]
        wtab = np.vstack((wtab, w))
        current_objval, prev_objval = obj_fn(X, y, w), obj_fn(X, y, wtab[k-1])
        if _bfgs_stopcondition(current_objval, prev_objval, stopping_eps):
            print(f"Stochastic L-BFGS optimiser: stopping condition reached at iteration {k}\n")
            break
        k += 1
    
    return w, wtab


def QP_KSVM(X, y, kernel_fn, **kernelparams):
    # TODO
    pass