import numpy as np
import util
# TODO:
#   - debug BFGS stopping conditions
#   - add SVRG option to update gradient with a randomly chosen inner-loop step

def GD(X, y, w, obj_fn, max_iter, use_prox=False):
    """
    Basic gradient descent implementation.

    Parameters
    ----------
    X: numpy.ndarray
        Data matrix
    y: numpy.ndarray
        Classification labels
    w: numpy.ndarray
        Initialisation of the solver
    obj_fn: function
        Loss function object from the linear_model script
    max_iter: int
        Number of iterations (i.e. number of descent steps), in this case, one iteration is one
        epoch i.e, one pass through the data.

    Returns
    -------
    weights: numpy.ndarray
        final iterate of the solver
    wtab: numpy.ndarray
        table of all the iterates
    """
    util.check_input_dims(X, y, w)
    stepsize = 1./util.smoothness(X, obj_fn.l2)
    wtab = np.copy(w)

    for _ in range(max_iter):
        if use_prox:
            w = util.prox(w-stepsize*obj_fn.grad(X, y, w), stepsize*obj_fn.l1)
        else:
            w -= stepsize*obj_fn.grad(X, y, w)
        wtab = np.vstack((wtab, w))

    return w, wtab


def SGD(X, y, w, obj_fn, max_iter, use_prox=False):
    """
    Stochastic Gradient Descent
    `max_iter` should normally be set to some integer multiple of the number of samples (note that
    for this algorithm it controls the number of data points iterated over, rather than the number
    of passes over the dataset (epochs) as in the vanilla gradient descent implementation).
    """
    util.check_input_dims(X, y, w)
    smoothness = util.smoothness(X, obj_fn.l2)
    wtab = np.copy(w)
    n = len(y)

    for k in range(max_iter):
        stepsize = 1./(smoothness*(int(1+k/n)**.6))
        i = np.random.randint(n)
        update = stepsize*obj_fn.grad(X, y, w, i)
        if use_prox:
            w = util.prox(w-update, stepsize*obj_fn.l1)
        else:
            w -= update
        if k%n == 0: # 1 epoch = n iterations
            wtab = np.vstack((wtab, w))

    return w, wtab


def SAGA(X, y, w, obj_fn, max_iter, strong_convexity=None):
    """
    Implementation of the SAGA gradient method introduced in the 2007 paper 'SAGA: A Fast
    Incremental Gradient Method with Support for Non-Strongly Convex Composite Objectives'
    (arXiv:1407.0202)
    """
    util.check_input_dims(X, y, w)
    n = len(y)
    smoothness = util.smoothness(X, obj_fn.l2)
    if strong_convexity is None:
        strong_convexity = obj_fn.l2
    stepsize = 1/(2*(strong_convexity*n+smoothness))

    wtab = np.copy(w)
    gtab = np.vstack([obj_fn.grad(X, y, w, i) for i in range(n)])

    for k in range(max_iter):
        j = np.random.randint(n)
        update_vec = obj_fn.grad(X, y, w, j)-gtab[j]+np.mean(gtab, axis=0)
        w = util.prox(w-stepsize*update_vec, stepsize*.1)
        if k%n == 0:
            wtab = np.vstack((wtab, w))

    return w, wtab


def SVRG(X, y, w, obj_fn, max_iter, strong_convexity=None, use_prox=False):
    """
    Stochastic Variance-Reduced Gradient method, from the paper 'Accelerating Stochastic
    Gradient Descent using Predictive Variance Reduction', published in NIPS 2013
    """
    util.check_input_dims(X, y, w)
    smoothness = util.smoothness(X, obj_fn.l2)
    if strong_convexity is None:
        strong_convexity = obj_fn.l2
    M = int(1.1*smoothness/strong_convexity)
    stepsize = 1./smoothness

    wtab = np.copy(w)

    for _ in range(max_iter):
        v0 = w
        v = np.copy(v0)
        grad_tmp = obj_fn.grad(X, y, v)
        for __ in range(M):
            i = np.random.randint(len(y))
            update_vec = obj_fn.grad(X, y, v, i)-obj_fn.grad(X, y, v0, i)+grad_tmp
            if use_prox:
                v = util.prox(v-stepsize*update_vec, stepsize*obj_fn.l1)
            else:
                v -= stepsize*update_vec
        w = v
        wtab = np.vstack((wtab, w))

    return w, wtab


def NM(X, y, w, obj_fn, max_iter, stopping_eps=1e-5):
    """
    Newton's method for logistic regression. The args `X`, `y`, `w`, `obj_fn`, `max_iter`, and
    `smoothness` have the same definitions as for gradient descent.

    Parameters
    ----------
    stopping_eps:  float
        Precision to be used in the stopping rule
    """
    util.check_input_dims(X, y, w)
    smoothness = util.smoothness(X, obj_fn.l2)
    stepsize = 1./smoothness
    
    wtab = np.copy(w)

    for _ in range(max_iter):
        g = obj_fn.grad(X, y, w)
        direction = np.linalg.solve(obj_fn.hess(X, y, w), g)
        w -= stepsize*direction
        wtab = np.vstack((wtab, w))
        if .5*np.dot(g, direction) <= stopping_eps:
            break

    return w, wtab


def BFGS(X, y, w, obj_fn, max_iter, smoothness, stopping_eps=1e-5):
    """
    Implementation of the Broyden-Fletcher-Goldfarb-Shanno algorithm, a type of quasi-Newton method,
    with the stopping rule based on the absolute change in the value of the objective function
    """
    util.check_input_dims(X, y, w)
    wtab = np.copy(w)
    H = np.eye(w.size) # initialise Hessian approximation
    stepsize = 1./smoothness

    for k in range(max_iter):
        w_prev = np.copy(w)
        g = obj_fn.grad(X, y, w)
        direction = np.dot(H, g)
        #print(f"\nIteration {k}: 2-norms; g:{dbnorm(g)}, d:{dbnorm(direction)}")
    
        # Newton-method style weight update
        w -= stepsize*direction
        wtab = np.vstack((wtab, w))

        # precision check for stopping condition
        current_objval, prev_objval = obj_fn(X, y, w), obj_fn(X, y, wtab[k])
        #print(f"Stopping condition check: current={current_objval}, previous={prev_objval}")
        if util.bfgs_stopcondition(current_objval, prev_objval, stopping_eps):
            print(f"BFGS optimiser: stopping condition reached at iteration {k}\n")
            break

        # update Hessian approximation
        try:
            H -= util.bfgs_hessapprox_update(H, w-w_prev, obj_fn.grad(X, y, w)-g)
        except TypeError:
            print(f"BFGS optimiser: Zero-displacement curvature at iteration {k} Hessian update: stopping descent")
            break

    return w, wtab


def LBFGS(X, y, w, obj_fn, max_iter, smoothness, memory_size=10, stopping_eps=1e-5):
    """
    This is a variant of the BFGS algorithm adapted for storage efficiency - it uses an implicit 
    representation of the update term for the Hessian approximation that gives it a linear memory
    requirement. It keeps a record of previous updates of the gradient and directly approximates
    the Hessian-gradient product instead of storing separate versions of both
    """
    util.check_input_dims(X, y, w)
    wtab = np.copy(w)
    stepsize = 1.0/smoothness
    p = w.size
    H = np.eye(p)
    z = obj_fn.grad(X, y, w) # initialisation of grad-Hess product
    disp, grad_disp = {}, {}

    for k in range(max_iter):
        w_prev = np.copy(w)
        g = obj_fn.grad(X, y, w)
        w -= stepsize*z
        disp[k%memory_size] = w-w_prev
        grad_disp[k%memory_size] = obj_fn(X, y, w)-g
        if k < memory_size:
            try:
                H -= util.bfgs_hessapprox_update(H, disp[k], grad_disp[k])
            except TypeError:
                print(f"L-BFGS optimiser: Zero-displacement curvature caught at iteration {k} Hessian update: stopping descent")
                break
            z = np.dot(H, g)
        else:
            z = util.lbfgs_recursion(k, memory_size, g, p, disp, grad_disp)
        wtab = np.vstack((wtab, w))
        current_objval, prev_objval = obj_fn(X, y, w), obj_fn(X, y, wtab[k-1])
        if util.bfgs_stopcondition(current_objval, prev_objval, stopping_eps):
            print(f"L-BFGS optimiser: stopping condition reached at iteration {k}\n")
            break

    return w, wtab


def SLBFGS(X, y, w, obj_fn, max_iter, smoothness, n_updates, memory_size, n_curve_updates, batch_size_grad, batch_size_hess, stopping_eps=1e-5):
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
    wtab = np.copy(w)
    stepsize = 1./smoothness
    n, p = y.size, w.size
    H = np.eye(p)
    x = w
    disp, grad_disp = {}, {}

    r = 0 # iterator for the Hessian curvature estimation updates
    for k in range(max_iter):
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
                x = np.mean(vtab, 0)#.reshape(p, 1) # reshape for dot product in Hessian
                disp[r] = x-x_prev

                # update gradient-Hessian product approximation
                grad_disp[r] = disp[r]*np.mean(
                    np.array(
                        [obj_fn.hess(X, y, x, j) for j in np.random.randint(n, size=batch_size_hess)]
                    )
                )

                if r < memory_size: # first `filling` of the memory with product estimations
                    try:
                        H -= util.bfgs_hessapprox_update(H, disp[r], grad_disp[r])
                    except TypeError:
                        print(f"Stochastic L-BFGS optimiser: Zero-displacement curvature caught at iteration {k} Hessian update: stopping descent")
                        break
                else:
                    H = util.lbfgs_recursion(r, memory_size, obj_fn.grad(X, y, x), p, disp, grad_disp)
                r += 1
        w = vtab[np.random.randint(n_updates)]
        wtab = np.vstack((wtab, w))
        current_objval, prev_objval = obj_fn(X, y, w), obj_fn(X, y, wtab[k-1])
        if util.bfgs_stopcondition(current_objval, prev_objval, stopping_eps):
            print(f"Stochastic L-BFGS optimiser: stopping condition reached at iteration {k}\n")
            break
    
    return w, wtab


def QP_KSVM(X, y, soft_margin_coef=1.0, kernel_fn=None, **kernelparams):
    """
    Uses cvxopt to find the optimal SVM hyperplane by solving the dual optimisation problem
    as a quadratic program - the solver returns the Lagrange multipliers of the dual
    problem.
    
    This function will return the hyperplane weights only in the linear kernel
    case, otherwise it will return the Lagrange multipliers, as computing the hyperplane
    explicitly requires projecting into the feature space on which the kernel can be defined
    as an inner product; for all non-trivial kernels, the hyperplane will be computed implicitly
    in the inner product space at prediction time.
    
    The `soft_margin_coef` argument can be set to None to do hard-margin classification.
    """
    from cvxopt import matrix
    from cvxopt.solvers import qp, options
    import kernels

    if kernel_fn is None:
        kernel_fn = kernels.linear
    
    n = X.shape[0]
    K = util.gram(X, kernel_fn, **kernelparams)
    
    I = np.eye(n)
    G, h = matrix(-1*I), np.zeros(n)

    options["show_progress"] = False
    qp_sol = np.ravel(qp(
        P=matrix(np.outer(y, y)*K),
        q=matrix(-1*np.ones(n)),
        G=matrix(G if soft_margin_coef is None else np.vstack((G, I))),
        h=matrix(h if soft_margin_coef is None else np.vstack((h, soft_margin_coef*np.ones(n))), (2*n, 1), "d"),
        A=matrix(y, (1, n), "d"),
        b=matrix(0.0)
    )["x"])

    # get support vectors and bias term
    is_sv = qp_sol > 1e-5
    sv_lms, support_vectors, sv_labels = qp_sol[is_sv], X[is_sv], y[is_sv]
    bias = np.mean(sv_labels-np.sum(sv_lms*sv_labels*np.array([K[is_sv,j] for j in np.arange(n)[is_sv]]), 0))

    res = np.sum(sv_lms*sv_labels*support_vectors, 1) if kernel_fn is kernels.linear else sv_lms

    return res, bias