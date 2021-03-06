import numpy as np
import util
import solvers
import kernels
# TODO:
#   - intercept terms in loss functions
#   - debug hinge GD

class LogisticLoss:

    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, w, X, y):
        logistic_input = y*np.dot(X, w)
        log_term = np.empty_like(logistic_input)
        pos_idx = logistic_input > 0.
        log_term[pos_idx] = -np.log(1.+np.exp(-logistic_input[pos_idx]))
        log_term[~pos_idx] = logistic_input[~pos_idx]-np.log(1.+np.exp(logistic_input[~pos_idx]))
        
        l1reg = self.l1*np.linalg.norm(w, 1) if self.l1 != 0. else 0.
        l2reg = 0.5*self.l2*np.dot(w, w) if self.l2 != 0. else 0.
        
        return -np.mean(log_term)+l1reg+l2reg

    def grad(self, X, y, w, batch=None):
        """
        Calculates the gradient of the loss for the given classifier weights.
        To calculate a mini-batch gradient, use the 'batch' argument to pass a list of indices to select,
        or a single integer to take the gradient at one data point only
        """
        util.check_input_dims(X, y, w)
        if batch is not None:
            X, y = X[batch], y[batch]

        logistic_term = y*(util.logistic_stable(y*np.dot(X, w))-1)
        if len(X.shape) == 1:
            res = logistic_term*X
        else:
            res = logistic_term@X
        if type(batch) is not int:
            res /= len(y)

        return res+self.l2*w

    def hess(self, X, y, w, batch=None):
        """
        Calculates the Hessian matrix of the loss for the given classifier weights.
        Mini-batches can be implemented similarly as for the grad function.
        """
        util.check_input_dims(X, y, w)
        if batch is not None:
            X, y = X[batch], y[batch]
        if len(X.shape) == 1:
            X.shape = (1, X.shape[0])

        logistic_terms = util.logistic_stable(y*np.dot(X, w))
        diag_terms = logistic_terms/(1-logistic_terms)
        D = np.diag(np.ravel(diag_terms))
        hessmat = np.dot(X.T, np.dot(D, X))

        # def _hessmat(X, y):
        #     expterm = np.exp(-y*np.dot(X, w))
        #     return (expterm/(1+expterm)**2)*np.outer(X, X)

        
        # reg = self.l2
        # if n > 1:
        #     d = X.shape[1]
        #     res = np.zeros((d, d))
        #     reg *= np.eye(d, d)
        #     for i in range(n):
        #         res += _hessmat(X[i,:], y[i])               
        # else:
        #     res = _hessmat(X, y)
        
        return hessmat+self.l2*np.eye(hessmat.shape[0])


class HingeLoss:

    def __init__(self, l2=0.05):
        self.l2 = l2

    def __call__(self, w, X, y):
        """
        Calculates the hinge loss for the given data & classification weights

        Parameters
        ----------
        X : numpy.ndarray
            Training data, size n x p (Gram matrix in the kernel SVM case)
        y : numpy.ndarray
            Classification labels corresponding to X, size 1 x n
        w : numpy.ndarray
            Weights calculated by the classifier, size p x 1
        """
        n = X.shape[0]
        loss = np.array([max(0., 1.-y[i]*np.dot(X[i], w)) for i in range(n)])

        return np.sum(loss)+.5*self.l2*np.dot(w, w)

    def grad(self, X, y, w, batch=None):
        util.check_input_dims(X, y, w)
        if batch is not None:
            X, y = X[batch], y[batch]

        try:
            res = np.empty_like(X)
            for i in range(X.shape[0]):
                res[i,:] = -y[i]*X[i] if y[i]*np.dot(X[i], w) >= 1 else np.zeros(X.shape[1])
        except IndexError:
            res = -y*X if y*np.dot(X, w) < 1 else np.zeros(len(w))

        return np.sum(res, 0)+self.l2*w


class LinearModel:

    def __init__(self, loss_fn, solver, svmkernel=None, l1=.01, l2=.01, max_iter=100):
        losses = {'logistic', 'hinge'}
        
        if loss_fn not in losses:
            raise ValueError(f'loss function argument must be one of {losses}')
        
        if loss_fn == 'hinge' and solver in {'nm', 'bfgs', 'lbfgs', 'slbfgs'}:
            # TODO: test svm with quasi-Newton methods and adjust this accordingly
            raise ValueError(f'the solver {solver.upper()}() is a second-order method which cannot be used for direct minimisation of the hinge loss function: try solver=`qp_ksvm`')
        elif loss_fn != 'hinge' and solver == 'qp_ksvm':
            raise ValueError('the quadratic programming solver only applies to SVM classification (loss_fn=`hinge`)')

        if svmkernel is not None and loss_fn != 'hinge':
            raise UserWarning('kernel function argument is applicable only to SVM classification (loss_fn=`hinge`)')
        elif svmkernel is None and loss_fn == 'hinge':
            svmkernel = 'linear'

        try:
            chosen_solver = getattr(solvers, solver.upper())
        except AttributeError:
            raise NameError('Invalid solver name')

        self.loss = LogisticLoss(l1, l2) if loss_fn == 'logistic' else HingeLoss(l2)
        self.solver = chosen_solver
        self.svmkernel = getattr(kernels, svmkernel) if loss_fn == 'hinge' else None
        self.max_iter = max_iter
        self.use_cvxopt_wrap_format = loss_fn == 'hinge' and solver == 'qp_ksvm'

    def fit(self, X, y, use_prox=False, random_init=False, **solver_kwargs):
        """
        Fits the model to the training data X and labels y by minimising the empirical risk using the
        specified loss function & regularisation coefficients
        """
        if not self.use_cvxopt_wrap_format:
            self._weights, self._wtab = self.solver(
                X, y,
                5*np.random.rand(X.shape[1]) if random_init else np.zeros(X.shape[1]),
                self.loss, **solver_kwargs
            )
        else:
            # currently the cvxopt wrapper needs this format
            self._weights, self._bias = self.solver(X, y, kernel_fn=self.svmkernel, **solver_kwargs)
        
        if self._weights.size == X.shape[1]: # check that it's not kernel SVM
            self.loss_val = self.loss(self._weights, X, y)

    def decision_function(self, X):
        """
        Predict confidence scores for the input data in X
        """
        if not hasattr(self, '_weights'):
            raise BaseException('Model has not been fitted yet')
        
        return 1.0/(1+np.exp(-np.dot(X, self._weights)))

    def predict(self, X, probas=None):
        """
        Outputs the model's class labels for the input data in X
        """
        if not hasattr(self, '_weights'):
            raise BaseException('Model has not been fitted yet')
        if probas is None:
            probas = self.decision_function(X)

        return np.array([[-1, 1][p >= 0.5] for p in probas])

