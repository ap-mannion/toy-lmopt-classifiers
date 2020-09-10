import numpy as np
import solvers
import kernels
# TODO:
# add cross-entropy loss


class LogisticLoss:

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, X, y, w):
        loss = np.log(1+np.exp(-y*np.dot(X, w)))
        if self.l1 != 0.0:
            loss += self.l1*np.linalg.norm(w, 1)
        if self.l2 != 0.0:
            loss += 0.5*self.l2*np.linalg.norm(w, 2)
        
        return np.mean(loss)

    def grad(self, X, y, w, batch=None):
        """
        Calculates the gradient of the loss for the given classifier weights.
        To calculate a mini-batch gradient, use the 'batch' argument to pass a list of indices to select,
        or a single integer to take the gradient at one data point only
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"""Number of samples in X ({X.shape[0]}) doesn't match the number
of samples in y ({y.shape[0]})""")
        if X.shape[1] != w.shape[0]:
            raise ValueError(f"""Dimension of X ({X.shape[1]}) doesn't match the dimension of w
({w.shape[0]})""")
        if batch is not None:
            X = X[batch,:]
            y = y[batch]

        res = sum(np.diag(-y/(1+np.exp(y*np.dot(X, w)))))/len(y)
        if X.shape[0] == 1:
            res *= X
        else:
            res = res@X

        return res+self.l2*w

    def hess(self, X, y, w, batch=None):
        """
        Calculates the Hessian matrix of the loss for the given classifier weights.
        Mini-batches can be implemented similarly as for the grad function.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"""Number of samples in X ({X.shape[0]}) doesn't match the number
of samples in y ({y.shape[0]})""")
        if X.shape[1] != w.shape[0]:
            raise ValueError(f"""Dimension of X ({X.shape[1]}) doesn't match the dimension of w
({w.shape[0]})""")
        if batch is not None:
            X = X[batch,:]
            y = y[batch]

        def _hessmat(X, y):
            expterm = np.exp(-y*np.dot(X, w))
            return (expterm/(1+expterm)**2)*np.outer(X, X)

        n = len(y)
        reg = self.l2
        if n > 1:
            d = X.shape[1]
            res = np.zeros((d, d))
            reg *= np.eye(d, d)
            for i in range(n):
                res += _hessmat(X[i,:], y[i])               
        else:
            res = _hessmat(X, y)
        
        return res+reg


class HingeLoss:

    def __init__(self, l2=0.05):
        self.l2 = l2

    def __call__(self, X, y, w):
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
        if len(w.shape) == 1:
            w = w.reshape((len(w), 1))
        n = X.shape[0]
        loss = np.empty(n)
        for i in range(n):
            loss[i] = max(0, 1-y[i]*np.dot(X[i,:], w))
        return np.sum(loss)+0.5*self.l2*np.linalg.norm(w, 2)

    def grad(self, X, y, w, batch=None):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"""Number of samples in X ({X.shape[0]}) doesn't match the number
of samples in y ({y.shape[0]})""")
        if X.shape[1] != w.shape[0]:
            raise ValueError(f"""Dimension of X ({X.shape[1]}) doesn't match the dimension of w
({w.shape[0]})""")
        if batch is not None:
            X = X[batch,:]
            y = y[batch]
        
        d = w.shape[0]
        res = np.empty(X.shape)
        for i in range(X.shape[0]):
            res[i,:] = y[i]*X[i,:] if y[i]*np.dot(X[i,:], w) >= 1 else 0

        return np.sum(res, 0)+self.l2*w

    def hess(self, w):
        d = w.shape[0]

        return np.zeros((d, d))


class LinearModel:

    def __init__(self, loss_fn, solver='gd', svmkernel='gaussian', l1=0.0, l2=0.01, max_iter=100):
        losses = {'logistic', 'hinge'}
        if loss_fn not in losses:
            raise ValueError('loss function argument must be one of %r'%losses)

        self.loss = LogisticLoss() if loss_fn == 'logistic' else HingeLoss()
        self.solver = getattr(solvers, solver.upper())
        self.svmkernel = getattr(kernels, svmkernel) if loss_fn == 'hinge' else None
        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter

    def fit(self, X, y, **solver_kwargs):
        """
        Fits the model to the training data X and labels y by minimising the empirical risk using the
        specified loss function & regularisation coefficients
        """
        self._weights, self._wtab = self.solver(np.zeros(X.shape[1]), self.loss, **solver_kwargs)
        self.loss_val = self.loss(X, y, self._weights)

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

