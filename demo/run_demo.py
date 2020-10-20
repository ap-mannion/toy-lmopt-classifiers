import sys
from argparse import ArgumentParser
import datasets
from plotting import learning_curve


def main(args):
    # Load data
    loader = getattr(datasets, '_'.join(['load', args.dataset, 'data']))
    X_train, X_test, y_train, y_test = loader(file_path=args.data_fp)

    # Logistic Regression models
    lr_models = {}
    for solver in {'gd', 'sgd', 'saga', 'svrg', 'nm', 'bfgs', 'lbfgs', 'slbfgs'}:
        lr_models[solver] = LinearModel(loss_fn='logistic', solver=solver)
    

    for solver, model in lr_models.items():
        fit_kwargs = {'max_iter':args.epochs, 'smoothness':1.0}
        if solver in {'saga', 'svrg'}:
            fit_kwargs['strong_convexity'] = args.sconv
        elif solver == 'slbfgs':
            fit_kwargs['n_updates'] = 10
            fit_kwargs['memory_sixe'] = 10
            fit_kwargs['n_curve_updates'] = 10
            fit_kwargs['batch_size_grad'] = 32
            fit_kwargs['batch_size_hess'] = 32

        model.fit(X_train, y_train, **fit_kwargs)

    learning_curve(**lr_models)


if __name__ == '__main__':
    sys.path.append('..')
    from linear_model import LinearModel

    parser = ArgumentParser()
    parser.add_argument('dataset', type=str, choices={'student', 'bioseq'})
    parser.add_argument('data_fp', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--sconv', type=float, default=0.5)

    main(parser.parse_args())