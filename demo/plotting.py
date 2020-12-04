import numpy as np
import matplotlib.pyplot as plt


def learning_curve(X, y, *models, **named_models):
    plt.figure(figsize=(20, 10))

    for model in models:
        plt.plot(get_curve(X, y, model))

    for name, model in named_models.items():
        plt.plot(get_curve(X, y, model), label=name)
        plt.legend(loc='best')
    
    plt.grid(True)
    plt.ylabel('Empirical Risk')
    plt.xlabel('Epochs')
    plt.title('Learning curves')

    plt.show()


def get_curve(X, y, model):
    if not hasattr(model, '_weights'):
        raise BaseException('Model has not been fitted yet')

    return [model.loss(w, X, y) for w in model._wtab]