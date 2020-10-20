import numpy as np
import matplotlib.pyplot as plt


def learning_curve(*models, **named_models):
    plt.figure(figsize=(20, 10))

    for model in models:
        plt.plot(get_curve(model))

    for name, model in named_models.items():
        plt.plot(get_curve(model), label=name)
    
    plt.grid(True)
    plt.ylabel('Empirical Risk')
    plt.xlabel('Epochs')
    plt.title('Learning curves')

    plt.show()


def get_curve(model):
    if not hasattr(model, '_weights'):
        raise BaseException('Model has not been fitted yet')

    return [model.loss(w) for w in model._wtab]