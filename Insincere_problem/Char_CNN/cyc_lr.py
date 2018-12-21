#https://arxiv.org/pdf/1608.03983.pdf
#/1506.01186.pdf

from keras import backend as K
from keras.callbacks import LearningRateScheduler
import numpy as np

def cyclical_learning_rates(iteration, step_size, max_lr, min_lr):

    def schedule(epoch):
        cycle = np.floor(1 + iteration/(2*step_size))
        x = np.abs(iteration/step_size-2*cycle)
        lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1-x))

        return lr 

    return LearningRateScheduler(schedule)
