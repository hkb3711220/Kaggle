#/1506.01186.pdf

from keras import backend as K
from keras.callbacks import Callback
import numpy as np

#https://arxiv.org/pdf/1608.03983.pdf
#/1506.01186.pdf

class cyclr(Callback):

    def __init__(self, max_lr, base_lr, step_size, gamma=None, mode='triangular'):

        self.max_lr =  max_lr
        self.base_lr = base_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        self.clr_iteration = 0
        self.trn_iteration = 0

        if self.mode == 'triangular':
            self.scale_fn = lambda x:1
        elif self.mode == 'triangular2':
            self.scale_fn = lambda x: 1/(2.**(x-1))
        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: self.gamma**(x)

        self.history = {}
        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):

        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):

        cycle = np.floor(1 + self.clr_iteration/(2*self.step_size))
        x = np.abs(self.clr_iteration/self.step_size-2*cycle)
        if self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x))*self.scale_fn(self.clr_iteration)
        else:
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x))*self.scale_fn(cycle)

        return lr

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs={}):
        logs = logs or {}
        self.trn_iteration +=1
        self.clr_iteration +=1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
