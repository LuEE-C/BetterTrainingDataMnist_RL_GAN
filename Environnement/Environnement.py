import numpy as np
import numba as nb
from lightgbm import LGBMClassifier
import os
from keras.datasets import mnist
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


class Environnement:
    def __init__(self, amount_per_class):
        self.amount_per_class = amount_per_class

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.astype(np.float32)
        self.x_test = self.x_test.astype(np.float32)

        self.x_train -= 127.5
        self.x_train /= 127.5
        self.x_test -= 127.5
        self.x_test /= 127.5

        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], 28, 28, 1))


        self.gbm_x_train = np.reshape(self.x_train, (self.x_train.shape[0], 28 * 28))
        self.gbm_x_test = np.reshape(self.x_test, (self.x_test.shape[0], 28 * 28))
        self.y_test, self.y_val = self.y_test[:self.gbm_x_test.shape[0]//3], self.y_test[self.gbm_x_test.shape[0]//3:]
        self.gbm_x_test, self.gbm_x_val = self.gbm_x_test[:self.gbm_x_test.shape[0]//3], self.gbm_x_test[self.gbm_x_test.shape[0]//3:]

        self.model = LGBMClassifier(objective='multiclass', num_class=10, n_jobs=1, min_child_samples=1,
                                    min_child_weight=0, min_data_in_bin=1, verbosity=-1, verbose=-1)

        self.class_indexes = [np.where(self.y_train == i) for i in range(10)]

    def get_values(self, actions, targets):
        if actions.mean() >= 1.0 or actions.mean() <= -1:
            print(actions.mean())
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        actions = np.reshape(actions, (actions.shape[0], 28*28))
        self.model.fit(actions, targets)
        pred_val = self.model.predict(self.gbm_x_val)
        pred_test = self.model.predict(self.gbm_x_test)
        val = f1_score(y_true=self.y_val, y_pred=pred_val, average=None)
        test = f1_score(y_true=self.y_test, y_pred=pred_test, average=None)

        return val, test

    def get_whole_training_set(self):
        return self.x_train, self.y_train

    def query_state(self):
        choices = np.array([np.random.choice(class_index[0], self.amount_per_class)
                            for class_index in self.class_indexes]).flatten()
        state = self.x_train[choices]
        targets = self.y_train[choices]
        self.model.fit(self.gbm_x_train[choices], self.y_train[choices])
        pred = self.model.predict(self.gbm_x_val)
        f1 = f1_score(y_true=self.y_val, y_pred=pred, average=None)

        return state, f1, targets

if __name__ == '__main__':
    env = Environnement(10)
    print(env.query_state())
