import numpy as np
import numba as nb
from lightgbm import LGBMClassifier
import os
from keras.datasets import mnist
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


class Environnement:
    def __init__(self, amount_per_class, amount_of_examples):
        self.dataset_x, self.dataset_y, self.dataset_f1, self.test_x, self.test_y = create_mnist_dataset(amount_per_class, amount_of_examples)
        self.index = 0
        self.dataset_x = np.reshape(self.dataset_x, (self.dataset_x.shape[0], self.dataset_x.shape[1], 28, 28, 1))
        # Keeping em positive
        self.dataset_x /= 255.0

        self.model = LGBMClassifier(objective='multiclass', num_class=10, n_jobs=1, min_child_samples=1,
                                    min_child_weight=0, min_data_in_bin=1, verbosity=-1, verbose=-1)

    def get_values(self, actions, targets):
        actions = np.reshape(actions, (actions.shape[0], 28*28)) * 255.0
        self.model.fit(actions, targets)
        pred = self.model.predict(self.test_x)
        return f1_score(self.test_y, pred, average=None)




    @nb.jit
    def query_state(self):

        state = self.dataset_x[self.index]
        f1 = self.dataset_f1[self.index]
        targets = self.dataset_y[self.index]

        self.index += 1
        # End of epoch, shuffle dataset for next epoch
        if self.index + 1 >= self.dataset_x.shape[0]:
            self.index = 0
            return state, f1, targets, True
        else:
            return state, f1, targets, False


def create_mnist_dataset(amount_per_class=50, how_many_training_examples=5000000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if os.path.isfile('dataset_x_' + str(amount_per_class) + '_' + str(how_many_training_examples) + '.npy'):
        dataset_x = np.load('dataset_x_' + str(amount_per_class) + '_' + str(how_many_training_examples) + '.npy')
        dataset_y = np.load('dataset_y_' + str(amount_per_class) + '_' + str(how_many_training_examples) + '.npy')
        dataset_f1 = np.load('dataset_f1_' + str(amount_per_class) + '_' + str(how_many_training_examples) + '.npy')

    else:
        x_train_gbm = np.reshape(x_train, (x_train.shape[0], 28 * 28))
        x_test_gbm = np.reshape(x_test, (x_test.shape[0], 28 * 28))
        dataset_x = np.zeros((how_many_training_examples, amount_per_class * 10, 28, 28))
        dataset_y = np.zeros((how_many_training_examples, amount_per_class * 10))
        dataset_f1 = np.zeros((how_many_training_examples, 10))

        class_indexes = [np.where(y_train == i) for i in range(10)]
        choices = [np.array([np.random.choice(class_index[0], amount_per_class) for class_index in class_indexes]).flatten() for _ in range(how_many_training_examples)]

        # Training with n_jobs >1 causes segfaults, lightgbm is faster then itself
        # This is definitely hardware dependant, has no issue on my laptop
        model = LGBMClassifier(objective='multiclass', num_class=10, n_jobs=1, min_child_samples=1,
                               min_child_weight=1e-10, min_data_in_bin=1, verbosity=-1, verbose=-1)
        # model = XGBClassifier(objective='multi:softmax', n_estimators=50)
        for i in range(how_many_training_examples):
            model.fit(x_train_gbm[choices[i]], y_train[choices[i]])
            pred = model.predict(x_test_gbm)
            dataset_x[i] = x_train[choices[i]]
            dataset_y[i] = y_train[choices[i]]
            dataset_f1[i] = f1_score(y_test, pred, average=None)
            print(i)

        np.save('dataset_x_' + str(amount_per_class) + '_' + str(how_many_training_examples), dataset_x)
        np.save('dataset_y_' + str(amount_per_class) + '_' + str(how_many_training_examples), dataset_y)
        np.save('dataset_f1_' + str(amount_per_class) + '_' + str(how_many_training_examples), dataset_f1)

    return dataset_x, dataset_y, dataset_f1, np.reshape(x_test, (x_test.shape[0], 28*28)), y_test

if __name__ == '__main__':
    env = Environnement(15, 50000)
    print(env.query_state(2))
