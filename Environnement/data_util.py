import numpy as np
import os
from time import time
from lightgbm import LGBMClassifier
from keras.datasets import mnist
from sklearn.metrics import f1_score, classification_report


def create_mnist_dataset(amount_per_class=50, how_many_training_examples=5000000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train_lgbm = np.reshape(x_train, (x_train.shape[0], 28 * 28))
    x_test_lgbm = np.reshape(x_test, (x_test.shape[0], 28 * 28))
    dataset_x = np.zeros((how_many_training_examples, amount_per_class * 10, 28, 28))
    dataset_y = np.zeros((how_many_training_examples, amount_per_class * 10))
    dataset_f1 = np.zeros((how_many_training_examples, 10))

    class_indexes = [np.where(y_train == i) for i in range(10)]


    start = time()
    model = LGBMClassifier(objective='multiclass', num_class=10)
    for i in range(how_many_training_examples):
        choices = [np.random.choice(class_index[0], amount_per_class) for class_index in class_indexes]
        choices = np.array(choices).flatten()

        model.fit(x_train_lgbm[choices], y_train[choices])
        pred = model.predict(x_test_lgbm)

        dataset_x[i] = x_train[choices]
        dataset_y[i] = y_train[choices]
        dataset_f1[i] = f1_score(y_test, pred, average=None)





    print(time() - start)


if __name__ == '__main__':
    create_mnist_dataset(10, 1000)