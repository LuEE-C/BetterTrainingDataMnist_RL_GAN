import os

import keras.backend as K
from keras import losses
import numba as nb
import numpy as np
from keras.layers import Input, Dense, Flatten, PReLU, BatchNormalization, Conv2D, UpSampling2D, MaxPool2D
from keras.models import Model

from Environnement.Environnement import Environnement
from LSTM_Model import LSTM_Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def policy_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value

    # Maybe some halfbaked normalization would be nice
    # something like advantage = advantage + 0.1 * advantage/(K.std(advantage) + 1e-10)

    # Fullbaked norm seems very unstable
    # advantage /= (K.std(advantage) + 1e-10)
    def loss(y_true, y_pred):
        prob = K.sum(y_pred * y_true)
        old_prob = K.sum(old_prediction * y_true)
        log_prob = K.mean(K.log(prob + 1e-10))

        r = K.mean(prob / (old_prob + 1e-10))

        entropy = K.sum(y_pred * K.log(y_pred + 1e-10), axis=-1)

        # So every element still looks like himself, iunno if that is all that needed
        mse_loss = losses.mean_squared_error(y_true=y_true, y_pred=y_pred)

        return mse_loss -log_prob * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage)) + 0.01 * entropy
    return loss


class Agent:
    def __init__(self, training_epochs=10, from_save=False, amount_per_class=10, amount_of_examples=1000):
        self.training_epochs = training_epochs
        self.environnement = Environnement(amount_per_class=amount_per_class, amount_of_examples=amount_of_examples)
        self.batch_size = amount_per_class * 10

        self.training_data = [[], [], [], []]

        # Bunch of placeholders values
        self.dummy_value = np.zeros((self.batch_size, 1))
        self.dummy_predictions = np.zeros((self.batch_size, 28, 28, 1))

        self.actor_critic = self._build_actor_critic()

        if from_save is True:
            self.actor_critic.load_weights('actor_critic')

    def _build_actor_critic(self):

        state_input = Input(shape=(28, 28, 1))

        # Used for loss function
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_predictions = Input(shape=(28, 28, 1))

        main_network = Conv2D(256, (3,3), padding='same')(state_input)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = MaxPool2D()(main_network)

        main_network = Conv2D(128, (3,3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = MaxPool2D()(main_network)

        main_network = Conv2D(64, (3,3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = UpSampling2D()(main_network)

        main_network = Conv2D(128, (3,3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)

        actor_next_word = UpSampling2D()(main_network)
        actor_next_word = Conv2D(1, (3,3), padding='same', activation='sigmoid')(actor_next_word)

        critic = Flatten()(main_network)

        critic_value = Dense(1)(critic)

        actor_critic = Model(inputs=[state_input, actual_value, predicted_value, old_predictions], outputs=[actor_next_word, critic_value])
        actor_critic.compile(optimizer='adam',
                      loss=[policy_loss(actual_value=actual_value,
                                        predicted_value=predicted_value,
                                        old_prediction=old_predictions
                                        ),
                            'mse'
                            ])

        actor_critic.summary()
        return actor_critic

    def train(self, epoch):

        value_list, policy_losses, critic_losses = [], [], []
        e = 0
        while e <= epoch:
            done = False
            print('Epoch :', e)
            batch_num = 0
            while done == False:


                batch_x, batch_f1, batch_y, done = self.environnement.query_state()
                old_predictions, predicted_values = self.actor_critic.predict([batch_x, self.dummy_value, self.dummy_value, self.dummy_predictions])
                actions = get_actions(old_predictions)
                values = self.get_values(actions, batch_f1, batch_y)
                value_list.append(np.mean(values))


                tmp_loss = np.zeros(shape=(self.training_epochs, 2))
                for i in range(self.training_epochs):
                    tmp_loss[i] = (self.actor_critic.train_on_batch([batch_x, values, predicted_values, old_predictions],
                                                                    [actions, values])[1:])
                policy_losses.append(np.mean(tmp_loss[:,0]))
                critic_losses.append(np.mean(tmp_loss[:,1]))


                if batch_num % 500 == 0:
                    print()
                    self.actor_critic.save_weights('actor_critic')
                    print('Batch number :', batch_num, '\tEpoch :', e, '\tAverage values :', np.mean(value_list))
                    print('Policy losses :', '%.5f' % np.mean(policy_losses),
                          '\tCritic losses :', '%.5f' % np.mean(critic_losses))
                    value_list, policy_losses, critic_losses = [], [], []

                batch_num += 1
            e += 1

    @nb.jit
    def get_values(self, actions, batch_f1, batch_y):
        class_values = self.environnement.get_values(actions, batch_y)
        class_values -= batch_f1
        values = np.zeros(batch_y.shape)

        for i in range(values.shape[0]):
            values[i] = class_values[int(batch_y[i])]
        return values


@nb.jit(nb.float32[:,:,:,:](nb.float32[:,:,:,:]))
def get_actions(old_predictions):
    actions = np.zeros((old_predictions.shape))
    # This can't be good
    for i in range(actions.shape[0]):
        for j in range(actions.shape[1]):
            for k in range(actions.shape[2]):
                actions[i,j,k,0] = np.random.choice([0,1], 1, p=[old_predictions[i,j,k,0], 1-old_predictions[i,j,k,0]])[0]
    return actions





if __name__ == '__main__':
    agent = Agent()
    agent.train(epoch=5000)