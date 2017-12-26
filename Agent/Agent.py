import os

import numba as nb
import numpy as np
import pickle

import keras.backend as K
from keras.layers import Input, Dense, PReLU, Flatten, Concatenate, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

from NoisyDense import NoisyDense
from DenseNet import DenseNet

from Environnement.Environnement import Environnement

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def policy_loss(actual_value, predicted_value, old_prediction, mask):
    advantage = actual_value - predicted_value

    def loss(y_true, y_pred):
        prob = K.mean(mask * (K.square(y_pred - y_true)))
        old_prob = K.mean(mask * (K.square(old_prediction - y_true)))

        # Prob instead of logprob cause logprob(0) is all over the place
        #log_prob = K.mean(K.log(prob + 1e-10))

        r = K.mean(prob / (old_prob + 1e-10))
        return - prob * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))
    return loss


class Agent:
    def __init__(self, training_epochs=10, from_save=False, amount_per_class=10):
        self.training_epochs = training_epochs
        self.environnement = Environnement(amount_per_class=amount_per_class)
        self.batch_size = amount_per_class * 10
        self.noise = .55

        # Bunch of placeholders values
        self.dummy_value = np.zeros((self.batch_size, 1))
        self.dummy_predictions = np.zeros((self.batch_size, 784))

        self.critic_losses, self.policy_losses = [], []

        self.replay_policy_losses, self.replay_critic_losses = [], []
        self.values, self.test_values = [], []
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        if from_save is True:
            self.actor.load_weights('actor')
            self.critic.load_weights('critic')

    def _build_actor(self):

        state_input = Input(shape=(28, 28, 1))

        # Used for loss function
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_predictions = Input(shape=(784,))
        mask = Input(shape=(784,))

        class_input = Input(shape=(1,))
        digit_class = Dense(128)(class_input)
        digit_class = PReLU()(digit_class)
        digit_class = Dense(128)(digit_class)
        digit_class = PReLU()(digit_class)

        main_network = Conv2D(64, (3,3), padding='same')(state_input)
        main_network = PReLU()(main_network)
        main_network = MaxPooling2D()(main_network)
        main_network = Conv2D(128, (3,3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = MaxPooling2D()(main_network)

        main_network = Flatten()(main_network)

        main_network = Concatenate()([main_network, digit_class])

        main_network = Dense(256)(main_network)
        main_network = PReLU()(main_network)

        actor = Dense(512)(main_network)
        actor = PReLU()(actor)
        # actor = NoisyDense(28 * 28, activation='tanh', sigma_init=0.5, name='actor_output')(actor)
        actor = Dense(28 * 28, activation='tanh')(actor)

        actor_model = Model(inputs=[state_input, actual_value, predicted_value, old_predictions, class_input, mask],
                             outputs=[actor])
        actor_model.compile(optimizer=Adam(),
                             loss=[policy_loss(actual_value=actual_value,
                                               predicted_value=predicted_value,
                                               old_prediction=old_predictions,
                                               mask=mask
                                               )
                                   ])

        actor_model.summary()
        return actor_model

    def _build_critic(self):

        state_input = Input(shape=(28, 28, 1))

        class_input = Input(shape=(1,))
        digit_class = Dense(128)(class_input)
        digit_class = PReLU()(digit_class)
        digit_class = Dense(128)(digit_class)
        digit_class = PReLU()(digit_class)

        action_input = Input(shape=(784,))
        action = Dense(256)(action_input)
        action = PReLU()(action)
        action = Dense(128)(action)
        action = PReLU()(action)


        main_network = Conv2D(64, (3,3), padding='same')(state_input)
        main_network = PReLU()(main_network)
        main_network = MaxPooling2D()(main_network)
        main_network = Conv2D(128, (3,3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = MaxPooling2D()(main_network)
        main_network = Conv2D(256, (3,3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = Flatten()(main_network)

        main_network = Concatenate()([main_network, digit_class, action])

        main_network = Dense(256)(main_network)
        main_network = PReLU()(main_network)

        critic = Dense(512)(main_network)
        critic = PReLU()(critic)
        critic = Dense(1)(critic)

        critic_model = Model(inputs=[state_input, class_input, action_input],
                             outputs=[critic])
        critic_model.compile(optimizer=Adam(),
                             loss='mse')

        critic_model.summary()
        return critic_model

    def print_average_weight(self):
        mean = np.mean(self.actor.get_layer('actor_output').get_weights()[1])
        # The exploration noise is relative to the learned amount of noise
        self.noise = 0.05 + .5 * mean
        return mean

    def train(self, epoch):

        value_list, test_values_list, policy_losses, critic_losses, classification_losses = [], [], [], [], []
        e = 0
        while e <= epoch:
            done = False
            print('Epoch :', e)
            batch_num = 0
            while done is False:

                batch_x, batch_f1, batch_y = self.environnement.query_state()
                old_predictions = self.actor.predict([batch_x, self.dummy_value, self.dummy_value, self.dummy_predictions, batch_y, self.dummy_predictions])
                actions, new_predictions, mask = self.get_actions(old_predictions, batch_x)
                predicted_values = self.critic.predict([batch_x, batch_y, mask])
                values, test = self.get_values(new_predictions, batch_f1, batch_y)
                value_list.append(np.mean(values))
                test_values_list.append(np.mean(test))

                tmp_loss = np.zeros(shape=(self.training_epochs, 2))
                for i in range(self.training_epochs):
                    tmp_loss[i, 0 ] = self.actor.train_on_batch([batch_x, values, predicted_values, old_predictions, batch_y, mask], [actions])
                    tmp_loss[i, 1] = self.critic.train_on_batch([batch_x, batch_y, mask], [values])

                # self.actor.get_layer("actor_output").sample_noise()
                policy_losses.append(np.mean(tmp_loss[:,0]))
                critic_losses.append(np.mean(tmp_loss[:,1]))

                if batch_num % (100000//self.batch_size) == 0:
                    self.noise = 0.05 + 0.25 / ((np.mean(value_list) + 3) ** 2)
                    self.save_losses(critic_losses, policy_losses, value_list, test_values_list)
                    self.print_most_recent_losses(batch_num, e, value_list, test_values_list)
                    value_list, test_values_list, policy_losses, critic_losses = [], [], [], []

                    self.actor.save_weights('actor')
                    self.critic.save_weights('critic')

                batch_num += 1
            e += 1


    def save_losses(self, critic_losses, policy_losses, values, test_values):
        self.critic_losses.append(np.mean(critic_losses))
        self.policy_losses.append(np.mean(policy_losses))
        self.values.append(np.mean(values))
        self.test_values.append(np.mean(test_values))
        pickle.dump(self.critic_losses, open('critic_loss.pkl', 'wb'))
        pickle.dump(self.policy_losses, open('policy_loss.pkl', 'wb'))
        pickle.dump(self.values, open('values.pkl', 'wb'))
        pickle.dump(self.test_values, open('test_values.pkl', 'wb'))

    def print_most_recent_losses(self, batch_num, e, value_list, test_values_list):
        print()
        print('Batch number :', batch_num, '\tEpoch :', e, '\tAverage values :', np.mean(value_list), '\tAverage test values :', np.mean(test_values_list))
        print('Policy losses :', '%.5f' % np.mean(self.policy_losses[-(100000//self.batch_size):]),
              '\tCritic losses :', '%.5f' % np.mean(self.critic_losses[-(100000//self.batch_size):]),
              '\tCurrent noise :', '%.5f' % self.noise,
              # '\tAverage Noisy Layer :','%.5f' % self.print_average_weight()
              )


    @nb.jit
    def get_values(self, actions, batch_f1, batch_y):
        class_values, test = self.environnement.get_values(actions, batch_y)
        class_values -= batch_f1
        values = np.zeros(batch_y.shape)
        test -= batch_f1

        for i in range(values.shape[0]):
            values[i] = class_values[int(batch_y[i])]

        normalizing_factor = np.nanmean(batch_f1)/2

        return values/normalizing_factor, test/normalizing_factor


    @nb.jit
    def get_actions(self, old_predictions, old_state):
        action = np.random.random_integers(0, 28*28 - 1, (old_predictions.shape[0], 1))
        actions = old_predictions
        mask = np.zeros(old_predictions.shape)
        new_predictions = np.reshape(old_state, (actions.shape))
        for i in range(action.shape[0]):
            actions[i, action[i]] += np.random.normal(loc=0, scale=self.noise, size=(1,))
            # Adding twice the amount instead of once so it can bring a 1 to a -1 and vice versa
            new_predictions[i, action[i]] += 2*actions[i, action[i]]
            mask[i, action[i]] = 1
        actions = np.clip(actions, -1, 1)
        new_predictions = np.clip(new_predictions, -1, 1)
        return actions, new_predictions, mask


if __name__ == '__main__':
    agent = Agent(amount_per_class=10)
    agent.train(epoch=5000)
