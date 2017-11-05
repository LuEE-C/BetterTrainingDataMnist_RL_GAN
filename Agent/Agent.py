import os

import numba as nb
import numpy as np
import pickle

import keras.backend as K
from keras.layers import Input, Dense, PReLU, BatchNormalization, Conv2D, Flatten
from keras.models import Model
from keras.optimizers import Adam

from NoisyDense import NoisyDense
from PriorityExperienceReplay import PriorityExperienceReplay

from Environnement.Environnement import Environnement

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def policy_loss(actual_value, predicted_value, old_prediction):
    advantage = actual_value - predicted_value

    def loss(y_true, y_pred):
        prob = K.mean(K.square(y_pred - y_true))
        old_prob = K.mean(K.square(old_prediction - y_true))
        log_prob = K.mean(K.log(prob + 1e-10))

        r = K.mean(prob / (old_prob + 1e-10))
        return - log_prob * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage))
    return loss


class Agent:
    def __init__(self, training_epochs=10, from_save=False, amount_per_class=10):
        self.training_epochs = training_epochs
        self.environnement = Environnement(amount_per_class=amount_per_class)
        self.priority_replay = PriorityExperienceReplay(500)
        self.batch_size = amount_per_class * 10
        self.noise = 0.5

        # Bunch of placeholders values
        self.dummy_value = np.zeros((self.batch_size, 1))
        self.dummy_predictions = np.zeros((self.batch_size, 784))

        self.critic_losses, self.policy_losses = [], []

        self.replay_policy_losses, self.replay_critic_losses = [], []
        self.values, self.test_values = [], []
        self.actor_critic = self._build_actor_critic()

        if from_save is True:
            self.actor_critic.load_weights('actor_critic')

    def _build_actor_critic(self):

        state_input = Input(shape=(28, 28, 1))

        # Used for loss function
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_predictions = Input(shape=(784,))

        main_network = Conv2D(128, (3,3), padding='same', strides=(2,2))(state_input)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)

        main_network = Conv2D(128, (3,3), padding='same', strides=(2,2))(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)

        main_network = Flatten()(main_network)

        actor = Dense(128)(main_network)
        actor = PReLU()(actor)
        actor = BatchNormalization()(actor)
        actor = Dense(128)(actor)
        actor = PReLU()(actor)
        actor = BatchNormalization()(actor)
        actor = NoisyDense(28*28, activation='tanh', name='actor_output')(actor)

        critic = Dense(128)(main_network)
        critic = PReLU()(critic)
        critic = BatchNormalization()(critic)
        critic = Dense(128)(critic)
        critic = PReLU()(critic)
        critic = BatchNormalization()(critic)
        critic = Dense(1)(critic)

        actor_critic = Model(inputs=[state_input, actual_value, predicted_value, old_predictions],
                             outputs=[actor, critic])
        actor_critic.compile(optimizer=Adam(),
                      loss=[policy_loss(actual_value=actual_value,
                                        predicted_value=predicted_value,
                                        old_prediction=old_predictions
                                        ),
                            'mse'
                            ])

        actor_critic.summary()
        return actor_critic

    def print_average_weight(self):
        return np.mean(self.actor_critic.get_layer('actor_output').get_weights()[1])

    def train(self, epoch):

        value_list, test_values_list, policy_losses, critic_losses, classification_losses = [], [], [], [], []
        e = 0
        while e <= epoch:
            done = False
            print('Epoch :', e)
            batch_num = 0
            while done is False:

                batch_x, batch_f1, batch_y = self.environnement.query_state()
                old_predictions, predicted_values = self.actor_critic.predict([batch_x, self.dummy_value, self.dummy_value, self.dummy_predictions])
                actions, new_predictions = get_actions(old_predictions, batch_x, self.noise)
                values, test = self.get_values(new_predictions, batch_f1, batch_y)
                value_list.append(np.mean(values))
                test_values_list.append(np.mean(test))

                tmp_loss = np.zeros(shape=(self.training_epochs, 2))
                for i in range(self.training_epochs):
                    tmp_loss[i] = (self.actor_critic.train_on_batch([batch_x, values, predicted_values, old_predictions],
                                                                    [actions, values])[1:])
                policy_losses.append(np.mean(tmp_loss[:,0]))
                critic_losses.append(np.mean(tmp_loss[:,1]))
                self.actor_critic.get_layer('actor_output').sample_noise()

                # self.priority_replay.add_elem(critic_losses[-1], [batch_x, values, predicted_values, old_predictions, actions, values, batch_y])


                if batch_num % 500 == 0:
                    # self.play_priority_replay(100)
                    self.save_losses(critic_losses, policy_losses, value_list, test_values_list)
                    self.print_most_recent_losses(batch_num, e, value_list, test_values_list)
                    value_list, test_values_list, policy_losses, critic_losses = [], [], [], []

                    self.actor_critic.save_weights('actor_critic')

                batch_num += 1
            e += 1

    def play_priority_replay(self, n=100):
        self.replay_policy_losses = []
        self.replay_critic_losses = []
        self.replay_classification_losses = []
        priority_batches = self.priority_replay.get_n_largest(n)

        for batches in priority_batches:
            for i in range(self.training_epochs):
                tmp_loss = np.zeros(shape=(self.training_epochs, 3))
                tmp_loss[i] = (self.actor_critic.train_on_batch(
                    [batches[1][0], batches[1][1], batches[1][2], batches[1][3]],
                    [batches[1][4], batches[1][5], batches[1][6]])[1:])
            self.replay_policy_losses.append(np.mean(tmp_loss[:, 0]))
            self.replay_critic_losses.append(np.mean(tmp_loss[:, 1]))
            self.replay_classification_losses.append(np.mean(tmp_loss[:, 2]))
            self.priority_replay.add_elem(self.replay_critic_losses[-1],
                                          [batches[1][0], batches[1][1], batches[1][2], batches[1][3],
                                           batches[1][4], batches[1][5], batches[1][6]])

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
        print('Policy losses :', '%.5f' % self.policy_losses[-1],
              '\tCritic losses :', '%.5f' % self.critic_losses[-1],
              '\tAverage Noisy Layer :','%.5f' % self.print_average_weight()
              # '\tReplay Policy losses :', '%.5f' % np.mean(self.replay_policy_losses),
              # '\tReplay Critic losses :', '%.5f' % np.mean(self.replay_critic_losses),
              # '\tReplay Classification losses :', '%.5f' % np.mean(self.replay_classification_losses)
              )


    @nb.jit
    def get_values(self, actions, batch_f1, batch_y):
        class_values, test = self.environnement.get_values(actions, batch_y)
        class_values -= batch_f1
        values = np.zeros(batch_y.shape)
        test -= batch_f1

        for i in range(values.shape[0]):
            values[i] = class_values[int(batch_y[i])]
        return values, test


@nb.jit
def get_actions(old_predictions, old_state, noise):
    actions = np.random.normal(loc=0, scale=noise, size=old_predictions.shape) + old_predictions
    actions = np.clip(actions, -1, 1)
    new_predictions = actions + np.reshape(old_state, (actions.shape))
    new_predictions = np.clip(new_predictions, -1, 1)
    return actions, new_predictions


if __name__ == '__main__':
    agent = Agent(amount_per_class=1)
    agent.train(epoch=5000)
