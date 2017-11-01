import os

import numba as nb
import numpy as np
import pickle

import keras.backend as K
from keras.losses import mean_squared_error
from keras.layers import Input, Dense, Flatten, PReLU, BatchNormalization, Conv2D, UpSampling2D, MaxPool2D, GlobalMaxPooling2D
from keras.models import Model
from random import randint

from PriorityExperienceReplay import PriorityExperienceReplay
from Environnement.Environnement import Environnement

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
        mse = K.mean(mean_squared_error(y_true, y_pred))
        return mse - log_prob * K.mean(K.minimum(r * advantage, K.clip(r, min_value=0.8, max_value=1.2) * advantage)) + 0.01 * entropy
    return loss


class Agent:
    def __init__(self, training_epochs=10, from_save=False, amount_per_class=10):
        self.training_epochs = training_epochs
        self.environnement = Environnement(amount_per_class=amount_per_class)
        self.priority_replay = PriorityExperienceReplay(500)
        self.batch_size = amount_per_class * 10

        # Bunch of placeholders values
        self.dummy_value = np.zeros((self.batch_size, 1))
        self.dummy_predictions = np.zeros((self.batch_size, 28, 28, 1))

        self.actor_critic = self._build_actor_critic()
        self.critic_losses, self.policy_losses, self.classification_losses = [], [], []

        self.replay_policy_losses, self.replay_critic_losses, self.replay_classification_losses = [], [], []


        if from_save is True:
            self.actor_critic.load_weights('actor_critic')
        if os.path.isfile('pre_trained_actor_critic'):
            self.actor_critic.load_weights('pre_trained_actor_critic')
        else:
            self.pretrain_actor_critic()
            self.actor_critic.load_weights('pre_trained_actor_critic')

    def _build_actor_critic(self):

        state_input = Input(shape=(28, 28, 1))
        # class_input = Input(shape=(1,))

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

        main_network = Conv2D(128, (3,3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = UpSampling2D()(main_network)

        main_network = Conv2D(256, (3,3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = UpSampling2D()(main_network)

        actor = Conv2D(1, (3,3), padding='same', activation='sigmoid')(main_network)

        flat = GlobalMaxPooling2D()(main_network)

        critic = Dense(1)(flat)
        input_class = Dense(10, activation='softmax')(flat)

        actor_critic = Model(inputs=[state_input, actual_value, predicted_value, old_predictions], outputs=[actor, critic, input_class])
        actor_critic.compile(optimizer='adam',
                      loss=[policy_loss(actual_value=actual_value,
                                        predicted_value=predicted_value,
                                        old_prediction=old_predictions
                                        ),
                            'mse',
                            'sparse_categorical_crossentropy'
                            ])

        actor_critic.summary()
        return actor_critic

    def pretrain_actor_critic(self):
        state_input = Input(shape=(28, 28, 1))
        # class_input = Input(shape=(1,))

        # Used for loss function
        actual_value = Input(shape=(1,))
        predicted_value = Input(shape=(1,))
        old_predictions = Input(shape=(28, 28, 1))

        main_network = Conv2D(256, (3, 3), padding='same')(state_input)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = MaxPool2D()(main_network)

        main_network = Conv2D(128, (3, 3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = MaxPool2D()(main_network)

        main_network = Conv2D(128, (3, 3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = UpSampling2D()(main_network)

        main_network = Conv2D(256, (3, 3), padding='same')(main_network)
        main_network = PReLU()(main_network)
        main_network = BatchNormalization()(main_network)
        main_network = UpSampling2D()(main_network)

        actor = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(main_network)

        flat = GlobalMaxPooling2D()(main_network)

        critic = Dense(1)(flat)
        input_class = Dense(10, activation='softmax')(flat)

        actor_critic = Model(inputs=[state_input, actual_value, predicted_value, old_predictions],
                             outputs=[actor, critic, input_class])
        actor_critic.compile(optimizer='adam',
                             loss=['mse',
                                   'mse',
                                   'sparse_categorical_crossentropy'
                                   ])

        actor_critic.summary()

        x_train, y_train = self.environnement.get_whole_training_set()

        actor_critic.fit([x_train, np.zeros((x_train.shape[0], 1)), np.zeros((x_train.shape[0], 1)), np.zeros(x_train.shape)], [x_train, np.zeros((x_train.shape[0], 1)), y_train], batch_size=128, epochs=20, verbose=1)
        actor_critic.save_weights('pre_trained_actor_critic')


    def train(self, epoch):

        value_list, policy_losses, critic_losses, classification_losses = [], [], [], []
        e = 0
        while e <= epoch:
            done = False
            print('Epoch :', e)
            batch_num = 0
            while done == False:


                batch_x, batch_f1, batch_y, done = self.environnement.query_state()
                old_predictions, predicted_values, _ = self.actor_critic.predict([batch_x, self.dummy_value, self.dummy_value, self.dummy_predictions])
                actions = get_actions(old_predictions)
                values = self.get_values(actions, batch_f1, batch_y)
                value_list.append(np.mean(values))


                tmp_loss = np.zeros(shape=(self.training_epochs, 3))
                for i in range(self.training_epochs):
                    tmp_loss[i] = (self.actor_critic.train_on_batch([batch_x, values, predicted_values, old_predictions],
                                                                    [actions, values, batch_y])[1:])
                policy_losses.append(np.mean(tmp_loss[:,0]))
                critic_losses.append(np.mean(tmp_loss[:,1]))
                classification_losses.append(np.mean(tmp_loss[:,2]))
                # self.priority_replay.add_elem(critic_losses[-1], [batch_x, values, predicted_values, old_predictions, actions, values, batch_y])


                if batch_num % 50 == 0:
                    # self.play_priority_replay(100)
                    self.save_losses(critic_losses, policy_losses, classification_losses)
                    self.print_most_recent_losses(batch_num, e, value_list)
                    value_list, policy_losses, critic_losses, classification_losses = [], [], [], []

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

    def save_losses(self, critic_losses, policy_losses, classification_losses):
        self.critic_losses.append(np.mean(critic_losses))
        self.policy_losses.append(np.mean(policy_losses))
        self.classification_losses.append(np.mean(classification_losses))
        pickle.dump(self.critic_losses, open('critic_loss.pkl', 'wb'))
        pickle.dump(self.policy_losses, open('policy_loss.pkl', 'wb'))
        pickle.dump(self.classification_losses, open('classification_loss.pkl', 'wb'))

    def print_most_recent_losses(self, batch_num, e, value_list):
        print()
        print('Batch number :', batch_num, '\tEpoch :', e, '\tAverage values :', np.mean(value_list))
        print('Policy losses :', '%.5f' % self.policy_losses[-1],
              '\tCritic losses :', '%.5f' % self.critic_losses[-1],
              '\tClassification losses :', '%.5f' % self.classification_losses[-1],
              '\tReplay Policy losses :', '%.5f' % np.mean(self.replay_policy_losses),
              '\tReplay Critic losses :', '%.5f' % np.mean(self.replay_critic_losses),
              '\tReplay Classification losses :', '%.5f' % np.mean(self.replay_classification_losses))


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
    agent = Agent(amount_per_class=15)
    agent.train(epoch=5000)
