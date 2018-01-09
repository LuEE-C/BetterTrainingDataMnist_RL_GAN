import os

import numba as nb
import numpy as np
import math

import keras.backend as K
from PriorityExperienceReplay.PriorityExperienceReplay import Experience

import tensorflow as tf

from Actor import ActorNetwork
from Critic import CriticNetwork

from Environnement.Environnement import Environnement

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Agent:
    def __init__(self, amount_per_class=10):
        self.environnement = Environnement(amount_per_class=amount_per_class)
        self.batch_size = amount_per_class * 10

        # Bunch of placeholders values
        self.dummy_value = np.zeros((self.batch_size, 1))
        self.dummy_predictions = np.zeros((self.batch_size, 784))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_learning_phase(1)
        K.set_session(self.sess)

        self.atoms = 51
        self.v_max = 2
        self.v_min = -2
        self.delta_z = (self.v_max - self.v_min) / float(self.atoms - 1)
        self.z_steps = np.array([self.v_min + i * self.delta_z for i in range(self.atoms)]).astype(np.float32)

        self.actor = ActorNetwork(self.sess, 28*28 + 10, 28*28, tau=0.001, lr=5*10e-5)
        self.critic = CriticNetwork(self.sess, 28*28 + 10, 28*28, tau=0.001, lr=5*10e-5)
        self.memory = Experience(memory_size=1000000, batch_size=self.batch_size*10, alpha=0.5)

    def train(self, epoch):

        e = 0
        while e <= epoch:
            done = False
            print('Epoch :', e)
            batch_num = 0
            while self.memory.tree.size < 10000:
                self.add_values_to_memory()

            while done is False:
                if batch_num % 4 == 0:
                    self.add_values_to_memory()
                self.train_loop()
                batch_num += 1

                if batch_num % (100000//self.batch_size) == 0:
                    batch_x, batch_f1, batch_y = self.environnement.query_state()
                    batch_y_prime = self.flatten_digit_class(batch_y)
                    pred_x = np.reshape(batch_x, (self.batch_size, 28 * 28))
                    pred_x = np.concatenate([pred_x, batch_y_prime], axis=1)
                    old_predictions = self.actor.model.predict([pred_x])
                    values, test_values = self.get_values(np.reshape(batch_x, old_predictions.shape) + 2 * old_predictions, batch_f1, batch_y)
                    print('Batch num :', batch_num, '\tValues :', np.mean(values), '\tTest values :', np.mean(test_values))

            e += 1

    def train_loop(self):
        states, actions, reward, indices = self.make_dataset()
        loss = self.critic.model.train_on_batch([states, actions], reward)

        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        # self.actor.target_train()
        # self.critic.target_train()
        self.memory.priority_update(indices, [loss for _ in range(len(indices))])

    def add_values_to_memory(self):
        batch_x, batch_f1, batch_y = self.environnement.query_state()
        batch_y_prime = self.flatten_digit_class(batch_y)
        pred_x = np.reshape(batch_x, (self.batch_size, 28 * 28))
        pred_x = np.concatenate([pred_x, batch_y_prime], axis=1)
        old_predictions = self.actor.model.predict(pred_x)
        actions, new_predictions = self.get_actions(old_predictions, batch_x)
        values, test = self.get_values(new_predictions, batch_f1, batch_y)

        for i in range(pred_x.shape[0]):
            self.memory.add([pred_x[i], actions[i], values[i]], 5)

    @nb.jit
    def flatten_digit_class(self, batch_y):
        batch_y_prime = np.zeros(shape=(self.batch_size, 10))
        for i in range(batch_y.shape[0]):
            batch_y_prime[batch_y[i]] = 1
        return batch_y_prime

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
        actions = old_predictions # + np.random.normal(loc=0, scale=1, size=old_predictions.shape)
        new_predictions = np.reshape(old_state, (actions.shape)) + actions
        actions = np.clip(actions, -1, 1)
        new_predictions = np.clip(new_predictions, -1, 1)
        return actions, new_predictions

    def make_dataset(self):
        data, weights, indices = self.memory.select(0.6)
        states, reward, actions = [], [], []
        for i in range(self.batch_size):
            states.append(data[i][0])
            actions.append(data[i][1])
            reward.append(data[i][2])
        states = np.array(states)
        reward = np.array(reward)
        actions = np.array(actions)
        return states, actions, reward, indices

    @nb.jit
    def update_m_prob(self, reward, m_prob, z):
        for i in range(self.batch_size):
            for j in range(self.atoms):
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[i,  int(m_l)] += z[i, j] * (m_u - bj)
                m_prob[i,  int(m_u)] += z[i, j] * (bj - m_l)

if __name__ == '__main__':
    agent = Agent(amount_per_class=1)
    agent.train(epoch=1)
