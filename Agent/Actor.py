from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Conv2D, Flatten, Reshape
from keras.models import Model
import tensorflow as tf
import keras.backend as K
import numpy as np
from NoisyDense import NoisyDense


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, tau, lr):
        self.sess = sess
        self.tau = tau
        self.lr = lr
        self.state_size = state_size
        self.action_size = action_size

        K.set_session(sess)

        self.model, self.weights, self.state = self.create_actor_network()
        # self.target_model, self.target_weights, self.target_state = self.create_actor_network()
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    # def target_train(self):
    #     actor_weights = self.model.get_weights()
    #     actor_target_weights = self.target_model.get_weights()
    #     for i in range(len(actor_weights)):
    #         actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
    #     self.target_model.set_weights(actor_target_weights)

    def get_average_random_weight(self):
        return np.mean(self.model.get_layer('out_actions').get_weights()[1])

    def create_actor_network(self):
        state_input = Input(shape=(self.state_size,))

        main_network = Reshape((28, 28, 1))(state_input)
        main_network = Conv2D(512, (3,3))(main_network)
        main_network = LeakyReLU()(main_network)
        main_network = Conv2D(256, (3,3))(main_network)
        main_network = LeakyReLU()(main_network)
        main_network = Conv2D(128, (3,3))(main_network)
        main_network = LeakyReLU()(main_network)
        main_network = Conv2D(64, (3,3))(main_network)
        main_network = LeakyReLU()(main_network)
        main_network = Flatten()(main_network)

        # main_network = Dense(2048)(state_input)
        # main_network = LeakyReLU()(main_network)
        # main_network = Dense(2048)(main_network)
        # main_network = LeakyReLU()(main_network)

        # outputs = Dense(28 * 28, activation='tanh', name='actions')(main_network)
        outputs = NoisyDense(28 * 28, activation='tanh', name='out_actions', sigma_init=1)(main_network)

        actor = Model(inputs=[state_input], outputs=outputs)
        actor.summary()

        return actor, actor.trainable_weights, state_input
