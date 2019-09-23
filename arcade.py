import gym
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import random

env = gym.make('Breakout-v0')
env.reset()
np.random.seed(1)

ACTIONS = [0, 1, 2, 3]
REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_INIT_SIZE = 1000
UPDATE_TARGET_ESTIMATOR_EVERY = 500
GAMMA = 0.99

for _ in range(1000):
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        break


def process_frame(frame):
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.crop_to_bounding_box(frame, 34, 0, 160, 160)
    frame = tf.image.resize(frame, [80, 80], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    frame = tf.squeeze(frame)
    return frame


def get_model():
    input_ = tf.keras.Input(shape=(80, 80, 4), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(16, 8, 4, activation="relu")(input_)
    x = tf.keras.layers.Conv2D(32, 4, 2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    output = tf.keras.layers.Dense(len(ACTIONS))(x)
    return tf.keras.Model(inputs=input_, outputs=output)


def create_copied_network(net_to_copy, copied_net):
    for i, j in zip(net_to_copy.trainable_variables, copied_net.trainable_variables):
        j.assign(i)


class DeepQLearningNet():
    def __init__(self, eps_start=1.0, eps_end=0.1, eps_dec_steps=50000):
        self.network = get_model()
        self.target_network = get_model()
        self.epsilons = np.linspace(eps_start, eps_end, eps_dec_steps)
        self.eps_dec_steps = eps_dec_steps
        self.opti_step = -1
        self.replay_memory = []
        self.nb_actions = len(ACTIONS)
        self.optimizer = tf.optimizers.RMSprop(0.00025, 0.99, 0.0, 1e-6)

    def policy(self, obs, epsilon):
        A = np.ones(self.nb_actions) * epsilon / self.nb_actions
        q_values = self.network.predict(np.expand_dims(obs, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    def train(self, nb_episodes=100, batch_size=32):
        losses = []
        for episode in range(nb_episodes):
            state = env.reset()
            state = process_frame(state)
            state = np.stack([state]*4, axis=2)
            done = False
            rew_sum = 0
            while not done:
                epsilon = self.epsilons[min(self.opti_step+1, self.eps_dec_steps-1)]
                if self.opti_step % UPDATE_TARGET_ESTIMATOR_EVERY == 0:
                    create_copied_network(self.network, self.target_network)
                action_probs = self.policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, rew, done, _ = env.step(ACTIONS[action])
                rew_sum += rew
                next_state = process_frame(next_state)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                if len(self.replay_memory) == REPLAY_MEMORY_SIZE:
                    self.replay_memory.pop(0)
                self.replay_memory.append((state, action, rew, next_state, done))
                if len(self.replay_memory) > REPLAY_MEMORY_INIT_SIZE:
                    training_batch = random.sample(self.replay_memory, batch_size)
                    states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*training_batch))
                    q_values_next_target = self.target_network.predict(next_states_batch)
                    t_best_actions = np.argmax(q_values_next_target, axis=1)
                    targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * GAMMA * q_values_next_target[np.arange(batch_size), t_best_actions]
                    with tf.GradientTape() as t:
                        states_batch = np.array(states_batch, dtype=np.float32)
                        output_batch = self.network(states_batch)
                        gather_indices = tf.range(batch_size) * tf.shape(output_batch)[1] + action_batch
                        action_predictions = tf.gather(tf.reshape(output_batch, [-1]), gather_indices)
                        loss = tf.reduce_mean(tf.square(targets_batch-action_predictions))
                        losses += [loss]
                        print('Ep.%d/%d | loss = %.3f | rew_sum = %.3f' % (episode, nb_episodes, loss, rew_sum))
                    gradients = t.gradient(loss, self.network.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
                    self.opti_step += 1
                state = next_state

    def play(self):
        done = False
        while not done:
            env.render()
            _, _, done, _ = env.step(env.action_space.sample())
            time.sleep(0.05)


if __name__ == "__main__":
    drlnet = DeepQLearningNet()
    drlnet.train()
    drlnet.play()
