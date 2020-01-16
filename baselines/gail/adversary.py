'''
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
'''
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym
import os
import os.path as osp


from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U
from baselines import logger

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class TransitionClassifier(object):
    def __init__(self, env, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = env.action_space.shape
        self.input_shape = tuple([o+a for o, a in zip(self.observation_shape, self.actions_shape)])
        self.num_actions = env.action_space.shape[0]
        self.hidden_size = hidden_size
        self.build_ph()
        # Build grpah
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
                                      self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward


class LinearReward(object):
    def __init__(self, env, expert_obs, expert_acs, scope="adversary",
                 simplex=False, favor_zero_expert_reward=False, recompute_expert_featexp=False, sqscale=.01,):
        self.scope = scope
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.observation_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
        
        self.expert_obs = expert_obs
        self.expert_acs = expert_acs

        self.simplex = simplex
        self.favor_zero_expert_reward = favor_zero_expert_reward
        self.recompute_expert_featexp = recompute_expert_featexp
        self.sqscale = sqscale
        
        with tf.variable_scope(scope):
            assert isinstance(self.action_space, gym.spaces.Box)
            self.input_rms = RunningMeanStd(shape=(self.observation_shape[0]+self.action_shape[0]))
        U.initialize()
        self.input_rms.update(np.concatenate([expert_obs, expert_acs], axis=1))

        self.expert_featexp = self._compute_featexp(expert_obs, expert_acs)
            
        feat_dim = self.expert_featexp.shape[0]
        if self.simplex:
            self.widx = np.random.randint(feat_dim)
        else:
            self.w = np.random.randn(feat_dim)
            self.w /= np.linalg.norm(self.w) + 1e-8
        
        self.reward_bound = 0.
        self.gap = 0.
        self.loss = tf.zeros([])
        self.loss_name = []
        self._fit_count = 0
        
    def fit(self, obs, acs, print_featexp=True, plot_featexp=False):
        curr_featexp = self._compute_featexp(obs, acs)

        if self.recompute_expert_featexp:
            self.expert_featexp = self._compute_featexp(self.expert_obs, self.expert_acs)

        # Debug
        if print_featexp:
            logger.info(curr_featexp.flatten())
            logger.info(self.expert_featexp.flatten())
        if plot_featexp:
            pass
            # self._fit_count += 1
            # plt.figure(dpi=100)
            # plt.plot(curr_featexp, marker='+', label='curr')
            # plt.plot(self.expert_featexp.flatten(), marker='*', label='expert')
            # plt.legend()
            # plt.grid()
            # save_dir = osp.join(logger.get_dir(),  'featexp')
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = osp.join(save_dir, '{}.png'.format(self._fit_count))
            # plt.savefig(save_path, bbox_inches='tight')
            # print('save featexp into: {}'.format(save_path))

        if self.simplex:
            v = curr_featexp - self.expert_featexp
            self.widx = np.argmin(v)
            self.gap = self.expert_featexp[self.widx] - curr_featexp[self.widx]
            return {'widx': self.widx, 'gap': self.gap}
        else:
            w = self.expert_featexp - curr_featexp
            l2 = np.linalg.norm(w)
            self.w = w / (l2 + 1e-8)
            self.gap = np.linalg.norm(self.expert_featexp - curr_featexp)
            return {'w_obs': self.w[:self.observation_shape[0]],
                    'w_acs': self.w[self.observation_shape[0]: self.observation_shape[0]+self.action_shape[0]],
                    'b_obs': self.w[-self.observation_shape[0]:],
                    'gap': self.gap}

    def get_reward(self, obs, acs):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
            
        feat = self._featurize(obs, acs)
        rew = (feat[:, self.widx] if self.simplex else feat.dot(self.w)) / float(feat.shape[1])
        
        if self.favor_zero_expert_reward:
            self.reward_bound = max(self.reward_bound, rew.max())
        else:
            self.reward_bound = min(self.reward_bound, rew.min())
        rew_shifted = rew - self.reward_bound
        return rew_shifted
    
    def _compute_featexp(self, obs, acs):
        return self._featurize(obs, acs).mean(axis=0)

    def _featurize(self, obs, acs):
        # normalize
        if isinstance(self.action_space, gym.spaces.Box):
            assert obs.ndim == 2 and acs.ndim == 2
            # normalize
            sess = tf.get_default_session()
            mean, std = sess.run([self.input_rms.mean, self.input_rms.std])
            inputs_normlized = (np.concatenate([obs, acs], axis=1) - mean) / std
            obs, acs = inputs_normlized[:, :obs.shape[1]], inputs_normlized[:, obs.shape[1]:]
            
            # Linear + Quadratic + Bias
            feat = [obs, acs, (self.sqscale*obs)**2, (self.sqscale*acs)**2]
            feat.append(np.ones([len(obs), 1]))
            feat = np.concatenate(feat, axis=1)
        else:
            raise NotImplementedError
        
        assert feat.ndim == 2 and feat.shape[0] == obs.shape[0]
        return feat

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        
    # def update_inputnorm(self, obs, acs):
    #     assert isinstance(self.action_space, gym.spaces.Box)
    #     self.obs_rms.update(np.concatenate([obs, acs], axis=1))
