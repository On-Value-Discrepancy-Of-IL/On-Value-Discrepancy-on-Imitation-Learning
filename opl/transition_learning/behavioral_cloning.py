import tensorflow as tf
import numpy as np
import os
import os.path as osp
import time

from baselines.common.input import observation_placeholder
from baselines.common.distributions import DiagGaussianPdType
from baselines.common.tf_util import get_session, initialize, save_variables, load_variables
from baselines.common.models import get_network_builder
from baselines.common.misc_util import set_global_seeds
from baselines import logger


class BCTransition(object):
    def __init__(
            self,
            env,
            network_fn,
            lr=5e-4,
            state_input=True,
            scope="BCTransition",
            max_grad_norm=10,
            residual_learning=True,
            deterministic_output=True
    ):
        self.scope = scope

        ob_space = env.observation_space
        ac_space = env.action_space

        env.reset()
        state_shape = env.unwrapped.get_save_variables().shape if state_input else ob_space.shape

        self.state_ph = tf.placeholder(tf.float32, [None, *state_shape], "state_ph")
        self.ac_ph = observation_placeholder(ac_space, name="ac_ph")
        self.state2_ph = tf.placeholder(tf.float32, [None, *state_shape], "state2_ph")
        self.sess = get_session()

        with tf.variable_scope(self.scope):
            inputs = tf.concat([self.state_ph, self.ac_ph], axis=-1)
            latent = network_fn(inputs)
            assert len(state_shape) == 1
            pdtype = DiagGaussianPdType(size=state_shape[0])
            self.pd, *_ = pdtype.pdfromlatent(latent)
            if deterministic_output:
                pd_output = self.pd.mode()
            else:
                pd_output = self.pd.sample()

            if residual_learning:
                self.loss = tf.reduce_mean(self.pd.neglogp(self.state2_ph - self.state_ph))
                self.output_tf = self.state_ph + pd_output
            else:
                self.loss = tf.reduce_mean(self.pd.neglogp(self.state2_ph))
                self.output_tf = pd_output

            self.mse = tf.reduce_mean(tf.norm(self.output_tf - self.state2_ph, axis=-1))

        self.params = tf.trainable_variables(self.scope)
        logger.info("\n***********************{}********************************".format(self.scope))
        for param in self.params:
            logger.info(param)
        logger.info("***********************{}********************************\n".format(self.scope))
        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        grads_and_var = self.trainer.compute_gradients(self.loss, self.params)
        grads, var = zip(*grads_and_var)

        self.grad_norm = tf.global_norm(grads)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self._train_iter = 0

        self.savedir = osp.join(logger.get_dir(), "models")
        os.makedirs(self.savedir, exist_ok=True)

        initialize()

    def train(self, states, acs, states2):
        feed_dict = {
            self.state_ph: states,
            self.ac_ph: acs,
            self.state2_ph: states2,
        }
        _, loss, mse, grad_norm = self.sess.run([self._train_op, self.loss, self.mse, self.grad_norm],
                                                feed_dict=feed_dict)
        self._train_iter += 1
        res = {'train/loss': loss, 'train/mse': mse, 'train/grad_norm': grad_norm}
        return res

    def valid(self, states, acs, states2):
        feed_dict = {
            self.state_ph: states,
            self.ac_ph: acs,
            self.state2_ph: states2,
        }
        loss, mse, grad_norm = self.sess.run([self.loss, self.mse, self.grad_norm], feed_dict=feed_dict)
        res = {'test/loss': loss, 'test/mse': mse, 'test/grad_norm': grad_norm}
        return res

    def step(self, state, ac):
        assert state.ndim == ac.ndim
        flat = False
        if state.ndim == 1:
            flat = True
            ob, ac = np.expand_dims(state, 0), np.expand_dims(ac, 0)
        state2 = self.sess.run(self.output_tf, feed_dict={self.state_ph: state, self.ac_ph: ac})
        if flat:
            return state2[0]
        else:
            return state

    def save(self, save_path=None):
        save_path = save_path or osp.join(self.savedir, "{}.model".format(self._train_iter))
        save_variables(save_path, variables=self.params, sess=self.sess)
        logger.info("save {} into: {}".format(self.scope, save_path))

    def load(self, load_path):
        load_variables(load_path, variables=self.params, sess=self.sess)
        logger.info("load {} into: {}".format(self.scope, load_path))


def learn(
        *,
        env,
        dataset,
        seed=2019,
        network='mlp',
        lr=5e-4,
        epoch=100,
        batch_size=64,
        log_interval=1000,
        save_interval=100,
        state_input=True,
        residual_learning=True,
        deterministic_output=False,
        **network_kwargs
):
    set_global_seeds(seed)

    network_builder = get_network_builder(network) if isinstance(network, str) else network
    network_fn = network_builder(**network_kwargs)

    assert callable(network_fn)

    model = BCTransition(env=env, network_fn=network_fn, lr=lr,
                         state_input=state_input,
                         residual_learning=residual_learning,
                         deterministic_output=deterministic_output)

    logger.info("\nRunning {} with the following kwargs:".format(model.scope))
    for key, value in locals().copy().items():
        logger.info(key, value)
    logger.info("\n")

    nb_samples = dataset.train_set.num_pairs
    nb_batch = nb_samples // batch_size
    nb_iter = 0
    tstart = time.time()
    print(epoch, nb_samples, nb_batch)
    for epoch_iter in range(epoch):
        # dataset will automatically shuffle
        for batch_iter in range(nb_batch):
            obs, acs, rs, dones, mus, states, states2 = dataset.get_next_batch(batch_size=batch_size, split='train')
            # USE STATES RATHER THAN OBS TO TRAIN TRANSITION MODEL!!!
            train_logs = model.train(states=states, acs=acs, states2=states2)
            nb_iter += 1
            if nb_iter % log_interval == 0 or nb_iter == 1:
                obs, acs, rs, dones, mus, states, states2 = dataset.get_next_batch(batch_size=batch_size, split='val')
                test_logs = model.valid(states=states, acs=acs, states2=states2)

                timesofar = time.time() - tstart
                logs = {
                    'time/fps': nb_iter // timesofar,
                    'time/timesofar': timesofar,
                    'time/epoch': epoch_iter,
                    'time/iter': nb_iter,
                }
                logs.update(train_logs)
                logs.update(test_logs)

                logger.info("***********Epoch:{}/{}******************".format(epoch_iter, epoch))
                for key, value in logs.items():
                    logger.record_tabular(key, value)
                logger.dump_tabular()
        if epoch_iter % save_interval == 0 or epoch_iter == 0:
            model.save()

    savepath = osp.join(logger.get_dir(), "models", "final.model")
    model.save(savepath)
    return model


