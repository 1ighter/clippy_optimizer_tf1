import tensorflow as tf


class ClippyAdagrad(tf.train.Optimizer):

    def __init__(self, learning_rate=0.001, initial_accumulator_value=0.1,
                 variable_relative_threshold=0.1, accumulator_relative_threshold=0.0,
                 absolute_threshold=1e-7, epsilon=1e-7, name='ClippyAdagrad', use_locking=False):

        super(ClippyAdagrad, self).__init__(name=name, use_locking=False)
        self._lr = learning_rate
        self._initial_accumulator_value = initial_accumulator_value
        self._var_relative_threshold = variable_relative_threshold
        self._accumulator_relative_threshold = accumulator_relative_threshold
        self._absolute_threshold = absolute_threshold
        self._epsilon = epsilon

        self._accumulators = {}

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):

        for (grad, var) in grads_and_vars:

            if var not in self._accumulators:
                self._accumulators[var] = tf.Variable(
                    initial_value=tf.constant(self._initial_accumulator_value, shape=var.get_shape(), dtype=var.dtype),
                    trainable=False)

            accumulator = self._accumulators[var]
            lr = tf.cast(self._lr, var.dtype)
            epsilon = tf.cast(self._epsilon, var.dtype)

            accumulator_update = tf.square(grad)
            accumulator_values = accumulator.assign_add(accumulator_update)

            precondition = tf.rsqrt(accumulator_values + epsilon)
            delta = lr * grad * precondition

            clipped_delta, _ = self._clip_by_thresholds(delta, var, precondition)

            var_update = var.assign_sub(clipped_delta)

        return tf.group(*[var_update for var_update, _ in grads_and_vars])

    def _clip_by_thresholds(self, grad, var, precondition):

        references = [var, precondition]
        relative_factors = [self._var_relative_threshold, self._accumulator_relative_threshold]

        max_norm = self._absolute_threshold
        for ref, relative_factor in zip(references, relative_factors):
            max_norm += relative_factor * tf.reduce_max(tf.abs(ref))

        grad_norm = tf.norm(grad)
        return tf.where(tf.greater(grad_norm, max_norm),
                        grad * (max_norm / grad_norm),
                        grad), max_norm

    def _apply_dense(self, grad, var):
        raise NotImplementedError

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError

    def _apply_sparse(self, grad, var):
        raise NotImplementedError

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError