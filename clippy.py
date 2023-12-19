import tensorflow as tf


class ClippyAdagrad(tf.train.Optimizer):

    def __init__(self, learning_rate=0.1, initial_accumulator_value=0.0,
                 variable_relative_threshold=0.5, accumulator_relative_threshold=0.0,
                 absolute_threshold=0.01, epsilon=1e-7, name='ClippyAdagrad', use_locking=False):

        super(ClippyAdagrad, self).__init__(name=name, use_locking=False)
        self._lr = learning_rate
        self._initial_accumulator_value = initial_accumulator_value
        self._var_relative_threshold = variable_relative_threshold
        self._accumulator_relative_threshold = accumulator_relative_threshold
        self._absolute_threshold = absolute_threshold
        self._epsilon = epsilon

        self._accumulators = {}

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):

        var_updates = []
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

            clipped_delta, _ = self.shrink_by_references(delta, [var, precondition],
                                                         [self._var_relative_threshold,
                                                          self._accumulator_relative_threshold],
                                                         self._absolute_threshold)

            var_update = var.assign_sub(clipped_delta)
            var_updates.append(var_update)

        if global_step is not None:
            with tf.control_dependencies(var_updates):
                global_step_update = tf.assign_add(global_step, 1)
                var_updates.append(global_step_update)

        return tf.group(*var_updates, name=name)

    def shrink_by_references(self, tensor, references, relative_factors, absolute_factor):
        if any(relative_factor < 0 for relative_factor in relative_factors):
            raise ValueError("relative_factors must all be non-negative.")
        if absolute_factor < 0:
            raise ValueError("absolute_factor must be non-negative.")
        if len(references) != len(relative_factors):
            raise ValueError("References and relative_factors must have the same length.")

        max_norm = absolute_factor
        for ref, relative_factor in zip(references, relative_factors):
            max_norm += relative_factor * tf.abs(ref)

        scale = tf.minimum(1.0, max_norm / tf.reduce_max(tf.abs(tensor)))
        shrinked_tensor = tensor * scale
        return tensor * scale, scale

    def _apply_dense(self, grad, var):
        raise NotImplementedError

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError

    def _apply_sparse(self, grad, var):
        raise NotImplementedError

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError
