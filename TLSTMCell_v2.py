from tensorflow.python.ops.rnn_cell_impl import RNNCell,LSTMStateTuple
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops,math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import dtypes
import tensorflow as tf
class TLSTMCell(RNNCell):

    def __init__(self,num_units,forget_bias=1.0,state_is_tuple=True,activation=None,reuse=None):
        super(TLSTMCell,self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self.num_units = num_units

    @property
    def state_size(self):
        cs_size = self.num_units * 1
        return (LSTMStateTuple(cs_size, 1*self.num_units)
                if self._state_is_tuple else 1 * self.num_units)

    @property
    def output_size(self):
        return self.num_units * 1


    def call(self, inputs, state):
        # print('state 0',inputs)
        sigmoid = math_ops.sigmoid
        tanh = math_ops.tanh
        if self._state_is_tuple:
            c0,h0 = state
        else:
            c0, h0 = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        # 时间差, 暂时转为浮点型
        # delt_t = float(array_ops.slice(delt_t,0,1)) 
        # text向量
        # text = array_ops.slice(inputs,1,128)

        # print('state 1')
        inputs_x = inputs[:,1:]
        delt_t = inputs[:,0:1]

        # print('state 1.1',inputs_x,h0)
        # 时间衰减部分
        with tf.variable_scope('1'):
            concat_time_x = _linear([inputs_x,h0],3*self.num_units,bias=True) 

        # print('state 1.2')   
        # 文本部分
        with tf.variable_scope('2'):         
            concat_x = _linear([inputs_x, h0],3*self.num_units,bias=True)

        # print('state 1.3')
        with tf.variable_scope('3'):         
            output_x = _linear([inputs_x,h0],self.num_units,bias=True)

        # print('state 2')

        # 时间衰减部分
        i00,j00,f00 = array_ops.split(value=concat_time_x,num_or_size_splits=3,axis=1) 
        # 文本部分
        i10, j10, f10 = array_ops.split(value=concat_x, num_or_size_splits=3, axis=1) 

        # print('state 2.1')
        # print(c0 * math_ops.exp(-1 * delt_t) * sigmoid(f00 + self._forget_bias))
        # print((1 - math_ops.exp(-1 * delt_t)) * sigmoid(i00) * tanh(j00))
        new_c0 = c0 * math_ops.exp(-1 * delt_t) * sigmoid(f00 + self._forget_bias) +  (1 - math_ops.exp(-1 * delt_t)) * sigmoid(i00) * tanh(j00)
        # new_c0 = c0 * sigmoid(f00 + self._forget_bias) 

        # print('state 2.2')
        new_c0 = new_c0 * sigmoid(f10 + self._forget_bias) + sigmoid(i10) * tanh(j10)
        
        # print('state 2.3')
        new_h0 = tanh(new_c0) * sigmoid(output_x)

        # print('state 3')

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c0,new_h0)
        else:
            new_state = array_ops.concat([new_c0,new_h0],1)

        # print('state 4')

        return new_h0,new_state

from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _linear(args,
            output_size,
            bias,
            weight_name=_WEIGHTS_VARIABLE_NAME,
            bias_name=_BIAS_VARIABLE_NAME,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        weight_name, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)

    # if the args is a single tensor then matmul it with weight
    # if the args is a list of tensors then concat them in axis of 1 and matmul
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          bias_name, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)