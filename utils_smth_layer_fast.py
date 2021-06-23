import numpy as np
import tensorflow as tf


#@tf.function  (input_signature=[\
#      tf.TensorSpec(shape=(None, 672), dtype=tf.float32), \
#      tf.TensorSpec(shape=(None, 672, 1), dtype=tf.float32), \
#      tf.TensorSpec(shape=None, dtype=tf.int32)  \
#      ] )
@tf.function
@tf.custom_gradient
# here we separate the solution and the gradient
def  sino_smth_tf_grad_nv_sep (y_tf, beta_tf,  nchans1) :

#  nchans  = tf.shape (y_tf) [-1]

  nchans = y_tf.get_shape().as_list() [-1]
  #  y_tf = tf.squeeze (y_tf)
  beta_tf  = tf.squeeze (beta_tf)

  fy = [None for i in range (nchans)]
  k =  [None for i in range (nchans)]
  xsol = [None for i in range (nchans)]

  dtype = tf.float64
  dtype = tf.float32

  weights = tf.ones_like (y_tf)
  batch_size = tf.shape (y_tf) [0]

  k[0] = tf.zeros (batch_size, dtype  = dtype)
  fy[0] = tf.zeros (batch_size, dtype = dtype)

  for i in range (0, nchans-1) :
    xsol[i] = (k[i]*fy[i] + weights [:, i]*y_tf[:, i] )  / (k[i] + weights[:, i]  + beta_tf[:, i])

    fy[i+1] = (k[i]*fy[i] + weights[:, i] * y_tf[:, i]) / (k[i] + weights[:, i])
    k[i+1] = (k[i] + weights[:, i]) * beta_tf[:, i]  / ( k[i]+ weights[:, i] + beta_tf[:, i] )
  for i in range ( nchans-1, nchans) :
    xsol[i] = (k[i]*fy[i] + weights [: , i]*y_tf[:, i] )  / (k[i] + weights[:, i] )
  for i in range (nchans-2, -1, -1) :
    xsol[i] += beta_tf[:, i]*xsol[i+1]  / (k[i] + weights[:, i] + beta_tf[:, i])

  xsol = tf.transpose ( tf.convert_to_tensor (xsol), (1, 0 ) )

  # related to gradient calculation

  def sino_grad (z_tf) :
    xzprod = tf.zeros (batch_size, dtype = dtype)

    dwdk = [ None for i in range (nchans)]
    dwdb = [ tf.zeros (batch_size, dtype = dtype) for i in range (nchans)]
    dwdfy = [ None for i in range (nchans)]
    dwdfz = [ None for i in range (nchans)]

    fz = [None for i in range (nchans)]
    fz [0] = tf.zeros (batch_size, dtype = dtype)

    for i in range (0, nchans-1) :

      dd =  weights[:, i]/ (k[i] + weights[:, i])
      aa = k[i] * dd
      bb = y_tf[:, i] - fy [i]
      cc = z_tf[:, i] - fz[i]

      fz [i+1] = (k[i]*fz[i] + z_tf[:, i]) / (k[i] + weights[:, i])

      xzprod +=  aa *  bb  * cc

      dadk = dd*dd

      dwdk[i] = dadk*bb*cc
      dwdfy[i] = -aa*cc
      dwdfz[i] = -aa*bb

    for i in range ( nchans-1, nchans) :

      dd =  weights[:, i]/ (k[i] + weights[:, i])
      aa = k[i] * dd
      bb = y_tf[:, i] - fy [i]
      cc = z_tf[:, i] - fz[i]

      xzprod +=  aa *  bb  * cc

      dadk = dd*dd

      dwdk[i] = dadk*bb*cc
      dwdfy[i] = -aa*cc
      dwdfz[i] = -aa*bb

    for i in range (nchans-2, -1, -1) :

      AA  = (beta_tf[:, i] / ( k[i] + weights[:, i] + beta_tf[:, i]) )**2
      BB  = weights[:, i] * ( y_tf[:, i] - fy [i]) / ((k[i] + weights[:, i])**2)
      CC  = weights[:, i] * ( z_tf[:, i] - fz [i]) / ((k[i] + weights[:, i])**2)

      dwdk [i] +=  dwdk[i+1]* AA - dwdfy[i+1] *BB - dwdfz[i+1] * CC
      dwdfy[i] += dwdfy[i+1] * k[i] / (k[i] + weights[:, i])
      dwdfz[i] += dwdfz[i+1] * k[i] / (k[i] + weights[:, i])

      dkdb =   ( ( k[i] + weights[:, i])/ (k[i] + weights[:, i] + beta_tf[:, i]))**2
      dwdb [i] =  -dwdk [i+1] * dkdb

    dwdb =   tf.expand_dims (tf.transpose (tf.convert_to_tensor (dwdb) , (1, 0) ), axis = 2)
    xzprod = tf.reduce_sum(y_tf*z_tf, axis = 1 ) - xzprod

    return None, dwdb,  None 

  return xsol, sino_grad 



class Sinosmth_grad_k(tf.keras.layers.Layer):

  def __init__(self,  nchans ):
    super(Sinosmth_grad_k, self).__init__()
    self.nchans = nchans 

  def build(self, input_shape):
      self.batch_size = input_shape[0]
      #var_init = tf.ones(self.batch_size, dtype=tf.float32)[..., None, None, None]
      #self.my_var = tf.Variable(var_init, trainable=False, validate_shape=True)
      super(Sinosmth_grad_k, self).build(input_shape)  # Be sure to call this at the end

  def call(self, inputs):

    y , beta = inputs 
    input_shape = y.shape
    #xsol = tf.map_fn ( lambda x  : sino_smth_tf (x[0], x[1], self.nchans) , [y, beta] , dtype = tf.float32) 
    xsol = sino_smth_tf_grad_nv_sep (y, beta, self.nchans ) 
    new_shape = self.compute_output_shape(input_shape) 

    #return tf.reshape (xsol, new_shape)
    return xsol

  def compute_output_shape(self, input_shape):
        return  input_shape

