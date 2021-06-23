from sys import argv, exit
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time 

from optparse import OptionParser

from utils_smth_layer_fast import *
from utils_cnn_model import *
from utils_tfds import  load_data_fbp
from utils_ramp_filter_tf import *

os.environ['PYTHONINSPECT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#plt.rcParams['figure.figsize'] = (8.0, 6.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if (len (argv) < 2) :
  print ("usage: {} epoch #".format (os.path.basename (argv[0])))
  print ('''
    -f  [ # of filters, default 128 ] 
    -b  [ # of blocks, default 2 ] 
    ''')
  exit (1)


parser = OptionParser()

parser.add_option("-c", "--chans", type = "int", dest="nchans", \
    help="num of chans", default=672)
parser.add_option("-v", "--views", type = "int", dest="nviews", \
    help="num of views", default=1160)
parser.add_option("-t", "--stepsize", type = "float", dest="lr", \
    help="learning rate", default=1.0e-4)
parser.add_option("-f", "--nfilters", type = "int", dest="nfilters", \
    help="num of filters", default=128)
parser.add_option("-b", "--nblocks", type = "int", dest="nblocks", \
    help="num of blocks", default=2)

(options, args) = parser.parse_args()

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

header = 8369
nchans =  672  
nviews = 1160 

# max train is around 15,000
num_train = 1 
num_val = 1 
num_test = 1 

nfilters = options.nfilters
nblocks = options.nblocks

train_fname = "./test_names.txt"  # a place holder for running test
test_fname = "./test_names.txt"
input_len = nchans 

initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=0.1)
inputs = tf.keras.layers.Input (shape = (input_len, 1 ))

is_training = False 
beta_o =  Conv1Dthin_d (inputs, nfilters, nblocks,  initializer, trainable=is_training )
beta_cnn =  tf.keras.Model (inputs = inputs, outputs=beta_o )
sinosmth_layer = Sinosmth_grad_k ( input_len)  

print (beta_cnn.summary())

num_epochs = 1 
batch_size = 1
steps_per_epoch = num_train// batch_size 
train_steps = num_epochs*steps_per_epoch
init_learning_rate = options.lr
print_every = 15
save_every = 1

learning_rate_fn = tf.optimizers.schedules.PolynomialDecay(init_learning_rate, train_steps, 0.5*init_learning_rate, 2)

def optimizer_init_fn():
#    return tf.keras.optimizers.SGD(learning_rate=learning_rate)
  return tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-8)

optimizer = optimizer_init_fn()


train_ds, val_ds, test_ds = load_data_fbp (train_fname, test_fname, num_train, num_val, num_test, batch_size = batch_size)


is_reload = True 

alpha = 0
apod = 2 # smooth 0, 1, 2

xdim = 512
fov_xdim = 200
fov_ydim = 200
lower = 0
upper = xdim - fov_xdim

xoff = 255.5
yoff = 255.5
fov  = 2*245
x_size = 0.975
xvec = (np.arange(xdim) - xoff ) * x_size
yvec = (np.arange(xdim) - yoff  ) * x_size
xmat, ymat = np.meshgrid (xvec, yvec)

# here we have a recon mask 
mask = np.where (xmat * xmat + ymat*ymat <= (fov/2)*(fov/2) , 1, 0 )
mask = tf.convert_to_tensor (mask, dtype = tf.float32)


checkpoint_directory = './checkpoint_dir/'

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=beta_cnn)
manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_directory, max_to_keep=50)

e0 = int(args[0] )  

test_loss = tf.keras.metrics.Mean(name='test_loss')
tt_loss = ()
et_loss = ()

start_time  = time.time ()
checkpoints_list = manager.checkpoints
t = 0 
t_ = 0

for epoch in range (e0, 1) :

  status = ckpt.restore(checkpoints_list [epoch] ) 
  print("Restored from epoch {} {}".format(epoch, checkpoints_list [epoch] )  )

  test_loss.reset_states()

  for x_scale, y_np, x_np, label_fname in test_ds : 

      loss = 0
      i = 0 

      x0 = random.randint (lower, upper)
      y0 = random.randint (lower, upper)

# for debugging maybe good to fix the roi location

      x0 = 255 - 100
      y0 = 255 - 100

      beta_o =  beta_cnn (x_scale[i]) 
      logits = sinosmth_layer ([x_np[i], beta_o])

      recon0_n = rcn_fbp (logits, x0, y0, apod ) # fixed fov dim inside

  # conversion to attenuation cm-1
      label = y_np[i, y0:y0+fov_ydim, x0:x0+fov_xdim]/ 1000.0 * 0.0183
      mask_roi  = mask [y0:y0+fov_ydim, x0:x0+fov_xdim]

      loss1 = tf.reduce_sum (((label - recon0_n)*mask_roi)**2)
      loss2 = alpha* tf.reduce_sum(beta_o*beta_o)
      loss += (loss1 + loss2 )

      loss /= batch_size
      test_loss.update_state (loss)

      end_time  = time.time ()
      t += 1

      et_loss += (loss.numpy() , ) 

      if (t % print_every ==0 ) :
        template = 'Iter {:6d}, Epoch {:4d}, Loss: {}, running avg_loss: {},  time {:6f}'
        print (template.format(t, epoch+1,
                                 loss.numpy(), test_loss.result().numpy(), end_time -start_time ))
        start_time = end_time

  tt_loss += (test_loss.result().numpy(), )

  print (tt_loss[-1] )

h2o = 0.0183
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,6))
hdl1 = ax[0].imshow (recon0_n.numpy() ) 
hdl2 = ax[1].imshow (label.numpy() ) 
hdl3 = ax[2].imshow (mask_roi.numpy() ) 
ax[0].set_title ('recon')
ax[1].set_title ('label')
ax[2].set_title ('mask [0/1]')
cw = 1024/1000*h2o
ww = 400/1000*h2o
hdl1.set_clim ( [cw - ww/2, cw+ww/2]) 
hdl2.set_clim ( [cw - ww/2, cw+ww/2]) 

plt.show(block = False)

