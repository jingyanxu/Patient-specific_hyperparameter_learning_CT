import tensorflow as tf
import random

def parse_function (fname_in, fname_label, nchans = 672, nviews = 1160, header = 8369) :
  imagestring = tf.strings.substr(tf.io.read_file (fname_in), pos=header, len=nchans*nviews*4)
  image =  tf.reshape(tf.io.decode_raw (imagestring, tf.float32 ), [nviews, nchans, 1]) 

  labelstring = tf.strings.substr (tf.io.read_file (fname_label), pos = header, len=nchans*nviews*4)
  label = tf.reshape(tf.io.decode_raw (labelstring, tf.float32 ), [nviews, nchans ]) 

  return image, label, fname_label 

def parse_function_fbp (fname_in, fname_label, nchans = 672, nviews = 1160, xdim = 512, header = 8369) :
  imagestring = tf.strings.substr(tf.io.read_file (fname_in), pos=header, len=nchans*nviews*4)
  image =  tf.reshape(tf.io.decode_raw (imagestring, tf.float32 ), [nviews, nchans, 1]) 

  labelstring = tf.strings.substr (tf.io.read_file (fname_label), pos = header, len=xdim*xdim*4)
  label = tf.reshape(tf.io.decode_raw (labelstring, tf.float32 ), [xdim, xdim ]) 

  return image, label, fname_label 


def train_preprocess2 (image, label, label_fname) :

  maxval = tf.math.reduce_max (image)
  minval = tf.math.reduce_min (image)
  image_s = (image - minval)/ (maxval - minval)
  label = label 

  return image_s, label, tf.squeeze (image ) , label_fname


def parse_fnames (fnames_train, fnames_test, num_train, num_val, num_test) : 

  f = open (fnames_train, 'r')
  lines = f.read().splitlines()
  f.close ()

  sublines = random.sample (lines, num_train+num_val)

  fnames_ny  =  [x.split ()[1] for x in sublines]
  fnames_label =  [x.split ()[0] for x in sublines]

  total = len (lines)
  print ("total_train: %g, num_train: %g, num_val: %g" %  (total, num_train, num_val ))

  train_fnames_ny, train_fnames_label = fnames_ny[0:num_train] , fnames_label[0:num_train]
  val_fnames_ny, val_fnames_label = fnames_ny[num_train:num_train+num_val] , \
                                                  fnames_label[num_train:num_train+num_val]

  del lines, sublines, fnames_ny, fnames_label

  f = open (fnames_test, 'r')
  lines = f.read().splitlines()
  f.close ()

#  sublines = random.sample (lines, num_test)
  sublines = lines [0: num_test]

  fnames_ny  =  [x.split ()[1] for x in sublines]
  fnames_label =  [x.split ()[0] for x in sublines]

  total = len (lines)
  print ("total_test: %g, num_test: %g" %  (total, num_test ))

  test_fnames_ny, test_fnames_label = fnames_ny , fnames_label

  return train_fnames_ny, train_fnames_label, val_fnames_ny, val_fnames_label, test_fnames_ny, test_fnames_label 



# do not change batch_size for now ,
def load_data_fbp (train_name, test_name, num_train, num_val, num_test, batch_size = 1) :

  train_fnames_ny, train_fnames_label, val_fnames_ny, val_fnames_label, test_fnames_ny, test_fnames_label = \
         parse_fnames (train_name, test_name, num_train, num_val, num_test) 

  with tf.device ('/cpu:0') :

    train_dataset = tf.data.Dataset.from_tensor_slices((train_fnames_ny, train_fnames_label))
    train_dataset = train_dataset.shuffle(num_train).repeat(1)
    train_dataset = train_dataset.map(parse_function_fbp, num_parallel_calls=4)
    train_dataset = train_dataset.map(train_preprocess2, num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True )
#    train_dataset = train_dataset.unbatch ()
    train_dataset = train_dataset.prefetch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_fnames_ny, val_fnames_label))
    val_dataset = val_dataset.map(parse_function_fbp, num_parallel_calls=4)
    val_dataset = val_dataset.map(train_preprocess2, num_parallel_calls=4)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
#    val_dataset = val_dataset.unbatch ()
    val_dataset = val_dataset.prefetch(1 ) 

    test_dataset = tf.data.Dataset.from_tensor_slices((test_fnames_ny, test_fnames_label))
    test_dataset = test_dataset.map(parse_function_fbp, num_parallel_calls=4)
    test_dataset = test_dataset.map(train_preprocess2, num_parallel_calls=4)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
#    test_dataset = test_dataset.unbatch ()
    test_dataset = test_dataset.prefetch(1 ) 

  return train_dataset, val_dataset, test_dataset 

