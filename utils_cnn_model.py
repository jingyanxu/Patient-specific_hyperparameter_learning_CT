import tensorflow as tf


def Conv1D512_d (previous, initializer ='he_normal', trainable = True )  :

#  inputs = tf.keras.layers.Input (shape = input_shape)
  beta_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', \
              kernel_initializer=initializer )  (previous)
  beta_1 = tf.keras.layers.BatchNormalization(trainable = trainable) (beta_1)
  beta_1 = tf.keras.layers.LeakyReLU() (beta_1)

  beta_11 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', \
              kernel_initializer=initializer )  (tf.concat ( [beta_1, previous ] , axis =-1) )
  beta_11 = tf.keras.layers.BatchNormalization( trainable = trainable) (beta_11)
  beta_11 = tf.keras.layers.LeakyReLU() (beta_11)

  beta_2 = tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', \
              activation='relu', kernel_initializer=initializer )  (tf.concat([beta_11, beta_1, previous], axis = -1 ) )
#  model = tf.keras.Model (inputs = inputs, outputs=beta_2)

  return beta_2 


def Conv1Dthin_d (previous, nb_filter, nblocks, initializer ='he_normal', trainable = True )  :

  all_previous = previous 
  for i in range (nblocks)  :  

    beta_1 = tf.keras.layers.Conv1D(filters=nb_filter, kernel_size=3, strides=1, padding='same', \
                kernel_initializer=initializer )  ( all_previous ) 
    beta_1 = tf.keras.layers.BatchNormalization(trainable = trainable) (beta_1)
    next = tf.keras.layers.LeakyReLU() (beta_1)

    all_previous = tf.concat ( [all_previous, next], axis = -1)

# the output layer

  next = tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', \
            activation = 'relu',  kernel_initializer=initializer )  ( all_previous  ) 


  return next 

