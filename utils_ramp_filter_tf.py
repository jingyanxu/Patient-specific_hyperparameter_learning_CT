import tensorflow as tf
import numpy as np

eps = 1e-4
roi_xdim = 200 
roi_ydim = 200 
#x0 = 152
#y0 = 109
nviews = 1160
nchans = 672
fov = 2*245

u_size =  1.4083
D = 1040.0
duD  = u_size/D

params = {}
params["start_angle"]  = 0
params["nreadings"] = nviews
params ["xdim"] = roi_xdim
params ["ydim"] = roi_ydim
params ["fov"] = fov

params ["x_size"] = 0.975
params ["y_size"] = 0.975
params ["s2d"] = D
params ["s2o"] = 570.0
params ["u_off"] = 336.625
params ["u_size"] = u_size
params ["num_u"] = nchans
params ["print_every"] = 4
params ["angles_per_set"] = nviews
params ["duD"] = duD
params ["xoff"] = 255.5
params ["yoff"] = 255.5 
params ['apod'] = 0

def vfunc (x )   : 

  y2 = tf.ones_like (x) * 0.5
  mask = tf.cast (tf.where(tf.abs(x) > eps, 1, 0), x.dtype)
  y1 = tf.math.divide_no_nan (tf.sin (x) , x) + tf.math.divide_no_nan( (tf.cos (x) - 1), (x*x) ) 
  y  = y1 * mask + (1 - mask) * y2

  return y 

def wfunc (x )   : 
  y2 = tf.zeros_like (x) 
  mask = tf.cast (tf.where ( tf.abs (x) > eps, 1, 0), x.dtype ) 
  y1  = tf.math.divide_no_nan(tf.cos (x) -1 , x ) 
  y = mask * y1 + (1 - mask) * y2

  return y 

def apply_ramp_filter (proj, apod  ) : 

  du = params['duD']
  nviews = params['nreadings'] 
  nchans =  params['num_u']
  uoff = params['u_off']
#  apod = params['apod']
#  apod = 0

  nchans_pad =  int(np.exp2 (np.ceil(np.log2 (nchans) )  ) )
  print (nchans, nchans_pad) 

#  ramp_filter = tf.zeros ((2*nchans_pad,))
  ramp_imag = tf.zeros ((2*nchans_pad,), dtype = tf.float64)
  rf_p2 = tf.zeros ( 2*nchans_pad -nchans+1-nchans, dtype = tf.float64)

  du = tf.cast (du, tf.float64)
  uoff = tf.constant( uoff, dtype=tf.float64)
  #  gamma=(-uoff + tf.cast (range(nchans), tf.float64))*du
  gamma=(-uoff + tf.range(nchans, dtype=tf.float64))*du
  cosweight= tf.cos(gamma)

  proj = tf.cast (proj, dtype = tf.float64) * cosweight
#  proj = proj 

  if (apod == 0) :
    hamming = tf.constant ( 0.5, dtype  = tf.float64)
  else : 
    hamming = tf.constant (1.0, dtype=tf.float64)

  if (apod == 1) :
    hamming = tf.constant (1.0, dtype=tf.float64)
  else : 
    hamming = tf.constant( 0.5, dtype=tf.float64)

  pi = tf.constant (np.pi, dtype = tf.float64)
  arg1 = pi*tf.range(nchans, dtype=tf.float64) 
  arg2 = pi*tf.range (-nchans+1, 0, dtype=tf.float64) 
  a1 = arg1 * du/pi   
  a2 = arg2 * du/pi

  w12 = tf.ones_like (a1[nchans :]) 
  w11 = (a1[1:nchans] / tf.sin ( a1[1:nchans]) )**2
  w1 = tf.concat ( [tf.constant ([1.], dtype=tf.float64), w11, w12], axis = -1) 

  w2 = (a2/tf.sin(a2))**2

  if (apod == 0  )  or (apod == 1) : 
    h1 = hamming * vfunc (arg1) + 0.5 * (1 - hamming)* (vfunc (pi + arg1) + vfunc (pi- arg1))
    h2 = hamming * vfunc (arg2) + 0.5 * (1 - hamming)* (vfunc (pi + arg2) + vfunc (pi- arg2))
    rf_p1  = h1 * w1
    rf_p3  = h2 * w2
    ramp_filter = tf.concat ( [rf_p1, rf_p2, rf_p3], axis=-1)   / (2 * du)
  else : 

    h1 = wfunc (arg1 + pi/2) - wfunc (arg1 - pi/2)
    h2 = wfunc (arg2 + pi/2) - wfunc (arg2 - pi/2)

    rf_p1 = h1*w1
    rf_p3 = h2 * w2
    ramp_filter = tf.concat ( [rf_p1, rf_p2, rf_p3], axis=-1)   / ( - 2 * du* pi)

  ramp_filter_fft = tf.signal.fft (tf.complex (ramp_filter, ramp_imag))
  proj_pad = tf.pad (proj , [[0, 0], [0, 2*nchans_pad -nchans] ], constant_values =0 )
  proj_pad_fft = tf.signal.fft( tf.complex (proj_pad, tf.zeros_like (proj_pad)) ) 
  proj_filter = tf.math.real (tf.signal.ifft(ramp_filter_fft * proj_pad_fft ) ) [:, 0:nchans]

  proj_filter *= cosweight*cosweight
  #proj_filter = tf.cast (proj_filter, dtype = tf.float32)


  return proj_filter , cosweight




# this is float32 version
def backprojection_nv  (proj, x0, y0, xdim=64, ydim=64 )  :

  start_angle  = params['start_angle'] 
  nreadings = params ['nreadings'] 
#  xdim = params['xdim'] 
#  ydim = params['ydim'] 
  fov = params ['fov']
  x_size = params['x_size']
  y_size = params['y_size']
  s2d = params['s2d'] 
  s2o = params['s2o'] 
  u_off = params['u_off'] 
  u_size = params['u_size'] 
  nchans = params['num_u'] 
  print_every = params['print_every']
  angles_per_set = params ["angles_per_set"]
  du = params ["duD"]
  #xoff = params ["xoff"] 
  #yoff = params ["yoff"] 

  xoff = params ["xoff"] - x0
  yoff = params ["yoff"] - y0

  dtype =  tf.float32

  proj = tf.cast (proj, dtype) 

  xvec = (tf.range (xdim, dtype=dtype)  - xoff  )*tf.constant (x_size, dtype=dtype )
  yvec = (tf.range (ydim, dtype=dtype)  - yoff  )*tf.constant (y_size, dtype=dtype)
  (xmat, ymat) = tf.meshgrid (xvec, yvec)
  #ymat  = ymat[::-1, :]

#  angles_per_set = 32
  nsubsets = nreadings//angles_per_set
  xmat1 = tf.expand_dims (xmat, axis=-1)
  ymat1 = tf.expand_dims (ymat, axis=-1)
  #idx = tf.reshape (tf.tile (tf.reshape(tf.range (angles_per_set), [-1,1]), [1, xdim*ydim]  ), [-1] )
  idx = tf.reshape (tf.tile (tf.range (angles_per_set), [xdim*ydim]), [-1] )

  pi = tf.constant (np.pi, dtype = dtype)
  viewangles = tf.range(nreadings, dtype = dtype)*2.0*pi/nreadings - start_angle/180*pi
  dangle = tf.abs (viewangles[1] - viewangles[0])
  bim = tf.zeros ( [ydim, xdim], dtype=dtype)

  for isubset in range(nsubsets ) :  
    if (isubset % print_every == 0) : 
      print (isubset, end = " " ) 
    iv0 = isubset * angles_per_set 
    iv1 = (isubset + 1) * angles_per_set 
    angles =  viewangles [iv0:iv1]
    e11, e12 = -tf.cos(angles) , -tf.sin(angles)
    e21, e22 = e12, -e11

    num = xmat1 *e21 + ymat1*e22
    den = s2o + xmat1*e11 + ymat1*e12

    coord = tf.reshape (tf.math.atan2 (num, den)*s2d/u_size + u_off, [  -1 ] ) 
    lower = tf.cast (tf.floor (coord), tf.int32) 
    upper = lower + 1
    lower = tf.clip_by_value (lower, 0, nchans-1)
    weights = coord - tf.cast (lower, dtype) 

    lower = tf.stack ([idx, tf.clip_by_value (lower, 0, nchans-1)], axis =1)
    upper = tf.stack([idx, tf.clip_by_value (upper, 0, nchans-1) ], axis =1)

    lim = tf.gather_nd (proj [iv0:iv1, :],  lower )
    uim = tf.gather_nd (proj [iv0:iv1, :],  upper )

    im1 = tf.reshape (lim * (1 - weights) + uim * weights , [ydim, xdim, angles_per_set] ) 
    bim += tf.reduce_sum (im1/den/den , axis = -1)

  bim *= s2o*pi/nreadings 
  bim = tf.cast (bim, dtype = tf.float32)

  return bim

def backprojection_nv_d  (proj, x0, y0, xdim=64, ydim=64)  :

  start_angle  = params['start_angle'] 
  nreadings = params ['nreadings'] 
#  xdim = params['xdim'] 
#  ydim = params['ydim'] 
  fov = params ['fov']
  x_size = params['x_size']
  y_size = params['y_size']
  s2d = params['s2d'] 
  s2o = params['s2o'] 
  u_off = params['u_off'] 
  u_size = params['u_size'] 
  nchans = params['num_u'] 
  print_every = params['print_every']
  angles_per_set = params ["angles_per_set"]
  du = params ["duD"]
  #xoff = params ["xoff"] 
  #yoff = params ["yoff"] 
  dtype =  tf.float64

  xoff = params ["xoff"] - tf.cast( x0, dtype = dtype)
  yoff = params ["yoff"] - tf.cast( y0, dtype = dtype)

  xvec = (tf.range (xdim, dtype=dtype)  - xoff )*tf.constant (x_size, dtype=dtype )
  yvec = (tf.range (ydim, dtype=dtype)  - yoff )*tf.constant (y_size, dtype=dtype)
  (xmat, ymat) = tf.meshgrid (xvec, yvec)
  #ymat  = ymat[::-1, :]

#  angles_per_set = 32
  nsubsets = nreadings//angles_per_set
  xmat1 = tf.expand_dims (xmat, axis=-1)
  ymat1 = tf.expand_dims (ymat, axis=-1)
  #idx = tf.reshape (tf.tile (tf.reshape(tf.range (angles_per_set), [-1,1]), [1, xdim*ydim]  ), [-1] )
  idx = tf.reshape (tf.tile (tf.range (angles_per_set), [xdim*ydim]), [-1] )

  pi = tf.constant (np.pi, dtype = dtype)
  viewangles = tf.range(nreadings, dtype = dtype)*2.0*pi/nreadings - start_angle/180*pi
  dangle = tf.abs (viewangles[1] - viewangles[0])
  bim = tf.zeros ( [ydim, xdim], dtype=dtype)

  for isubset in range(nsubsets ) :  
    if (isubset % print_every == 0) : 
      print (isubset, end = " " ) 
    iv0 = isubset * angles_per_set 
    iv1 = (isubset + 1) * angles_per_set 
    angles =  viewangles [iv0:iv1]
    e11, e12 = -tf.cos(angles) , -tf.sin(angles)
    e21, e22 = e12, -e11

    num = xmat1 *e21 + ymat1*e22
    den = s2o + xmat1*e11 + ymat1*e12

    coord = tf.reshape (tf.math.atan2 (num, den)*s2d/u_size + u_off, [  -1 ] ) 
    lower = tf.cast (tf.floor (coord), tf.int32) 
    upper = lower + 1
    lower = tf.clip_by_value (lower, 0, nchans-1)
    weights = coord - tf.cast (lower, dtype) 

    lower = tf.stack ([idx, tf.clip_by_value (lower, 0, nchans-1)], axis =1)
    upper = tf.stack([idx, tf.clip_by_value (upper, 0, nchans-1) ], axis =1)

    lim = tf.gather_nd (proj [iv0:iv1, :],  lower )
    uim = tf.gather_nd (proj [iv0:iv1, :],  upper )

    im1 = tf.reshape (lim * (1 - weights) + uim * weights , [ydim, xdim, angles_per_set] ) 
    bim += tf.reduce_sum (im1/den/den , axis = -1)

  bim *= s2o*pi/nreadings 
  bim = tf.cast (bim, dtype = tf.float32)

  return bim

#tf.TensorSpec(shape=None, dtype=tf.int32) 
#@tf.function
@tf.function(
input_signature=[ tf.TensorSpec(shape=None, dtype=tf.float32),  \
tf.TensorSpec(shape=None, dtype=tf.float32),  \
tf.TensorSpec(shape=None, dtype=tf.float32) , \
tf.TensorSpec(shape=None, dtype=tf.int32) ])
def rcn_fbp (proj, x0, y0, apod  ) :   

  proj_f, cosweight = apply_ramp_filter (proj, apod  )  
  rcn =  backprojection_nv  (proj_f, x0, y0, 200, 200 ) 
  return rcn

@tf.function(
input_signature=[ tf.TensorSpec(shape=None, dtype=tf.float32),  \
tf.TensorSpec(shape=None, dtype=tf.float32),  \
tf.TensorSpec(shape=None, dtype=tf.float32) , \
tf.TensorSpec(shape=None, dtype=tf.int32), \
tf.TensorSpec(shape=None, dtype=tf.int32), \
tf.TensorSpec(shape=None, dtype=tf.int32) ])
def rcn_fbp_d (proj, x0, y0, apod, xdim, ydim ) :   

  proj_f, cosweight = apply_ramp_filter (proj, apod )  
  rcn =  backprojection_nv_d  (proj_f, x0, y0, xdim, ydim) 
  return rcn




