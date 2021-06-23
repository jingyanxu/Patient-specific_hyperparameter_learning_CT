tested under tf2.?
model compiling may take some time 

********************************************************************************************
utils_tfds.py is to use tensorflow dataset  for manipulating training/testing samples/labels
other file names should be self explanatory 

********************************************************************************************
copy all files to a directory: call it "loc", maintain the directory structure
then cd to directory "loc" and run the following : 

python ./run_test.py 0

the included screenshot shows the output of running the code as is, with 3 images
the reconstruction using the proposed method (with the pretrained model), 
the label
and a 0/1 mask. In this particular case, the mask is an image of all 1 (hence no contrast). 

********************************************************************************************
the included data consist of one noise-free reconstruction (label) and one noisy projection data.
method to read the image files should be apparent from utils_tfds.py
  for both files, there is a custom header of size 8369 bytes; the remainder is the float32 data; 
  each data file size is 8369 + 4 (float32) * (data elements in image) + 1   (bytes)
