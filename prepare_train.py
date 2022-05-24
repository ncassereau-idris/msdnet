import numpy as np
from skimage import io
import os
import sys


#dir name
mydir = sys.argv[1]

#prefix name
prefix = sys.argv[2]

# name of the high quality volume in 3D stack file
namehq='%s/hq.tif' %mydir

# name of the low quality volume in 3D stack file
namelq='%s/lq.tif' %mydir

print('creating folders ...')
# create folder to put high quality data as slices
os.mkdir('%s/%s_trainhq/' %(mydir,prefix))   # for training high quality data
os.mkdir('%s/%s_valhq/' %(mydir,prefix))	    # for validation high quality data
os.mkdir('%s/%s_trainlq/' %(mydir,prefix))   # for training low quality data
os.mkdir('%s/%s_vallq/' %(mydir,prefix))	    # for validation low quality data
os.mkdir('%s/%s_apply/' %(mydir,prefix))	    # for applying training

print('read volumes ...')
# read the high quality volume 
imhq=io.imread(namehq)

# read the low quality volume 
imlq=io.imread(namelq)

print('make slicing ...')
# find how many slices to keep for validation 10% 
value = int (np.size(imhq,0)*0.1)

# make slice for high quality volume for training (keep last 10% file for validation)
for i in range (0,np.size(imhq,0)-value):
     name='%s/%s_trainhq/hq_%0.4d.tif' %(mydir,prefix,i)
     io.imsave(name,imhq[i,:,:])

# make slice for low quality volume for training (keep last 10% file for validation)
for i in range (0,np.size(imlq,0)-value):
     name='%s/%s_trainlq/lq_%0.4d.tif' %(mydir,prefix,i)
     io.imsave(name,imlq[i,:,:])

# make slice for high quality volume for validation 
for i in range (np.size(imhq,0)-value,np.size(imhq,0)):
     name='%s/%s_valhq/hq_%0.4d.tif' %(mydir,prefix,i)
     io.imsave(name,imhq[i,:,:])

# make slice for low quality volume for validation 
for i in range (np.size(imlq,0)-value,np.size(imlq,0)):
     name='%s/%s_vallq/lq_%0.4d.tif' %(mydir,prefix,i)
     io.imsave(name,imlq[i,:,:])

# make slice to apply training     
for i in range (0,np.size(imlq,0)):
     name='%s/%s_apply/lq_%0.4d.tif' %(mydir,prefix,i)
     io.imsave(name,imlq[i,:,:])
    
