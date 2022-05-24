#-----------------------------------------------------------------------
#Copyright 2019 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/msdnet/
#License: MIT
#
#This file is part of MSDNet, a Python implementation of the
#Mixed-Scale Dense Convolutional Neural Network.
#-----------------------------------------------------------------------

"""
Example 05: Train a network for regression (tomography)
=======================================================

This script trains a MS-D network for regression (i.e. denoising/artifact removal)
Run generatedata.py first to generate required training data.
"""

# Import code
import msdnet
from pathlib import Path
import sys

#dir name
mydir = sys.argv[1]

#prefix name
prefix = sys.argv[2]

namelogfile='%s/logs/log_%s.txt' %(mydir,prefix)
nameimagelogfile='%s/logs/image_%s' %(mydir,prefix)
nametrainfile='%s/logs/train_file_%s.h5' %(mydir,prefix)


# Define dilations in [1,10] as in paper.
dilations = msdnet.dilations.IncrementDilations(10)

# Create main network object for regression, with 100 layers,
# [1,10] dilations, 5 input channels (5 slices), 1 output channel, using
# the GPU (set gpu=False to use CPU)
#n = msdnet.network.MSDNet(100, dilations, 5, 1, gpu=True)

n = msdnet.network.MSDNet(100, dilations, 5, 1, gpu=True)

# Initialize network parameters
n.initialize()

print('read training files ...')

# Define training data
# First, create lists of input files (low quality) and target files (high quality)
#flsin = sorted((Path('tomo_trainlq') / 'lq').glob('*.tif'))
#flstg = sorted((Path('tomo_trainhq') / 'hq').glob('*.tif'))

# Create list of datapoints (i.e. input/target pairs)
dats = []
namelow='%s/%s_trainlq/*.tif' %(mydir,prefix)
namehigh='%s/%s_trainhq/*.tif' %(mydir,prefix)
dats = msdnet.utils.load_simple_data(namelow,namehigh, augment=False)

# Convert input slices to input slabs (i.e. multiple slices as input)
dats = msdnet.data.convert_to_slabs(dats, 2, flip=True)
# Augment data by rotating and flipping
dats_augm = [msdnet.data.RotateAndFlipDataPoint(d) for d in dats]
    
# Normalize input and output of network to zero mean and unit variance using
# training data images
n.normalizeinout(dats)

# Use image batches of a single image
bprov = msdnet.data.BatchProvider(dats,1)

print('read validation files ...')

# Define validation data (not using augmentation)
namelow2='%s/%s_vallq/*.tif' %(mydir,prefix)
namehigh2='%s/%s_valhq/*.tif' %(mydir,prefix)
datsv = msdnet.utils.load_simple_data(namelow2,namehigh2, augment=False)

# Convert input slices to input slabs (i.e. multiple slices as input)
datsv = msdnet.data.convert_to_slabs(datsv, 2, flip=False)

# Select loss function
l2loss = msdnet.loss.L2Loss()

# Validate with loss function
val = msdnet.validate.LossValidation(datsv, loss=l2loss)

# Use ADAM training algorithms
t = msdnet.train.AdamAlgorithm(n, loss=l2loss)

# Log error metrics to console
consolelog = msdnet.loggers.ConsoleLogger()
# Log error metrics to file
filelog = msdnet.loggers.FileLogger(namelogfile)
# Log typical, worst, and best images to image files
imagelog = msdnet.loggers.ImageLogger(nameimagelogfile, onlyifbetter=False, period=1, chan_in=2)

# Train network until program is stopped manually
# Network parameters are saved in regr_params.h5
# Validation is run after every len(datsv) (=256)
# training steps.
print('starting training')
msdnet.train.train(n, t, val, bprov, nametrainfile,loggers=[consolelog,filelog,imagelog], val_every=len(datsv))
