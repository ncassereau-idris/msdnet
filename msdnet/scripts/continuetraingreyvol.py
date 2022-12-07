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
Example 07: Train a network for segmentation (tomography)
=========================================================

This script trains a MS-D network for segmentation (i.e. labeling)
Run generatedata.py first to generate required training data.
"""

# Import code
import msdnet
import tifffile
import numpy as np


def continuetraingreyvol(mydir, prefixlow, prefixhigh):
    namelogfile='%s/logc_%s.txt' %(mydir,prefixlow)
    nameimagelogfile='%s/imagec_%s' %(mydir,prefixlow)
    nameimagelogfilesingle='%s/image_singlec_%s' %(mydir,prefixlow)
    nametrainfile='%s/train_filec_%s.h5' %(mydir,prefixlow)
    nametrainfilecheck='%s/train_file_%s.checkpoint' %(mydir,prefixlow)

    namelow='%s/%s.tif' %(mydir,prefixlow)
    namehigh='%s/%s.tif' %(mydir,prefixhigh)

    # Define dilations in [1,10] as in paper.
    dilations = msdnet.dilations.IncrementDilations(10)

    # Create main network object for segmentation, with 100 layers,
    # [1,10] dilations, 5 input channels (5 slices), 4 output channels (one for each label), 
    # using the GPU (set gpu=False to use CPU)
    n = msdnet.network.MSDNet(100, dilations, 5, 1, gpu=True)

    # Initialize network parameters
    n.initialize()

    # Define training data
    # First, create lists of input files (low quality) and target files (labels)
    arrin = tifffile.imread(namelow)
    arrtg = tifffile.imread(namehigh)
    print(arrin.shape)
    print(arrtg.shape)
    # Create list of datapoints (i.e. input/target pairs)
    dats = []
    for i in range(arrin.shape[0]):
        # Create datapoint with file names
        d = msdnet.data.ArrayDataPoint(arrin[i:i+1].astype(np.float32), arrtg[i:i+1].astype(np.float32))
        # Convert datapoint to one-hot, using labels 0, 1, 2, and 3,
        # which are the labels given in each label TIFF file.
        # d_oh = msdnet.data.OneHotDataPoint(d, [0,255])
        # Add datapoint to list
        dats.append(d)
    # Note: The above can also be achieved using a utility function for such 'simple' cases:
    # dats = msdnet.utils.load_simple_data('tomo_train/lowqual/*.tiff', 'tomo_train/label/*.tiff', augment=False, labels=[0,1,2,3])

    # Convert input slices to input slabs (i.e. multiple slices as input)
    datst = msdnet.data.convert_to_slabs(dats, 2, flip=True)
    datsnf = msdnet.data.convert_to_slabs(dats, 2, flip=False)
    # Augment data by rotating and flipping
    datsv = []
    dats_augm = []
    for i in range(len(dats)):
        if i%10==5:
            datsv.append(datsnf[i])
        else:
            dats_augm.append(msdnet.data.RotateAndFlipDataPoint(datst[i]))
        
    # Normalize input and output of network to zero mean and unit variance using
    # training data images
    n.normalizeinout(datsv)

    # Use image batches of a single image
    bprov = msdnet.data.BatchProvider(dats_augm,1)

    # Load network, training algorithm, and validation object from checkpoint of previous training
    n, t, val = msdnet.train.restore_training(nametrainfilecheck, msdnet.network.MSDNet, msdnet.train.AdamAlgorithm, msdnet.validate.MSEValidation, datsv, gpu=True)

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
    imagelog = msdnet.loggers.ImageLogger(nameimagelogfile, onlyifbetter=True, chan_in=2)

    # Train network until program is stopped manually
    # Network parameters are saved in regr_params.h5
    # Validation is run after every len(datsv) (=256)
    # training steps.
    print('starting training')
    msdnet.train.train(n, t, val, bprov, nametrainfile,loggers=[consolelog,filelog,imagelog], val_every=len(datsv))


if __name__ == "__main__":
    import sys
    #dir name
    mydir = sys.argv[1]
    #prefix name lowquality volume
    prefixlow = sys.argv[2]
    #prefix name high volume
    prefixhigh = sys.argv[3]
    continuetraingreyvol(mydir, prefixlow, prefixhigh)