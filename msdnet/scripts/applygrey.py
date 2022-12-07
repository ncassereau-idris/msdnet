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
Example 06: Apply trained network for regression (tomography)
=============================================================

This script applies a trained MS-D network for regression (i.e. denoising/artifact removal)
Run generatedata_tomography.py first to generate required training data and train_regr_tomography.py to train
a network.
"""

# Import code
import msdnet
from pathlib import Path
import imageio
import tqdm


def applygrey(mydir, prefix):
    nameh5='%s/train_file_%s.h5' %(mydir,prefix)


    # Make folder for output
    outfolder = Path('%s/volIA_%s' %(mydir,prefix))
    outfolder.mkdir(exist_ok=True)

    # Load network from file
    n = msdnet.network.MSDNet.from_file(nameh5, gpu=True)

    # Process all test images
    flsin = sorted((Path('%s/%s_apply' %(mydir,prefix))).glob('*.tif'))

    dats = [msdnet.data.ImageFileDataPoint(str(f)) for f in flsin]
    # Convert input slices to input slabs (i.e. multiple slices as input)
    dats = msdnet.data.convert_to_slabs(dats, 2, flip=False)
    for i in tqdm.trange(len(flsin)):
        # Compute network output
        output = n.forward(dats[i].input)
        # Save network output to file
        namesave='IA_%s_%04d.tiff' %(prefix,i)
        imageio.imsave(outfolder / namesave , output[0])

if __name__ == "__main__":
    import sys

    #dir name
    mydir = sys.argv[1]
    #prefix name
    prefix = sys.argv[2]
    applygrey(mydir, prefix)