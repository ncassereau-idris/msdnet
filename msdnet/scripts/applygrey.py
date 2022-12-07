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
import argparse
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str, action="store", required=True)
    parser.add_argument("samplename", type=str, action="store", required=True)
    parser.add_argument("volname", type=str, action="store", required=True)
    return parser


def applygrey(username, samplename, volname):
    path_prefix = Path(username, samplename)
    nameh5 = path_prefix / f"train_file_{volname}.h5"


    # Make folder for output
    outfolder = path_prefix / f"volIA_{volname}"
    outfolder.mkdir(exist_ok=True)

    # Load network from file
    n = msdnet.network.MSDNet.from_file(nameh5, gpu=True)

    # Process all test images
    flsin = sorted((path_prefix / f"{volname}_apply").glob('*.tif'))

    dats = [msdnet.data.ImageFileDataPoint(str(f)) for f in flsin]
    # Convert input slices to input slabs (i.e. multiple slices as input)
    dats = msdnet.data.convert_to_slabs(dats, 2, flip=False)
    for i in tqdm.trange(len(flsin)):
        # Compute network output
        output = n.forward(dats[i].input)
        # Save network output to file
        namesave='IA_%s_%04d.tiff' %(volname,i)
        imageio.imsave(outfolder / namesave , output[0])

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    applygrey(
        username=args.username,
        samplename=args.samplename,
        volname=args.volname
    )
