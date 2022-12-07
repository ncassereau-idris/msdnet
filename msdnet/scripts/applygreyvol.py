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

import msdnet
from pathlib import Path
import imageio
import numpy as np
import tifffile
import tqdm
import argparse
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str, action="store", required=True)
    parser.add_argument("samplename", type=str, action="store", required=True)
    parser.add_argument("lowvolname", type=str, action="store", required=True)
    return parser


def applygreyvol(username, samplename, lowvolname):
    path_prefix = Path(username, samplename)
    nameh5 = path_prefix / f"train_file_{lowvolname}.h5"

    namelow = path_prefix / f"{lowvolname}.tif"
    nameout = path_prefix / f"IA_{lowvolname}.tif"

    arrin = tifffile.imread(namelow)
    dats = []
    for i in range(arrin.shape[0]):
        d = msdnet.data.ArrayDataPoint(arrin[i:i+1].astype(np.float32))
        dats.append(d)
    datsnf = msdnet.data.convert_to_slabs(dats, 2, flip=False)

    # Load network from file
    n = msdnet.network.MSDNet.from_file(nameh5, gpu=True)

    out = np.zeros(arrin.shape, dtype=np.uint16)

    for i in tqdm.trange(len(datsnf)):
        # Compute network output
        output = n.forward(datsnf[i].input)
        out[i] = output[0]

    tifffile.imsave(nameout, np.float32(out))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    applygreyvol(
        username=args.username,
        samplename=args.samplename,
        lowvolname=args.lowvolname
    )
