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
import argparse
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str, action="store", required=True)
    parser.add_argument("samplename", type=str, action="store", required=True)
    parser.add_argument("volname", type=str, action="store", required=True)
    parser.add_argument("--nbdil", type=int, default=10, action="store")
    parser.add_argument("--nblayer", type=int, default=100, action="store")
    parser.add_argument("--nbchan", type=int, default=5, action="store", help="mettre 5 pour faire du 2.5 D ou 1 pour des slices")
    parser.add_argument("--nblab", type=int, default=1, action="store", help="garder 1 pour un entrainement sur niveaux de gris")
    return parser


def traingrey(username, samplename, volname, nbdil=10, nblayer=100, nbchan=5, nblab=1):
    path_prefix = Path(username, samplename)
    path_prefix_logs = path_prefix / "logs"
    namelogfile = path_prefix_logs / f"log_{volname}.txt"
    nameimagelogfile = path_prefix_logs / f"image_{volname}.txt"
    nametrainfile = path_prefix_logs / f"train_file_{volname}.h5"


    # Define dilations in [1,10] as in paper.
    dilations = msdnet.dilations.IncrementDilations(nbdil)

    # Create main network object for regression, with 100 layers,
    # [1,10] dilations, 5 input channels (5 slices), 1 output channel, using
    # the GPU (set gpu=False to use CPU)
    #n = msdnet.network.MSDNet(100, dilations, 5, 1, gpu=True)

    n = msdnet.network.MSDNet(nblayer, dilations, nbchan, nblab, gpu=True)

    # Initialize network parameters
    n.initialize()

    print('read training files ...')

    # Define training data
    # First, create lists of input files (low quality) and target files (high quality)
    #flsin = sorted((Path('tomo_trainlq') / 'lq').glob('*.tif'))
    #flstg = sorted((Path('tomo_trainhq') / 'hq').glob('*.tif'))

    # Create list of datapoints (i.e. input/target pairs)
    dats = []
    namelow = path_prefix / f"{volname}_trainlq" / "*.tif"
    namehigh = path_prefix / f"{volname}_trainhq" / "*.tif"
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
    namelow2 = path_prefix / f"{volname}_vallq" / "*.tif"
    namehigh2 = path_prefix / f"{volname}_valhq" / "*.tif"
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

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    traingrey(
        username=args.username,
        samplename=args.samplename,
        volname=args.volname,
        nbdil=args.nbdil,
        nblayer=args.nblayer,
        nbchan=args.nbchan,
        nblab=args.nblab
    )