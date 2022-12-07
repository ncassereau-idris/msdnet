import numpy as np
from skimage import io
import os
import argparse
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str, action="store", required=True)
    parser.add_argument("samplename", type=str, action="store", required=True)
    parser.add_argument("volname", type=str, action="store", required=True)
    return parser

def prepare_train(username, samplename, volname):
    path_prefix = Path(username, samplename)
    # name of the high quality volume in 3D stack file
    namehq = path_prefix / "hq.tif"

    # name of the low quality volume in 3D stack file
    namelq = path_prefix / "lq.tif"

    print('creating folders ...')
    # create folder to put high quality data as slices
    os.mkdir(path_prefix / f"{volname}_trainhq")   # for training high quality data
    os.mkdir(path_prefix / f"{volname}_valhq")	   # for validation high quality data
    os.mkdir(path_prefix / f"{volname}_trainlq")   # for training low quality data
    os.mkdir(path_prefix / f"{volname}_vallq")	   # for validation low quality data
    os.mkdir(path_prefix / f"{volname}_apply")	   # for applying training

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
        name = path_prefix / f"{volname}_trainhq" / f"hq_{i:0.4d}.tif"
        io.imsave(name,imhq[i,:,:])

    # make slice for low quality volume for training (keep last 10% file for validation)
    for i in range (0,np.size(imlq,0)-value):
        name = path_prefix / f"{volname}_trainlq" / f"lq_{i:0.4d}.tif"
        io.imsave(name,imlq[i,:,:])

    # make slice for high quality volume for validation 
    for i in range (np.size(imhq,0)-value,np.size(imhq,0)):
        name = path_prefix / f"{volname}_valhq" / f"hq_{i:0.4d}.tif"
        io.imsave(name,imhq[i,:,:])

    # make slice for low quality volume for validation 
    for i in range (np.size(imlq,0)-value,np.size(imlq,0)):
        name = path_prefix / f"{volname}_vallq" / f"lq_{i:0.4d}.tif"
        io.imsave(name,imlq[i,:,:])

    # make slice to apply training     
    for i in range (0,np.size(imlq,0)):
        name = path_prefix / f"{volname}_apply" / f"lq_{i:0.4d}.tif"
        io.imsave(name,imlq[i,:,:])
     
if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    prepare_train(
        username=args.username,
        samplename=args.samplename,
        volname=args.volname
    )