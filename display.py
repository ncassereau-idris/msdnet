import matplotlib.pyplot as plt
import argparse
import os
import glob
import numpy as np
import functools


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, action="store")
    parser.add_argument("prefix", type=str, action="store")
    return parser


def read_log(directory, prefix):
    filename = os.path.join(directory, f"log_{prefix}.txt")

    with open(filename, "r") as logfile:
        lines = logfile.readlines()
    # Split a line in 3 cols: step, error, best_error. Ditch best error
    lines = [line.split(", ")[:-1] for line in lines]

    steps, errors = zip(*lines)

    # Only extract values
    steps = [int(step.split(": ")[1]) for step in steps]
    errors = [float(err.split(": ")[1]) for err in errors]

    return steps, errors


def get_empty_pic(directory, prefix):
    # Get an image the same size as saved pictures but empty (as a placeholder)
    path = os.path.join(directory, f"image_{prefix}_step_*_best.png")
    first_file = glob.glob(path)[0]
    pic = plt.imread(first_file)
    return np.zeros_like(pic)


@functools.cache
def get_image(directory, prefix, step):
    path = os.path.join(directory, f"image_{prefix}_step_{step:09}_best.png")
    return plt.imread(path)


def main(args):
    directory, prefix = args.directory, args.prefix

    steps, errors = read_log(directory, prefix)
    placeholder = get_empty_pic(directory, prefix)

    fig, (ax_image, ax_loss) = plt.subplots(2, 1)
    imshow = ax_image.imshow(placeholder, vmin=0, vmax=1, cmap='gray')
    ax_image.axis('off')

    ax_loss.plot(steps, errors, marker="", ls="-", color="b")
    line, = ax_loss.plot(steps, errors, marker="o", ls="", color="b", label="Loss")
    ax_loss.legend()

    def hover(event):
        ind = line.contains(event)[1]["ind"]
        if len(ind) == 1:
            index = ind[0]
            correct_step = steps[index]
            im = get_image(directory, prefix, correct_step)
            imshow.set_data(im)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()


if __name__ ==  "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
