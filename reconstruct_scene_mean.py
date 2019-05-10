import numpy as np
import pandas as pd

import os, sys, argparse

from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle

import modules.config as cfg
import modules.metrics as metrics

from joblib import dump, load

from PIL import Image

def reconstruct(_scene_name):
    
    # construct the empty output image
    output_image = np.empty([cfg.number_of_rows, cfg.number_of_columns])

    # load scene and its `n` first pixel value data
    scene_path = os.path.join(cfg.folder_scenes_path, _scene_name)

    for id_column in range(cfg.number_of_columns):

        folder_path = os.path.join(scene_path, str(id_column))

        for id_row in range(cfg.number_of_rows):
            
            pixel_filename = _scene_name + '_' + str(id_column) + '_' + str(id_row) + ".dat"
            pixel_file_path = os.path.join(folder_path, pixel_filename)
            
            with open(pixel_file_path, 'r') as f:

                # predict the expected pixel value
                lines = [float(l) for l in f.readlines()]
                mean = sum(lines) / float(len(lines))

            output_image[id_row, id_column] = mean

        print("{0:.2f}%".format(id_column / cfg.number_of_columns * 100))
        sys.stdout.write("\033[F")

    return output_image

def main():

    parser = argparse.ArgumentParser(description="Train model and saved it")

    parser.add_argument('--scene', type=str, help='Scene name to reconstruct', choices=cfg.scenes_list)
    parser.add_argument('--image_name', type=str, help="The ouput image name")

    args = parser.parse_args()

    param_scene_name = args.scene
    param_image_name = args.image_name

    output_image = reconstruct(param_scene_name)

    if not os.path.exists(cfg.reconstructed_folder):
        os.makedirs(cfg.reconstructed_folder)

    image_path = os.path.join(cfg.reconstructed_folder, param_image_name)

    img = Image.fromarray(np.uint8(output_image))
    img.save(image_path)

if __name__== "__main__":
    main()
