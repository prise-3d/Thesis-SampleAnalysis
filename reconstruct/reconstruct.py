# main imports
import numpy as np
import pandas as pd
import os, sys, argparse

# models imports
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
from joblib import dump, load

# image processing imports
from ipfml import metrics
from PIL import Image

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from features import compute_feature

def reconstruct(_scene_name, _model_path, _n, _feature_choice):
    
    # construct the empty output image
    output_image = np.empty([cfg.number_of_rows, cfg.number_of_columns])

    # load the trained model
    clf = load(_model_path)

    # load scene and its `n` first pixel value data
    scene_path = os.path.join(cfg.dataset_path, _scene_name)

    for id_column in range(cfg.number_of_columns):

        folder_path = os.path.join(scene_path, str(id_column))

        pixels = []

        for id_row in range(cfg.number_of_rows):
            
            pixel_filename = _scene_name + '_' + str(id_column) + '_' + str(id_row) + ".dat"
            pixel_file_path = os.path.join(folder_path, pixel_filename)
            
            with open(pixel_file_path, 'r') as f:

                # predict the expected pixel value
                lines = [float(l)/255. for l in f.readlines()]
                pixel_values = lines[0:int(_n)]

                data = compute_feature(_feature_choice, pixel_values)

                pixels.append(data)

        # predict column pixels and fill image column by column
        pixels_predicted = clf.predict(pixels)

        # change normalized predicted value to pixel value
        pixels_predicted = pixels_predicted*255.

        for id_pixel, pixel in enumerate(pixels_predicted):
            output_image[id_pixel, id_column] = pixel

        print("{0:.2f}%".format(id_column / cfg.number_of_columns * 100))
        sys.stdout.write("\033[F")

    return output_image

def main():

    parser = argparse.ArgumentParser(description="Train model and saved it")

    parser.add_argument('--scene', type=str, help='Scene name to reconstruct', choices=cfg.scenes_list)
    parser.add_argument('--model_path', type=str, help='Model file path')
    parser.add_argument('--n', type=str, help='Number of pixel values approximated to keep')
    parser.add_argument('--feature', type=str, help='Feature choice to compute from samples', choices=cfg.features_list)
    parser.add_argument('--image_name', type=str, help="The ouput image name")

    args = parser.parse_args()

    param_scene_name = args.scene
    param_n          = args.n
    param_feature    = args.feature
    param_model_path = args.model_path
    param_image_name = args.image_name

    # get default value of `n` param
    if not param_n:
        param_n = param_model_path.split('_')[0]

    output_image = reconstruct(param_scene_name, param_model_path, param_n, param_feature)

    if not os.path.exists(cfg.reconstructed_folder):
        os.makedirs(cfg.reconstructed_folder)

    image_path = os.path.join(cfg.reconstructed_folder, param_image_name)

    img = Image.fromarray(np.uint8(output_image))
    img.save(image_path)

if __name__== "__main__":
    main()
